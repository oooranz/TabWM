"""Prediction heads."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

NEG_INF = -1e9

MAX_CATEGORIES = 16


class PointerHead(nn.Module):

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(
        self, slots: torch.Tensor, tokens: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        logits = torch.einsum("bnd,bsd->bns", self.q(slots), self.k(tokens)) * self.scale
        mask = candidates[:, None, :]
        empty = ~candidates.any(dim=1)
        if empty.any():
            mask = mask.clone()
            mask[empty, :, 0] = True
        return logits.masked_fill(~mask, NEG_INF)


class TypedValueHead(nn.Module):

    def __init__(self, d_model: int, max_categories: int = MAX_CATEGORIES) -> None:
        super().__init__()
        self.max_categories = max_categories
        self.numeric = nn.Linear(d_model, 1)
        self.categorical = nn.Linear(d_model, max_categories)
        self.boolean = nn.Linear(d_model, 1)

    def forward(self, slots: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "numeric": self.numeric(slots).squeeze(-1),
            "categorical": self.categorical(slots),
            "boolean": self.boolean(slots).squeeze(-1),
        }


def select_offset(offsets: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    if offsets.shape[-1] == 1:
        return offsets.squeeze(-1)
    return offsets.gather(-1, idx[..., None]).squeeze(-1)


class BinnedNumericHead(nn.Module):

    def __init__(self, d_model: int, n_side: int = 32, lo: float = 0.015,
                 hi: float = 8.0, shared_offset: bool = True) -> None:
        super().__init__()
        self.n_side = n_side
        self.n_bins = 2 * n_side + 1
        self.bin_logits = nn.Linear(d_model, self.n_bins)


        self.offsets = nn.Linear(d_model, 1 if shared_offset else self.n_bins)

        edges = torch.logspace(math.log10(lo), math.log10(hi), n_side + 1)
        centers = torch.sqrt(edges[:-1] * edges[1:])
        widths = edges[1:] - edges[:-1]
        self.register_buffer("edges", edges)
        self.register_buffer(
            "centers", torch.cat([-centers.flip(0), torch.zeros(1), centers]))
        self.register_buffer(
            "widths", torch.cat([widths.flip(0), torch.full((1,), 2 * lo), widths]))

    def target_bin(self, delta: torch.Tensor) -> torch.Tensor:
        mag = delta.abs()


        idx = torch.searchsorted(self.edges.contiguous(), mag.contiguous().detach(),
                                 right=True) - 1
        idx = idx.clamp(0, self.n_side - 1)
        inside = mag >= self.edges[0]
        signed = torch.where(delta < 0, self.n_side - 1 - idx, self.n_side + 1 + idx)
        return torch.where(inside, signed, torch.full_like(signed, self.n_side))

    def target_offset(self, delta: torch.Tensor, tgt_bin: torch.Tensor) -> torch.Tensor:
        c = self.centers.to(delta.dtype)[tgt_bin]
        w = self.widths.to(delta.dtype)[tgt_bin].clamp(min=1e-8)
        return ((delta - c) / (0.5 * w)).clamp(-1.0, 1.0)

    def target_dist(self, delta: torch.Tensor, sigma_scale: float = 0.75
                    ) -> torch.Tensor:
        signed = torch.cat([-self.edges.flip(0), self.edges]).to(delta.dtype)
        tb = self.target_bin(delta)
        sigma = (sigma_scale * self.widths.to(delta.dtype)[tb]).clamp(min=1e-8)


        z = (signed.expand(*delta.shape, -1) - delta[..., None]) / sigma[..., None]
        z = z.clone()
        z[..., 0] = -float("inf")
        z[..., -1] = float("inf")
        cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
        p = (cdf[..., 1:] - cdf[..., :-1]).clamp(min=0)
        return p / p.sum(-1, keepdim=True).clamp(min=1e-12)

    def decode(self, logits: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        idx = logits.argmax(dim=-1)
        off = torch.tanh(select_offset(offsets, idx))
        c = self.centers.to(logits.dtype)[idx]
        w = self.widths.to(logits.dtype)[idx]
        return c + off * 0.5 * w

    def forward(self, slots: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"bin_logits": self.bin_logits(slots),
                "bin_offsets": self.offsets(slots)}


def pad_class_mask(n_classes: torch.Tensor, max_categories: int = MAX_CATEGORIES) -> torch.Tensor:
    ar = torch.arange(max_categories, device=n_classes.device)
    return ar.expand(*n_classes.shape, max_categories) < n_classes.clamp(min=1)[..., None]


class AuxHeads(nn.Module):

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.reward = nn.Linear(d_model, 1)
        self.applicable = nn.Linear(d_model, 1)
        self.n_edits = nn.Linear(d_model, 1)

    def forward(self, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "reward": self.reward(pooled).squeeze(-1),
            "applicable": self.applicable(pooled).squeeze(-1),
            "n_edits": self.n_edits(pooled).squeeze(-1),
        }


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask[..., None].to(x.dtype)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def gather_pointer_targets(
    logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    B, N, S = logits.shape
    flat_logits = logits.reshape(B * N, S)
    flat_targets = targets.reshape(B * N).clone()
    keep = valid.reshape(B * N) & (flat_targets >= 0)
    flat_targets[~keep] = 0
    loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    return (loss * keep).sum() / keep.sum().clamp(min=1)
