"""Relational Transformer encoder."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from tabwm.model.role_embeddings import RoleTransitionEmbedding


_flex_compiled = None


def _flex(q, k, v, block_mask):
    global _flex_compiled
    if _flex_compiled is None:
        _flex_compiled = torch.compile(flex_attention)
    return _flex_compiled(q, k, v, block_mask=block_mask)

SEM_TYPE_ORDER = ("number", "text", "datetime", "boolean")


@dataclass
class RTConfig:

    num_blocks: int = 12
    d_model: int = 256
    d_text: int = 384
    num_heads: int = 8
    d_ff: int = 1024
    embedding_model: str = "all-MiniLM-L12-v2"

    @classmethod
    def from_json(cls, path) -> RTConfig:
        import json
        from pathlib import Path

        raw = json.loads(Path(path).read_text())
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})


class MaskedAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, block_mask=None, bool_mask=None):
        q = rearrange(self.wq(x), "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(self.wk(x), "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(self.wv(x), "b s (h d) -> b h s d", h=self.num_heads)
        q, k = self.q_norm(q), self.k_norm(k)

        if block_mask is not None:
            x = _flex(q, k, v, block_mask)
        elif bool_mask is not None:


            m = bool_mask.unsqueeze(1)
            eye = torch.eye(m.shape[-1], dtype=torch.bool, device=m.device)
            empty = ~m.any(-1, keepdim=True)
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=m | (empty & eye))
            x = x.masked_fill(empty, 0.0)
        else:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                x = F.scaled_dot_product_attention(q, k, v)

        return self.wo(rearrange(x, "b h s d -> b s (h d)"))


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RelationalBlock(nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int) -> None:
        super().__init__()
        self.attn_types = ["col", "feat", "nbr"]
        self.norms = nn.ModuleDict({l: nn.RMSNorm(d_model) for l in [*self.attn_types, "ffn"]})
        self.attns = nn.ModuleDict(
            {l: MaskedAttention(d_model, num_heads) for l in self.attn_types}
        )
        self.ffn = FFN(d_model, d_ff)

    def forward(self, x, block_masks=None, bool_masks=None):
        for name in self.attn_types:
            x = x + self.attns[name](
                self.norms[name](x),
                block_mask=None if block_masks is None else block_masks[name],
                bool_mask=None if bool_masks is None else bool_masks[name],
            )
        return x + self.ffn(self.norms["ffn"](x))


def _make_block_mask(mask, batch_size, seq_len, device):
    def _mod(b, h, q_idx, kv_idx):
        return mask[b, q_idx, kv_idx]

    return create_block_mask(
        mask_mod=_mod, B=batch_size, H=None, Q_LEN=seq_len, KV_LEN=seq_len,
        device=device, _compile=True,
    )


def relational_masks(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    node_idxs = batch["node_idxs"]
    f2p = batch["f2p_nbr_idxs"]
    is_padding = batch["is_padding"]

    pad = (~is_padding[:, :, None]) & (~is_padding[:, None, :])
    same_node = node_idxs[:, :, None] == node_idxs[:, None, :]
    kv_in_f2p = (node_idxs[:, None, :, None] == f2p[:, :, None, :]).any(-1)
    q_in_f2p = (node_idxs[:, :, None, None] == f2p[:, None, :, :]).any(-1)
    same_col_table = (batch["col_name_idxs"][:, :, None] == batch["col_name_idxs"][:, None, :]) & (
        batch["table_name_idxs"][:, :, None] == batch["table_name_idxs"][:, None, :]
    )
    return {
        "feat": ((same_node | kv_in_f2p) & pad).contiguous(),
        "nbr": (q_in_f2p & pad).contiguous(),
        "col": (same_col_table & pad).contiguous(),
    }


class TabWMEncoder(nn.Module):

    def __init__(self, config: RTConfig) -> None:
        super().__init__()
        d_model, d_text = config.d_model, config.d_text
        self.config = config

        self.enc_dict = nn.ModuleDict(
            {
                "number": nn.Linear(1, d_model, bias=True),
                "text": nn.Linear(d_text, d_model, bias=True),
                "datetime": nn.Linear(1, d_model, bias=True),
                "col_name": nn.Linear(d_text, d_model, bias=True),
                "boolean": nn.Linear(1, d_model, bias=True),
            }
        )
        self.dec_dict = nn.ModuleDict(
            {
                "number": nn.Linear(d_model, 1, bias=True),
                "text": nn.Linear(d_model, d_text, bias=True),
                "datetime": nn.Linear(d_model, 1, bias=True),
                "boolean": nn.Linear(d_model, 1, bias=True),
            }
        )
        self.norm_dict = nn.ModuleDict(
            {
                t: nn.RMSNorm(d_model)
                for t in ("number", "text", "datetime", "col_name", "boolean")
            }
        )
        self.mask_embs = nn.ParameterDict(
            {t: nn.Parameter(torch.randn(d_model)) for t in SEM_TYPE_ORDER}
        )
        self.blocks = nn.ModuleList(
            [RelationalBlock(d_model, config.num_heads, config.d_ff) for _ in range(config.num_blocks)]
        )
        self.norm_out = nn.RMSNorm(d_model)


        self.role_transition = RoleTransitionEmbedding(d_model)
        self.d_model = d_model


    def embed(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        is_padding = batch["is_padding"]
        keep = (~is_padding)[..., None]
        x = self.norm_dict["col_name"](self.enc_dict["col_name"](batch["col_name_values"])) * keep

        sem_types = batch["sem_types"]
        masks = batch.get("masks")
        for i, t in enumerate(SEM_TYPE_ORDER):
            is_t = (sem_types == i) & ~is_padding
            if masks is not None:
                unmasked, masked = is_t & ~masks, is_t & masks
            else:
                unmasked, masked = is_t, None
            x = x + self.norm_dict[t](self.enc_dict[t](batch[f"{t}_values"])) * unmasked[..., None]
            if masked is not None:
                x = x + self.mask_embs[t] * masked[..., None]

        return self.role_transition(x, batch["role_ids"], batch["transition_ids"])


    def forward(
        self, batch: dict[str, torch.Tensor], *, use_flex: bool | None = None
    ) -> torch.Tensor:
        x = self.embed(batch)
        masks = relational_masks(batch)
        B, S = batch["node_idxs"].shape
        device = x.device

        if use_flex is None:
            use_flex = device.type == "cuda"

        if use_flex:
            make = partial(_make_block_mask, batch_size=B, seq_len=S, device=device)
            block_masks = {k: make(v) for k, v in masks.items()}
            for block in self.blocks:
                x = block(x, block_masks=block_masks)
        else:
            for block in self.blocks:
                x = block(x, bool_masks=masks)

        return self.norm_out(x)


def load_rt_checkpoint(
    encoder: TabWMEncoder, path, *, strict_backbone: bool = True
) -> dict[str, int]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state and not any(
        key.endswith(".weight") for key in state
    ):
        state = state["model"]

    result = encoder.load_state_dict(state, strict=False)
    unexpected = list(result.unexpected_keys)
    missing = [
        key
        for key in result.missing_keys
        if not key.startswith("role_transition.")
    ]
    if strict_backbone and (unexpected or missing):
        raise ValueError(
            f"checkpoint does not match the backbone: {len(missing)} missing "
            f"{missing[:4]}, {len(unexpected)} unexpected {unexpected[:4]}"
        )
    return {
        "loaded": len(state),
        "missing_backbone": len(missing),
        "unexpected": len(unexpected),
        "randomly_initialised": len(
            [
                key
                for key in result.missing_keys
                if key.startswith("role_transition.")
            ]
        ),
    }
