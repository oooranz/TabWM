"""Numeric value prediction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from tabwm.data.collate import Role, SEM_NUMBER, SEM_TEXT
from tabwm.data.delta import ColumnStats
from tabwm.model.delta_decoder import DecoderOutput, _CrossBlock


def visible_pairs(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    names = (
        "role_ids", "transition_ids", "node_idxs", "table_name_idxs",
        "col_name_idxs", "sem_types", "is_padding", "masks",
    )
    arrays = {name: batch[name].detach().cpu().numpy() for name in names}
    rows = []
    for b in range(len(arrays["role_ids"])):
        before = {}
        def key(i):
            return tuple(int(arrays[name][b, i]) for name in names[1:5])
        visible = ~(arrays["is_padding"][b] | arrays["masks"][b])
        for i in np.flatnonzero(visible & (arrays["sem_types"][b] == SEM_NUMBER)):
            if arrays["role_ids"][b, i] == Role.STATE_BEFORE and arrays["transition_ids"][b, i] > 0:
                before[key(i)] = i
        pairs = []
        for i in np.flatnonzero(visible & (arrays["sem_types"][b] == SEM_NUMBER)):
            if arrays["role_ids"][b, i] == Role.STATE_DELTA_DEMO and key(i) in before:
                pairs.append((before[key(i)], i))
        rows.append(pairs)
    count = max(1, max(map(len, rows), default=0))
    positions = np.full((len(rows), count, 2), -1, dtype=np.int64)
    for b, pairs in enumerate(rows):
        if pairs:
            positions[b, :len(pairs)] = pairs
    pos = torch.as_tensor(positions, device=batch["role_ids"].device)
    return pos[..., 0], pos[..., 1]


def gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return x[torch.arange(x.shape[0], device=x.device)[:, None], idx.clamp_min(0)]


@dataclass(frozen=True)
class ValueDecoderConfig:
    width: int = 256
    layers: int = 2
    heads: int = 8
    transformation_proposals: bool = True


class ContextualValueDecoder(nn.Module):

    def __init__(self, d_model: int, d_text: int, config: ValueDecoderConfig):
        super().__init__()
        width = config.width
        self.config = config
        self.action = nn.Linear(d_model + d_text + 1, width)
        self.query = nn.Linear(2 * d_model + 3 + width, width)
        self.evidence = nn.Linear(2 * d_model + 3 + width, width)
        self.blocks = nn.ModuleList(
            [_CrossBlock(width, config.heads, 4 * width) for _ in range(config.layers)]
        )
        self.null = nn.Parameter(torch.zeros(1, 1, width))
        self.out = nn.Linear(width, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        if config.transformation_proposals:
            self.proposal_query = nn.Linear(width, width, bias=False)
            self.proposal_key = nn.Linear(width, width, bias=False)
            self.proposal_gate = nn.Linear(width, 2)
            nn.init.zeros_(self.proposal_gate.weight)
            nn.init.zeros_(self.proposal_gate.bias)

    def forward(
        self,
        hidden,
        base: DecoderOutput,
        batch,
        *,
        target_positions: torch.Tensor | None = None,
        pair_positions=None,
    ):

        hidden = hidden.float()
        slots = base.slots.float()
        numbers = batch["number_values_exact"].squeeze(-1).float()
        before, after = (
            pair_positions if pair_positions is not None else visible_pairs(batch)
        )
        valid = (before >= 0) & (after >= 0)

        predicted_target = base.cell_logits.argmax(-1)
        target = predicted_target
        if target_positions is not None:
            target_positions = target_positions.to(device=predicted_target.device)
            target = torch.where(target_positions >= 0, target_positions, predicted_target)
        current = gather(numbers, target)
        initial = base.value_numeric.float()

        transition_ids = batch["transition_ids"].clamp_min(0)
        n_transitions = int(transition_ids.max().item()) + 1
        action_mask = (
            (batch["role_ids"] == Role.ACTION)
            & ~batch["is_padding"]
            & ~batch["masks"]
        )
        text = batch["text_values"].float()
        text = text * (batch["sem_types"] == SEM_TEXT)[..., None]
        numeric = numbers * (batch["sem_types"] == SEM_NUMBER)
        action_features = torch.cat((hidden, text, numeric[..., None]), -1)
        action_features = action_features * action_mask[..., None]
        pooled = action_features.new_zeros(
            (action_features.shape[0], n_transitions, action_features.shape[-1])
        )
        pooled.scatter_add_(
            1, transition_ids[..., None].expand_as(action_features), action_features
        )
        counts = action_features.new_zeros((action_features.shape[0], n_transitions))
        counts.scatter_add_(1, transition_ids, action_mask.float())
        actions = self.action(pooled / counts.clamp_min(1)[..., None])
        query_action = actions[:, :1].expand(-1, target.shape[1], -1)
        demo_action = gather(actions, gather(transition_ids, after))

        query = self.query(
            torch.cat(
                (
                    slots,
                    gather(hidden, target),
                    current[..., None],
                    initial[..., None],
                    (initial - current)[..., None],
                    query_action,
                ),
                -1,
            )
        )
        before_value, after_value = gather(numbers, before), gather(numbers, after)
        evidence = self.evidence(
            torch.cat(
                (
                    gather(hidden, before),
                    gather(hidden, after),
                    before_value[..., None],
                    after_value[..., None],
                    (after_value - before_value)[..., None],
                    demo_action,
                ),
                -1,
            )
        )
        evidence = torch.cat(
            (self.null.expand(evidence.shape[0], -1, -1), evidence), 1
        )
        padding = torch.cat((torch.zeros_like(valid[:, :1]), ~valid), 1)
        for block in self.blocks:
            query = block(query, evidence, padding)

        eligible = gather(batch["sem_types"], target) == SEM_NUMBER
        eligible &= gather(batch["role_ids"], target) == Role.STATE_QUERY
        eligible &= ~gather(batch["is_padding"] | batch["masks"], target)
        eligible &= valid.any(1)[:, None]

        correction = self.out(query).squeeze(-1)
        if self.config.transformation_proposals:
            same_column = valid[:, None, :].expand(-1, target.shape[1], -1).clone()
            for name in ("table_name_idxs", "col_name_idxs"):
                same_column &= (
                    gather(batch[name], target)[..., None]
                    == gather(batch[name], after)[:, None, :]
                )
            same_column &= (
                (before_value.abs() < ColumnStats.MAX_SIGMA)
                & (after_value.abs() < ColumnStats.MAX_SIGMA)
            )[:, None, :]
            same_column &= (current.abs() < ColumnStats.MAX_SIGMA)[..., None]

            logits = torch.einsum(
                "bnw,bpw->bnp",
                self.proposal_query(query),
                self.proposal_key(evidence[:, 1:]),
            ) / self.config.width**0.5
            weights = logits.masked_fill(~same_column, -1e4).softmax(-1)
            weights = weights * same_column
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
            shift_value, linear_value, linear_ok = transformation_candidates(
                before_value, after_value, current, weights
            )
            incumbent = initial + correction
            gates = self.proposal_gate(query).tanh()
            proposal = gates[..., 0] * (shift_value - incumbent)
            proposal += gates[..., 1] * (linear_value - incumbent) * linear_ok
            correction = correction + proposal * same_column.any(-1)

        return correction * eligible


def transformation_candidates(before, after, current, weights):

    before = before[:, None, :]
    after = after[:, None, :]
    mean_before = (weights * before).sum(-1)
    mean_after = (weights * after).sum(-1)
    centered_before = before - mean_before[..., None]
    centered_after = after - mean_after[..., None]
    variance = (weights * centered_before.square()).sum(-1)
    covariance = (weights * centered_before * centered_after).sum(-1)
    slope = (covariance / variance.clamp_min(1e-4)).clamp(-20.0, 20.0)
    shift_value = current + (weights * (after - before)).sum(-1)
    linear_value = mean_after + slope * (current - mean_before)
    linear_ok = (variance > 1e-4) & ((weights > 0).sum(-1) >= 2)
    return shift_value, linear_value, linear_ok
