"""Structured edit decoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from tabwm.data.collate import Role
from tabwm.data.delta import N_OPS
from tabwm.model.heads import (
    BinnedNumericHead,
    MAX_CATEGORIES,
    AuxHeads,
    PointerHead,
    TypedValueHead,
    masked_mean,
)


@dataclass
class DecoderOutput:

    op_logits: torch.Tensor
    exist_logits: torch.Tensor
    cell_logits: torch.Tensor
    row_logits: torch.Tensor
    table_logits: torch.Tensor
    link_logits: torch.Tensor
    value_numeric: torch.Tensor
    value_categorical: torch.Tensor
    value_boolean: torch.Tensor
    aux: dict[str, torch.Tensor]
    slots: torch.Tensor


    bin_logits: torch.Tensor | None = None
    bin_offsets: torch.Tensor | None = None
    cell_value: torch.Tensor | None = None

    @property
    def n_slots(self) -> int:
        return self.op_logits.shape[1]


class _CrossBlock(nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.norm_self = nn.RMSNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm_cross = nn.RMSNorm(d_model)
        self.norm_kv = nn.RMSNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm_ffn = nn.RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.SiLU(), nn.Linear(d_ff, d_model)
        )

    def forward(self, slots, tokens, key_padding_mask):
        h = self.norm_self(slots)
        slots = slots + self.self_attn(h, h, h, need_weights=False)[0]
        q = self.norm_cross(slots)
        kv = self.norm_kv(tokens)
        slots = slots + self.cross_attn(
            q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False
        )[0]
        return slots + self.ffn(self.norm_ffn(slots))


class DeltaDecoder(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_slots: int = 16,
        n_layers: int = 2,
        num_heads: int = 8,
        d_ff: int | None = None,
        max_categories: int = MAX_CATEGORIES,
        numeric_head: str = "scalar",
        shared_offset: bool = True,
    ) -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.n_slots = n_slots
        self.slot_embed = nn.Parameter(torch.randn(n_slots, d_model) * 0.02)
        self.layers = nn.ModuleList(
            [_CrossBlock(d_model, num_heads, d_ff) for _ in range(n_layers)]
        )
        self.norm_out = nn.RMSNorm(d_model)

        self.op_head = nn.Linear(d_model, N_OPS)
        self.exist_head = nn.Linear(d_model, 1)
        self.pointers = nn.ModuleDict(
            {name: PointerHead(d_model) for name in ("cell", "row", "table", "link")}
        )
        self.values = TypedValueHead(d_model, max_categories)


        self.numeric_head = numeric_head
        self.binned = BinnedNumericHead(d_model, shared_offset=shared_offset) if numeric_head == "binned" else None
        self.aux = AuxHeads(d_model)

    def forward(
        self, hidden: torch.Tensor, batch: dict[str, torch.Tensor]
    ) -> DecoderOutput:
        B = hidden.shape[0]
        is_padding = batch["is_padding"]
        roles = batch["role_ids"]

        slots = self.slot_embed[None].expand(B, -1, -1)
        for layer in self.layers:
            slots = layer(slots, hidden, key_padding_mask=is_padding)
        slots = self.norm_out(slots)


        is_query_cell = (roles == Role.STATE_QUERY) & ~is_padding
        is_row = batch["is_row_anchor"] & ~is_padding
        is_table = (roles == Role.TABLE_ANCHOR) & ~is_padding

        values = self.values(slots)
        pooled = masked_mean(hidden, ~is_padding)

        cell_logits = self.pointers["cell"](slots, hidden, is_query_cell)
        bin_logits = bin_offsets = cell_value = None
        numeric = values["numeric"]
        if self.binned is not None:
            b = self.binned(slots)
            bin_logits, bin_offsets = b["bin_logits"], b["bin_offsets"]


            nums = batch["number_values_exact"].squeeze(-1)
            idx = cell_logits.argmax(dim=-1)
            cell_value = nums.gather(1, idx)
            numeric = cell_value + self.binned.decode(bin_logits, bin_offsets)

        return DecoderOutput(
            op_logits=self.op_head(slots),
            exist_logits=self.exist_head(slots).squeeze(-1),
            cell_logits=cell_logits,
            row_logits=self.pointers["row"](slots, hidden, is_row),
            table_logits=self.pointers["table"](slots, hidden, is_table),
            link_logits=self.pointers["link"](slots, hidden, is_row),
            value_numeric=numeric,
            bin_logits=bin_logits,
            bin_offsets=bin_offsets,
            cell_value=cell_value,
            value_categorical=values["categorical"],
            value_boolean=values["boolean"],
            aux=self.aux(pooled),
            slots=slots,
        )


def decode_predictions(
    out: DecoderOutput, *, threshold: float = 0.5, max_edits: int | None = None
) -> list[list[PredictedEditRaw]]:
    exist = torch.sigmoid(out.exist_logits).detach()
    ops = out.op_logits.argmax(-1)
    cells = out.cell_logits.argmax(-1)
    rows = out.row_logits.argmax(-1)
    tables = out.table_logits.argmax(-1)
    links = out.link_logits.argmax(-1)
    cats = out.value_categorical.argmax(-1)
    bools = (torch.sigmoid(out.value_boolean) > 0.5).long()

    batch_out: list[list[PredictedEditRaw]] = []
    for b in range(out.op_logits.shape[0]):
        keep = (exist[b] >= threshold).nonzero().flatten().tolist()
        keep.sort(key=lambda i: -float(exist[b, i]))
        if max_edits is not None:
            keep = keep[:max_edits]
        batch_out.append(
            [
                PredictedEditRaw(
                    slot=i,
                    op=int(ops[b, i]),
                    cell_pos=int(cells[b, i]),
                    row_pos=int(rows[b, i]),
                    table_pos=int(tables[b, i]),
                    link_pos=int(links[b, i]),
                    value_numeric=float(out.value_numeric[b, i].detach()),
                    value_categorical=int(cats[b, i]),
                    value_boolean=int(bools[b, i]),
                    confidence=float(exist[b, i]),
                )
                for i in keep
            ]
        )
    return batch_out


@dataclass(frozen=True)
class PredictedEditRaw:

    slot: int
    op: int
    cell_pos: int
    row_pos: int
    table_pos: int
    link_pos: int
    value_numeric: float
    value_categorical: int
    value_boolean: int
    confidence: float
