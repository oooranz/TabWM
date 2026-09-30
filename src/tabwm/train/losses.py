"""Matching and training losses."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from tabwm.data.delta import OP_TO_IDX, ValueKind
from tabwm.model.delta_decoder import DecoderOutput
from tabwm.model.heads import gather_pointer_targets, select_offset
from tabwm.world.state import EditOp

NOOP_IDX = OP_TO_IDX[EditOp.NOOP]
_POINTER_OPS = {
    OP_TO_IDX[EditOp.UPDATE_VALUE]: "cell",
    OP_TO_IDX[EditOp.UPDATE_STATE]: "cell",
    OP_TO_IDX[EditOp.DELETE_ROW]: "row",
    OP_TO_IDX[EditOp.INSERT_ROW]: "table",
    OP_TO_IDX[EditOp.EMIT_EVENT]: "table",
}


@dataclass
class LossWeights:

    edit: float = 1.0
    pointer: float = 1.0
    value: float = 1.0
    exist: float = 1.0
    constraint: float = 0.5
    reward: float = 0.1
    applicable: float = 0.1


    cost_op: float = 1.0
    cost_pointer: float = 2.0
    cost_value: float = 0.5


    value_relative: bool = False
    value_rel_scale: float = 0.033
    value_tol_lo: float = 0.01
    value_tol_hi: float = 0.1
    value_ordinal: bool = False
    value_ordinal_sigma: float = 0.75
    value_bins_weight: float = 1.0


@dataclass
class LossOutput:
    total: torch.Tensor
    parts: dict[str, torch.Tensor] = field(default_factory=dict)
    stats: dict[str, float] = field(default_factory=dict)

    def item(self) -> float:
        return float(self.total.detach())

    def log_dict(self) -> dict[str, float]:
        out = {k: float(v.detach()) for k, v in self.parts.items()}
        out["loss"] = self.item()
        out.update(self.stats)
        return out


@torch.no_grad()
def hungarian_match(
    out: DecoderOutput, batch: dict[str, torch.Tensor], w: LossWeights
) -> list[tuple[np.ndarray, np.ndarray]]:
    B, N, _ = out.op_logits.shape
    op_lp = out.op_logits.log_softmax(-1)
    cell_lp = out.cell_logits.log_softmax(-1)
    row_lp = out.row_logits.log_softmax(-1)
    table_lp = out.table_logits.log_softmax(-1)
    cat_lp = out.value_categorical.log_softmax(-1)

    gold_op = batch["edit_op"]
    valid = batch["edit_valid"]
    pointer_lp = {"cell": cell_lp, "row": row_lp, "table": table_lp}
    pos_key = {
        "cell": batch["edit_cell_pos"],
        "row": batch["edit_row_pos"],
        "table": batch["edit_table_pos"],
    }

    matches: list[tuple[np.ndarray, np.ndarray]] = []
    for b in range(B):
        m = int(valid[b].sum())
        if m == 0:
            matches.append((np.zeros(0, int), np.zeros(0, int)))
            continue

        cost = torch.zeros(N, m, device=out.op_logits.device)
        for j in range(m):
            op = int(gold_op[b, j])
            cost[:, j] -= w.cost_op * op_lp[b, :, op]

            kind = _POINTER_OPS.get(op)
            if kind is not None:
                p = int(pos_key[kind][b, j])
                if p >= 0:
                    cost[:, j] -= w.cost_pointer * pointer_lp[kind][b, :, p]

            vk = int(batch["edit_value_kind"][b, j])
            if vk == ValueKind.NUMERIC:
                cost[:, j] += w.cost_value * (
                    out.value_numeric[b] - batch["edit_value_num"][b, j]
                ).abs()
            elif vk == ValueKind.CATEGORICAL:
                cost[:, j] -= w.cost_value * cat_lp[b, :, int(batch["edit_value_class"][b, j])]
            elif vk == ValueKind.BOOLEAN:
                target = float(batch["edit_value_class"][b, j])
                cost[:, j] += w.cost_value * F.binary_cross_entropy_with_logits(
                    out.value_boolean[b], torch.full_like(out.value_boolean[b], target),
                    reduction="none",
                )

        rows, cols = linear_sum_assignment(cost.float().cpu().numpy())
        matches.append((rows, cols))
    return matches


_BINNED: list = []


def register_binned_head(head) -> None:
    _BINNED.clear()
    _BINNED.append(head)


def _binned_head(out):
    if not _BINNED:
        raise RuntimeError("binned numeric head not registered; call "
                           "losses.register_binned_head(model.decoder.binned) "
                           "after building the model")
    return _BINNED[0]


def edit_set_loss(
    out: DecoderOutput,
    batch: dict[str, torch.Tensor],
    weights: LossWeights | None = None,
    *,
    matches: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> LossOutput:
    w = weights or LossWeights()
    matches = matches if matches is not None else hungarian_match(out, batch, w)
    device = out.op_logits.device
    B, N, _ = out.op_logits.shape


    tgt_op = torch.full((B, N), NOOP_IDX, dtype=torch.long, device=device)
    tgt_exist = torch.zeros(B, N, device=device)
    tgt_cell = torch.full((B, N), -1, dtype=torch.long, device=device)
    tgt_row = torch.full((B, N), -1, dtype=torch.long, device=device)
    tgt_table = torch.full((B, N), -1, dtype=torch.long, device=device)
    tgt_num = torch.zeros(B, N, device=device)
    tgt_tol = torch.zeros(B, N, device=device)
    tgt_class = torch.zeros(B, N, dtype=torch.long, device=device)
    tgt_kind = torch.zeros(B, N, dtype=torch.long, device=device)
    matched = torch.zeros(B, N, dtype=torch.bool, device=device)

    for b, (slots, golds) in enumerate(matches):
        for s, g in zip(slots, golds):
            tgt_op[b, s] = batch["edit_op"][b, g]
            tgt_exist[b, s] = 1.0
            tgt_cell[b, s] = batch["edit_cell_pos"][b, g]
            tgt_row[b, s] = batch["edit_row_pos"][b, g]
            tgt_table[b, s] = batch["edit_table_pos"][b, g]
            tgt_num[b, s] = batch["edit_value_num"][b, g]
            tgt_tol[b, s] = batch["edit_value_tol_z"][b, g]
            tgt_class[b, s] = batch["edit_value_class"][b, g]
            tgt_kind[b, s] = batch["edit_value_kind"][b, g]
            matched[b, s] = True

    parts: dict[str, torch.Tensor] = {}


    parts["op"] = F.cross_entropy(out.op_logits.reshape(B * N, -1), tgt_op.reshape(B * N))
    parts["exist"] = F.binary_cross_entropy_with_logits(out.exist_logits, tgt_exist)


    is_cell_op = matched & ((tgt_op == OP_TO_IDX[EditOp.UPDATE_VALUE]) |
                            (tgt_op == OP_TO_IDX[EditOp.UPDATE_STATE]))
    is_row_op = matched & (tgt_op == OP_TO_IDX[EditOp.DELETE_ROW])
    is_table_op = matched & ((tgt_op == OP_TO_IDX[EditOp.INSERT_ROW]) |
                             (tgt_op == OP_TO_IDX[EditOp.EMIT_EVENT]))
    parts["ptr_cell"] = gather_pointer_targets(out.cell_logits, tgt_cell, is_cell_op)
    parts["ptr_row"] = gather_pointer_targets(out.row_logits, tgt_row, is_row_op)
    parts["ptr_table"] = gather_pointer_targets(out.table_logits, tgt_table, is_table_op)


    num_sel = matched & (tgt_kind == ValueKind.NUMERIC)
    cat_sel = matched & (tgt_kind == ValueKind.CATEGORICAL)
    bool_sel = matched & (tgt_kind == ValueKind.BOOLEAN)

    if out.bin_logits is None:
        if w.value_relative:


            tol = tgt_tol.clamp(w.value_tol_lo, w.value_tol_hi)
            wt = (w.value_rel_scale / tol).clamp(min=1.0)
            parts["value_num"] = _masked(
                F.huber_loss(out.value_numeric, tgt_num, reduction="none") * wt,
                num_sel)
        else:
            parts["value_num"] = _masked(
                F.huber_loss(out.value_numeric, tgt_num, reduction="none"), num_sel
            )
    else:


        nums = batch["number_values_exact"].squeeze(-1)
        v_cur = nums.gather(1, tgt_cell.clamp(min=0))
        delta = (tgt_num - v_cur).float()
        head = _binned_head(out)
        tgt_bin = head.target_bin(delta)
        tgt_off = head.target_offset(delta, tgt_bin)
        if w.value_ordinal:
            tgt_dist = head.target_dist(delta, w.value_ordinal_sigma)
            ce = -(tgt_dist * F.log_softmax(out.bin_logits.float(), dim=-1)).sum(-1)
        else:
            ce = F.cross_entropy(
                out.bin_logits.reshape(-1, out.bin_logits.shape[-1]),
                tgt_bin.reshape(-1), reduction="none",
            ).view_as(tgt_bin)
        parts["value_num"] = w.value_bins_weight * _masked(ce, num_sel)
        pred_off = torch.tanh(select_offset(out.bin_offsets, tgt_bin))
        parts["value_off"] = _masked(
            F.huber_loss(pred_off, tgt_off.to(pred_off.dtype), reduction="none"),
            num_sel)
        parts["value_refined"] = _masked(
            F.huber_loss(out.value_numeric, tgt_num, reduction="none"), num_sel
        )


    n_cat_matched = float((matched & (tgt_kind == ValueKind.CATEGORICAL)).sum())
    cat_sel = cat_sel & _in_domain(batch, tgt_cell, tgt_class)
    cat_logits = _mask_to_domain(out.value_categorical, batch, tgt_cell)
    parts["value_cat"] = _masked(
        F.cross_entropy(
            cat_logits.reshape(B * N, -1),
            tgt_class.clamp(min=0).reshape(B * N),
            reduction="none",
        ).reshape(B, N),
        cat_sel,
    )
    parts["value_bool"] = _masked(
        F.binary_cross_entropy_with_logits(
            out.value_boolean, tgt_class.float(), reduction="none"
        ),
        bool_sel,
    )


    parts["constraint"] = state_machine_constraint_loss(out, batch, tgt_cell, cat_sel)


    if "aux_reward" in batch:
        parts["reward"] = F.huber_loss(out.aux["reward"], batch["aux_reward"])
    if "aux_applicable" in batch:
        parts["applicable"] = F.binary_cross_entropy_with_logits(
            out.aux["applicable"], batch["aux_applicable"].float()
        )

    total = (
        w.edit * parts["op"]
        + w.exist * parts["exist"]
        + w.pointer * (parts["ptr_cell"] + parts["ptr_row"] + parts["ptr_table"])
        + w.value * (parts["value_num"] + parts["value_cat"] + parts["value_bool"]
                     + parts.get("value_off", torch.zeros((), device=device))
                     + parts.get("value_refined", torch.zeros((), device=device)))
        + w.constraint * parts["constraint"]
        + w.reward * parts.get("reward", torch.zeros((), device=device))
        + w.applicable * parts.get("applicable", torch.zeros((), device=device))
    )

    n_gold = float(batch["edit_valid"].sum())
    stats = {
        "cat_unscorable_frac": (
            1.0 - float(cat_sel.sum()) / n_cat_matched if n_cat_matched else 0.0
        ),
        "n_gold_edits": n_gold / B,
        "n_matched": float(matched.sum()) / B,
        "frac_in_context": (
            float(batch["edit_in_context"].sum()) / max(n_gold, 1.0) if n_gold else 1.0
        ),
    }
    return LossOutput(total=total, parts=parts, stats=stats)


def _masked(per_slot: torch.Tensor, sel: torch.Tensor) -> torch.Tensor:
    n = sel.sum()
    if n == 0:
        return (per_slot * 0.0).sum()
    return (per_slot * sel).sum() / n


def _in_domain(
    batch: dict[str, torch.Tensor], tgt_cell: torch.Tensor, tgt_class: torch.Tensor
) -> torch.Tensor:
    n_classes = _gather_per_slot(batch["cell_n_classes"], tgt_cell)
    return (tgt_cell >= 0) & (tgt_class < n_classes) & (n_classes > 0)


def _mask_to_domain(
    logits: torch.Tensor, batch: dict[str, torch.Tensor], tgt_cell: torch.Tensor
) -> torch.Tensor:
    from tabwm.model.heads import NEG_INF

    n_classes = _gather_per_slot(batch["cell_n_classes"], tgt_cell)
    ar = torch.arange(logits.shape[-1], device=logits.device)
    allowed = ar < n_classes.clamp(min=1)[..., None]
    return logits.masked_fill(~allowed, NEG_INF)


def _gather_per_slot(per_token: torch.Tensor, tgt_pos: torch.Tensor) -> torch.Tensor:
    safe = tgt_pos.clamp(min=0)
    return torch.gather(per_token, 1, safe)


def state_machine_constraint_loss(
    out: DecoderOutput,
    batch: dict[str, torch.Tensor],
    tgt_cell: torch.Tensor,
    cat_sel: torch.Tensor,
) -> torch.Tensor:
    legal = batch.get("cell_legal_classes")
    if legal is None or cat_sel.sum() == 0:
        return (out.value_categorical * 0.0).sum()

    B, N, C = out.value_categorical.shape
    idx = tgt_cell.clamp(min=0)[..., None].expand(B, N, legal.shape[-1])
    legal_per_slot = torch.gather(legal, 1, idx)[..., :C]


    logp = out.value_categorical.log_softmax(-1)
    legal_mass = torch.logsumexp(
        logp.masked_fill(~legal_per_slot, -1e9), dim=-1
    )
    return _masked(-legal_mass, cat_sel & legal_per_slot.any(-1))


@torch.no_grad()
def edit_set_accuracy(
    out: DecoderOutput,
    batch: dict[str, torch.Tensor],
    matches: list[tuple[np.ndarray, np.ndarray]],
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    exist = torch.sigmoid(out.exist_logits) >= threshold
    pred_op = out.op_logits.argmax(-1)
    pred_cell = out.cell_logits.argmax(-1)

    tp = fp = fn = 0
    op_correct = ptr_correct = n = 0
    for b, (slots, golds) in enumerate(matches):
        n_gold = int(batch["edit_valid"][b].sum())
        n_pred = int(exist[b].sum())
        hit = 0
        for s, g in zip(slots, golds):
            if not exist[b, s]:
                continue
            gold_op = int(batch["edit_op"][b, g])
            ok_op = int(pred_op[b, s]) == gold_op
            gold_cell = int(batch["edit_cell_pos"][b, g])
            ok_ptr = gold_cell < 0 or int(pred_cell[b, s]) == gold_cell
            op_correct += ok_op
            ptr_correct += ok_ptr
            n += 1
            hit += ok_op and ok_ptr
        tp += hit
        fp += max(0, n_pred - hit)
        fn += max(0, n_gold - hit)

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {
        "edit_precision": prec,
        "edit_recall": rec,
        "edit_f1": 2 * prec * rec / max(prec + rec, 1e-9),
        "op_acc": op_correct / max(n, 1),
        "pointer_acc": ptr_correct / max(n, 1),
        "pred_edits_per_task": float(exist.sum()) / exist.shape[0],
    }
