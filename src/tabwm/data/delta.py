"""Edit encoding and decoding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from tabwm.world.schema import Column, ColumnKind, Schema
from tabwm.world.state import Edit, EditOp, RelationalState, StateDelta, clip_to_column

EDIT_OPS: tuple[EditOp, ...] = (
    EditOp.NOOP,
    EditOp.UPDATE_VALUE,
    EditOp.UPDATE_STATE,
    EditOp.INSERT_ROW,
    EditOp.DELETE_ROW,
    EditOp.EMIT_EVENT,
)
OP_TO_IDX = {op: i for i, op in enumerate(EDIT_OPS)}
N_OPS = len(EDIT_OPS)

NO_POINTER = -1


class ValueKind:

    NONE = 0
    NUMERIC = 1
    CATEGORICAL = 2
    BOOLEAN = 3


def value_kind_of(col: Column) -> int:
    if col.kind is ColumnKind.CATEGORICAL:
        return ValueKind.CATEGORICAL
    if col.kind is ColumnKind.BOOL:
        return ValueKind.BOOLEAN
    return ValueKind.NUMERIC


@dataclass
class ColumnStats:

    stats: dict[tuple[str, str], tuple[float, float]] = field(default_factory=dict)

    MAX_SIGMA: ClassVar[float] = 50.0

    @classmethod
    def from_states(
        cls, states: list[RelationalState], schema: Schema, *, min_span_frac: float = 0.02
    ) -> ColumnStats:
        out: dict[tuple[str, str], tuple[float, float]] = {}
        for table in schema:
            for col in table.columns:
                if col.kind is ColumnKind.CATEGORICAL or col.kind is ColumnKind.BOOL:
                    continue
                span = float(col.hi or 1.0) - float(col.lo or 0.0)
                floor = max(min_span_frac * abs(span), 1e-6)
                chunks = [
                    s.tables[table.name].columns[col.name]
                    for s in states
                    if table.name in s.tables and s.n_rows(table.name) > 0
                ]
                if not chunks:
                    out[(table.name, col.name)] = (0.0, max(floor, 1.0))
                    continue
                vals = np.concatenate(chunks).astype(np.float64)
                out[(table.name, col.name)] = (float(vals.mean()), max(float(vals.std()), floor))
        return cls(stats=out)

    def get(self, table: str, column: str) -> tuple[float, float]:
        return self.stats.get((table, column), (0.0, 1.0))

    def normalize(self, table: str, column: str, value: float) -> float:
        mean, std = self.get(table, column)
        z = (float(value) - mean) / std
        return float(np.clip(z, -self.MAX_SIGMA, self.MAX_SIGMA))

    def denormalize(self, table: str, column: str, value: float) -> float:
        mean, std = self.get(table, column)
        return float(value) * std + mean


def encode_cell(col: Column, raw: float, table: str, stats: ColumnStats) -> float:
    if col.kind is ColumnKind.CATEGORICAL:
        return float(raw)
    if col.kind is ColumnKind.BOOL:
        return float(raw)
    return stats.normalize(table, col.name, raw)


def decode_cell(col: Column, encoded: float, table: str, stats: ColumnStats) -> float:
    if col.kind is ColumnKind.CATEGORICAL:
        return clip_to_column(col, round(float(encoded)))
    if col.kind is ColumnKind.BOOL:
        return clip_to_column(col, float(encoded))
    return clip_to_column(col, stats.denormalize(table, col.name, encoded))


def _tolerance_z(stats: ColumnStats, table: str, column: str, value: float,
                 rtol: float = 0.02, atol: float = 1e-3) -> float:
    entry = stats.stats.get((table, column))
    if entry is None:
        return 0.0
    _, std = entry
    return float(max(atol, rtol * abs(float(value))) / max(std, 1e-12))


@dataclass(frozen=True)
class EditTarget:

    op: int
    cell_pos: int = NO_POINTER
    row_pos: int = NO_POINTER
    table_pos: int = NO_POINTER
    link_pos: int = NO_POINTER
    value_kind: int = ValueKind.NONE
    value_num: float = 0.0
    value_tol_z: float = 0.0
    value_class: int = 0
    n_classes: int = 0
    table: str = ""
    column: str = ""
    row_id: int = -1
    in_context: bool = True

    @property
    def edit_op(self) -> EditOp:
        return EDIT_OPS[self.op]


@dataclass
class TokenIndex:

    cell: dict[tuple[str, int, str], int] = field(default_factory=dict)
    row: dict[tuple[str, int], int] = field(default_factory=dict)
    table: dict[str, int] = field(default_factory=dict)

    def cell_pos(self, table: str, row_id: int, column: str) -> int:
        return self.cell.get((table, int(row_id), column), NO_POINTER)

    def row_pos(self, table: str, row_id: int) -> int:
        return self.row.get((table, int(row_id)), NO_POINTER)

    def table_pos(self, table: str) -> int:
        return self.table.get(table, NO_POINTER)


def encode_targets(
    delta: StateDelta,
    schema: Schema,
    index: TokenIndex,
    stats: ColumnStats,
    *,
    max_slots: int = 16,
) -> tuple[list[EditTarget], dict[str, int]]:
    targets: list[EditTarget] = []
    diag = {"total": 0, "out_of_context": 0, "dropped": 0, "unresolved": 0}

    for e in delta.dedup().edits:
        if e.op is EditOp.NOOP:
            continue
        diag["total"] += 1
        if len(targets) >= max_slots:
            diag["dropped"] += 1
            continue
        target = _encode_one(e, schema, index, stats)
        if target is None:
            diag["unresolved"] += 1
            continue
        if not target.in_context:
            diag["out_of_context"] += 1
        targets.append(target)
    return targets, diag


def _encode_one(
    e: Edit, schema: Schema, index: TokenIndex, stats: ColumnStats
) -> EditTarget | None:
    table_pos = index.table_pos(e.table)

    if e.op in (EditOp.UPDATE_VALUE, EditOp.UPDATE_STATE):
        try:
            col = schema.table(e.table).column(e.column)
        except (KeyError, StopIteration):
            return None
        cell_pos = index.cell_pos(e.table, e.row_id, e.column)
        row_pos = index.row_pos(e.table, e.row_id)
        kind = value_kind_of(col)
        return EditTarget(
            op=OP_TO_IDX[e.op],
            cell_pos=cell_pos,
            row_pos=row_pos,
            table_pos=table_pos,
            value_kind=kind,
            value_num=(
                stats.normalize(e.table, e.column, e.value)
                if kind is ValueKind.NUMERIC
                else 0.0
            ),
            value_tol_z=(
                _tolerance_z(stats, e.table, e.column, e.value)
                if kind is ValueKind.NUMERIC
                else 0.0
            ),
            value_class=int(e.value) if kind != ValueKind.NUMERIC else 0,
            n_classes=col.n_categories if kind == ValueKind.CATEGORICAL else 2,
            table=e.table,
            column=e.column,
            row_id=int(e.row_id),
            in_context=cell_pos != NO_POINTER,
        )

    if e.op is EditOp.DELETE_ROW:
        row_pos = index.row_pos(e.table, e.row_id)
        return EditTarget(
            op=OP_TO_IDX[e.op],
            row_pos=row_pos,
            table_pos=table_pos,
            table=e.table,
            row_id=int(e.row_id),
            in_context=row_pos != NO_POINTER,
        )


    link_pos = NO_POINTER
    try:
        tab = schema.table(e.table)
    except KeyError:
        return None
    for fk in tab.foreign_keys:
        for name, val in e.values:
            if name == fk.column:
                link_pos = index.row_pos(fk.ref_table, int(val))
                break
        if link_pos != NO_POINTER:
            break
    return EditTarget(
        op=OP_TO_IDX[e.op],
        table_pos=table_pos,
        link_pos=link_pos,
        table=e.table,
        row_id=int(e.row_id),
        in_context=table_pos != NO_POINTER,
    )


@dataclass(frozen=True)
class PredictedEdit:

    op: int
    cell_pos: int = NO_POINTER
    row_pos: int = NO_POINTER
    table_pos: int = NO_POINTER
    link_pos: int = NO_POINTER
    value_num: float = 0.0
    value_class: int = 0
    confidence: float = 1.0


def decode_edits(
    predictions: list[PredictedEdit],
    reverse: dict[int, tuple[str, int, str]],
    table_of_pos: dict[int, str],
    schema: Schema,
    stats: ColumnStats,
    state: RelationalState,
    *,
    threshold: float = 0.5,
) -> StateDelta:
    delta = StateDelta()
    for p in predictions:
        op = EDIT_OPS[p.op] if 0 <= p.op < N_OPS else EditOp.NOOP
        if op is EditOp.NOOP or p.confidence < threshold:
            continue

        if op in (EditOp.UPDATE_VALUE, EditOp.UPDATE_STATE):
            ref = reverse.get(p.cell_pos)
            if ref is None:
                continue
            table, row_id, column = ref
            try:
                col = schema.table(table).column(column)
            except KeyError:
                continue
            kind = value_kind_of(col)
            raw = (
                decode_cell(col, p.value_num, table, stats)
                if kind == ValueKind.NUMERIC
                else clip_to_column(col, float(p.value_class))
            )


            true_op = EditOp.UPDATE_STATE if col.is_state else EditOp.UPDATE_VALUE
            delta.add(
                Edit(op=true_op, table=table, row_id=row_id, column=column, value=raw)
            )

        elif op is EditOp.DELETE_ROW:
            ref = reverse.get(p.row_pos)
            if ref is None:
                continue
            table, row_id, _ = ref
            delta.add(Edit(op=EditOp.DELETE_ROW, table=table, row_id=row_id))

        else:
            table = table_of_pos.get(p.table_pos)
            if table is None:
                continue
            try:
                tab = schema.table(table)
            except KeyError:
                continue
            link = reverse.get(p.link_pos)
            values: list[tuple[str, float]] = []
            for c in tab.columns:
                if c.name == tab.pk:
                    continue
                v = 0.0
                fk = next((f for f in tab.foreign_keys if f.column == c.name), None)
                if fk is not None and link is not None and link[0] == fk.ref_table:
                    v = float(link[1])
                values.append((c.name, clip_to_column(c, v)))
            delta.add(
                Edit(
                    op=EditOp.EMIT_EVENT if op is EditOp.EMIT_EVENT else EditOp.INSERT_ROW,
                    table=table,
                    row_id=state.peek_id(table),
                    values=tuple(values),
                )
            )
    return delta


def targets_to_delta(targets: list[EditTarget], schema: Schema, stats: ColumnStats) -> StateDelta:
    delta = StateDelta()
    for t in targets:
        op = t.edit_op
        if op in (EditOp.UPDATE_VALUE, EditOp.UPDATE_STATE):
            col = schema.table(t.table).column(t.column)
            raw = (
                decode_cell(col, t.value_num, t.table, stats)
                if t.value_kind == ValueKind.NUMERIC
                else clip_to_column(col, float(t.value_class))
            )
            delta.add(Edit(op=op, table=t.table, row_id=t.row_id, column=t.column, value=raw))
        elif op is EditOp.DELETE_ROW:
            delta.add(Edit(op=op, table=t.table, row_id=t.row_id))
    return delta
