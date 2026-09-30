"""Relational states and edits."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from tabwm.world.schema import Column, ColumnKind, Schema, TableKind

NULL_ID = -1


def dtype_for(kind: ColumnKind) -> np.dtype:
    return np.dtype(np.float64) if kind is ColumnKind.NUMBER else np.dtype(np.int64)


class EditOp(str, Enum):

    UPDATE_VALUE = "UPDATE_VALUE"
    UPDATE_STATE = "UPDATE_STATE"
    INSERT_ROW = "INSERT_ROW"
    DELETE_ROW = "DELETE_ROW"
    EMIT_EVENT = "EMIT_EVENT"
    NOOP = "NOOP"


_ROW_OPS = (EditOp.INSERT_ROW, EditOp.EMIT_EVENT)


@dataclass(frozen=True)
class Edit:

    op: EditOp
    table: str = ""
    row_id: int = NULL_ID
    column: str = ""
    value: float = 0.0
    values: tuple[tuple[str, float], ...] = ()

    def key(self) -> tuple:
        return (self.op.value, self.table, int(self.row_id), self.column)

    def canonical(self) -> tuple:
        return (*self.key(), round(float(self.value), 9), tuple(sorted(self.values)))

    @property
    def is_structural(self) -> bool:
        return self.op in (EditOp.INSERT_ROW, EditOp.EMIT_EVENT, EditOp.DELETE_ROW)

    def __repr__(self) -> str:
        if self.op is EditOp.NOOP:
            return "NOOP"
        if self.op in _ROW_OPS:
            vals = ", ".join(f"{k}={v:g}" for k, v in self.values)
            return f"{self.op.value}({self.table}#{self.row_id}: {vals})"
        if self.op is EditOp.DELETE_ROW:
            return f"DELETE_ROW({self.table}#{self.row_id})"
        return f"{self.op.value}({self.table}#{self.row_id}.{self.column}={self.value:g})"


@dataclass
class StateDelta:

    edits: list[Edit] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.edits)

    def __iter__(self) -> Iterator[Edit]:
        return iter(self.edits)

    def __bool__(self) -> bool:
        return bool(self.edits)

    def add(self, edit: Edit) -> None:
        if edit.op is not EditOp.NOOP:
            self.edits.append(edit)

    def extend(self, edits: Iterable[Edit]) -> None:
        for e in edits:
            self.add(e)

    def dedup(self) -> StateDelta:
        last: dict[tuple, Edit] = {}
        for e in self.edits:
            last[e.key()] = e
        return StateDelta(edits=list(last.values()))

    def canonical(self) -> tuple:
        return tuple(sorted(e.canonical() for e in self.dedup().edits))

    def by_op(self) -> dict[EditOp, list[Edit]]:
        out: dict[EditOp, list[Edit]] = {}
        for e in self.edits:
            out.setdefault(e.op, []).append(e)
        return out

    def touched_rows(self) -> set[tuple[str, int]]:
        return {(e.table, int(e.row_id)) for e in self.edits if e.op is not EditOp.NOOP}

    def touched_tables(self) -> set[str]:
        return {e.table for e in self.edits if e.op is not EditOp.NOOP}

    def __eq__(self, other: object) -> bool:
        return isinstance(other, StateDelta) and self.canonical() == other.canonical()

    def __repr__(self) -> str:
        return f"StateDelta({len(self.edits)} edits: {self.edits!r})"


@dataclass
class TableState:

    columns: dict[str, np.ndarray]
    pk: str

    def __post_init__(self) -> None:
        self._index: dict[int, int] = {}
        self._reindex()

    def _reindex(self) -> None:
        self._index = {int(v): i for i, v in enumerate(self.columns[self.pk])}

    @property
    def n_rows(self) -> int:
        return len(self.columns[self.pk])

    @property
    def row_ids(self) -> np.ndarray:
        return self.columns[self.pk]

    def has_row(self, row_id: int) -> bool:
        return int(row_id) in self._index

    def pos(self, row_id: int) -> int:
        try:
            return self._index[int(row_id)]
        except KeyError as exc:
            raise KeyError(f"no row {row_id} (pk={self.pk})") from exc

    def get(self, row_id: int, column: str) -> float:
        return float(self.columns[column][self.pos(row_id)])

    def set(self, row_id: int, column: str, value: float) -> None:
        arr = self.columns[column]
        p = self.pos(row_id)
        arr[p] = value
        if column == self.pk:


            self._reindex()

    def append(self, values: dict[str, float]) -> None:
        for name, arr in self.columns.items():
            v = values.get(name, NULL_ID if name != self.pk else None)
            if v is None:
                raise ValueError(f"missing value for {name!r} on insert")
            self.columns[name] = np.append(arr, np.asarray(v, dtype=arr.dtype))
        self._index[int(values[self.pk])] = self.n_rows - 1

    def drop(self, row_id: int) -> None:
        p = self.pos(row_id)
        for name, arr in self.columns.items():
            self.columns[name] = np.delete(arr, p)
        self._reindex()

    def copy(self) -> TableState:
        return TableState(columns={k: v.copy() for k, v in self.columns.items()}, pk=self.pk)


@dataclass
class RelationalState:

    tables: dict[str, TableState]
    time: int = 0
    _next_id: dict[str, int] = field(default_factory=dict)


    @classmethod
    def empty(cls, schema: Schema) -> RelationalState:
        tables = {
            t.name: TableState(
                columns={c.name: np.zeros(0, dtype=dtype_for(c.kind)) for c in t.columns},
                pk=t.pk,
            )
            for t in schema
        }
        return cls(tables=tables, _next_id={t.name: 0 for t in schema})

    def copy(self) -> RelationalState:
        return RelationalState(
            tables={k: v.copy() for k, v in self.tables.items()},
            time=self.time,
            _next_id=dict(self._next_id),
        )


    def __getitem__(self, table: str) -> TableState:
        return self.tables[table]

    def n_rows(self, table: str) -> int:
        return self.tables[table].n_rows

    def row_ids(self, table: str) -> np.ndarray:
        return self.tables[table].row_ids

    def has_row(self, table: str, row_id: int) -> bool:
        return table in self.tables and self.tables[table].has_row(row_id)

    def get(self, table: str, row_id: int, column: str) -> float:
        return self.tables[table].get(row_id, column)

    def set(self, table: str, row_id: int, column: str, value: float) -> None:
        self.tables[table].set(row_id, column, value)

    def peek_id(self, table: str) -> int:
        return self._next_id.get(table, 0)

    def fresh_id(self, table: str) -> int:
        rid = self._next_id.get(table, 0)
        self._next_id[table] = rid + 1
        return rid


    def parent_id(self, schema: Schema, table: str, row_id: int, parent_table: str) -> int:
        fk = schema.table(table).fk_to(parent_table)
        if fk is None:
            return NULL_ID
        return int(self.get(table, row_id, fk.column))

    def child_ids(
        self, schema: Schema, table: str, row_id: int, child_table: str
    ) -> np.ndarray:
        fk = schema.table(child_table).fk_to(table)
        if fk is None:
            return np.zeros(0, dtype=np.int64)
        ts = self.tables[child_table]
        mask = ts.columns[fk.column] == int(row_id)
        return ts.row_ids[mask]


    def apply(
        self, delta: StateDelta, *, strict: bool = True, advance_time: bool = True
    ) -> RelationalState:
        out = self.copy()
        out.apply_(delta, strict=strict)
        if advance_time:
            out.time = self.time + 1
        return out

    def apply_(self, delta: StateDelta, *, strict: bool = True) -> None:


        ordered = sorted(
            delta.dedup().edits,
            key=lambda e: (
                0 if e.op in _ROW_OPS else (2 if e.op is EditOp.DELETE_ROW else 1)
            ),
        )
        for e in ordered:
            if e.op is EditOp.NOOP:
                continue
            ts = self.tables.get(e.table)
            if ts is None:
                if strict:
                    raise KeyError(f"unknown table {e.table!r}")
                continue
            if e.op in _ROW_OPS:
                if ts.has_row(e.row_id):
                    if strict:
                        raise ValueError(f"row {e.table}#{e.row_id} already exists")
                    continue
                values = {ts.pk: float(e.row_id), **dict(e.values)}
                ts.append(values)
                self._next_id[e.table] = max(self._next_id.get(e.table, 0), int(e.row_id) + 1)
            elif e.op is EditOp.DELETE_ROW:
                if not ts.has_row(e.row_id):
                    if strict:
                        raise KeyError(f"cannot delete missing {e.table}#{e.row_id}")
                    continue
                ts.drop(e.row_id)
            else:
                if not ts.has_row(e.row_id) or e.column not in ts.columns:
                    if strict:
                        raise KeyError(f"cannot update {e.table}#{e.row_id}.{e.column}")
                    continue
                ts.set(e.row_id, e.column, e.value)


    def checksum(self) -> str:
        h = hashlib.blake2b(digest_size=16)
        h.update(str(self.time).encode())
        for tname in sorted(self.tables):
            ts = self.tables[tname]
            h.update(tname.encode())
            order = np.argsort(ts.row_ids, kind="stable")
            for cname in sorted(ts.columns):
                h.update(cname.encode())
                h.update(np.ascontiguousarray(ts.columns[cname][order]).tobytes())
        return h.hexdigest()

    def total_rows(self) -> int:
        return sum(ts.n_rows for ts in self.tables.values())

    def total_cells(self) -> int:
        return sum(ts.n_rows * len(ts.columns) for ts in self.tables.values())

    def describe(self, schema: Schema | None = None, max_rows: int = 6) -> str:
        lines = [f"RelationalState(t={self.time}, {self.total_rows()} rows)"]
        for tname in sorted(self.tables):
            ts = self.tables[tname]
            cols = list(ts.columns)
            lines.append(f"  {tname} ({ts.n_rows} rows): {', '.join(cols)}")
            for i in range(min(max_rows, ts.n_rows)):
                cells = []
                for c in cols:
                    v = ts.columns[c][i]
                    if schema is not None:
                        col = schema.table(tname).column(c)
                        if col.kind is ColumnKind.CATEGORICAL and col.categories:
                            idx = int(v)
                            v = col.categories[idx] if 0 <= idx < len(col.categories) else v
                    cells.append(f"{c}={v}")
                lines.append(f"      {' '.join(str(x) for x in cells)}")
            if ts.n_rows > max_rows:
                lines.append(f"      … {ts.n_rows - max_rows} more")
        return "\n".join(lines)


def diff_states(before: RelationalState, after: RelationalState, schema: Schema) -> StateDelta:
    delta = StateDelta()
    for table in schema:
        b, a = before.tables[table.name], after.tables[table.name]
        b_ids, a_ids = set(map(int, b.row_ids)), set(map(int, a.row_ids))
        is_event = table.kind is TableKind.EVENT

        for rid in sorted(a_ids - b_ids):
            values = tuple(
                (c.name, float(a.get(rid, c.name)))
                for c in table.columns
                if c.name != table.pk
            )
            delta.add(
                Edit(
                    op=EditOp.EMIT_EVENT if is_event else EditOp.INSERT_ROW,
                    table=table.name,
                    row_id=rid,
                    values=values,
                )
            )
        for rid in sorted(b_ids - a_ids):
            delta.add(Edit(op=EditOp.DELETE_ROW, table=table.name, row_id=rid))
        for rid in sorted(a_ids & b_ids):
            for c in table.columns:
                if c.name == table.pk:
                    continue
                bv, av = b.get(rid, c.name), a.get(rid, c.name)
                if bv != av:
                    delta.add(
                        Edit(
                            op=EditOp.UPDATE_STATE if c.is_state else EditOp.UPDATE_VALUE,
                            table=table.name,
                            row_id=rid,
                            column=c.name,
                            value=float(av),
                        )
                    )
    return delta


def states_equal(a: RelationalState, b: RelationalState) -> bool:
    return a.checksum() == b.checksum()


def clip_to_column(col: Column, value: float) -> float:
    if col.kind is ColumnKind.BOOL:
        return float(1 if value > 0.5 else 0)
    if col.kind is ColumnKind.CATEGORICAL:
        k = col.n_categories
        return float(min(max(int(round(value)), 0), k - 1))
    if col.kind is ColumnKind.INT:
        v = float(int(round(value)))
    else:
        v = float(value)
    if col.lo is not None:
        v = max(v, col.lo)
    if col.hi is not None:
        v = min(v, col.hi)
    if col.kind in (ColumnKind.INT, ColumnKind.DATETIME):
        v = float(int(round(v)))
    return v
