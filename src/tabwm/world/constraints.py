"""WorldRel state constraints."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from tabwm.world.schema import ColumnKind, Schema
from tabwm.world.state import NULL_ID, RelationalState, clip_to_column


class ViolationKind(str, Enum):
    PK_DUPLICATE = "pk_duplicate"
    FK_DANGLING = "fk_dangling"
    RANGE = "range"
    CATEGORY = "category"
    AGGREGATE = "aggregate"
    STATE_TRANSITION = "state_transition"
    MISSING_ROW = "missing_row"


@dataclass(frozen=True)
class Violation:
    kind: ViolationKind
    table: str
    column: str = ""
    row_id: int = NULL_ID
    detail: str = ""

    def __repr__(self) -> str:
        loc = f"{self.table}" + (f"#{self.row_id}" if self.row_id != NULL_ID else "")
        loc += f".{self.column}" if self.column else ""
        return f"{self.kind.value}({loc}{': ' + self.detail if self.detail else ''})"


AGGREGATE_TOL = 1e-6


def check_state(
    state: RelationalState, schema: Schema, *, tol: float = AGGREGATE_TOL
) -> list[Violation]:
    out: list[Violation] = []
    for table in schema:
        ts = state.tables.get(table.name)
        if ts is None:
            out.append(Violation(ViolationKind.MISSING_ROW, table.name, detail="table absent"))
            continue

        ids = ts.row_ids
        if len(np.unique(ids)) != len(ids):
            out.append(
                Violation(ViolationKind.PK_DUPLICATE, table.name, table.pk,
                          detail=f"{len(ids) - len(np.unique(ids))} duplicates")
            )

        for fk in table.foreign_keys:
            vals = ts.columns[fk.column]
            ref = set(int(x) for x in state.row_ids(fk.ref_table))
            for v in np.unique(vals):
                iv = int(v)
                if iv != NULL_ID and iv not in ref:
                    out.append(
                        Violation(ViolationKind.FK_DANGLING, table.name, fk.column,
                                  detail=f"{iv} not in {fk.ref_table}")
                    )

        for col in table.columns:
            if col.name in (table.pk, *table.fk_columns):
                continue
            vals = ts.columns[col.name]
            if len(vals) == 0:
                continue
            if col.kind is ColumnKind.CATEGORICAL:
                bad = (vals < 0) | (vals >= col.n_categories)
                if bad.any():
                    out.append(
                        Violation(ViolationKind.CATEGORY, table.name, col.name,
                                  detail=f"{int(bad.sum())} out of [0,{col.n_categories})")
                    )
            elif col.kind is ColumnKind.BOOL:
                bad = (vals != 0) & (vals != 1)
                if bad.any():
                    out.append(
                        Violation(ViolationKind.RANGE, table.name, col.name,
                                  detail=f"{int(bad.sum())} non-boolean")
                    )
            elif col.lo is not None and col.hi is not None:
                bad = (vals < col.lo - tol) | (vals > col.hi + tol)
                if bad.any():
                    out.append(
                        Violation(ViolationKind.RANGE, table.name, col.name,
                                  detail=f"{int(bad.sum())} outside [{col.lo:g},{col.hi:g}]")
                    )

            if col.is_aggregate:
                out.extend(_check_aggregate(state, schema, table.name, col, tol))
    return out


def _check_aggregate(state, schema, table_name, col, tol) -> list[Violation]:
    from tabwm.world.mechanisms import _aggregate

    child_table, child_col, how = col.aggregate_of
    out: list[Violation] = []
    for rid in state.row_ids(table_name):
        rid = int(rid)
        kids = state.child_ids(schema, table_name, rid, child_table)


        want = clip_to_column(col, _aggregate(state, child_table, child_col, kids, how))
        got = state.get(table_name, rid, col.name)
        if abs(got - want) > max(tol, tol * abs(want)):
            out.append(
                Violation(ViolationKind.AGGREGATE, table_name, col.name, row_id=rid,
                          detail=f"{got:g} != {how}(children)={want:g}")
            )
    return out


def reachable_states(col) -> dict[int, set[int]]:
    edges = col.state_machine or ()
    k = col.n_categories
    reach: dict[int, set[int]] = {i: set() for i in range(k)}
    for f, t in edges:
        reach[f].add(t)
    changed = True
    while changed:
        changed = False
        for i in range(k):
            grown = set(reach[i])
            for j in list(reach[i]):
                grown |= reach[j]
            if grown != reach[i]:
                reach[i] = grown
                changed = True
    return reach


def check_transition(
    before: RelationalState, after: RelationalState, schema: Schema
) -> list[Violation]:
    out: list[Violation] = []
    for table in schema:
        b, a = before.tables.get(table.name), after.tables.get(table.name)
        if b is None or a is None:
            continue
        for col in table.columns:
            if not col.is_state:
                continue
            reach = reachable_states(col)
            for rid in a.row_ids:
                rid = int(rid)
                if not b.has_row(rid):
                    continue
                f, t = int(b.get(rid, col.name)), int(a.get(rid, col.name))
                if f != t and t not in reach.get(f, set()):
                    out.append(
                        Violation(ViolationKind.STATE_TRANSITION, table.name, col.name,
                                  row_id=rid, detail=f"{f}→{t} unreachable")
                    )
    return out


def violation_summary(violations: list[Violation]) -> dict[str, int]:
    out = {k.value: 0 for k in ViolationKind}
    for v in violations:
        out[v.kind.value] += 1
    return out


def is_valid(state: RelationalState, schema: Schema) -> bool:
    return not check_state(state, schema)
