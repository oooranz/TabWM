"""Context sampling."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from tabwm.world.actions import ActionInstance, ActionSchema
from tabwm.world.schema import Schema
from tabwm.world.state import NULL_ID, RelationalState, StateDelta

NodeRef = tuple[str, int]


@dataclass(frozen=True)
class ContextParams:

    max_cells: int = 384
    max_depth: int = 3
    max_children_per_hop: int = 6
    max_rows_per_table: int = 24
    n_background_rows: int = 4
    include_table_anchors: bool = True


@dataclass
class ContextSample:

    nodes: list[NodeRef]
    depth: dict[NodeRef, int]
    seeds: list[NodeRef]
    n_cells: int
    truncated: bool = False
    tables: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.nodes)

    def __contains__(self, ref: NodeRef) -> bool:
        return ref in self.depth

    def covers(self, delta: StateDelta) -> tuple[int, int]:
        from tabwm.world.state import EditOp

        hit = 0
        for e in delta:
            if e.op in (EditOp.INSERT_ROW, EditOp.EMIT_EVENT):
                hit += e.table in self.tables
            elif e.op is not EditOp.NOOP:
                hit += (e.table, int(e.row_id)) in self.depth
        return hit, len(delta.edits)

    def coverage(self, delta: StateDelta) -> float:
        hit, total = self.covers(delta)
        return hit / total if total else 1.0

    def max_depth_reached(self) -> int:
        return max(self.depth.values(), default=0)


def action_seeds(action: ActionInstance, schema_of: ActionSchema) -> list[NodeRef]:
    return [
        (table, row_id)
        for _, table, row_id in action.entity_pointers(schema_of)
        if row_id != NULL_ID
    ]


def sample_context(
    state: RelationalState,
    schema: Schema,
    seeds: list[NodeRef],
    params: ContextParams,
    rng: np.random.Generator,
) -> ContextSample:
    n_cols = {t.name: len(t.columns) for t in schema}
    nodes: list[NodeRef] = []
    depth: dict[NodeRef, int] = {}
    per_table: dict[str, int] = {}
    cells = 0
    truncated = False

    def admit(ref: NodeRef, d: int) -> bool:
        nonlocal cells, truncated
        if ref in depth:
            return False
        table, row_id = ref
        if not state.has_row(table, row_id):
            return False
        if per_table.get(table, 0) >= params.max_rows_per_table:
            return False
        cost = n_cols[table]
        if cells + cost > params.max_cells:
            truncated = True
            return False
        depth[ref] = d
        nodes.append(ref)
        per_table[table] = per_table.get(table, 0) + 1
        cells += cost
        return True

    queue: deque[tuple[NodeRef, int]] = deque()
    live_seeds = [s for s in seeds if state.has_row(*s)]
    for ref in live_seeds:
        if admit(ref, 0):
            queue.append((ref, 0))

    while queue and not truncated:
        (table, row_id), d = queue.popleft()
        if d >= params.max_depth:
            continue

        for parent in schema.parents(table):
            pid = state.parent_id(schema, table, row_id, parent)
            if pid != NULL_ID and admit((parent, pid), d + 1):
                queue.append(((parent, pid), d + 1))

        for child in schema.children(table):
            kids = state.child_ids(schema, table, row_id, child)
            if len(kids) > params.max_children_per_hop:


                kids = kids[rng.choice(len(kids), params.max_children_per_hop, replace=False)]
            for kid in kids:
                ref = (child, int(kid))
                if admit(ref, d + 1):
                    queue.append((ref, d + 1))


    for _ in range(params.n_background_rows):
        if truncated:
            break
        table = str(rng.choice(schema.table_names))
        ids = state.row_ids(table)
        if len(ids) == 0:
            continue
        admit((table, int(ids[int(rng.integers(len(ids)))])), params.max_depth + 1)

    return ContextSample(
        nodes=nodes,
        depth=depth,
        seeds=live_seeds,
        n_cells=cells,
        truncated=truncated,
        tables=tuple(schema.table_names) if params.include_table_anchors
        else tuple(sorted({t for t, _ in nodes})),
    )


def sample_context_for_action(
    state: RelationalState,
    schema: Schema,
    action: ActionInstance,
    action_schema: ActionSchema,
    params: ContextParams,
    rng: np.random.Generator,
) -> ContextSample:
    seeds = action_seeds(action, action_schema)
    if not seeds:


        table = str(rng.choice(schema.table_names))
        ids = state.row_ids(table)
        seeds = [(table, int(ids[0]))] if len(ids) else []
    return sample_context(state, schema, seeds, params, rng)


def shallow_context(depth: int) -> ContextParams:
    return ContextParams(max_depth=depth)


def context_stats(sample: ContextSample, schema: Schema) -> dict[str, float]:
    by_depth: dict[int, int] = {}
    for d in sample.depth.values():
        by_depth[d] = by_depth.get(d, 0) + 1
    return {
        "n_nodes": len(sample.nodes),
        "n_cells": sample.n_cells,
        "n_tables": len({t for t, _ in sample.nodes}),
        "max_depth": sample.max_depth_reached(),
        "truncated": float(sample.truncated),
        "nodes_at_depth_0": by_depth.get(0, 0),
        "nodes_beyond_depth_1": sum(c for d, c in by_depth.items() if d > 1),
    }
