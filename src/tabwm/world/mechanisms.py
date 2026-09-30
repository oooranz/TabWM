"""WorldRel action mechanisms."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

import numpy as np

from tabwm.world.actions import ActionSchema, ArgKind
from tabwm.world.schema import ColumnKind, Schema, TableKind
from tabwm.world.state import Edit, EditOp, RelationalState, clip_to_column


class MechanismFamily(str, Enum):
    LOCAL = "local"
    CROSS_TABLE = "cross_table"
    MULTI_HOP = "multi_hop"
    CASCADE = "cascade"
    AGGREGATE = "aggregate"
    STATE_MACHINE = "state_machine"
    THRESHOLD = "threshold"
    STOCHASTIC = "stochastic"
    DELAYED = "delayed"
    EVENT = "event"
    STRUCTURAL = "structural"
    REGIME = "regime"


@dataclass
class EffectContext:

    state: RelationalState
    schema: Schema
    action: ActionSchema
    bindings: dict[str, float]
    rng: np.random.Generator
    regime: int = 0

    def entity(self, arg: str) -> tuple[str, int] | None:
        try:
            spec = self.action.arg(arg)
        except KeyError:
            return None
        if spec.kind is not ArgKind.ENTITY:
            return None
        rid = int(self.bindings.get(arg, -1))
        if not self.state.has_row(spec.table, rid):
            return None
        return spec.table, rid

    def scalar(self, arg: str, default: float = 0.0) -> float:
        return float(self.bindings.get(arg, default))


class Effect(abc.ABC):

    family: ClassVar[MechanismFamily]

    @abc.abstractmethod
    def apply(self, ctx: EffectContext) -> list[Edit]:
        pass

    @abc.abstractmethod
    def describe(self) -> str: ...

    def families(self) -> set[MechanismFamily]:
        return {self.family}


def _write(
    ctx: EffectContext, table: str, row_id: int, column: str, raw: float
) -> list[Edit]:
    col = ctx.schema.table(table).column(column)
    val = clip_to_column(col, raw)
    cur = ctx.state.get(table, row_id, column)
    if val == cur:
        return []
    if col.is_state and (int(cur), int(val)) not in set(col.state_machine or ()):
        return []
    op = EditOp.UPDATE_STATE if col.is_state else EditOp.UPDATE_VALUE
    return [Edit(op=op, table=table, row_id=row_id, column=column, value=val)]


def _walk(ctx: EffectContext, path: tuple[str, ...], row_id: int) -> list[int]:
    frontier = [row_id]
    for src, dst in zip(path, path[1:]):
        nxt: list[int] = []
        going_up = dst in ctx.schema.parents(src)
        for rid in frontier:
            if going_up:
                pid = ctx.state.parent_id(ctx.schema, src, rid, dst)
                if pid >= 0 and ctx.state.has_row(dst, pid):
                    nxt.append(pid)
            else:
                nxt.extend(int(x) for x in ctx.state.child_ids(ctx.schema, src, rid, dst))
        frontier = nxt
        if not frontier:
            break
    return frontier


@dataclass(frozen=True)
class LocalUpdate(Effect):

    family: ClassVar[MechanismFamily] = MechanismFamily.LOCAL

    arg: str
    table: str
    column: str
    a: float = 1.0
    b: float = 0.0
    c: float = 0.0
    scalar_arg: str = ""
    noise: float = 0.0

    def apply(self, ctx: EffectContext) -> list[Edit]:
        ent = ctx.entity(self.arg)
        if ent is None:
            return []
        table, rid = ent
        if table != self.table:
            return []
        x = ctx.state.get(table, rid, self.column)
        u = ctx.scalar(self.scalar_arg) if self.scalar_arg else 0.0
        raw = self.a * x + self.b * u + self.c
        if self.noise:
            raw += float(ctx.rng.normal(0.0, self.noise))
        return _write(ctx, table, rid, self.column, raw)

    def describe(self) -> str:
        u = f" + {self.b:g}·{self.scalar_arg}" if self.scalar_arg else ""
        eps = f" + N(0,{self.noise:g})" if self.noise else ""
        return f"{self.table}.{self.column} := {self.a:g}·x{u} + {self.c:g}{eps}"


@dataclass(frozen=True)
class CrossTableUpdate(Effect):

    family: ClassVar[MechanismFamily] = MechanismFamily.CROSS_TABLE

    arg: str
    path: tuple[str, ...]
    column: str
    a: float = 1.0
    b: float = 0.0
    c: float = 0.0
    scalar_arg: str = ""
    max_targets: int = 8

    @property
    def hops(self) -> int:
        return len(self.path) - 1

    def families(self) -> set[MechanismFamily]:
        return {MechanismFamily.MULTI_HOP if self.hops >= 2 else MechanismFamily.CROSS_TABLE}

    def apply(self, ctx: EffectContext) -> list[Edit]:
        ent = ctx.entity(self.arg)
        if ent is None or ent[0] != self.path[0]:
            return []
        targets = _walk(ctx, self.path, ent[1])[: self.max_targets]
        dst = self.path[-1]
        u = ctx.scalar(self.scalar_arg) if self.scalar_arg else 0.0
        edits: list[Edit] = []
        for rid in targets:
            x = ctx.state.get(dst, rid, self.column)
            edits.extend(_write(ctx, dst, rid, self.column, self.a * x + self.b * u + self.c))
        return edits

    def describe(self) -> str:
        u = f" + {self.b:g}·{self.scalar_arg}" if self.scalar_arg else ""
        return f"{'→'.join(self.path)}.{self.column} := {self.a:g}·x{u} + {self.c:g}"


@dataclass(frozen=True)
class Cascade(Effect):

    family: ClassVar[MechanismFamily] = MechanismFamily.CASCADE

    arg: str
    path: tuple[str, ...]
    column: str
    to_index: int
    max_targets: int = 12

    def apply(self, ctx: EffectContext) -> list[Edit]:
        ent = ctx.entity(self.arg)
        if ent is None or ent[0] != self.path[0]:
            return []
        dst = self.path[-1]
        edits: list[Edit] = []
        for rid in _walk(ctx, self.path, ent[1])[: self.max_targets]:
            edits.extend(_write(ctx, dst, rid, self.column, float(self.to_index)))
        return edits

    def describe(self) -> str:
        return f"cascade {'→'.join(self.path)}.{self.column} := {self.to_index}"


@dataclass(frozen=True)
class AggregateRecompute(Effect):

    family: ClassVar[MechanismFamily] = MechanismFamily.AGGREGATE

    table: str
    column: str
    arg: str = ""

    def apply(self, ctx: EffectContext) -> list[Edit]:
        col = ctx.schema.table(self.table).column(self.column)
        if col.aggregate_of is None:
            return []
        child_table, child_col, how = col.aggregate_of
        targets = self._targets(ctx, child_table)
        edits: list[Edit] = []
        for rid in targets:
            kids = ctx.state.child_ids(ctx.schema, self.table, rid, child_table)
            edits.extend(
                _write(ctx, self.table, rid, self.column,
                       _aggregate(ctx.state, child_table, child_col, kids, how))
            )
        return edits

    def _targets(self, ctx: EffectContext, child_table: str) -> list[int]:
        if self.arg:
            ent = ctx.entity(self.arg)
            if ent is None:
                return []
            table, rid = ent
            if table == self.table:
                return [rid]
            path = ctx.schema.fk_path(table, self.table)
            return _walk(ctx, path, rid) if path else []
        return [int(x) for x in ctx.state.row_ids(self.table)]

    def describe(self) -> str:
        return f"{self.table}.{self.column} := agg(children)"


def _aggregate(
    state: RelationalState, table: str, column: str, row_ids: np.ndarray, how: str
) -> float:
    if how == "count":
        return float(len(row_ids))
    if len(row_ids) == 0:
        return 0.0
    ts = state.tables[table]
    vals = np.array([ts.get(int(r), column) for r in row_ids], dtype=np.float64)
    if how == "sum":
        return float(vals.sum())
    if how == "mean":
        return float(vals.mean())
    return float(vals.max())


@dataclass(frozen=True)
class StateTransition(Effect):

    family: ClassVar[MechanismFamily] = MechanismFamily.STATE_MACHINE

    arg: str
    table: str
    column: str
    transition_map: tuple[int, ...]

    def apply(self, ctx: EffectContext) -> list[Edit]:
        ent = ctx.entity(self.arg)
        if ent is None or ent[0] != self.table:
            return []
        _, rid = ent
        cur = int(ctx.state.get(self.table, rid, self.column))
        if not 0 <= cur < len(self.transition_map):
            return []
        nxt = self.transition_map[cur]
        if nxt == cur:
            return []
        return _write(ctx, self.table, rid, self.column, float(nxt))

    def describe(self) -> str:
        moves = ", ".join(
            f"{i}→{t}" for i, t in enumerate(self.transition_map) if i != t
        )
        return f"{self.table}.{self.column}: {moves}"


@dataclass(frozen=True)
class ThresholdTrigger(Effect):

    family: ClassVar[MechanismFamily] = MechanismFamily.THRESHOLD

    arg: str
    table: str
    watch_column: str
    threshold: float
    above: bool
    set_column: str
    set_value: float

    def apply(self, ctx: EffectContext) -> list[Edit]:
        ent = ctx.entity(self.arg)
        if ent is None or ent[0] != self.table:
            return []
        _, rid = ent
        v = ctx.state.get(self.table, rid, self.watch_column)
        fired = v >= self.threshold if self.above else v <= self.threshold
        if not fired:
            return []
        return _write(ctx, self.table, rid, self.set_column, self.set_value)

    def describe(self) -> str:
        rel = ">=" if self.above else "<="
        return (f"if {self.table}.{self.watch_column} {rel} {self.threshold:g} "
                f"then {self.set_column} := {self.set_value:g}")


@dataclass(frozen=True)
class StochasticChoice(Effect):

    family: ClassVar[MechanismFamily] = MechanismFamily.STOCHASTIC

    p: float
    then: Effect
    otherwise: Effect | None = None

    def families(self) -> set[MechanismFamily]:
        out = {self.family} | self.then.families()
        if self.otherwise is not None:
            out |= self.otherwise.families()
        return out

    def apply(self, ctx: EffectContext) -> list[Edit]:
        if ctx.rng.random() < self.p:
            return self.then.apply(ctx)
        return self.otherwise.apply(ctx) if self.otherwise is not None else []

    def describe(self) -> str:
        alt = f" else {self.otherwise.describe()}" if self.otherwise else ""
        return f"w.p. {self.p:.2f} [{self.then.describe()}]{alt}"


@dataclass(frozen=True)
class RegimeSwitch(Effect):

    family: ClassVar[MechanismFamily] = MechanismFamily.REGIME

    branches: tuple[Effect, ...]

    def families(self) -> set[MechanismFamily]:
        out = {self.family}
        for b in self.branches:
            out |= b.families()
        return out

    def apply(self, ctx: EffectContext) -> list[Edit]:
        if not self.branches:
            return []
        return self.branches[ctx.regime % len(self.branches)].apply(ctx)

    def describe(self) -> str:
        return " | ".join(f"r{i}: {b.describe()}" for i, b in enumerate(self.branches))


@dataclass(frozen=True)
class EmitEvent(Effect):

    family: ClassVar[MechanismFamily] = MechanismFamily.EVENT

    arg: str
    event_table: str
    fk_column: str
    value_column: str = ""
    a: float = 1.0
    c: float = 0.0
    scalar_arg: str = ""

    def apply(self, ctx: EffectContext) -> list[Edit]:
        ent = ctx.entity(self.arg)
        if ent is None:
            return []
        _, rid = ent
        tab = ctx.schema.table(self.event_table)
        new_id = ctx.state.peek_id(self.event_table)
        values: list[tuple[str, float]] = []
        u = ctx.scalar(self.scalar_arg) if self.scalar_arg else 0.0
        for c in tab.columns:
            if c.name == tab.pk:
                continue
            if c.name == self.fk_column:
                values.append((c.name, float(rid)))
            elif c.name == self.value_column:
                values.append((c.name, clip_to_column(c, self.a * u + self.c)))
            elif c.kind is ColumnKind.DATETIME:
                values.append((c.name, float(ctx.state.time)))
            else:
                values.append((c.name, clip_to_column(c, 0.0)))
        return [
            Edit(op=EditOp.EMIT_EVENT, table=self.event_table, row_id=new_id,
                 values=tuple(values))
        ]

    def describe(self) -> str:
        return f"emit {self.event_table}({self.fk_column}←{self.arg})"


@dataclass(frozen=True)
class DeleteRow(Effect):

    family: ClassVar[MechanismFamily] = MechanismFamily.STRUCTURAL

    arg: str
    table: str

    def apply(self, ctx: EffectContext) -> list[Edit]:
        ent = ctx.entity(self.arg)
        if ent is None or ent[0] != self.table:
            return []
        return [Edit(op=EditOp.DELETE_ROW, table=self.table, row_id=ent[1])]

    def describe(self) -> str:
        return f"delete {self.table}[{self.arg}]"


@dataclass(frozen=True)
class ScheduleDelayed(Effect):

    family: ClassVar[MechanismFamily] = MechanismFamily.DELAYED

    arg: str
    pending_table: str
    fk_column: str
    target_column_id: int
    delay: int
    amount: float = 0.0
    scalar_arg: str = ""
    b: float = 0.0

    def apply(self, ctx: EffectContext) -> list[Edit]:
        ent = ctx.entity(self.arg)
        if ent is None:
            return []
        _, rid = ent
        tab = ctx.schema.table(self.pending_table)
        u = ctx.scalar(self.scalar_arg) if self.scalar_arg else 0.0
        payload = {
            self.fk_column: float(rid),
            "due_at": float(ctx.state.time + self.delay),
            "amount": self.amount + self.b * u,
            "op_id": float(self.target_column_id),
        }
        values = tuple(
            (c.name, clip_to_column(c, payload.get(c.name, 0.0)))
            for c in tab.columns
            if c.name != tab.pk
        )
        return [
            Edit(op=EditOp.EMIT_EVENT, table=self.pending_table,
                 row_id=ctx.state.peek_id(self.pending_table), values=values)
        ]

    def describe(self) -> str:
        return f"schedule {self.pending_table} (+{self.delay} ticks, col#{self.target_column_id})"


@dataclass(frozen=True)
class PendingSpec:

    pending_table: str
    target_table: str
    fk_column: str
    target_columns: tuple[str, ...]
    a: float = 1.0

    def fire(self, ctx: EffectContext) -> list[Edit]:
        ts = ctx.state.tables[self.pending_table]
        if ts.n_rows == 0:
            return []
        due = ts.columns["due_at"] <= ctx.state.time
        edits: list[Edit] = []
        for pos in np.flatnonzero(due):
            pid = int(ts.row_ids[pos])
            target = int(ts.columns[self.fk_column][pos])
            amount = float(ts.columns["amount"][pos])
            col_id = int(ts.columns["op_id"][pos]) % max(1, len(self.target_columns))
            column = self.target_columns[col_id]
            if ctx.state.has_row(self.target_table, target):
                x = ctx.state.get(self.target_table, target, column)
                edits.extend(_write(ctx, self.target_table, target, column, x + self.a * amount))
            edits.append(
                Edit(op=EditOp.DELETE_ROW, table=self.pending_table, row_id=pid)
            )
        return edits

    def describe(self) -> str:
        return (f"fire {self.pending_table} → {self.target_table}"
                f".{{{','.join(self.target_columns)}}} (+{self.a:g}·amount)")


@dataclass(frozen=True)
class ExogenousDrift(Effect):

    family: ClassVar[MechanismFamily] = MechanismFamily.LOCAL

    table: str
    column: str
    rate: float
    p: float = 1.0
    max_rows: int = 6

    def apply(self, ctx: EffectContext) -> list[Edit]:
        ids = ctx.state.row_ids(self.table)
        if len(ids) == 0:
            return []
        n = min(self.max_rows, len(ids))
        picked = ctx.rng.choice(len(ids), size=n, replace=False)
        edits: list[Edit] = []
        for i in picked:
            if ctx.rng.random() >= self.p:
                continue
            rid = int(ids[int(i)])
            x = ctx.state.get(self.table, rid, self.column)
            edits.extend(_write(ctx, self.table, rid, self.column, x + self.rate))
        return edits

    def describe(self) -> str:
        return f"drift {self.table}.{self.column} += {self.rate:g} (p={self.p:.2f})"


@dataclass(frozen=True)
class MechanismParams:

    n_effects: tuple[int, int] = (1, 4)
    p_local: float = 1.00
    p_state_machine: float = 0.60
    p_cross_table: float = 0.55
    p_multi_hop: float = 0.30
    p_cascade: float = 0.30
    p_aggregate: float = 0.50
    p_threshold: float = 0.30
    p_stochastic: float = 0.20
    p_delayed: float = 0.20
    p_event: float = 0.30
    p_structural: float = 0.06
    p_regime: float = 0.15
    p_noise: float = 0.20
    noise_scale: tuple[float, float] = (0.01, 0.15)
    delay_range: tuple[int, int] = (1, 4)
    n_regimes: tuple[int, int] = (2, 3)
    p_exogenous: float = 0.35
    max_hops: int = 3


def _num_columns(schema: Schema, table: str):
    return [
        c for c in schema.table(table).data_columns((ColumnKind.NUMBER, ColumnKind.INT))
        if not c.is_aggregate
    ]


def _state_columns(schema: Schema, table: str):
    return [c for c in schema.table(table).data_columns((ColumnKind.CATEGORICAL,)) if c.is_state]


def _cat_columns(schema: Schema, table: str):
    return list(schema.table(table).data_columns((ColumnKind.CATEGORICAL,)))


def sample_effects(
    rng: np.random.Generator,
    schema: Schema,
    action: ActionSchema,
    params: MechanismParams,
) -> tuple[Effect, ...]:
    table = action.primary_table
    if not table:
        return ()
    scalars = [a.name for a in action.args if a.kind is ArgKind.SCALAR]
    scalar = scalars[0] if scalars else ""
    effects: list[Effect] = []
    n_target = int(rng.integers(params.n_effects[0], params.n_effects[1] + 1))

    def noise() -> float:
        if rng.random() >= params.p_noise:
            return 0.0
        lo, hi = params.noise_scale
        return float(rng.uniform(lo, hi))


    nums = _num_columns(schema, table)
    if nums and rng.random() < params.p_local:
        c = nums[int(rng.integers(len(nums)))]
        style = rng.random()
        if style < 0.45:
            a, b, const = 1.0, (float(rng.uniform(0.3, 1.5)) if scalar else 0.0), \
                float(rng.uniform(-8, 8))
        elif style < 0.75:
            a, b, const = float(rng.uniform(0.4, 1.6)), 0.0, 0.0
        else:
            a, b, const = 0.0, 0.0, float(rng.uniform(0.0, float(c.hi or 1.0) * 0.6))
        effects.append(
            LocalUpdate(arg="target", table=table, column=c.name, a=a, b=b, c=const,
                        scalar_arg=scalar if b else "", noise=noise())
        )


    states = _state_columns(schema, table)
    if states and rng.random() < params.p_state_machine:
        c = states[int(rng.integers(len(states)))]
        tmap = []
        for i in range(c.n_categories):
            legal = c.legal_transitions(i)
            tmap.append(int(rng.choice(legal)) if legal else i)
        effects.append(
            StateTransition(arg="target", table=table, column=c.name,
                            transition_map=tuple(tmap))
        )


    neighbours = [*schema.parents(table), *schema.children(table)]
    if neighbours and rng.random() < params.p_cross_table:
        path = _sample_path(rng, schema, table, params)
        dst_nums = _num_columns(schema, path[-1]) if path else []
        if path and dst_nums:
            c = dst_nums[int(rng.integers(len(dst_nums)))]
            effects.append(
                CrossTableUpdate(
                    arg="target", path=path, column=c.name,
                    a=1.0, b=float(rng.uniform(-1.2, 1.2)) if scalar else 0.0,
                    c=float(rng.uniform(-5, 5)), scalar_arg=scalar,
                )
            )


    kids = schema.children(table)
    if kids and rng.random() < params.p_cascade:
        kid = str(rng.choice(kids))
        kid_cats = _cat_columns(schema, kid)
        if kid_cats:
            c = kid_cats[int(rng.integers(len(kid_cats)))]
            effects.append(
                Cascade(arg="target", path=(table, kid), column=c.name,
                        to_index=int(rng.integers(c.n_categories)))
            )


    if rng.random() < params.p_aggregate:
        touched = {table, *(e.path[-1] for e in effects if isinstance(e, CrossTableUpdate)),
                   *(e.path[-1] for e in effects if isinstance(e, Cascade))}
        for parent in schema.tables:
            for c in parent.columns:
                if c.is_aggregate and c.aggregate_of[0] in touched:
                    effects.append(
                        AggregateRecompute(table=parent.name, column=c.name)
                    )
                    break


    if nums and rng.random() < params.p_threshold:
        watch = nums[int(rng.integers(len(nums)))]
        cats = _cat_columns(schema, table)
        bools = list(schema.table(table).data_columns((ColumnKind.BOOL,)))
        lo, hi = float(watch.lo or 0.0), float(watch.hi or 1.0)
        if cats or bools:
            if cats and (not bools or rng.random() < 0.6):
                c = cats[int(rng.integers(len(cats)))]
                set_value = float(rng.integers(c.n_categories))
            else:
                c = bools[int(rng.integers(len(bools)))]
                set_value = 1.0
            effects.append(
                ThresholdTrigger(
                    arg="target", table=table, watch_column=watch.name,
                    threshold=float(rng.uniform(lo + 0.2 * (hi - lo), lo + 0.8 * (hi - lo))),
                    above=bool(rng.random() < 0.6),
                    set_column=c.name, set_value=set_value,
                )
            )


    event_kids = [k for k in kids if schema.table(k).kind is TableKind.EVENT]
    if event_kids and rng.random() < params.p_event:
        ev = str(rng.choice(event_kids))
        fk = schema.table(ev).fk_to(table)
        ev_nums = _num_columns(schema, ev)
        if fk is not None:
            effects.append(
                EmitEvent(
                    arg="target", event_table=ev, fk_column=fk.column,
                    value_column=ev_nums[0].name if ev_nums else "",
                    a=float(rng.uniform(0.5, 2.0)), c=float(rng.uniform(0, 5)),
                    scalar_arg=scalar,
                )
            )


    if rng.random() < params.p_structural and schema.table(table).kind is TableKind.EVENT:
        effects.append(DeleteRow(arg="target", table=table))


    if len(effects) > n_target + 2:
        effects = effects[: n_target + 2]


    if effects and rng.random() < params.p_stochastic:
        i = int(rng.integers(len(effects)))
        effects[i] = StochasticChoice(p=float(rng.uniform(0.25, 0.85)), then=effects[i])
    if effects and rng.random() < params.p_regime:
        i = int(rng.integers(len(effects)))
        base = effects[i]
        effects[i] = RegimeSwitch(branches=(base, _perturb(rng, base)))

    return tuple(effects)


def _sample_path(
    rng: np.random.Generator, schema: Schema, table: str, params: MechanismParams
) -> tuple[str, ...]:
    hops = 1
    while hops < params.max_hops and rng.random() < params.p_multi_hop:
        hops += 1
    path = [table]
    for _ in range(hops):
        head = path[-1]
        nbrs = [t for t in (*schema.parents(head), *schema.children(head)) if t not in path]
        if not nbrs:
            break
        path.append(str(rng.choice(nbrs)))
    return tuple(path)


def _perturb(rng: np.random.Generator, e: Effect) -> Effect:
    from dataclasses import replace as _replace

    if isinstance(e, LocalUpdate):
        return _replace(e, a=e.a * float(rng.uniform(0.3, 1.8)),
                        c=e.c + float(rng.uniform(-10, 10)))
    if isinstance(e, CrossTableUpdate):
        return _replace(e, a=e.a * float(rng.uniform(0.3, 1.8)),
                        c=e.c + float(rng.uniform(-6, 6)))
    if isinstance(e, ThresholdTrigger):
        return _replace(e, threshold=e.threshold * float(rng.uniform(0.4, 1.6)))
    if isinstance(e, Cascade):
        return _replace(e, to_index=e.to_index)
    return e


def sample_exogenous(
    rng: np.random.Generator, schema: Schema, params: MechanismParams
) -> tuple[ExogenousDrift, ...]:
    if rng.random() >= params.p_exogenous:
        return ()
    out: list[ExogenousDrift] = []
    for _ in range(1 + int(rng.random() < 0.3)):
        table = str(rng.choice(schema.table_names))
        nums = _num_columns(schema, table)
        if not nums:
            continue
        c = nums[int(rng.integers(len(nums)))]
        span = float(c.hi or 1.0) - float(c.lo or 0.0)
        out.append(
            ExogenousDrift(
                table=table, column=c.name,
                rate=float(rng.uniform(-0.05, 0.05) * span),
                p=float(rng.uniform(0.3, 1.0)),
                max_rows=int(rng.integers(1, 5)),
            )
        )
    return tuple(out)
