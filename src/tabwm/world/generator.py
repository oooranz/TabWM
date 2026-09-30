"""Executable WorldRel worlds."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from tabwm.world.actions import (
    ActionInstance,
    ActionSchema,
    ArgKind,
    sample_action_args,
    sample_action_name,
    sample_preconditions,
)
from tabwm.world.constraints import Violation, check_state, check_transition
from tabwm.world.mechanisms import (
    EffectContext,
    ExogenousDrift,
    MechanismFamily,
    MechanismParams,
    PendingSpec,
    ScheduleDelayed,
    _aggregate,
    sample_effects,
    sample_exogenous,
)
from tabwm.world.schema import (
    Column,
    ColumnKind,
    ForeignKey,
    Schema,
    SchemaParams,
    Table,
    TableKind,
    sample_schema,
)
from tabwm.world.state import (
    NULL_ID,
    Edit,
    EditOp,
    RelationalState,
    StateDelta,
    clip_to_column,
    diff_states,
    dtype_for,
)

WORLDREL_VERSION = "0.1.0"


@dataclass(frozen=True)
class ActionParams:

    n_actions: tuple[int, int] = (4, 24)
    p_precondition: float = 0.45
    p_action_cost: float = 0.5
    action_cost_range: tuple[float, float] = (0.0, 2.0)


@dataclass(frozen=True)
class RewardParams:

    n_terms: tuple[int, int] = (1, 3)
    weight_range: tuple[float, float] = (-1.0, 1.0)


@dataclass(frozen=True)
class WorldConfig:

    schema: SchemaParams = field(default_factory=SchemaParams)
    mechanisms: MechanismParams = field(default_factory=MechanismParams)
    actions: ActionParams = field(default_factory=ActionParams)
    reward: RewardParams = field(default_factory=RewardParams)
    version: str = WORLDREL_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.blake2b(blob.encode(), digest_size=8).hexdigest()


@dataclass(frozen=True)
class RewardSpec:

    terms: tuple[tuple[str, str, float], ...] = ()
    action_costs: tuple[float, ...] = ()

    def potential(self, state) -> float:
        total = 0.0
        for table, column, w in self.terms:
            ts = state.tables.get(table)
            if ts is None or ts.n_rows == 0:
                continue
            total += w * float(np.sum(ts.columns[column]))
        return total

    def cost(self, aid: int) -> float:
        return self.action_costs[aid] if aid < len(self.action_costs) else 0.0

    def describe(self) -> str:
        return " + ".join(f"{w:+.2f}·Σ{t}.{c}" for t, c, w in self.terms) or "(none)"


@dataclass
class StepResult:

    state: RelationalState
    delta: StateDelta
    reward: float
    done: bool
    applicable: bool
    families: frozenset[MechanismFamily]
    violations: list[Violation] = field(default_factory=list)
    exogenous_delta: StateDelta = field(default_factory=StateDelta)

    @property
    def n_edits(self) -> int:
        return len(self.delta)

    @property
    def full_delta(self) -> StateDelta:
        return StateDelta(edits=[*self.delta.edits, *self.exogenous_delta.edits]).dedup()


@dataclass
class World:

    world_id: int
    seed: int
    schema: Schema
    actions: tuple[ActionSchema, ...]
    exogenous: tuple[ExogenousDrift, ...] = ()
    pending: tuple[PendingSpec, ...] = ()
    reward_spec: RewardSpec = field(default_factory=RewardSpec)
    regime: int = 0
    n_regimes: int = 1
    config: WorldConfig = field(default_factory=WorldConfig)

    _state: RelationalState | None = field(default=None, repr=False)
    _rng: np.random.Generator | None = field(default=None, repr=False)


    @property
    def prior_hash(self) -> str:
        return self.config.hash()

    def action(self, aid: int) -> ActionSchema:
        return self.actions[aid]

    @property
    def n_actions(self) -> int:
        return len(self.actions)

    def mechanism_families(self) -> frozenset[MechanismFamily]:
        out: set[MechanismFamily] = set()
        for a in self.actions:
            for e in a.effects:
                out |= e.families()
        if self.exogenous:
            out.add(MechanismFamily.LOCAL)
        if self.pending:
            out.add(MechanismFamily.DELAYED)
        if self.n_regimes > 1:
            out.add(MechanismFamily.REGIME)
        return frozenset(out)


    def reset(self, seed: int | None = None) -> RelationalState:
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, 1, 0 if seed is None else seed])
        )
        self._rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, 2, 0 if seed is None else seed])
        )
        self._state = sample_initial_state(rng, self.schema)
        return self._state

    @property
    def state(self) -> RelationalState:
        if self._state is None:
            return self.reset()
        return self._state

    def step(self, action: ActionInstance) -> StepResult:
        if self._state is None or self._rng is None:
            self.reset()
        result = self.transition(self._state, action, self._rng)
        self._state = result.state
        return result


    def transition(
        self,
        state: RelationalState,
        action: ActionInstance,
        rng: np.random.Generator | None = None,
        *,
        check: bool = False,
    ) -> StepResult:
        rng = rng if rng is not None else np.random.default_rng(self.seed)
        before = state
        work = state.copy()
        work.time = state.time + 1
        families: set[MechanismFamily] = set()

        def ctx(bindings: dict[str, float], schema_of: ActionSchema) -> EffectContext:
            return EffectContext(
                state=work, schema=self.schema, action=schema_of,
                bindings=bindings, rng=rng, regime=self.regime,
            )

        blank = ActionSchema(aid=-1, name="_system", args=())


        for spec in self.pending:
            edits = spec.fire(ctx({}, blank))
            if edits:
                families.add(MechanismFamily.DELAYED)
                work.apply_(_StateDelta(edits), strict=False)


        aschema = self.actions[action.aid]
        applicable = aschema.is_applicable(work, self.schema, action.bindings)
        if applicable:
            for effect in aschema.effects:
                edits = effect.apply(ctx(action.bindings, aschema))
                if edits:
                    families |= effect.families()
                    work.apply_(_StateDelta(edits), strict=False)


        families |= self._repair(work)


        delta = diff_states(before, work, self.schema)


        exo_before = work.copy()
        for drift in self.exogenous:
            edits = drift.apply(ctx({}, blank))
            if edits:
                work.apply_(_StateDelta(edits), strict=False)
        if self.exogenous:
            self._repair(work)
        exogenous_delta = diff_states(exo_before, work, self.schema)

        reward = (
            self.reward_spec.potential(work)
            - self.reward_spec.potential(before)
            - (self.reward_spec.cost(action.aid) if applicable else 0.0)
        )
        violations = (
            check_state(work, self.schema) + check_transition(before, work, self.schema)
            if check
            else []
        )
        return StepResult(
            exogenous_delta=exogenous_delta,
            state=work, delta=delta, reward=float(reward), done=False,
            applicable=applicable, families=frozenset(families), violations=violations,
        )

    def _repair(self, work: RelationalState) -> set[MechanismFamily]:
        fired: set[MechanismFamily] = set()
        if _repair_referential(work, self.schema):
            fired.add(MechanismFamily.CASCADE)
        if _repair_aggregates(work, self.schema):
            fired.add(MechanismFamily.AGGREGATE)
        return fired


    def bindable_actions(self, state: RelationalState) -> list[int]:
        out: list[int] = []
        for k, sch in enumerate(self.actions):
            if all(state.n_rows(a.table) > 0 for a in sch.entity_args):
                out.append(k)
        return out

    def sample_action(
        self,
        state: RelationalState,
        rng: np.random.Generator,
        *,
        aid: int | None = None,
        valid: bool = True,
        require_applicable: bool = False,
        tries: int = 8,
    ) -> ActionInstance | None:
        if aid is not None:
            pool = [aid]
        else:
            pool = self.bindable_actions(state) if valid else list(range(self.n_actions))
            if not pool:
                return None
        fallback: ActionInstance | None = None
        for _ in range(tries):
            k = pool[int(rng.integers(len(pool)))]
            sch = self.actions[k]
            b = sch.sample_bindings(state, rng, valid=valid)
            if b is None:
                continue
            inst = ActionInstance(aid=k, bindings=b)
            if not require_applicable or sch.is_applicable(state, self.schema, b):
                return inst
            fallback = fallback or inst


        return fallback

    def candidate_actions(
        self, state: RelationalState, rng: np.random.Generator, n: int
    ) -> list[ActionInstance]:
        out: list[ActionInstance] = []
        for _ in range(n * 4):
            if len(out) >= n:
                break
            inst = self.sample_action(state, rng, require_applicable=True)
            if inst is not None:
                out.append(inst)
        return out


    def describe(self, *, anonymous: bool = False) -> str:
        lines = [
            f"World(id={self.world_id}, seed={self.seed}, prior={self.prior_hash}, "
            f"regime={self.regime}/{self.n_regimes})",
            self.schema.describe(),
            f"reward: {self.reward_spec.describe()}",
            f"actions ({self.n_actions}):",
        ]
        for a in self.actions:
            lines.append(f"  {a.describe(anonymous=anonymous)}")
            for e in a.effects:
                lines.append(f"      · {e.describe()}")
        for d in self.exogenous:
            lines.append(f"  exogenous: {d.describe()}")
        for p in self.pending:
            lines.append(f"  pending: {p.describe()}")
        lines.append(f"families: {sorted(f.value for f in self.mechanism_families())}")
        return "\n".join(lines)

    def with_regime(self, regime: int) -> World:
        return replace(self, regime=regime % max(1, self.n_regimes), _state=None, _rng=None)


def _StateDelta(edits: list) -> StateDelta:
    return StateDelta(edits=list(edits))


def _repair_referential(state, schema: Schema) -> list:
    edits: list = []
    for _ in range(len(schema)):
        round_edits: list = []
        for table in schema:
            ts = state.tables[table.name]
            if ts.n_rows == 0:
                continue
            doomed: set[int] = set()
            for fk in table.foreign_keys:
                ref = set(int(x) for x in state.row_ids(fk.ref_table))
                vals = ts.columns[fk.column]
                for pos in range(len(vals)):
                    v = int(vals[pos])
                    if v != NULL_ID and v not in ref:
                        doomed.add(int(ts.row_ids[pos]))
            for rid in sorted(doomed):
                round_edits.append(Edit(op=EditOp.DELETE_ROW, table=table.name, row_id=rid))
        if not round_edits:
            break
        state.apply_(_StateDelta(round_edits), strict=False)
        edits.extend(round_edits)
    return edits


def _repair_aggregates(state, schema: Schema) -> list:
    edits: list = []
    for tname in reversed(schema.topo_order()):
        table = schema.table(tname)
        for col in table.columns:
            if not col.is_aggregate:
                continue
            child_table, child_col, how = col.aggregate_of
            ts = state.tables[tname]
            for rid in list(int(x) for x in ts.row_ids):
                kids = state.child_ids(schema, tname, rid, child_table)
                want = clip_to_column(col, _aggregate(state, child_table, child_col, kids, how))
                if want != state.get(tname, rid, col.name):
                    e = Edit(op=EditOp.UPDATE_VALUE, table=tname, row_id=rid,
                             column=col.name, value=want)
                    state.apply_(_StateDelta([e]), strict=False)
                    edits.append(e)
    return edits


def sample_initial_state(rng: np.random.Generator, schema: Schema):
    state = RelationalState.empty(schema)
    for tname in schema.topo_order():
        table = schema.table(tname)
        n = table.n_rows_init
        cols: dict[str, np.ndarray] = {}
        cols[table.pk] = np.arange(n, dtype=np.int64)
        for fk in table.foreign_keys:
            parent_ids = state.row_ids(fk.ref_table)
            if len(parent_ids) == 0:
                cols[fk.column] = np.full(n, NULL_ID, dtype=np.int64)
            else:
                cols[fk.column] = parent_ids[rng.integers(0, len(parent_ids), size=n)].astype(
                    np.int64
                )
        for col in table.columns:
            if col.name in cols:
                continue
            cols[col.name] = _sample_column(rng, col, n)
        state.tables[tname].columns = cols
        state.tables[tname]._reindex()
        state._next_id[tname] = n

    _repair_aggregates(state, schema)
    return state


def _sample_column(rng: np.random.Generator, col: Column, n: int) -> np.ndarray:
    if n == 0:
        return np.zeros(0, dtype=dtype_for(col.kind))
    if col.kind is ColumnKind.BOOL:
        p = float(rng.uniform(0.15, 0.85))
        return (rng.random(n) < p).astype(np.int64)
    if col.kind is ColumnKind.CATEGORICAL:
        k = col.n_categories
        if col.is_state:

            w = np.exp(-np.arange(k) * float(rng.uniform(0.3, 1.2)))
        else:
            w = rng.dirichlet(np.ones(k) * float(rng.uniform(0.4, 3.0)))
        w = w / w.sum()
        return rng.choice(k, size=n, p=w).astype(np.int64)
    lo, hi = float(col.lo or 0.0), float(col.hi or 1.0)
    shape = rng.random()
    if shape < 0.35:
        raw = rng.uniform(lo, hi, size=n)
    elif shape < 0.70:
        mu = rng.uniform(lo + 0.15 * (hi - lo), lo + 0.85 * (hi - lo))
        raw = rng.normal(mu, 0.12 * (hi - lo), size=n)
    else:
        raw = lo + (hi - lo) * rng.beta(float(rng.uniform(0.6, 3.0)),
                                        float(rng.uniform(0.6, 3.0)), size=n)
    raw = np.clip(raw, lo, hi)
    if col.kind in (ColumnKind.INT, ColumnKind.DATETIME):
        return np.rint(raw).astype(np.int64)
    return raw.astype(np.float64)


def _add_pending_table(
    rng: np.random.Generator, schema: Schema, target: str
) -> tuple[Schema, PendingSpec] | None:
    tab = schema.table(target)
    nums = [c for c in tab.data_columns((ColumnKind.NUMBER, ColumnKind.INT)) if not c.is_aggregate]
    if not nums:
        return None
    name = f"pending_{target}"
    if name in schema.table_names:
        return None
    fk_col = f"{target}_id"
    pending = Table(
        name=name,
        kind=TableKind.EVENT,
        pk="id",
        columns=(
            Column(name="id", kind=ColumnKind.INT, lo=0, hi=1e9),
            Column(name=fk_col, kind=ColumnKind.INT, lo=0, hi=1e9),
            Column(name="due_at", kind=ColumnKind.INT, lo=0, hi=1e6),
            Column(name="amount", kind=ColumnKind.NUMBER, lo=-1e4, hi=1e4),
            Column(name="op_id", kind=ColumnKind.INT, lo=0, hi=64),
        ),
        foreign_keys=(ForeignKey(column=fk_col, ref_table=target),),
        n_rows_init=0,
    )
    spec = PendingSpec(
        pending_table=name,
        target_table=target,
        fk_column=fk_col,
        target_columns=tuple(c.name for c in nums[:4]),
        a=float(rng.uniform(0.5, 1.5)),
    )
    return Schema(tables=(*schema.tables, pending), anonymous=schema.anonymous), spec


def sample_world(
    seed: int,
    config: WorldConfig | None = None,
    *,
    world_id: int | None = None,
    regime: int | None = None,
    schema: Schema | None = None,
) -> World:
    config = config or WorldConfig()
    ss = np.random.SeedSequence(seed)
    r_schema, r_actions, r_mech, r_reward = (
        np.random.default_rng(s) for s in ss.spawn(4)
    )

    schema = schema if schema is not None else _schema_with_actionable_table(
        r_schema, config.schema)


    pending: list[PendingSpec] = []
    if r_mech.random() < config.mechanisms.p_delayed:
        target = str(r_mech.choice(schema.table_names))
        got = _add_pending_table(r_mech, schema, target)
        if got is not None:
            schema, spec = got
            pending.append(spec)

    n_regimes = (
        int(r_mech.integers(config.mechanisms.n_regimes[0], config.mechanisms.n_regimes[1] + 1))
        if r_mech.random() < config.mechanisms.p_regime
        else 1
    )

    actions = _sample_actions(r_actions, r_mech, schema, config, pending)
    exogenous = sample_exogenous(r_mech, schema, config.mechanisms)
    reward_spec = _sample_reward(r_reward, schema, config, len(actions))

    return World(
        world_id=seed if world_id is None else world_id,
        seed=seed,
        schema=schema,
        actions=actions,
        exogenous=exogenous,
        pending=tuple(pending),
        reward_spec=reward_spec,
        regime=(int(r_mech.integers(n_regimes)) if regime is None else regime % n_regimes),
        n_regimes=n_regimes,
        config=config,
    )


def _schema_with_actionable_table(
    rng: np.random.Generator, params: SchemaParams, tries: int = 12
) -> Schema:
    schema = sample_schema(rng, params)
    for _ in range(tries):
        if any(t.data_columns() for t in schema):
            return schema
        schema = sample_schema(rng, params)
    return schema


def _sample_actions(
    r_actions: np.random.Generator,
    r_mech: np.random.Generator,
    schema: Schema,
    config: WorldConfig,
    pending: list[PendingSpec],
) -> tuple[ActionSchema, ...]:
    ap = config.actions
    n = int(r_actions.integers(ap.n_actions[0], ap.n_actions[1] + 1))


    actionable = [t.name for t in schema if t.data_columns() and t.n_rows_init > 0]
    if not actionable:
        actionable = [t.name for t in schema if t.n_rows_init > 0] or list(schema.table_names)
    used_names: set[str] = set()
    out: list[ActionSchema] = []
    for aid in range(n):
        table = str(r_actions.choice(actionable))
        args = sample_action_args(r_actions, schema, table)
        ent = args[0]
        pre = sample_preconditions(r_actions, schema, ent, ap.p_precondition)
        draft = ActionSchema(
            aid=aid, name=sample_action_name(r_actions, table, used_names),
            args=args, preconditions=pre,
        )
        effects = list(sample_effects(r_mech, schema, draft, config.mechanisms))

        for spec in pending:
            if spec.target_table == table and r_mech.random() < config.mechanisms.p_delayed:
                scalars = [a.name for a in args if a.kind is ArgKind.SCALAR]
                effects.append(
                    ScheduleDelayed(
                        arg="target", pending_table=spec.pending_table,
                        fk_column=spec.fk_column,
                        target_column_id=int(r_mech.integers(len(spec.target_columns))),
                        delay=int(r_mech.integers(
                            config.mechanisms.delay_range[0],
                            config.mechanisms.delay_range[1] + 1)),
                        amount=float(r_mech.uniform(-15, 15)),
                        scalar_arg=scalars[0] if scalars else "",
                        b=float(r_mech.uniform(-1.0, 1.0)) if scalars else 0.0,
                    )
                )
                break
        cost = (
            float(r_actions.uniform(*ap.action_cost_range))
            if r_actions.random() < ap.p_action_cost
            else 0.0
        )
        out.append(replace(draft, effects=tuple(effects), reward_weight=cost))
    return tuple(out)


def _sample_reward(
    rng: np.random.Generator, schema: Schema, config: WorldConfig, n_actions: int
) -> RewardSpec:
    rp = config.reward
    candidates = [
        (t.name, c.name)
        for t in schema
        for c in t.data_columns((ColumnKind.NUMBER, ColumnKind.INT))
    ]
    if not candidates:
        return RewardSpec(action_costs=tuple([0.0] * n_actions))
    k = min(len(candidates), int(rng.integers(rp.n_terms[0], rp.n_terms[1] + 1)))
    picks = rng.choice(len(candidates), size=k, replace=False)
    terms = tuple(
        (*candidates[int(i)], float(rng.uniform(*rp.weight_range))) for i in picks
    )
    costs = tuple(float(rng.uniform(0.0, 1.0)) for _ in range(n_actions))
    return RewardSpec(terms=terms, action_costs=costs)


def sample_worlds(
    seeds: Iterable[int], config: WorldConfig | None = None
) -> list[World]:
    return [sample_world(s, config) for s in seeds]
