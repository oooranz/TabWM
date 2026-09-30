"""WorldRel actions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from tabwm.world.schema import ColumnKind, Schema
from tabwm.world.state import RelationalState

if TYPE_CHECKING:
    from tabwm.world.mechanisms import Effect


class ArgKind(str, Enum):
    ENTITY = "entity"
    SCALAR = "scalar"
    CATEGORY = "category"


@dataclass(frozen=True)
class ArgSpec:
    name: str
    kind: ArgKind
    table: str = ""
    lo: float = 0.0
    hi: float = 1.0
    is_int: bool = False
    categories: tuple[str, ...] = ()

    def sample(
        self, state: RelationalState, rng: np.random.Generator, *, valid: bool = True
    ) -> float | None:
        if self.kind is ArgKind.ENTITY:
            ids = state.row_ids(self.table)
            if not valid:
                return float(state.peek_id(self.table) + 1000)
            if len(ids) == 0:
                return None
            return float(ids[int(rng.integers(len(ids)))])
        if self.kind is ArgKind.CATEGORY:
            k = len(self.categories)
            return float(rng.integers(k)) if valid else float(k + 3)
        span = self.hi - self.lo
        if not valid:
            return float(self.hi + span * (0.5 + rng.random()))
        v = self.lo + rng.random() * span
        return float(int(round(v))) if self.is_int else float(v)


class PreconditionKind(str, Enum):
    COLUMN_IN = "column_in"
    COLUMN_CMP = "column_cmp"
    HAS_CHILDREN = "has_children"


@dataclass(frozen=True)
class Precondition:
    kind: PreconditionKind
    arg: str
    column: str = ""
    allowed: tuple[int, ...] = ()
    op: str = ">="
    threshold: float = 0.0
    child_table: str = ""
    min_count: int = 1

    def holds(
        self,
        state: RelationalState,
        schema: Schema,
        action: ActionSchema,
        bindings: dict[str, float],
    ) -> bool:
        spec = action.arg(self.arg)
        row_id = int(bindings[self.arg])
        if not state.has_row(spec.table, row_id):
            return False
        if self.kind is PreconditionKind.HAS_CHILDREN:
            kids = state.child_ids(schema, spec.table, row_id, self.child_table)
            return len(kids) >= self.min_count
        v = state.get(spec.table, row_id, self.column)
        if self.kind is PreconditionKind.COLUMN_IN:
            return int(v) in self.allowed
        if self.op == ">=":
            return v >= self.threshold
        if self.op == "<=":
            return v <= self.threshold
        if self.op == ">":
            return v > self.threshold
        return v < self.threshold

    def describe(self) -> str:
        if self.kind is PreconditionKind.HAS_CHILDREN:
            return f"|{self.arg}.{self.child_table}| >= {self.min_count}"
        if self.kind is PreconditionKind.COLUMN_IN:
            return f"{self.arg}.{self.column} in {self.allowed}"
        return f"{self.arg}.{self.column} {self.op} {self.threshold:g}"


@dataclass(frozen=True)
class ActionSchema:

    aid: int
    name: str
    args: tuple[ArgSpec, ...]
    preconditions: tuple[Precondition, ...] = ()
    effects: tuple[Effect, ...] = ()
    reward_weight: float = 0.0

    @property
    def anon_name(self) -> str:
        return f"ACTION_{self.aid}"

    def label(self, *, anonymous: bool) -> str:
        return self.anon_name if anonymous else self.name

    def arg(self, name: str) -> ArgSpec:
        for a in self.args:
            if a.name == name:
                return a
        raise KeyError(f"{self.name}.{name}")

    @property
    def entity_args(self) -> tuple[ArgSpec, ...]:
        return tuple(a for a in self.args if a.kind is ArgKind.ENTITY)

    @property
    def primary_table(self) -> str:
        ents = self.entity_args
        return ents[0].table if ents else ""

    def sample_bindings(
        self,
        state: RelationalState,
        rng: np.random.Generator,
        *,
        valid: bool = True,
    ) -> dict[str, float] | None:
        out: dict[str, float] = {}
        for spec in self.args:
            v = spec.sample(state, rng, valid=valid)
            if v is None:
                return None
            out[spec.name] = v
        return out

    def is_applicable(
        self, state: RelationalState, schema: Schema, bindings: dict[str, float]
    ) -> bool:
        for spec in self.entity_args:
            if not state.has_row(spec.table, int(bindings[spec.name])):
                return False
        return all(p.holds(state, schema, self, bindings) for p in self.preconditions)

    def describe(self, *, anonymous: bool = False) -> str:
        args = ", ".join(
            f"{a.name}:{a.kind.value}" + (f"→{a.table}" if a.table else "")
            for a in self.args
        )
        pre = "; ".join(p.describe() for p in self.preconditions)
        head = f"{self.label(anonymous=anonymous)}({args})"
        return head + (f"  [pre: {pre}]" if pre else "")


@dataclass(frozen=True)
class ActionInstance:

    aid: int
    bindings: dict[str, float] = field(default_factory=dict)

    def entity_pointers(self, schema_of: ActionSchema) -> tuple[tuple[str, str, int], ...]:
        return tuple(
            (a.name, a.table, int(self.bindings[a.name])) for a in schema_of.entity_args
        )

    def describe(self, schema_of: ActionSchema, *, anonymous: bool = False) -> str:
        parts = []
        for a in schema_of.args:
            v = self.bindings[a.name]
            if a.kind is ArgKind.ENTITY:
                parts.append(f"{a.name}={a.table}#{int(v)}")
            elif a.kind is ArgKind.CATEGORY:
                idx = int(v)
                nm = a.categories[idx] if 0 <= idx < len(a.categories) else f"?{idx}"
                parts.append(f"{a.name}={nm if not anonymous else idx}")
            else:
                parts.append(f"{a.name}={v:g}")
        return f"{schema_of.label(anonymous=anonymous)}({', '.join(parts)})"

    def key(self) -> tuple:
        return (self.aid, tuple(sorted((k, round(float(v), 9)) for k, v in self.bindings.items())))


_ACTION_VERBS = (
    "cancel", "approve", "submit", "settle", "refund", "ship", "restock",
    "transfer", "escalate", "close", "reserve", "release", "adjust", "flag",
    "merge", "split", "renew", "suspend", "activate", "archive", "audit",
    "dispatch", "reconcile", "expire",
)


def sample_action_name(
    rng: np.random.Generator, table: str, used: set[str]
) -> str:
    verb = str(rng.choice(_ACTION_VERBS))
    base = f"{verb}_{table.rstrip('s')}"
    name = base
    k = 2
    while name in used:
        name = f"{base}_{k}"
        k += 1
    used.add(name)
    return name


def sample_preconditions(
    rng: np.random.Generator,
    schema: Schema,
    ent: ArgSpec,
    p_precondition: float,
) -> tuple[Precondition, ...]:
    if rng.random() >= p_precondition:
        return ()
    table = schema.table(ent.table)
    out: list[Precondition] = []

    cats = [c for c in table.data_columns((ColumnKind.CATEGORICAL,))]
    nums = [c for c in table.data_columns((ColumnKind.NUMBER, ColumnKind.INT))]
    kids = schema.children(ent.table)

    choices: list[str] = []
    if cats:
        choices.append("in")
    if nums:
        choices.append("cmp")
    if kids:
        choices.append("kids")
    if not choices:
        return ()

    for _ in range(1 + int(rng.random() < 0.25)):
        which = str(rng.choice(choices))
        if which == "in":
            c = cats[int(rng.integers(len(cats)))]
            k = c.n_categories
            n_allowed = int(rng.integers(1, max(2, k)))
            allowed = tuple(sorted(int(x) for x in rng.choice(k, size=n_allowed, replace=False)))
            out.append(
                Precondition(kind=PreconditionKind.COLUMN_IN, arg=ent.name,
                             column=c.name, allowed=allowed)
            )
        elif which == "cmp":
            c = nums[int(rng.integers(len(nums)))]
            lo, hi = float(c.lo or 0.0), float(c.hi or 1.0)
            out.append(
                Precondition(
                    kind=PreconditionKind.COLUMN_CMP, arg=ent.name, column=c.name,
                    op=str(rng.choice([">=", "<="])),
                    threshold=float(lo + (hi - lo) * rng.uniform(0.05, 0.45)),
                )
            )
        else:
            out.append(
                Precondition(
                    kind=PreconditionKind.HAS_CHILDREN, arg=ent.name,
                    child_table=str(rng.choice(kids)), min_count=1,
                )
            )

    seen: set[tuple] = set()
    uniq: list[Precondition] = []
    for p in out:
        k = (p.kind, p.column, p.child_table)
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return tuple(uniq)


def sample_action_args(
    rng: np.random.Generator,
    schema: Schema,
    primary_table: str,
    *,
    p_second_entity: float = 0.30,
    p_scalar: float = 0.55,
    p_category: float = 0.25,
) -> tuple[ArgSpec, ...]:
    args: list[ArgSpec] = [
        ArgSpec(name="target", kind=ArgKind.ENTITY, table=primary_table)
    ]
    others = [t for t in schema.table_names if t != primary_table]
    if others and rng.random() < p_second_entity:
        args.append(
            ArgSpec(name="other", kind=ArgKind.ENTITY, table=str(rng.choice(others)))
        )
    if rng.random() < p_scalar:
        hi = float(rng.choice([10.0, 50.0, 100.0]))
        args.append(
            ArgSpec(name="amount", kind=ArgKind.SCALAR, lo=0.0, hi=hi,
                    is_int=bool(rng.random() < 0.5))
        )
    tab = schema.table(primary_table)
    cats = tab.data_columns((ColumnKind.CATEGORICAL,))
    if cats and rng.random() < p_category:
        c = cats[int(rng.integers(len(cats)))]
        args.append(
            ArgSpec(name="mode", kind=ArgKind.CATEGORY, categories=c.categories or ())
        )
    return tuple(args)


def sample_action_sequence_ids(
    rng: np.random.Generator, n_actions: int, *, permute: bool = False
) -> Sequence[int]:
    ids = list(range(n_actions))
    if permute:
        rng.shuffle(ids)
    return ids
