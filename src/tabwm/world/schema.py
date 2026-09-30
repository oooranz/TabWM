"""WorldRel relational schemas."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from enum import Enum

import numpy as np


class ColumnKind(str, Enum):

    NUMBER = "number"
    INT = "int"
    BOOL = "bool"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"

    @property
    def is_numeric(self) -> bool:
        return self in (ColumnKind.NUMBER, ColumnKind.INT)


class TableKind(str, Enum):

    ENTITY = "entity"
    EVENT = "event"


@dataclass(frozen=True)
class Column:

    name: str
    kind: ColumnKind
    lo: float | None = None
    hi: float | None = None
    categories: tuple[str, ...] | None = None
    state_machine: tuple[tuple[int, int], ...] | None = None
    aggregate_of: tuple[str, str, str] | None = None

    def __post_init__(self) -> None:
        if self.kind is ColumnKind.CATEGORICAL and not self.categories:
            raise ValueError(f"categorical column {self.name!r} needs categories")
        if self.kind.is_numeric and (self.lo is None or self.hi is None):
            raise ValueError(f"numeric column {self.name!r} needs lo/hi bounds")

    @property
    def n_categories(self) -> int:
        return len(self.categories) if self.categories else 0

    @property
    def is_state(self) -> bool:
        return self.state_machine is not None

    @property
    def is_aggregate(self) -> bool:
        return self.aggregate_of is not None

    def legal_transitions(self, from_idx: int) -> tuple[int, ...]:
        if self.state_machine is None:
            raise ValueError(f"{self.name!r} is not a state machine")
        return tuple(t for f, t in self.state_machine if f == from_idx)


@dataclass(frozen=True)
class ForeignKey:

    column: str
    ref_table: str


@dataclass(frozen=True)
class Table:
    name: str
    kind: TableKind
    pk: str
    columns: tuple[Column, ...]
    foreign_keys: tuple[ForeignKey, ...] = ()
    n_rows_init: int = 0

    def __post_init__(self) -> None:
        names = [c.name for c in self.columns]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate column names in {self.name!r}: {names}")
        if self.pk not in names:
            raise ValueError(f"pk {self.pk!r} not among columns of {self.name!r}")
        for fk in self.foreign_keys:
            if fk.column not in names:
                raise ValueError(f"fk column {fk.column!r} not in {self.name!r}")

    def column(self, name: str) -> Column:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(f"{self.name}.{name}")

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)

    @property
    def fk_columns(self) -> tuple[str, ...]:
        return tuple(fk.column for fk in self.foreign_keys)

    def fk_to(self, ref_table: str) -> ForeignKey | None:
        for fk in self.foreign_keys:
            if fk.ref_table == ref_table:
                return fk
        return None

    def data_columns(self, kinds: tuple[ColumnKind, ...] | None = None) -> tuple[Column, ...]:
        skip = {self.pk, *self.fk_columns}
        out = tuple(c for c in self.columns if c.name not in skip)
        if kinds is not None:
            out = tuple(c for c in out if c.kind in kinds)
        return out


@dataclass(frozen=True)
class Schema:

    tables: tuple[Table, ...]
    anonymous: bool = False

    def __post_init__(self) -> None:
        names = [t.name for t in self.tables]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate table names: {names}")
        known = set(names)
        for t in self.tables:
            for fk in t.foreign_keys:
                if fk.ref_table not in known:
                    raise ValueError(f"{t.name}.{fk.column} references unknown {fk.ref_table!r}")


    def table(self, name: str) -> Table:
        for t in self.tables:
            if t.name == name:
                return t
        raise KeyError(name)

    @property
    def table_names(self) -> tuple[str, ...]:
        return tuple(t.name for t in self.tables)

    def __iter__(self) -> Iterator[Table]:
        return iter(self.tables)

    def __len__(self) -> int:
        return len(self.tables)

    def parents(self, name: str) -> tuple[str, ...]:
        return tuple(fk.ref_table for fk in self.table(name).foreign_keys)

    def children(self, name: str) -> tuple[str, ...]:
        return tuple(t.name for t in self.tables if any(fk.ref_table == name for fk in t.foreign_keys))

    def topo_order(self) -> tuple[str, ...]:
        remaining = {t.name: set(self.parents(t.name)) for t in self.tables}
        order: list[str] = []
        while remaining:
            ready = sorted(n for n, deps in remaining.items() if not deps - set(order))
            if not ready:
                raise ValueError(f"cycle in FK graph among {sorted(remaining)}")
            order.extend(ready)
            for n in ready:
                remaining.pop(n)
        return tuple(order)

    def fk_depth(self) -> int:
        memo: dict[str, int] = {}

        def depth(n: str) -> int:
            if n not in memo:
                memo[n] = 0
                memo[n] = 1 + max((depth(p) for p in self.parents(n)), default=-1)
            return memo[n]

        return max((depth(t.name) for t in self.tables), default=0)

    def fk_path(self, src: str, dst: str, max_len: int = 4) -> tuple[str, ...] | None:
        frontier: list[tuple[str, ...]] = [(src,)]
        seen = {src}
        while frontier:
            nxt: list[tuple[str, ...]] = []
            for path in frontier:
                if len(path) > max_len:
                    return None
                head = path[-1]
                for nb in (*self.parents(head), *self.children(head)):
                    if nb == dst:
                        return (*path, nb)
                    if nb not in seen:
                        seen.add(nb)
                        nxt.append((*path, nb))
            frontier = nxt
        return None


    def anonymized(self) -> Schema:
        if self.anonymous:
            return self
        tmap = {t.name: f"table_{i}" for i, t in enumerate(self.tables)}
        cmaps = {
            t.name: {c.name: f"col_{j}" for j, c in enumerate(t.columns)} for t in self.tables
        }

        def anon_agg(agg: tuple[str, str, str] | None) -> tuple[str, str, str] | None:
            if agg is None:
                return None
            child_tbl, child_col, how = agg
            return (tmap[child_tbl], cmaps[child_tbl][child_col], how)

        new_tables = []
        for t in self.tables:
            cmap = cmaps[t.name]
            cols = tuple(
                replace(
                    c,
                    name=cmap[c.name],
                    categories=(
                        tuple(f"cat_{k}" for k in range(len(c.categories)))
                        if c.categories
                        else None
                    ),
                    aggregate_of=anon_agg(c.aggregate_of),
                )
                for c in t.columns
            )
            new_tables.append(
                replace(
                    t,
                    name=tmap[t.name],
                    pk=cmap[t.pk],
                    columns=cols,
                    foreign_keys=tuple(
                        ForeignKey(column=cmap[fk.column], ref_table=tmap[fk.ref_table])
                        for fk in t.foreign_keys
                    ),
                )
            )
        return Schema(tables=tuple(new_tables), anonymous=True)

    def rename_map(self) -> dict[str, dict[str, str]]:
        tmap = {t.name: f"table_{i}" for i, t in enumerate(self.tables)}
        out: dict[str, dict[str, str]] = {"__tables__": tmap}
        for t in self.tables:
            out[t.name] = {c.name: f"col_{j}" for j, c in enumerate(t.columns)}
        return out

    def describe(self) -> str:
        lines = [f"Schema({len(self.tables)} tables, depth={self.fk_depth()})"]
        for t in self.topo_order():
            tab = self.table(t)
            fks = ", ".join(f"{fk.column}→{fk.ref_table}" for fk in tab.foreign_keys)
            lines.append(f"  {tab.name} [{tab.kind.value}, n={tab.n_rows_init}] pk={tab.pk}"
                         + (f" fk=({fks})" if fks else ""))
            for c in tab.columns:
                bits = [c.kind.value]
                if c.categories:
                    bits.append(f"{len(c.categories)} cats")
                if c.is_state:
                    bits.append("state-machine")
                if c.is_aggregate:
                    bits.append(f"agg={c.aggregate_of}")
                if c.kind.is_numeric:
                    bits.append(f"[{c.lo:g},{c.hi:g}]")
                lines.append(f"      {c.name}: {', '.join(bits)}")
        return "\n".join(lines)


@dataclass(frozen=True)
class SchemaParams:

    n_tables: tuple[int, int] = (2, 12)
    n_tables_weights: tuple[tuple[int, int, float], ...] | None = None
    n_data_columns: tuple[int, int] = (2, 7)
    entity_rows: tuple[int, int] = (8, 40)
    event_rows: tuple[int, int] = (0, 12)
    event_table_frac: float = 0.35
    n_categories: tuple[int, int] = (2, 5)
    p_state_machine: float = 0.55
    p_aggregate: float = 0.45
    p_bool: float = 0.18
    p_int: float = 0.30
    p_datetime: float = 0.08
    extra_fk_frac: float = 0.25
    layout: tuple[str, ...] = ("tree", "reverse_tree", "layered", "preferential")

    @classmethod
    def join_empirical(cls, **overrides) -> SchemaParams:
        return cls(
            n_tables=(2, 24),
            n_tables_weights=(
                (2, 2, 0.166), (3, 4, 0.374), (5, 8, 0.310),
                (9, 12, 0.080), (13, 20, 0.047), (21, 40, 0.023),
            ),
            **overrides,
        )


_TABLE_WORDS = (
    "customers", "orders", "payments", "inventory", "shipments", "invoices",
    "accounts", "transactions", "tickets", "products", "suppliers", "warehouses",
    "employees", "reservations", "claims", "shipmentlines", "batches", "routes",
)
_NUM_WORDS = (
    "amount", "quantity", "balance", "price", "weight", "score", "credit",
    "stock", "total", "discount", "risk", "capacity", "fee", "units",
)
_CAT_WORDS = ("status", "stage", "tier", "category", "priority", "region", "state")
_BOOL_WORDS = ("is_active", "is_flagged", "is_paid", "is_locked", "is_open")
_TIME_WORDS = ("created_at", "updated_at", "due_at", "closed_at")
_STATUS_SETS = (
    ("pending", "approved", "fulfilled"),
    ("open", "in_progress", "closed"),
    ("draft", "submitted", "settled", "refunded"),
    ("new", "active", "suspended", "terminated"),
    ("queued", "running", "done", "failed"),
)


def _unique(rng: np.random.Generator, words: tuple[str, ...], n: int, used: set[str]) -> list[str]:
    pool = [w for w in words if w not in used]
    rng.shuffle(pool)
    out: list[str] = []
    for i in range(n):
        if pool:
            w = pool.pop()
        else:
            w = f"{words[i % len(words)]}_{i}"
            while w in used:
                w += "_x"
        used.add(w)
        out.append(w)
    return out


def _sample_n_tables(rng: np.random.Generator, params: SchemaParams) -> int:
    if not params.n_tables_weights:
        return int(rng.integers(params.n_tables[0], params.n_tables[1] + 1))
    los, his, ws = zip(*params.n_tables_weights)
    p = np.asarray(ws, dtype=np.float64)
    i = int(rng.choice(len(ws), p=p / p.sum()))
    return int(rng.integers(los[i], his[i] + 1))


def _sample_fk_edges(
    rng: np.random.Generator, n: int, params: SchemaParams
) -> list[tuple[int, int]]:
    if n < 2:
        return []
    layout = str(rng.choice(params.layout))
    edges: list[tuple[int, int]] = []

    if layout == "tree":
        for child in range(1, n):
            edges.append((child, int(rng.integers(0, child))))
    elif layout == "reverse_tree":

        for child in range(1, n):
            parent = 0 if rng.random() < 0.6 else int(rng.integers(0, child))
            edges.append((child, parent))
    elif layout == "layered":
        depth = max(2, int(rng.integers(2, min(5, n) + 1)))
        layers: list[list[int]] = [[] for _ in range(depth)]
        for i in range(n):
            layers[min(i, depth - 1) if i < depth else int(rng.integers(1, depth))].append(i)
        for li in range(1, depth):
            for child in layers[li]:
                if layers[li - 1]:
                    edges.append((child, int(rng.choice(layers[li - 1]))))
    else:
        deg = np.ones(n)
        for child in range(1, n):
            p = deg[:child] / deg[:child].sum()
            parent = int(rng.choice(child, p=p))
            deg[parent] += 1
            edges.append((child, parent))

    n_extra = int(round(params.extra_fk_frac * (n - 1)))
    for _ in range(n_extra):
        child = int(rng.integers(1, n))
        parent = int(rng.integers(0, child))
        edges.append((child, parent))


    rank = np.argsort(rng.permutation(n))
    oriented: set[tuple[int, int]] = set()
    for child, parent in edges:
        if child == parent:
            continue
        if rank[parent] > rank[child]:
            child, parent = parent, child
        oriented.add((child, parent))
    return sorted(oriented)


def sample_schema(rng: np.random.Generator, params: SchemaParams | None = None) -> Schema:
    params = params or SchemaParams()
    n = _sample_n_tables(rng, params)
    edges = _sample_fk_edges(rng, n, params)
    used_tables: set[str] = set()
    table_names = _unique(rng, _TABLE_WORDS, n, used_tables)
    return schema_from_topology(rng, table_names, edges, params)


def schema_from_topology(
    rng: np.random.Generator,
    table_names: list[str],
    edges: list[tuple[int, int]],
    params: SchemaParams | None = None,
    *,
    pk_names: list[str] | None = None,
    fk_columns: dict[tuple[int, int], str] | None = None,
) -> Schema:
    params = params or SchemaParams()
    n = len(table_names)
    parents_of: dict[int, list[int]] = {i: [] for i in range(n)}
    for child, parent in edges:
        parents_of[child].append(parent)
    has_children = {p for _, p in edges}


    kinds = []
    for i in range(n):
        is_leaf = i not in has_children and parents_of[i]
        kinds.append(
            TableKind.EVENT if (is_leaf and rng.random() < params.event_table_frac)
            else TableKind.ENTITY
        )


    drafts: list[dict] = []
    for i, name in enumerate(table_names):
        used_cols: set[str] = set()
        pk = (pk_names[i] if pk_names else "id")
        cols: list[Column] = [Column(name=pk, kind=ColumnKind.INT, lo=0, hi=1e9)]
        used_cols.add(pk)
        fks: list[ForeignKey] = []
        for p in parents_of[i]:
            col = (fk_columns or {}).get((i, p)) or f"{table_names[p]}_id"
            while col in used_cols:
                col += "_2"
            used_cols.add(col)
            cols.append(Column(name=col, kind=ColumnKind.INT, lo=0, hi=1e9))
            fks.append(ForeignKey(column=col, ref_table=table_names[p]))

        n_data = int(rng.integers(params.n_data_columns[0], params.n_data_columns[1] + 1))
        for _ in range(n_data):
            u = rng.random()
            if u < params.p_bool:
                cname = _unique(rng, _BOOL_WORDS, 1, used_cols)[0]
                cols.append(Column(name=cname, kind=ColumnKind.BOOL))
            elif u < params.p_bool + params.p_datetime:
                cname = _unique(rng, _TIME_WORDS, 1, used_cols)[0]
                cols.append(Column(name=cname, kind=ColumnKind.DATETIME, lo=0, hi=1e6))
            elif u < params.p_bool + params.p_datetime + 0.34:
                cname = _unique(rng, _CAT_WORDS, 1, used_cols)[0]
                if rng.random() < 0.6:
                    cats = tuple(_STATUS_SETS[int(rng.integers(len(_STATUS_SETS)))])
                else:
                    k = int(rng.integers(params.n_categories[0], params.n_categories[1] + 1))
                    cats = tuple(f"{cname}_{j}" for j in range(k))
                sm = None
                if rng.random() < params.p_state_machine:
                    sm = _sample_state_machine(rng, len(cats))
                cols.append(
                    Column(name=cname, kind=ColumnKind.CATEGORICAL, categories=cats,
                           state_machine=sm)
                )
            else:
                cname = _unique(rng, _NUM_WORDS, 1, used_cols)[0]
                is_int = rng.random() < params.p_int
                hi = float(rng.choice([50, 100, 500, 1000]))
                cols.append(
                    Column(
                        name=cname,
                        kind=ColumnKind.INT if is_int else ColumnKind.NUMBER,
                        lo=0.0,
                        hi=hi,
                    )
                )
        lo, hi = (params.event_rows if kinds[i] is TableKind.EVENT else params.entity_rows)
        drafts.append(
            dict(
                name=name,
                kind=kinds[i],
                pk=pk,
                columns=cols,
                foreign_keys=tuple(fks),
                n_rows_init=int(rng.integers(lo, hi + 1)),
            )
        )


    child_of: dict[str, list[str]] = {t: [] for t in table_names}
    for child, parent in edges:
        child_of[table_names[parent]].append(table_names[child])
    for d in drafts:
        kids = child_of[d["name"]]
        if not kids or rng.random() >= params.p_aggregate:
            continue
        cands = [
            c for c in d["columns"]
            if c.kind.is_numeric and c.name != d["pk"]
            and c.name not in {fk.column for fk in d["foreign_keys"]}
        ]
        if not cands:
            continue
        kid = str(rng.choice(kids))
        kid_draft = next(x for x in drafts if x["name"] == kid)
        kid_nums = [
            c for c in kid_draft["columns"]
            if c.kind.is_numeric and c.name != kid_draft["pk"]
            and c.name not in {fk.column for fk in kid_draft["foreign_keys"]}
        ]
        how = str(rng.choice(["sum", "count"] if kid_nums else ["count"]))
        src_col = str(rng.choice([c.name for c in kid_nums])) if kid_nums else kid_draft["pk"]
        tgt = cands[int(rng.integers(len(cands)))]

        d["columns"] = [
            replace(tgt, hi=max(tgt.hi or 0.0, 1e5), aggregate_of=(kid, src_col, how))
            if c.name == tgt.name else c
            for c in d["columns"]
        ]

    tables = tuple(
        Table(
            name=d["name"], kind=d["kind"], pk=d["pk"],
            columns=tuple(d["columns"]), foreign_keys=d["foreign_keys"],
            n_rows_init=d["n_rows_init"],
        )
        for d in drafts
    )
    return Schema(tables=tables)


def _sample_state_machine(rng: np.random.Generator, k: int) -> tuple[tuple[int, int], ...]:
    edges: set[tuple[int, int]] = {(i, i + 1) for i in range(k - 1)}
    for i in range(k):
        if rng.random() < 0.30:
            edges.add((i, i))
        if rng.random() < 0.25 and i >= 2:
            edges.add((i, int(rng.integers(0, i))))
        if rng.random() < 0.20 and i + 2 < k:
            edges.add((i, i + 2))
    return tuple(sorted(edges))
