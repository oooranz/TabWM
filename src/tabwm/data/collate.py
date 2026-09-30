"""Relational task batching."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Protocol

import numpy as np

from tabwm.data.delta import (
    ColumnStats,
    EditTarget,
    TokenIndex,
    encode_cell,
    encode_targets,
)
from tabwm.data.sampler import ContextParams, ContextSample, sample_context_for_action
from tabwm.data.transition import TabWMTask, TransitionExample
from tabwm.world.actions import ActionInstance, ActionSchema, ArgKind
from tabwm.world.constraints import reachable_states
from tabwm.world.schema import Column, ColumnKind, Schema
from tabwm.world.state import EditOp, RelationalState


SEM_NUMBER, SEM_TEXT, SEM_DATETIME, SEM_BOOLEAN = 0, 1, 2, 3

from tabwm.model.heads import MAX_CATEGORIES

MAX_F2P_NBRS = 5


class Role:

    PAD = 0
    STATE_BEFORE = 1
    ACTION = 2
    STATE_DELTA_DEMO = 3
    STATE_QUERY = 4
    TABLE_ANCHOR = 5
    N_ROLES = 6


QUERY_TRANSITION = 0

ACTION_TABLE = "__action__"
DELETED_COL = "__deleted__"
INSERTED_COL = "__inserted__"
ANCHOR_COL = "__table__"


class TextEmbedder(Protocol):

    d_text: int

    def __call__(self, texts: list[str]) -> np.ndarray: ...


class HashEmbedder:

    def __init__(self, d_text: int = 384) -> None:
        self.d_text = d_text
        self._cache: dict[str, np.ndarray] = {}

    def __call__(self, texts: list[str]) -> np.ndarray:
        out = np.empty((len(texts), self.d_text), dtype=np.float32)
        for i, t in enumerate(texts):
            v = self._cache.get(t)
            if v is None:
                seed = int.from_bytes(hashlib.blake2b(t.encode(), digest_size=8).digest(), "big")
                v = np.random.default_rng(seed).normal(0, 1, self.d_text).astype(np.float32)
                v /= np.linalg.norm(v) + 1e-8
                self._cache[t] = v
            out[i] = v
        return out


class SentenceEmbedder:

    def __init__(self, model_name: str = "all-MiniLM-L12-v2", device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(f"sentence-transformers/{model_name}", device=device)
        self.d_text = int(self.model.get_sentence_embedding_dimension())
        self._cache: dict[str, np.ndarray] = {}

    def __call__(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            embs = self.model.encode(missing, convert_to_numpy=True, show_progress_bar=False)
            for t, e in zip(missing, embs):
                self._cache[t] = e.astype(np.float32)
        return np.stack([self._cache[t] for t in texts])


@dataclass(frozen=True)
class LengthMixture:

    lengths: tuple[int, ...] = (1024, 2048, 4096)
    weights: tuple[float, ...] = (0.62, 0.26, 0.12)
    p_step_up: float = 0.30

    def sample(self, rng: np.random.Generator, min_length: int = 0) -> int:
        eligible = [i for i, n in enumerate(self.lengths) if n >= min_length]
        if not eligible:
            return max(self.lengths)
        if min_length > 0:
            i = eligible[0]
            if i + 1 < len(self.lengths) and rng.random() < self.p_step_up:
                i += 1
            return int(self.lengths[i])
        p = np.asarray([self.weights[i] for i in eligible], dtype=np.float64)
        return int(self.lengths[eligible[int(rng.choice(len(eligible), p=p / p.sum()))]])

    def expected_cost(self) -> float:
        base = self.lengths[0] ** 2
        return sum(w * (n**2) / base for n, w in zip(self.lengths, self.weights))


@dataclass(frozen=True)
class CollateParams:
    ctx_len: int = 1024
    length_mixture: LengthMixture | None = None
    d_text: int = 384
    query_budget_frac: float = 0.40
    max_edit_slots: int = 16
    context: ContextParams = field(default_factory=ContextParams)
    min_demo_cells: int = 32
    length_buckets: tuple[int, ...] = (256, 512, 1024, 2048, 4096, 8192)
    pad_to: int = 128
    max_demo_delta_cells: int = 8


def _round_up(n: int, m: int) -> int:
    return ((n + m - 1) // m) * m


def _bucket_length(used: int, params: CollateParams) -> int:
    if not params.length_buckets:
        return max(params.pad_to, _round_up(used, params.pad_to))
    for b in sorted(params.length_buckets):
        if b >= used:
            return b
    return max(params.length_buckets)


def required_ctx_len(tasks: list, params: CollateParams) -> int:
    k_max = max((t.k for t in tasks), default=0)
    if k_max == 0:
        return 0
    demo_frac = 1.0 - params.query_budget_frac
    return int(np.ceil(k_max * params.min_demo_cells / max(demo_frac, 1e-9)))


@dataclass
class TabWMBatch:


    node_idxs: np.ndarray
    f2p_nbr_idxs: np.ndarray
    table_name_idxs: np.ndarray
    col_name_idxs: np.ndarray
    col_name_values: np.ndarray
    text_values: np.ndarray
    number_values: np.ndarray
    datetime_values: np.ndarray
    boolean_values: np.ndarray
    sem_types: np.ndarray
    is_padding: np.ndarray
    masks: np.ndarray


    role_ids: np.ndarray
    transition_ids: np.ndarray
    is_row_anchor: np.ndarray
    cell_n_classes: np.ndarray
    cell_legal_classes: np.ndarray


    edit_op: np.ndarray
    edit_cell_pos: np.ndarray
    edit_row_pos: np.ndarray
    edit_table_pos: np.ndarray
    edit_link_pos: np.ndarray
    edit_value_kind: np.ndarray
    edit_value_num: np.ndarray
    edit_value_tol_z: np.ndarray
    edit_value_class: np.ndarray
    edit_valid: np.ndarray
    edit_in_context: np.ndarray


    tasks: list[TabWMTask] = field(default_factory=list)
    indices: list[TokenIndex] = field(default_factory=list)
    stats: list[ColumnStats] = field(default_factory=list)
    reverse: list[dict[int, tuple[str, int, str]]] = field(default_factory=list)
    table_of_pos: list[dict[int, str]] = field(default_factory=list)
    diagnostics: list[dict] = field(default_factory=list)

    @property
    def batch_size(self) -> int:
        return self.node_idxs.shape[0]

    @property
    def seq_len(self) -> int:
        return self.node_idxs.shape[1]

    def torch(self, device: str = "cpu", dtype=None):
        import torch

        bf = dtype or torch.bfloat16
        t = lambda x, d=None: torch.as_tensor(x, device=device) if d is None else (
            torch.as_tensor(x, device=device).to(d)
        )
        return {
            "node_idxs": t(self.node_idxs),
            "f2p_nbr_idxs": t(self.f2p_nbr_idxs),
            "table_name_idxs": t(self.table_name_idxs),
            "col_name_idxs": t(self.col_name_idxs),
            "col_name_values": t(self.col_name_values, bf),
            "text_values": t(self.text_values, bf),
            "number_values": t(self.number_values, bf),


            "number_values_exact": t(self.number_values, torch.float32),
            "datetime_values": t(self.datetime_values, bf),
            "boolean_values": t(self.boolean_values, bf),
            "sem_types": t(self.sem_types),
            "is_padding": t(self.is_padding),
            "masks": t(self.masks),
            "role_ids": t(self.role_ids),
            "transition_ids": t(self.transition_ids),
            "is_row_anchor": t(self.is_row_anchor),
            "cell_n_classes": t(self.cell_n_classes),
            "cell_legal_classes": t(self.cell_legal_classes),
            "edit_op": t(self.edit_op),
            "edit_cell_pos": t(self.edit_cell_pos),
            "edit_row_pos": t(self.edit_row_pos),
            "edit_table_pos": t(self.edit_table_pos),
            "edit_link_pos": t(self.edit_link_pos),
            "edit_value_kind": t(self.edit_value_kind),
            "edit_value_num": t(self.edit_value_num, torch.float32),
            "edit_value_tol_z": t(self.edit_value_tol_z, torch.float32),
            "edit_value_class": t(self.edit_value_class),
            "edit_valid": t(self.edit_valid),
            "edit_in_context": t(self.edit_in_context),
        }


@dataclass
class _Seq:

    params: CollateParams
    node_idxs: list[int] = field(default_factory=list)
    f2p: list[list[int]] = field(default_factory=list)
    table_idx: list[int] = field(default_factory=list)
    col_idx: list[int] = field(default_factory=list)
    col_text: list[str] = field(default_factory=list)
    val_text: list[str] = field(default_factory=list)
    number: list[float] = field(default_factory=list)
    datetime: list[float] = field(default_factory=list)
    boolean: list[float] = field(default_factory=list)
    sem: list[int] = field(default_factory=list)
    role: list[int] = field(default_factory=list)
    trans: list[int] = field(default_factory=list)
    is_row_anchor: list[bool] = field(default_factory=list)
    n_classes: list[int] = field(default_factory=list)
    legal_classes: list[tuple[int, ...]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.node_idxs)

    def add(
        self,
        *,
        node: int,
        f2p: list[int],
        table_idx: int,
        col_idx: int,
        col_text: str,
        role: int,
        trans: int,
        sem: int = SEM_NUMBER,
        number: float = 0.0,
        datetime: float = 0.0,
        boolean: float = 0.0,
        val_text: str = "",
        is_row_anchor: bool = False,
        n_classes: int = 0,
        legal_classes: tuple[int, ...] = (),
    ) -> int:
        pos = len(self.node_idxs)
        self.node_idxs.append(node)
        self.f2p.append((f2p + [-1] * MAX_F2P_NBRS)[:MAX_F2P_NBRS])
        self.table_idx.append(table_idx)
        self.col_idx.append(col_idx)
        self.col_text.append(col_text)
        self.val_text.append(val_text)
        self.number.append(number)
        self.datetime.append(datetime)
        self.boolean.append(boolean)
        self.sem.append(sem)
        self.role.append(role)
        self.trans.append(trans)
        self.is_row_anchor.append(is_row_anchor)
        self.n_classes.append(n_classes)
        self.legal_classes.append(legal_classes)
        return pos

    def room(self) -> int:
        return self.params.ctx_len - len(self)


class _Vocab:

    def __init__(self) -> None:
        self.tables: dict[str, int] = {}
        self.columns: dict[tuple[str, str], int] = {}

    def table(self, name: str) -> int:
        return self.tables.setdefault(name, len(self.tables))

    def column(self, table: str, column: str) -> int:
        return self.columns.setdefault((table, column), len(self.columns))


def _cell_sem_and_value(
    col: Column, raw: float, table: str, stats: ColumnStats
) -> tuple[int, dict]:
    if col.kind is ColumnKind.CATEGORICAL:
        idx = int(raw)
        name = (
            col.categories[idx]
            if col.categories and 0 <= idx < len(col.categories)
            else f"cat_{idx}"
        )
        return SEM_TEXT, {"val_text": f"{col.name}={name}"}
    if col.kind is ColumnKind.BOOL:
        return SEM_BOOLEAN, {"boolean": float(raw)}
    if col.kind is ColumnKind.DATETIME:
        return SEM_DATETIME, {"datetime": encode_cell(col, raw, table, stats)}
    return SEM_NUMBER, {"number": encode_cell(col, raw, table, stats)}


def _class_metadata(col: Column, raw: float) -> tuple[int, tuple[int, ...]]:
    if col.kind is ColumnKind.BOOL:
        return 2, (0, 1)
    if col.kind is not ColumnKind.CATEGORICAL:
        return 0, ()
    k = col.n_categories
    if not col.is_state:
        return k, tuple(range(k))
    cur = int(raw)
    reach = reachable_states(col).get(cur, set())
    return k, tuple(sorted({cur, *reach}))


def _emit_row(
    seq: _Seq,
    state: RelationalState,
    schema: Schema,
    table: str,
    row_id: int,
    node: int,
    node_of: dict[tuple[str, int], int],
    vocab: _Vocab,
    stats: ColumnStats,
    role: int,
    trans: int,
    index: TokenIndex | None,
    reverse: dict[int, tuple[str, int, str]] | None,
) -> None:
    tab = schema.table(table)
    fk_nodes = []
    for fk in tab.foreign_keys:
        pid = int(state.get(table, row_id, fk.column))
        n = node_of.get((fk.ref_table, pid))
        if n is not None:
            fk_nodes.append(n)
    t_idx = vocab.table(table)

    for col in tab.columns:
        if seq.room() <= 0:
            return
        raw = state.get(table, row_id, col.name)
        sem, kw = _cell_sem_and_value(col, raw, table, stats)
        n_cls, legal = _class_metadata(col, raw)
        pos = seq.add(
            node=node, f2p=fk_nodes, table_idx=t_idx,
            col_idx=vocab.column(table, col.name),
            col_text=f"{col.name} of {table}", role=role, trans=trans, sem=sem,


            is_row_anchor=(col.name == tab.pk and role == Role.STATE_QUERY),
            n_classes=n_cls, legal_classes=legal,
            **kw,
        )
        if index is not None:
            index.cell[(table, row_id, col.name)] = pos
            if col.name == tab.pk:
                index.row[(table, row_id)] = pos
        if reverse is not None:
            reverse[pos] = (table, row_id, col.name)


def _emit_action(
    seq: _Seq,
    action: ActionInstance,
    aschema: ActionSchema,
    node: int,
    node_of: dict[tuple[str, int], int],
    vocab: _Vocab,
    trans: int,
    anonymous_actions: bool,
) -> None:
    ptr_nodes = [
        n
        for _, table, row_id in action.entity_pointers(aschema)
        if (n := node_of.get((table, row_id))) is not None
    ]
    t_idx = vocab.table(ACTION_TABLE)
    label = aschema.label(anonymous=anonymous_actions)

    if seq.room() > 0:
        seq.add(
            node=node, f2p=ptr_nodes, table_idx=t_idx,
            col_idx=vocab.column(ACTION_TABLE, "type"),
            col_text=f"type of {ACTION_TABLE}", role=Role.ACTION, trans=trans,
            sem=SEM_TEXT, val_text=label,
        )
    for spec in aschema.args:
        if seq.room() <= 0:
            return
        v = float(action.bindings.get(spec.name, 0.0))
        if spec.kind is ArgKind.ENTITY:
            continue
        if spec.kind is ArgKind.CATEGORY:
            idx = int(v)
            name = (
                spec.categories[idx]
                if spec.categories and 0 <= idx < len(spec.categories)
                else f"cat_{idx}"
            )
            seq.add(
                node=node, f2p=ptr_nodes, table_idx=t_idx,
                col_idx=vocab.column(ACTION_TABLE, spec.name),
                col_text=f"{spec.name} of {ACTION_TABLE}", role=Role.ACTION, trans=trans,
                sem=SEM_TEXT, val_text=f"{spec.name}={name}",
            )
        else:
            span = max(spec.hi - spec.lo, 1e-6)
            seq.add(
                node=node, f2p=ptr_nodes, table_idx=t_idx,
                col_idx=vocab.column(ACTION_TABLE, spec.name),
                col_text=f"{spec.name} of {ACTION_TABLE}", role=Role.ACTION, trans=trans,
                sem=SEM_NUMBER, number=float((v - spec.lo) / span),
            )


def _emit_demo_delta(
    seq: _Seq,
    ex: TransitionExample,
    schema: Schema,
    node_of: dict[tuple[str, int], int],
    next_node: int,
    vocab: _Vocab,
    stats: ColumnStats,
    trans: int,
    limit: int,
) -> int:
    shown = 0
    for e in ex.delta.dedup().edits:
        if shown >= limit or seq.room() <= 0:
            break
        if e.op in (EditOp.UPDATE_VALUE, EditOp.UPDATE_STATE):
            node = node_of.get((e.table, int(e.row_id)))
            if node is None:
                continue
            try:
                col = schema.table(e.table).column(e.column)
            except KeyError:
                continue
            sem, kw = _cell_sem_and_value(col, e.value, e.table, stats)
            seq.add(
                node=node, f2p=[], table_idx=vocab.table(e.table),
                col_idx=vocab.column(e.table, e.column),
                col_text=f"{e.column} of {e.table}", role=Role.STATE_DELTA_DEMO,
                trans=trans, sem=sem, **kw,
            )
            shown += 1
        elif e.op is EditOp.DELETE_ROW:
            node = node_of.get((e.table, int(e.row_id)))
            if node is None:
                continue
            seq.add(
                node=node, f2p=[], table_idx=vocab.table(e.table),
                col_idx=vocab.column(e.table, DELETED_COL),
                col_text=f"{DELETED_COL} of {e.table}", role=Role.STATE_DELTA_DEMO,
                trans=trans, sem=SEM_BOOLEAN, boolean=1.0,
            )
            shown += 1
        else:
            parent_nodes = []
            tab = schema.table(e.table)
            for fk in tab.foreign_keys:
                for name, val in e.values:
                    if name == fk.column:
                        n = node_of.get((fk.ref_table, int(val)))
                        if n is not None:
                            parent_nodes.append(n)
            seq.add(
                node=next_node, f2p=parent_nodes, table_idx=vocab.table(e.table),
                col_idx=vocab.column(e.table, INSERTED_COL),
                col_text=f"{INSERTED_COL} of {e.table}", role=Role.STATE_DELTA_DEMO,
                trans=trans, sem=SEM_BOOLEAN, boolean=1.0,
            )
            next_node += 1
            shown += 1
    return next_node


def encode_task(
    task: TabWMTask,
    params: CollateParams,
    rng: np.random.Generator,
) -> tuple[_Seq, TokenIndex, ColumnStats, dict[int, tuple[str, int, str]], dict[int, str], list[EditTarget], dict]:
    schema = task.schema.anonymized() if task.naming.anonymous_schema else task.schema
    rename = task.schema.rename_map() if task.naming.anonymous_schema else None

    def tname(t: str) -> str:
        return rename["__tables__"][t] if rename else t

    def cname(t: str, c: str) -> str:
        return rename[t][c] if rename else c


    observable = [ex.before_state for ex in task.context] + [task.query_state]
    stats_raw = ColumnStats.from_states(observable, task.schema)
    stats = (
        ColumnStats(
            stats={(tname(t), cname(t, c)): v for (t, c), v in stats_raw.stats.items()}
        )
        if rename
        else stats_raw
    )

    seq = _Seq(params=params)
    vocab = _Vocab()
    next_node = 0
    query_budget = int(params.ctx_len * params.query_budget_frac)
    demo_budget = params.ctx_len - query_budget


    max_k_fit = max(1, demo_budget // params.min_demo_cells)
    effective_k = min(task.k, max_k_fit)
    context = task.context[:effective_k]
    per_demo = demo_budget // max(1, effective_k) if effective_k else 0


    demo_diag = {"demo_context_truncated": 0, "demo_coverage": []}
    for j0, ex in enumerate(context):
        j = j0 + 1


        remaining = max(1, effective_k - j0)
        target_len = min(demo_budget, len(seq) + max(per_demo, (demo_budget - len(seq)) // remaining))
        sample = sample_context_for_action(
            ex.before_state, task.schema, ex.action, ex.action_schema,
            params.context, rng,
        )
        demo_diag["demo_coverage"].append(sample.coverage(ex.delta))
        node_of: dict[tuple[str, int], int] = {}
        for table, row_id in sample.nodes:
            node_of[(table, row_id)] = next_node
            next_node += 1
        for table, row_id in sample.nodes:
            if len(seq) >= target_len:
                demo_diag["demo_context_truncated"] += 1
                break
            _emit_row(
                seq, ex.before_state, task.schema, table, row_id,
                node_of[(table, row_id)], node_of, vocab, stats_raw,
                Role.STATE_BEFORE, j, None, None,
            )
        if len(seq) < demo_budget:
            _emit_action(seq, ex.action, ex.action_schema, next_node, node_of, vocab, j,
                         task.naming.anonymous_actions)
            next_node += 1
            next_node = _emit_demo_delta(
                seq, ex, task.schema, node_of, next_node, vocab, stats_raw, j,
                params.max_demo_delta_cells,
            )


    q_trans = QUERY_TRANSITION
    index = TokenIndex()
    reverse: dict[int, tuple[str, int, str]] = {}
    table_of_pos: dict[int, str] = {}


    for table in task.schema.table_names:
        if seq.room() <= 0:
            break
        pos = seq.add(
            node=next_node, f2p=[], table_idx=vocab.table(table),
            col_idx=vocab.column(table, ANCHOR_COL),
            col_text=f"{ANCHOR_COL} of {table}", role=Role.TABLE_ANCHOR, trans=q_trans,
            sem=SEM_NUMBER, number=float(task.query_state.n_rows(table)) / 100.0,
        )
        index.table[table] = pos
        table_of_pos[pos] = table
        next_node += 1

    q_sample: ContextSample = sample_context_for_action(
        task.query_state, task.schema, task.query_action, task.query_action_schema,
        params.context, rng,
    )
    node_of = {}
    for table, row_id in q_sample.nodes:
        node_of[(table, row_id)] = next_node
        next_node += 1
    for table, row_id in q_sample.nodes:
        if seq.room() <= 0:
            break
        _emit_row(
            seq, task.query_state, task.schema, table, row_id,
            node_of[(table, row_id)], node_of, vocab, stats_raw,
            Role.STATE_QUERY, q_trans, index, reverse,
        )
    _emit_action(seq, task.query_action, task.query_action_schema, next_node, node_of,
                 vocab, q_trans, task.naming.anonymous_actions)
    next_node += 1

    targets, tdiag = encode_targets(
        task.target_delta, task.schema, index, stats_raw, max_slots=params.max_edit_slots
    )

    diag = {
        **tdiag,
        "seq_len": len(seq),
        "k": task.k,
        "effective_k": effective_k,
        "k_truncated": task.k - effective_k,
        "query_context_nodes": len(q_sample.nodes),
        "query_context_truncated": q_sample.truncated,
        "query_coverage": q_sample.coverage(task.target_delta),
        "query_max_depth": q_sample.max_depth_reached(),
        "demo_coverage": (
            float(np.mean(demo_diag["demo_coverage"])) if demo_diag["demo_coverage"] else 1.0
        ),
        "naming": task.naming.name,
    }


    if rename:
        seq.col_text = [_rename_col_text(s, rename) for s in seq.col_text]

    _prune_dangling_pointers(seq)
    return seq, index, stats_raw, reverse, table_of_pos, targets, diag


def _prune_dangling_pointers(seq: _Seq) -> None:
    present = set(seq.node_idxs)
    for i, row in enumerate(seq.f2p):
        kept = [n for n in row if n in present]
        if len(kept) != sum(1 for n in row if n >= 0):
            seq.f2p[i] = (kept + [-1] * MAX_F2P_NBRS)[:MAX_F2P_NBRS]


def _rename_col_text(text: str, rename: dict) -> str:
    if " of " not in text:
        return text
    col, table = text.split(" of ", 1)
    tmap = rename["__tables__"]
    if table not in tmap:
        return text
    cmap = rename.get(table, {})
    return f"{cmap.get(col, col)} of {tmap[table]}"


def collate(
    tasks: list[TabWMTask],
    embedder: TextEmbedder,
    params: CollateParams | None = None,
    rng: np.random.Generator | None = None,
) -> TabWMBatch:
    params = params or CollateParams(d_text=embedder.d_text)
    rng = rng or np.random.default_rng(0)
    if params.length_mixture is not None:


        params = replace(
            params,
            ctx_len=params.length_mixture.sample(
                rng, min_length=required_ctx_len(tasks, params)
            ),
        )


    all_strings: list[str] = []
    encoded = []
    for task in tasks:
        seq, index, stats, reverse, tpos, targets, diag = encode_task(task, params, rng)
        encoded.append((seq, index, stats, reverse, tpos, targets, diag))
        all_strings.extend(seq.col_text)
        all_strings.extend(s for s in seq.val_text if s)


    used = max((len(s) for s, *_ in encoded), default=0)
    S = min(params.ctx_len, _bucket_length(used, params))
    B, M, D = len(tasks), params.max_edit_slots, embedder.d_text

    out = TabWMBatch(
        node_idxs=np.zeros((B, S), np.int64),
        f2p_nbr_idxs=np.full((B, S, MAX_F2P_NBRS), -1, np.int64),
        table_name_idxs=np.zeros((B, S), np.int64),
        col_name_idxs=np.zeros((B, S), np.int64),
        col_name_values=np.zeros((B, S, D), np.float32),
        text_values=np.zeros((B, S, D), np.float32),
        number_values=np.zeros((B, S, 1), np.float32),
        datetime_values=np.zeros((B, S, 1), np.float32),
        boolean_values=np.zeros((B, S, 1), np.float32),
        sem_types=np.zeros((B, S), np.int64),
        is_padding=np.ones((B, S), bool),
        masks=np.zeros((B, S), bool),
        role_ids=np.zeros((B, S), np.int64),
        transition_ids=np.zeros((B, S), np.int64),
        is_row_anchor=np.zeros((B, S), bool),
        cell_n_classes=np.zeros((B, S), np.int64),
        cell_legal_classes=np.zeros((B, S, MAX_CATEGORIES), bool),
        edit_op=np.zeros((B, M), np.int64),
        edit_cell_pos=np.full((B, M), -1, np.int64),
        edit_row_pos=np.full((B, M), -1, np.int64),
        edit_table_pos=np.full((B, M), -1, np.int64),
        edit_link_pos=np.full((B, M), -1, np.int64),
        edit_value_kind=np.zeros((B, M), np.int64),
        edit_value_num=np.zeros((B, M), np.float32),
        edit_value_tol_z=np.zeros((B, M), np.float32),
        edit_value_class=np.zeros((B, M), np.int64),
        edit_valid=np.zeros((B, M), bool),
        edit_in_context=np.zeros((B, M), bool),
    )

    if all_strings:
        uniq = list(dict.fromkeys(all_strings))
        lut = dict(zip(uniq, embedder(uniq)))
    else:
        lut = {}
    zero = np.zeros(D, np.float32)

    for b, (seq, index, stats, reverse, tpos, targets, diag) in enumerate(encoded):
        n = min(len(seq), S)
        out.node_idxs[b, :n] = seq.node_idxs[:n]
        out.f2p_nbr_idxs[b, :n] = np.asarray(seq.f2p[:n], np.int64)
        out.table_name_idxs[b, :n] = seq.table_idx[:n]
        out.col_name_idxs[b, :n] = seq.col_idx[:n]
        out.col_name_values[b, :n] = np.stack([lut[s] for s in seq.col_text[:n]])
        out.text_values[b, :n] = np.stack(
            [lut[s] if s else zero for s in seq.val_text[:n]]
        )
        out.number_values[b, :n, 0] = seq.number[:n]
        out.datetime_values[b, :n, 0] = seq.datetime[:n]
        out.boolean_values[b, :n, 0] = seq.boolean[:n]
        out.sem_types[b, :n] = seq.sem[:n]
        out.is_padding[b, :n] = False
        out.role_ids[b, :n] = seq.role[:n]
        out.transition_ids[b, :n] = seq.trans[:n]
        out.is_row_anchor[b, :n] = seq.is_row_anchor[:n]
        out.cell_n_classes[b, :n] = seq.n_classes[:n]
        for p, legal in enumerate(seq.legal_classes[:n]):
            for c in legal:
                if c < MAX_CATEGORIES:
                    out.cell_legal_classes[b, p, c] = True

        for m, t in enumerate(targets[:M]):
            out.edit_op[b, m] = t.op
            out.edit_cell_pos[b, m] = t.cell_pos
            out.edit_row_pos[b, m] = t.row_pos
            out.edit_table_pos[b, m] = t.table_pos
            out.edit_link_pos[b, m] = t.link_pos
            out.edit_value_kind[b, m] = t.value_kind
            out.edit_value_num[b, m] = t.value_num
            out.edit_value_tol_z[b, m] = t.value_tol_z
            out.edit_value_class[b, m] = t.value_class
            out.edit_valid[b, m] = True
            out.edit_in_context[b, m] = t.in_context

        out.tasks.append(tasks[b])
        out.indices.append(index)
        out.stats.append(stats)
        out.reverse.append(reverse)
        out.table_of_pos.append(tpos)
        out.diagnostics.append(diag)

    return out


def batch_stats(batch: TabWMBatch) -> dict[str, float]:
    d = batch.diagnostics
    if not d:
        return {}
    return {
        "batch_size": batch.batch_size,
        "seq_len": batch.seq_len,
        "fill_rate": float((~batch.is_padding).mean()),
        "mean_k": float(np.mean([x["k"] for x in d])),
        "mean_edits": float(batch.edit_valid.sum(1).mean()),
        "query_coverage": float(np.mean([x["query_coverage"] for x in d])),
        "demo_coverage": float(np.mean([x["demo_coverage"] for x in d])),
        "out_of_context_edits": int(sum(x["out_of_context"] for x in d)),
        "dropped_edits": int(sum(x["dropped"] for x in d)),
        "query_truncated_frac": float(np.mean([x["query_context_truncated"] for x in d])),
    }
