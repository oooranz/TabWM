"""Behavioral task sampling."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tabwm.world.actions import ActionInstance, ActionSchema
from tabwm.world.generator import World
from tabwm.world.mechanisms import MechanismFamily
from tabwm.world.rollout import Policy, Trajectory, rollout
from tabwm.world.schema import Schema
from tabwm.world.state import RelationalState, StateDelta


@dataclass(frozen=True)
class NamingMode:

    anonymous_schema: bool = False
    anonymous_actions: bool = False

    @property
    def name(self) -> str:
        s = "anon" if self.anonymous_schema else "sem"
        a = "anon" if self.anonymous_actions else "sem"
        return f"schema:{s}/action:{a}"


SEMANTIC = NamingMode(False, False)
ANON_ACTIONS = NamingMode(False, True)
ANON_SCHEMA = NamingMode(True, False)
FULLY_ANONYMOUS = NamingMode(True, True)
ALL_NAMING_MODES = (SEMANTIC, ANON_SCHEMA, ANON_ACTIONS, FULLY_ANONYMOUS)


@dataclass
class TransitionExample:

    world_id: int
    t: int
    before_state: RelationalState
    action: ActionInstance
    action_schema: ActionSchema
    delta: StateDelta
    reward: float | None = None
    done: bool = False
    applicable: bool = True
    families: frozenset[MechanismFamily] = frozenset()
    episode: int = -1

    @property
    def n_edits(self) -> int:
        return len(self.delta)

    @property
    def is_noop(self) -> bool:
        return len(self.delta) == 0


@dataclass(frozen=True)
class CounterQuery:

    action: ActionInstance
    action_schema: ActionSchema
    delta: StateDelta
    reward: float | None = None
    applicable: bool = True
    families: frozenset[MechanismFamily] = frozenset()


@dataclass
class TabWMTask:

    context: list[TransitionExample]
    query_state: RelationalState
    query_action: ActionInstance
    query_action_schema: ActionSchema
    target_delta: StateDelta
    schema: Schema
    world_id: int
    naming: NamingMode = SEMANTIC
    query_reward: float | None = None
    query_applicable: bool = True
    query_families: frozenset[MechanismFamily] = frozenset()
    meta: dict = field(default_factory=dict)

    counter_queries: list[CounterQuery] = field(default_factory=list)

    @property
    def k(self) -> int:
        return len(self.context)

    @property
    def n_queries(self) -> int:
        return 1 + len(self.counter_queries)

    def all_queries(self) -> list[tuple[ActionInstance, ActionSchema, StateDelta]]:
        return [
            (self.query_action, self.query_action_schema, self.target_delta),
            *((c.action, c.action_schema, c.delta) for c in self.counter_queries),
        ]

    def counterfactually_distinct(self) -> bool:
        return any(c.delta != self.target_delta for c in self.counter_queries)

    def query_example(self) -> TransitionExample:
        return TransitionExample(
            world_id=self.world_id,
            t=-1,
            before_state=self.query_state,
            action=self.query_action,
            action_schema=self.query_action_schema,
            delta=self.target_delta,
            reward=self.query_reward,
            applicable=self.query_applicable,
            families=self.query_families,
        )

    def action_ids_in_context(self) -> set[int]:
        return {ex.action.aid for ex in self.context}

    def query_action_demonstrated(self) -> bool:
        return self.query_action.aid in self.action_ids_in_context()

    def summary(self) -> dict:
        return {
            "world_id": self.world_id,
            "k": self.k,
            "naming": self.naming.name,
            "n_target_edits": len(self.target_delta),
            "query_applicable": self.query_applicable,
            "query_action_demonstrated": self.query_action_demonstrated(),
            "families": sorted(f.value for f in self.query_families),
        }


K_SWEEP = (0, 1, 2, 4, 8, 16, 32, 64)


@dataclass(frozen=True)
class TaskParams:

    k_choices: tuple[int, ...] = K_SWEEP
    k_weights: tuple[float, ...] | None = None

    episode_window: int = 4
    n_episodes: int = 20
    horizon: int = 48
    require_nonempty_query: float = 0.75
    n_counter_queries: int = 0
    counter_query_tries: int = 8
    naming_modes: tuple[NamingMode, ...] = ALL_NAMING_MODES
    naming_weights: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)

    def sample_k(self, rng: np.random.Generator) -> int:
        p = (
            np.asarray(self.k_weights, dtype=np.float64)
            if self.k_weights is not None
            else np.ones(len(self.k_choices))
        )
        p = p / p.sum()
        return int(self.k_choices[int(rng.choice(len(self.k_choices), p=p))])

    def sample_naming(self, rng: np.random.Generator) -> NamingMode:
        p = np.asarray(self.naming_weights, dtype=np.float64)
        p = p / p.sum()
        return self.naming_modes[int(rng.choice(len(self.naming_modes), p=p))]


def trajectory_to_examples(
    traj: Trajectory, world: World, episode: int = -1
) -> list[TransitionExample]:
    return [
        TransitionExample(
            world_id=world.world_id,
            t=tr.t,
            before_state=tr.state,
            action=tr.action,
            action_schema=world.actions[tr.action.aid],
            delta=tr.delta,
            reward=tr.reward,
            applicable=tr.applicable,
            families=tr.families,
            episode=episode,
        )
        for tr in traj
    ]


@dataclass
class EpisodePool:

    world_id: int
    episodes: list[list[TransitionExample]] = field(default_factory=list)

    @property
    def n_transitions(self) -> int:
        return sum(len(e) for e in self.episodes)

    def max_k(self) -> int:
        if len(self.episodes) < 2:
            return 0
        return self.n_transitions - max(len(e) for e in self.episodes)


def sample_episode_pool(
    world: World,
    params: TaskParams | None = None,
    *,
    policy: Policy | None = None,
    seed_offset: int = 0,
) -> EpisodePool:
    params = params or TaskParams()
    episodes = []
    for e in range(params.n_episodes):
        traj = rollout(
            world, horizon=params.episode_window, seed=seed_offset + e, policy=policy
        )
        ex = trajectory_to_examples(traj, world, episode=e)
        if ex:
            episodes.append(ex)
    return EpisodePool(world_id=world.world_id, episodes=episodes)


def sample_task(
    world: World,
    rng: np.random.Generator,
    params: TaskParams | None = None,
    *,
    pool: EpisodePool | None = None,
    trajectory: Trajectory | None = None,
    policy: Policy | None = None,
    k: int | None = None,
    naming: NamingMode | None = None,
    episode_seed: int = 0,
) -> TabWMTask | None:
    params = params or TaskParams()

    if trajectory is not None:
        return _sample_task_single_trajectory(
            world, rng, params, trajectory, k=k, naming=naming, episode_seed=episode_seed
        )

    if pool is None:
        pool = sample_episode_pool(world, params, policy=policy, seed_offset=episode_seed * 1000)
    if len(pool.episodes) < 2:
        return None

    k = params.sample_k(rng) if k is None else k
    k = min(k, pool.max_k())
    naming = params.sample_naming(rng) if naming is None else naming

    q_ep = int(rng.integers(len(pool.episodes)))
    query = _pick_query(pool.episodes[q_ep], rng, params)

    demo_source = [
        ex for i, ep in enumerate(pool.episodes) if i != q_ep for ex in ep
    ]
    if k and len(demo_source) >= k:
        idx = rng.choice(len(demo_source), size=k, replace=False)


        context = [demo_source[int(i)] for i in sorted(idx)]
    else:
        context = demo_source[:k]

    return _build_task(world, context, query, naming, params, episode_seed, rng)


def _sample_task_single_trajectory(
    world, rng, params, trajectory, *, k, naming, episode_seed
) -> TabWMTask | None:
    examples = trajectory_to_examples(trajectory, world)
    if len(examples) < 2:
        return None
    k = params.sample_k(rng) if k is None else k
    k = min(k, len(examples) - 1)
    naming = params.sample_naming(rng) if naming is None else naming

    query = _pick_query(examples, rng, params)
    remaining = [ex for ex in examples if ex is not query]
    if k and len(remaining) >= k:
        idx = sorted(rng.choice(len(remaining), size=k, replace=False))
        context = [remaining[int(i)] for i in idx]
    else:
        context = remaining[:k]
    return _build_task(world, context, query, naming, params, episode_seed, rng)


def sample_counter_queries(
    world: World,
    query: TransitionExample,
    rng: np.random.Generator,
    params: TaskParams,
) -> list[CounterQuery]:
    if params.n_counter_queries <= 0:
        return []
    out: list[CounterQuery] = []
    seen = {query.action.key()}
    for _ in range(params.counter_query_tries):
        if len(out) >= params.n_counter_queries:
            break
        alt = world.sample_action(query.before_state, rng, require_applicable=True)
        if alt is None or alt.key() in seen:
            continue
        seen.add(alt.key())
        res = world.transition(
            query.before_state, alt, np.random.default_rng(int(rng.integers(1 << 31)))
        )
        if not res.applicable:
            continue
        out.append(
            CounterQuery(
                action=alt,
                action_schema=world.actions[alt.aid],
                delta=res.delta,
                reward=res.reward,
                applicable=res.applicable,
                families=res.families,
            )
        )
    return out


def _build_task(world, context, query, naming, params, episode_seed, rng=None) -> TabWMTask:
    counters = (
        sample_counter_queries(world, query, rng, params)
        if rng is not None and params.n_counter_queries > 0
        else []
    )
    return TabWMTask(
        counter_queries=counters,
        context=context,
        query_state=query.before_state,
        query_action=query.action,
        query_action_schema=query.action_schema,
        target_delta=query.delta,
        schema=world.schema,
        world_id=world.world_id,
        naming=naming,
        query_reward=query.reward,
        query_applicable=query.applicable,
        query_families=query.families,
        meta={
            "episode_seed": episode_seed,
            "query_t": query.t,
            "query_episode": query.episode,
            "prior_hash": world.prior_hash,
        },
    )


def _pick_query(
    examples: list[TransitionExample], rng: np.random.Generator, params: TaskParams
) -> TransitionExample:
    if rng.random() < params.require_nonempty_query:
        candidates = [ex for ex in examples if not ex.is_noop]
        if candidates:
            return candidates[int(rng.integers(len(candidates)))]
    return examples[int(rng.integers(len(examples)))]


def sample_tasks(
    seeds: list[int],
    rng: np.random.Generator,
    params: TaskParams | None = None,
    *,
    config=None,
    tasks_per_world: int = 1,
    policy: Policy | None = None,
) -> list[TabWMTask]:
    from tabwm.world.generator import sample_world

    out: list[TabWMTask] = []
    params = params or TaskParams()
    for seed in seeds:
        world = sample_world(seed, config)
        pool = sample_episode_pool(world, params, policy=policy)
        for _ in range(tasks_per_world):
            task = sample_task(world, rng, params, pool=pool)
            if task is not None:
                out.append(task)
    return out
