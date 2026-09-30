"""WorldRel trajectories."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from tabwm.world.actions import ActionInstance
from tabwm.world.generator import World
from tabwm.world.mechanisms import MechanismFamily
from tabwm.world.state import RelationalState, StateDelta


class Policy(Protocol):

    def __call__(
        self, world: World, state: RelationalState, rng: np.random.Generator
    ) -> ActionInstance | None: ...


def random_valid_policy(
    world: World, state: RelationalState, rng: np.random.Generator
) -> ActionInstance | None:
    return world.sample_action(state, rng, valid=True, require_applicable=False)


def random_applicable_policy(
    world: World, state: RelationalState, rng: np.random.Generator
) -> ActionInstance | None:
    return world.sample_action(state, rng, valid=True, require_applicable=True)


def random_invalid_policy(
    world: World, state: RelationalState, rng: np.random.Generator
) -> ActionInstance | None:
    return world.sample_action(state, rng, valid=False, require_applicable=False)


def make_greedy_policy(n_candidates: int = 6) -> Policy:

    def policy(world, state, rng):
        cands = world.candidate_actions(state, rng, n_candidates)
        if not cands:
            return random_valid_policy(world, state, rng)
        best, best_r = None, -np.inf
        for a in cands:
            r = world.transition(state, a, np.random.default_rng(int(rng.integers(1 << 31))))
            if r.reward > best_r:
                best, best_r = a, r.reward
        return best

    return policy


def make_rare_action_policy() -> Policy:
    counts: Counter[int] = Counter()

    def policy(world, state, rng):
        weights = np.array(
            [1.0 / (1.0 + counts[a]) for a in range(world.n_actions)], dtype=np.float64
        )
        weights /= weights.sum()
        aid = int(rng.choice(world.n_actions, p=weights))
        inst = world.sample_action(state, rng, aid=aid, require_applicable=False)
        if inst is not None:
            counts[inst.aid] += 1
        return inst

    return policy


def make_goal_directed_policy(n_candidates: int = 6, horizon: int = 3) -> Policy:

    def policy(world, state, rng):
        cands = world.candidate_actions(state, rng, n_candidates)
        if not cands:
            return random_valid_policy(world, state, rng)
        best, best_r = None, -np.inf
        for a in cands:
            total, s, act = 0.0, state, a
            sub = np.random.default_rng(int(rng.integers(1 << 31)))
            for _ in range(horizon):
                res = world.transition(s, act, sub)
                total += res.reward
                s = res.state
                nxt = world.sample_action(s, sub, require_applicable=True)
                if nxt is None:
                    break
                act = nxt
            if total > best_r:
                best, best_r = a, total
        return best

    return policy


def mixture_policy(
    weights: dict[str, float] | None = None, **kwargs
) -> Policy:
    weights = weights or {
        "random_valid": 0.35,
        "random_applicable": 0.25,
        "rare_action": 0.20,
        "greedy": 0.10,
        "goal_directed": 0.05,
        "random_invalid": 0.05,
    }
    table: dict[str, Policy] = {
        "random_valid": random_valid_policy,
        "random_applicable": random_applicable_policy,
        "random_invalid": random_invalid_policy,
        "rare_action": make_rare_action_policy(),
        "greedy": make_greedy_policy(kwargs.get("n_candidates", 6)),
        "goal_directed": make_goal_directed_policy(kwargs.get("n_candidates", 6)),
    }
    names = list(weights)
    probs = np.array([weights[n] for n in names], dtype=np.float64)
    probs /= probs.sum()

    def policy(world, state, rng):
        name = names[int(rng.choice(len(names), p=probs))]
        return table[name](world, state, rng)

    return policy


@dataclass
class Transition:

    t: int
    state: RelationalState
    action: ActionInstance
    delta: StateDelta
    next_state: RelationalState
    reward: float
    applicable: bool
    families: frozenset[MechanismFamily]
    exogenous_delta: StateDelta = field(default_factory=StateDelta)

    @property
    def n_edits(self) -> int:
        return len(self.delta)

    @property
    def full_delta(self) -> StateDelta:
        return StateDelta(edits=[*self.delta.edits, *self.exogenous_delta.edits]).dedup()


@dataclass
class Trajectory:
    world_id: int
    seed: int
    transitions: list[Transition] = field(default_factory=list)
    initial_state: RelationalState | None = None

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, i: int) -> Transition:
        return self.transitions[i]

    def __iter__(self):
        return iter(self.transitions)


    def action_coverage(self, n_actions: int) -> float:
        seen = {tr.action.aid for tr in self.transitions}
        return len(seen) / max(1, n_actions)

    def applicable_rate(self) -> float:
        if not self.transitions:
            return 0.0
        return sum(tr.applicable for tr in self.transitions) / len(self.transitions)

    def nonempty_rate(self) -> float:
        if not self.transitions:
            return 0.0
        return sum(bool(tr.n_edits) for tr in self.transitions) / len(self.transitions)

    def mean_edits(self) -> float:
        if not self.transitions:
            return 0.0
        return float(np.mean([tr.n_edits for tr in self.transitions]))

    def family_counts(self) -> dict[str, int]:
        out: Counter[str] = Counter()
        for tr in self.transitions:
            for f in tr.families:
                out[f.value] += 1
        return dict(out)

    def summary(self, n_actions: int) -> dict[str, float | dict]:
        return {
            "steps": len(self),
            "action_coverage": round(self.action_coverage(n_actions), 3),
            "applicable_rate": round(self.applicable_rate(), 3),
            "nonempty_rate": round(self.nonempty_rate(), 3),
            "mean_edits": round(self.mean_edits(), 2),
            "families": self.family_counts(),
        }


def rollout(
    world: World,
    horizon: int,
    *,
    policy: Policy | None = None,
    seed: int = 0,
    check: bool = False,
    keep_states: bool = True,
) -> Trajectory:
    policy = policy or mixture_policy()
    rng = np.random.default_rng(np.random.SeedSequence([world.seed, 7, seed]))
    state = world.reset(seed=seed)
    traj = Trajectory(world_id=world.world_id, seed=seed, initial_state=state.copy())

    for t in range(horizon):
        action = policy(world, state, rng)
        if action is None:


            action = random_valid_policy(world, state, rng)
        if action is None:
            break
        res = world.transition(state, action, rng, check=check)
        if check and res.violations:
            raise AssertionError(
                f"world {world.world_id} step {t}: {res.violations[:4]}"
            )
        traj.transitions.append(
            Transition(
                t=t,
                state=state.copy() if keep_states else state,
                action=action,
                delta=res.delta,
                next_state=res.state.copy() if keep_states else res.state,
                reward=res.reward,
                applicable=res.applicable,
                families=res.families,
                exogenous_delta=res.exogenous_delta,
            )
        )
        state = res.state
    world._state = state
    return traj


def rollout_actions(
    world: World,
    state: RelationalState,
    actions: list[ActionInstance],
    rng: np.random.Generator,
) -> tuple[RelationalState, float]:
    total = 0.0
    for a in actions:
        res = world.transition(state, a, rng)
        total += res.reward
        state = res.state
    return state, total
