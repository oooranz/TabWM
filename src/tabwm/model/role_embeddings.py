"""Relational role embeddings."""

from __future__ import annotations

import torch
from torch import nn

from tabwm.data.collate import Role

MAX_TRANSITIONS = 128


class RoleTransitionEmbedding(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_roles: int = Role.N_ROLES,
        max_transitions: int = MAX_TRANSITIONS,
        init_scale: float = 0.02,
    ) -> None:
        super().__init__()
        self.role = nn.Embedding(n_roles, d_model)
        self.transition = nn.Embedding(max_transitions, d_model)
        self.max_transitions = max_transitions
        nn.init.normal_(self.role.weight, std=init_scale)
        nn.init.normal_(self.transition.weight, std=init_scale)
        with torch.no_grad():
            self.role.weight[Role.PAD].zero_()

    def forward(
        self, x: torch.Tensor, role_ids: torch.Tensor, transition_ids: torch.Tensor
    ) -> torch.Tensor:
        t = transition_ids.clamp(0, self.max_transitions - 1)
        return x + self.role(role_ids) + self.transition(t)


def shuffle_transition_ids(
    transition_ids: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    out = transition_ids.clone()
    for b in range(out.shape[0]):
        ids = torch.unique(out[b])
        ids = ids[ids > 0]
        if ids.numel() < 2:
            continue
        perm = ids[torch.randperm(ids.numel(), generator=generator, device=ids.device)]
        mapping = {int(a): int(c) for a, c in zip(ids.tolist(), perm.tolist())}
        flat = out[b]
        new = flat.clone()
        for a, c in mapping.items():
            new[flat == a] = c
        out[b] = new
    return out
