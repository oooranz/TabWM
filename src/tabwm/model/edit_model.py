"""Structured edit prediction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tabwm.data.collate import CollateParams, TabWMBatch, TextEmbedder, collate
from tabwm.data.delta import PredictedEdit, decode_edits
from tabwm.data.transition import (
    NamingMode,
    TabWMTask,
    TransitionExample,
)
from tabwm.model.delta_decoder import DecoderOutput, DeltaDecoder, decode_predictions
from tabwm.model.rt_backbone import RTConfig, TabWMEncoder, load_rt_checkpoint
from tabwm.world.actions import ActionInstance
from tabwm.world.generator import World
from tabwm.world.state import RelationalState, StateDelta


@dataclass
class EditModelConfig:
    rt: RTConfig = field(default_factory=RTConfig)
    n_slots: int = 16
    decoder_layers: int = 2
    decoder_heads: int = 8
    decoder_d_ff: int | None = None
    numeric_head: str = "scalar"
    shared_offset: bool = True

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)


class EditModel(nn.Module):

    def __init__(self, config: EditModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or EditModelConfig()
        self.encoder = TabWMEncoder(self.config.rt)
        self.decoder = DeltaDecoder(
            d_model=self.config.rt.d_model,
            n_slots=self.config.n_slots,
            n_layers=self.config.decoder_layers,
            num_heads=self.config.decoder_heads,
            d_ff=self.config.decoder_d_ff,
            numeric_head=self.config.numeric_head,
            shared_offset=self.config.shared_offset,
        )

    @classmethod
    def from_plurel_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        config: EditModelConfig | None = None,
        *,
        checkpoint_file: str = "synthetic-pretrain_rdb_1024_size_4b.pt",
        strict_backbone: bool = True,
    ) -> tuple[EditModel, dict[str, int]]:
        directory = Path(checkpoint_dir)
        cfg = config or EditModelConfig(
            rt=RTConfig.from_json(directory / "config.json")
        )
        model = cls(cfg)
        info = load_rt_checkpoint(
            model.encoder,
            directory / checkpoint_file,
            strict_backbone=strict_backbone,
        )
        info["decoder_params"] = sum(
            parameter.numel() for parameter in model.decoder.parameters()
        )
        return model, info

    def parameter_summary(self) -> dict[str, float]:
        enc = sum(p.numel() for p in self.encoder.parameters())
        new = sum(
            p.numel() for n, p in self.encoder.named_parameters()
            if n.startswith("role_transition")
        )
        dec = sum(p.numel() for p in self.decoder.parameters())
        return {
            "backbone_M": round((enc - new) / 1e6, 3),
            "role_transition_M": round(new / 1e6, 4),
            "decoder_M": round(dec / 1e6, 3),
            "total_M": round((enc + dec) / 1e6, 3),
        }


    def forward(
        self, batch: dict[str, torch.Tensor], *, use_flex: bool | None = None
    ) -> DecoderOutput:
        hidden = self.encoder(batch, use_flex=use_flex)
        return self.decoder(hidden, batch)


    @torch.no_grad()
    def predict_deltas(
        self,
        batch: TabWMBatch,
        *,
        device: str = "cpu",
        threshold: float = 0.5,
        use_flex: bool | None = None,
    ) -> list[StateDelta]:
        tensors = batch.torch(device=device, dtype=torch.float32)
        out = self(tensors, use_flex=use_flex)
        raw = decode_predictions(out, threshold=threshold)

        deltas: list[StateDelta] = []
        for b, slots in enumerate(raw):
            preds = [
                PredictedEdit(
                    op=r.op, cell_pos=r.cell_pos, row_pos=r.row_pos,
                    table_pos=r.table_pos, link_pos=r.link_pos,
                    value_num=r.value_numeric,
                    value_class=(
                        r.value_boolean
                        if _is_boolean_target(batch, b, r.cell_pos)
                        else r.value_categorical
                    ),
                    confidence=r.confidence,
                )
                for r in slots
            ]
            deltas.append(
                decode_edits(
                    preds,
                    reverse=batch.reverse[b],
                    table_of_pos=batch.table_of_pos[b],
                    schema=batch.tasks[b].schema,
                    stats=batch.stats[b],
                    state=batch.tasks[b].query_state,
                    threshold=threshold,
                )
            )
        return deltas

    @torch.no_grad()
    def rollout(
        self,
        world: World,
        state: RelationalState,
        actions: list[ActionInstance],
        context: list[TransitionExample],
        embedder: TextEmbedder,
        *,
        params: CollateParams | None = None,
        naming: NamingMode | None = None,
        device: str = "cpu",
        rng: np.random.Generator | None = None,
        threshold: float = 0.5,
        use_flex: bool | None = None,
    ) -> tuple[list[StateDelta], list[RelationalState]]:
        from tabwm.data.transition import SEMANTIC

        params = params or CollateParams(d_text=embedder.d_text)
        rng = rng or np.random.default_rng(0)
        naming = naming or SEMANTIC

        deltas: list[StateDelta] = []
        states: list[RelationalState] = []
        current = state

        for action in actions:
            task = TabWMTask(
                context=context,
                query_state=current,
                query_action=action,
                query_action_schema=world.actions[action.aid],
                target_delta=StateDelta(),
                schema=world.schema,
                world_id=world.world_id,
                naming=naming,
            )
            batch = collate([task], embedder, params, rng)
            delta = self.predict_deltas(
                batch, device=device, threshold=threshold, use_flex=use_flex
            )[0]
            current = current.apply(delta, strict=False)
            deltas.append(delta)
            states.append(current)

        return deltas, states


def _is_boolean_target(batch: TabWMBatch, b: int, cell_pos: int) -> bool:
    from tabwm.world.schema import ColumnKind

    ref = batch.reverse[b].get(cell_pos)
    if ref is None:
        return False
    table, _, column = ref
    try:
        return batch.tasks[b].schema.table(table).column(column).kind is ColumnKind.BOOL
    except KeyError:
        return False
