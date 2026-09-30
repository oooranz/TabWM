"""TabWM model and checkpoint loading."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file

from tabwm.model.contextual_value_decoder import ContextualValueDecoder, ValueDecoderConfig
from tabwm.model.edit_model import EditModel, EditModelConfig
from tabwm.model.rt_backbone import RTConfig


@dataclass
class TabWMConfig(EditModelConfig):
    value_decoder_width: int = 256
    value_decoder_layers: int = 2
    value_decoder_heads: int = 8
    value_decoder_transformation_proposals: bool = True

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> TabWMConfig:
        values = dict(raw)
        if "transformation_proposals" in values:
            values["value_decoder_transformation_proposals"] = values.pop(
                "transformation_proposals"
            )
        rt = values.get("rt")
        if isinstance(rt, Mapping):
            values["rt"] = RTConfig(
                **{key: value for key, value in rt.items() if key in RTConfig.__dataclass_fields__}
            )
        return cls(**{key: value for key, value in values.items() if key in cls.__dataclass_fields__})


class TabWM(EditModel):
    def __init__(self, config: TabWMConfig | None = None) -> None:
        cfg = config or TabWMConfig()
        super().__init__(cfg)
        self.config = cfg
        self.value_decoder = ContextualValueDecoder(
            cfg.rt.d_model,
            cfg.rt.d_text,
            ValueDecoderConfig(
                width=cfg.value_decoder_width,
                layers=cfg.value_decoder_layers,
                heads=cfg.value_decoder_heads,
                transformation_proposals=cfg.value_decoder_transformation_proposals,
            ),
        )

    def forward_base(self, batch: dict[str, torch.Tensor], *, use_flex: bool | None = None):
        hidden = self.encoder(batch, use_flex=use_flex)
        return hidden, self.decoder(hidden, batch)

    def apply_value_decoder(
        self,
        hidden: torch.Tensor,
        base,
        batch: dict[str, torch.Tensor],
        *,
        target_positions: torch.Tensor | None = None,
    ):
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            correction = self.value_decoder(
                hidden, base, batch, target_positions=target_positions
            )
        return replace(base, value_numeric=base.value_numeric.float() + correction)

    def forward(self, batch: dict[str, torch.Tensor], *, use_flex: bool | None = None):
        hidden, base = self.forward_base(batch, use_flex=use_flex)
        return self.apply_value_decoder(hidden, base, batch)

    def parameter_summary(self) -> dict[str, float]:
        summary = super().parameter_summary()
        value = sum(parameter.numel() for parameter in self.value_decoder.parameters())
        summary["value_decoder_M"] = round(value / 1e6, 3)
        summary["total_M"] = round(summary["total_M"] + value / 1e6, 3)
        return summary

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, *, map_location: str = "cpu"
    ) -> tuple[TabWM, dict]:
        blob = torch.load(Path(path), map_location=map_location, weights_only=False)
        meta = blob.get("meta") or {}
        raw_config = meta.get("model_config")
        if not isinstance(raw_config, Mapping):
            raise ValueError("checkpoint has no model_config")
        model = cls(TabWMConfig.from_dict(raw_config))
        model.load_state_dict(blob["model"], strict=True)
        return model, meta

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str | Path,
        *,
        device: str = "cpu",
        revision: str | None = None,
    ) -> TabWM:
        path = Path(model_id_or_path)
        if not path.exists():
            from huggingface_hub import snapshot_download

            path = Path(
                snapshot_download(
                    repo_id=str(model_id_or_path),
                    revision=revision,
                    allow_patterns=["config.json", "model.safetensors"],
                )
            )
        with (path / "config.json").open() as handle:
            raw = json.load(handle)
        model = cls(TabWMConfig.from_dict(raw.get("model", raw)))
        model.load_state_dict(
            load_file(path / "model.safetensors", device=device), strict=True
        )
        return model.to(device).eval()
