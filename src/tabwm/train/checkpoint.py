"""Training checkpoints."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

CHECKPOINT_VERSION = 1


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def rng_state() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _as_cpu_byte(t: Any) -> torch.Tensor:
    return torch.as_tensor(t).detach().to(device="cpu", dtype=torch.uint8)


def restore_rng(state: dict[str, Any]) -> None:
    torch.set_rng_state(_as_cpu_byte(state["torch"]))
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        cuda_states = [_as_cpu_byte(s) for s in state["torch_cuda"]]


        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_states)
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    step: int = 0,
    world_config: Any = None,
    train_config: Any = None,
    metrics: dict | None = None,
    counters: dict | None = None,
    extra: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    train_settings = _to_jsonable(train_config)
    if isinstance(train_settings, dict):
        train_settings.pop("out_dir", None)
        train_settings.pop("run_name", None)

    meta = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "step": step,
        "world_prior_hash": getattr(world_config, "hash", lambda: None)(),
        "worldrel_version": getattr(world_config, "version", None),
        "model_config": _to_jsonable(getattr(model, "config", None)),
        "world_config": _to_jsonable(world_config),
        "train_config": train_settings,
        "counters": counters or {},
        "metrics": metrics or {},
        **(extra or {}),
    }

    torch.save(
        {
            "meta": meta,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "rng": rng_state(),
        },
        path,
    )
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
    return path


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    world_config: Any = None,
    map_location: str = "cpu",
    allow_prior_mismatch: bool = False,
    restore_rng_state: bool = True,
) -> dict:
    blob = torch.load(Path(path), map_location=map_location, weights_only=False)
    meta = blob["meta"]

    if world_config is not None and not allow_prior_mismatch:
        want = world_config.hash()
        got = meta.get("world_prior_hash")
        if got is not None and got != want:
            raise ValueError(
                f"world prior changed since this checkpoint ({got} -> {want}). "
                "Resuming would mix two data distributions in one curve; pass "
                "allow_prior_mismatch=True only if that is genuinely intended."
            )

    model.load_state_dict(blob["model"])
    if optimizer is not None and blob.get("optimizer") is not None:
        optimizer.load_state_dict(blob["optimizer"])
    if scheduler is not None and blob.get("scheduler") is not None:
        scheduler.load_state_dict(blob["scheduler"])
    if restore_rng_state and blob.get("rng") is not None:
        restore_rng(blob["rng"])
    return meta


def latest_checkpoint(directory: str | Path, pattern: str = "step_*.pt") -> Path | None:
    files = sorted(Path(directory).glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None
