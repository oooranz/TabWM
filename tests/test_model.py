from __future__ import annotations

import os

import pytest

from tabwm.model.tabwm import TabWM


def test_released_checkpoint_loads_strictly() -> None:
    model_path = os.environ.get("TABWM_MODEL_PATH")
    if model_path is None:
        pytest.skip("set TABWM_MODEL_PATH to run the checkpoint test")
    model = TabWM.from_pretrained(model_path)
    assert sum(parameter.numel() for parameter in model.parameters()) == 24_680_164

