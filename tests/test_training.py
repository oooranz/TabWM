"""Small integration checks for the training and inference path."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from tabwm.data.collate import CollateParams, HashEmbedder, collate
from tabwm.data.delta import ValueKind
from tabwm.data.transition import TaskParams, sample_task
from tabwm.model.contextual_value_decoder import visible_pairs
from tabwm.model.rt_backbone import RTConfig
from tabwm.model.tabwm import TabWM, TabWMConfig
from tabwm.train.losses import LossWeights, register_binned_head
from tabwm.train.trainer import TrainConfig, Trainer
from tabwm.world.generator import sample_world
from tabwm.world.state import StateDelta


def make_model() -> TabWM:
    torch.manual_seed(7)
    model = TabWM(
        TabWMConfig(
            rt=RTConfig(num_blocks=1, d_model=32, d_text=384, num_heads=4, d_ff=64),
            n_slots=8,
            decoder_layers=1,
            decoder_heads=4,
            numeric_head="binned",
            value_decoder_width=32,
            value_decoder_layers=1,
            value_decoder_heads=4,
        )
    ).float()
    register_binned_head(model.decoder.binned)
    return model


def make_batch():
    tasks = []
    for seed in range(128):
        task = sample_task(sample_world(seed), np.random.default_rng(seed), TaskParams(), k=4)
        if task is not None:
            tasks.append(task)
        if len(tasks) < 8:
            continue
        batch = collate(
            tasks, HashEmbedder(), CollateParams(ctx_len=384), np.random.default_rng(91)
        )
        tensors = batch.torch(dtype=torch.float32)
        has_numeric = bool(
            (tensors["edit_valid"] & (tensors["edit_value_kind"] == ValueKind.NUMERIC)).any()
        )
        if has_numeric and bool((visible_pairs(tensors)[0] >= 0).any()):
            return batch, tensors
        tasks = tasks[4:]
    raise AssertionError("No task had both a numeric target and visible evidence")


class CoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.batch, cls.tensors = make_batch()

    def test_value_decoder_starts_as_identity(self) -> None:
        model = make_model().eval()
        with torch.no_grad():
            hidden, base = model.forward_base(self.tensors, use_flex=False)
            prediction = model.apply_value_decoder(hidden, base, self.tensors)
        self.assertTrue(torch.equal(prediction.value_numeric, base.value_numeric.float()))
        self.assertTrue(torch.equal(prediction.cell_logits, base.cell_logits))

    def test_one_step_inference(self) -> None:
        model = make_model().eval()
        predictions = model.predict_deltas(self.batch, device="cpu", use_flex=False)
        self.assertEqual(len(predictions), self.batch.batch_size)
        self.assertTrue(all(isinstance(delta, StateDelta) for delta in predictions))

    def test_query_target_is_not_a_model_input(self) -> None:
        task = self.batch.tasks[0]
        params = CollateParams(ctx_len=384)
        original = collate([task], HashEmbedder(), params, np.random.default_rng(91))
        changed = collate(
            [replace(task, target_delta=StateDelta())],
            HashEmbedder(), params, np.random.default_rng(91),
        )
        model = make_model().eval()
        with torch.no_grad():
            a = model(original.torch(dtype=torch.float32), use_flex=False)
            b = model(changed.torch(dtype=torch.float32), use_flex=False)
        for name in ("op_logits", "cell_logits", "value_numeric"):
            self.assertTrue(torch.equal(getattr(a, name), getattr(b, name)), name)

    def test_joint_gradient_and_checkpoint(self) -> None:
        model = make_model()
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model,
                TrainConfig(
                    max_steps=2,
                    batch_size=self.batch.batch_size,
                    num_workers=0,
                    precision="fp32",
                    value_teacher_steps=2,
                    out_dir=directory,
                ),
                loss_weights=LossWeights(value_ordinal=True, value_bins_weight=0.3),
                device="cpu",
            )
            logs = trainer.step_on_batch(self.batch)
            self.assertTrue(np.isfinite(logs["loss"]))
            self.assertGreater(logs["value_refined"], 0.0)
            self.assertEqual(logs["value_teacher_ratio"], 1.0)
            for prefix in ("encoder.", "decoder.", "value_decoder."):
                self.assertTrue(any(
                    name.startswith(prefix)
                    and parameter.grad is not None
                    and bool((parameter.grad != 0).any())
                    for name, parameter in model.named_parameters()
                ), prefix)
            trainer.optimizer_step()
            trainer.counters.step = 1
            path = trainer.save("step_1.pt")
            loaded, meta = TabWM.from_checkpoint(Path(path))
            self.assertEqual(meta["step"], 1)
            self.assertNotIn("git_commit", meta)
            self.assertNotIn("out_dir", meta["train_config"])
            self.assertNotIn("run_name", meta["train_config"])
            self.assertNotIn(directory, repr(meta))
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, loaded.state_dict()[name]), name)
            model.eval()
            loaded.eval()
            with torch.no_grad():
                expected = model(self.tensors, use_flex=False)
                actual = loaded(self.tensors, use_flex=False)
            for name in ("op_logits", "cell_logits", "value_numeric"):
                self.assertTrue(torch.equal(getattr(expected, name), getattr(actual, name)), name)


if __name__ == "__main__":
    unittest.main()
