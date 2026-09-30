"""TabWM training loop."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from tabwm.data.collate import (
    CollateParams,
    HashEmbedder,
    TabWMBatch,
    TextEmbedder,
    collate,
    required_ctx_len,
)
from tabwm.data.transition import TaskParams, sample_episode_pool, sample_task
from tabwm.train.checkpoint import save_checkpoint
from tabwm.train.losses import LossWeights, edit_set_accuracy, edit_set_loss, hungarian_match
from tabwm.world.generator import WorldConfig, sample_world
from tabwm.world.rollout import Policy


@dataclass
class TrainConfig:

    max_steps: int = 20_000
    batch_size: int = 16
    grad_accum: int = 1
    lr: float = 3e-4
    lr_backbone_mult: float = 0.25
    weight_decay: float = 0.01
    warmup_steps: int = 500
    grad_clip: float = 1.0
    precision: str = "bf16"

    value_teacher_start: float = 1.0
    value_teacher_end: float = 0.0
    value_teacher_steps: int = 50_000

    seed: int = 0
    world_seed_start: int = 0
    world_seed_end: int = 10_000_000
    tasks_per_world: int = 2
    num_workers: int = 8
    prefetch_factor: int = 4

    attention_budget: int = 7e7.__int__()

    def batch_for_length(self, seq_len: int) -> int:
        if seq_len <= 0:
            return self.batch_size
        return max(1, min(self.batch_size, self.attention_budget // (seq_len * seq_len)))

    log_every: int = 20
    checkpoint_every: int = 2_000
    out_dir: str = "runs/tabwm"
    run_name: str = "dev"

    activation_checkpointing: bool = False
    compile_model: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class WorldTaskStream(IterableDataset):

    def __init__(
        self,
        train: TrainConfig,
        world_config: WorldConfig,
        task_params: TaskParams,
        collate_params: CollateParams,
        embedder: TextEmbedder | None = None,
        policy: Policy | None = None,
    ) -> None:
        self.train = train
        self.world_config = world_config
        self.task_params = task_params
        self.collate_params = collate_params
        self.embedder = embedder
        self.policy = policy

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        nworkers = info.num_workers if info else 1

        embedder = self.embedder or HashEmbedder(self.collate_params.d_text)
        rng = np.random.default_rng(np.random.SeedSequence([self.train.seed, wid]))

        seed = self.train.world_seed_start + wid
        buffer: list = []
        while True:
            world = sample_world(seed, self.world_config)
            pool = sample_episode_pool(world, self.task_params, policy=self.policy)
            for _ in range(self.train.tasks_per_world):
                task = sample_task(world, rng, self.task_params, pool=pool)
                if task is not None:
                    buffer.append(task)


                params = self.collate_params
                if params.length_mixture is not None:
                    need = required_ctx_len(buffer, params) if buffer else 0
                    seq_len = params.length_mixture.sample(rng, min_length=need)
                    params = replace(params, ctx_len=seq_len, length_mixture=None)
                else:
                    seq_len = params.ctx_len
                n = self.train.batch_for_length(seq_len)

                if len(buffer) >= n:
                    yield collate(buffer[:n], embedder, params, rng)
                    buffer = buffer[n:]
            seed += nworkers
            if seed >= self.train.world_seed_end:
                seed = self.train.world_seed_start + wid


def _identity(x):
    return x[0]


def make_loader(stream: WorldTaskStream, train: TrainConfig) -> DataLoader:
    return DataLoader(
        stream,
        batch_size=1,
        num_workers=train.num_workers,
        prefetch_factor=train.prefetch_factor if train.num_workers else None,
        collate_fn=_identity,
        persistent_workers=bool(train.num_workers),
    )


@dataclass
class RunCounters:
    step: int = 0
    tasks: int = 0
    tokens: int = 0
    gold_edits: int = 0
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train: TrainConfig | None = None,
        world_config: WorldConfig | None = None,
        task_params: TaskParams | None = None,
        collate_params: CollateParams | None = None,
        loss_weights: LossWeights | None = None,
        device: str = "cuda",
    ) -> None:
        self.train_config = train or TrainConfig()
        self.world_config = world_config or WorldConfig()
        self.task_params = task_params or TaskParams()
        self.collate_params = collate_params or CollateParams()
        self.loss_weights = loss_weights or LossWeights()
        self.device = device
        self.model = model.to(device)
        self.counters = RunCounters()

        self.out_dir = Path(self.train_config.out_dir) / self.train_config.run_name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.out_dir / "metrics.jsonl"

        self.optimizer = self._build_optimizer()
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self._lr_lambda)
        self.amp_dtype = (
            torch.bfloat16 if self.train_config.precision == "bf16" else torch.float32
        )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        cfg = self.train_config
        backbone, new = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            is_new = (
                name.startswith(("decoder.", "value_decoder."))
                or "role_transition" in name
            )
            (new if is_new else backbone).append(p)
        return torch.optim.AdamW(
            [
                {"params": backbone, "lr": cfg.lr * cfg.lr_backbone_mult},
                {"params": new, "lr": cfg.lr},
            ],
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95),
        )

    def _lr_lambda(self, step: int) -> float:
        cfg = self.train_config
        if step < cfg.warmup_steps:
            return (step + 1) / max(cfg.warmup_steps, 1)
        progress = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))


    def _value_teacher_ratio(self) -> float:
        cfg = self.train_config
        if cfg.value_teacher_steps <= 0:
            return float(cfg.value_teacher_end)
        progress = min(max(self.counters.step / cfg.value_teacher_steps, 0.0), 1.0)
        return float(
            cfg.value_teacher_start
            + progress * (cfg.value_teacher_end - cfg.value_teacher_start)
        )

    @torch.no_grad()
    def _teacher_value_positions(self, base, batch, matches, ratio: float):
        if ratio <= 0.0:
            return None
        from tabwm.data.delta import ValueKind

        predicted = base.cell_logits.argmax(-1)
        route = torch.full_like(predicted, -1)
        for b, (slots, golds) in enumerate(matches):
            for slot, gold in zip(slots, golds):
                if int(batch["edit_value_kind"][b, gold]) != ValueKind.NUMERIC:
                    continue
                cell = int(batch["edit_cell_pos"][b, gold])
                if cell < 0:
                    continue
                if ratio >= 1.0 or float(torch.rand((), device=route.device)) < ratio:
                    route[b, int(slot)] = cell
        return route

    def step_on_batch(self, batch: TabWMBatch, *, train: bool = True) -> dict[str, float]:
        tensors = batch.torch(device=self.device, dtype=torch.float32)
        with torch.autocast(
            device_type=self.device.split(":")[0], dtype=self.amp_dtype,
            enabled=self.amp_dtype != torch.float32,
        ):
            hidden, base = self.model.forward_base(tensors)

        base_for_matching = _to_float32(base)
        matches = hungarian_match(base_for_matching, tensors, self.loss_weights)
        ratio = self._value_teacher_ratio() if train else 0.0
        target_positions = self._teacher_value_positions(
            base_for_matching, tensors, matches, ratio
        )
        out = self.model.apply_value_decoder(
            hidden, base, tensors, target_positions=target_positions
        )
        out = _to_float32(out)
        loss = edit_set_loss(out, tensors, self.loss_weights, matches=matches)

        if train:
            (loss.total / self.train_config.grad_accum).backward()

        logs = loss.log_dict()
        logs.update(edit_set_accuracy(out, tensors, matches))
        logs["seq_len"] = batch.seq_len
        logs["mean_k"] = float(np.mean([d["k"] for d in batch.diagnostics]))
        logs["value_teacher_ratio"] = ratio
        return logs

    def optimizer_step(self) -> float:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.train_config.grad_clip
        )
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        return float(grad_norm)


    def fit(self, loader: DataLoader | None = None) -> RunCounters:
        cfg = self.train_config
        if loader is None:
            stream = WorldTaskStream(
                cfg, self.world_config, self.task_params, self.collate_params
            )
            loader = make_loader(stream, cfg)

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        t0 = time.time()
        window: list[dict[str, float]] = []

        it = iter(loader)
        while self.counters.step < cfg.max_steps:
            for _ in range(cfg.grad_accum):
                batch = next(it)
                logs = self.step_on_batch(batch)
                window.append(logs)
                self.counters.tasks += batch.batch_size
                self.counters.tokens += int((~batch.is_padding).sum())
                self.counters.gold_edits += int(batch.edit_valid.sum())

            grad_norm = self.optimizer_step()
            self.counters.step += 1
            self.counters.seconds = time.time() - t0

            if self.counters.step % cfg.log_every == 0:
                self._log(window, grad_norm)
                window = []
            if cfg.checkpoint_every and self.counters.step % cfg.checkpoint_every == 0:
                self.save(f"step_{self.counters.step}.pt")

        self.save("final.pt")
        return self.counters

    def _log(self, window: list[dict[str, float]], grad_norm: float) -> None:
        agg = {k: float(np.mean([w[k] for w in window])) for k in window[0]}
        record = {
            "step": self.counters.step,
            "grad_norm": grad_norm,
            "lr": self.scheduler.get_last_lr()[-1],


            "vram_peak_gb": (
                torch.cuda.max_memory_allocated() / 2**30
                if torch.cuda.is_available() else 0.0
            ),
            "tasks": self.counters.tasks,
            "tokens": self.counters.tokens,
            "tasks_per_s": self.counters.tasks / max(self.counters.seconds, 1e-9),
            **agg,
        }
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(
            f"step {record['step']:>6} | loss {record['loss']:7.4f} "
            f"| f1 {record['edit_f1']:.3f} | op {record['op_acc']:.3f} "
            f"| ptr {record['pointer_acc']:.3f} | {record['tasks_per_s']:.1f} task/s",
            flush=True,
        )

    def save(self, name: str) -> Path:
        return save_checkpoint(
            self.out_dir / name,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.counters.step,
            world_config=self.world_config,
            train_config=self.train_config,
            counters=self.counters.as_dict(),
            extra={
                "task_params": asdict(self.task_params),
                "collate_params": asdict(self.collate_params),
                "loss_weights": asdict(self.loss_weights),
                "model_family": type(self.model).__name__,


            },
        )


def _to_float32(out):
    from dataclasses import replace

    def f(x):
        return x.float() if torch.is_tensor(x) and x.is_floating_point() else x

    return replace(
        out,
        op_logits=f(out.op_logits),
        exist_logits=f(out.exist_logits),
        cell_logits=f(out.cell_logits),
        row_logits=f(out.row_logits),
        table_logits=f(out.table_logits),
        link_logits=f(out.link_logits),
        value_numeric=f(out.value_numeric),
        value_categorical=f(out.value_categorical),
        value_boolean=f(out.value_boolean),
        aux={k: f(v) for k, v in out.aux.items()},
        slots=f(out.slots),
    )
