"""Train TabWM on WorldRel tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tabwm.data.collate import CollateParams, LengthMixture
from tabwm.data.transition import TaskParams
from tabwm.model.rt_backbone import RTConfig
from tabwm.model.tabwm import TabWM, TabWMConfig
from tabwm.train.checkpoint import latest_checkpoint, load_checkpoint
from tabwm.train.losses import LossWeights, register_binned_head
from tabwm.train.trainer import TrainConfig, Trainer
from tabwm.world.generator import WorldConfig

K_WEIGHTS = (0.04, 0.04, 0.06, 0.08, 0.12, 0.18, 0.24, 0.24)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", type=Path, default=Path("plurel"))
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--name", default="tabwm")
    parser.add_argument("--steps", type=int, default=600_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--encoder-lr-mult", type=float, default=0.25)
    parser.add_argument("--warmup", type=int, default=5_000)
    parser.add_argument("--teacher-steps", type=int, default=50_000)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    run_dir = args.output / args.name
    saved = latest_checkpoint(run_dir) if args.resume else None
    if args.resume and saved is None:
        raise SystemExit(f"No step checkpoint found in {run_dir}")
    if saved is None:
        rt = RTConfig.from_json(args.pretrained / "config.json")
        config = TabWMConfig(rt=rt, numeric_head="binned")
        model, info = TabWM.from_plurel_checkpoint(args.pretrained, config=config)
        print(f"Loaded {info['loaded']} pretrained tensors")
    else:
        blob = torch.load(saved, map_location="cpu", weights_only=False)
        config = TabWMConfig.from_dict(blob["meta"]["model_config"])
        model = TabWM(config)
        del blob

    register_binned_head(model.decoder.binned)
    train_config = TrainConfig(
        max_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_backbone_mult=args.encoder_lr_mult,
        warmup_steps=args.warmup,
        value_teacher_steps=args.teacher_steps,
        num_workers=args.workers,
        checkpoint_every=args.checkpoint_every,
        out_dir=str(args.output),
        run_name=args.name,
        seed=args.seed,
        world_seed_start=0,
        world_seed_end=8_000_000,
    )
    prior = WorldConfig()
    trainer = Trainer(
        model,
        train_config,
        prior,
        TaskParams(k_weights=K_WEIGHTS),
        CollateParams(length_mixture=LengthMixture()),
        LossWeights(value_ordinal=True, value_bins_weight=0.3),
        device=args.device,
    )

    if saved is not None:
        meta = load_checkpoint(
            saved,
            model=model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            world_config=prior,
            map_location=args.device,
        )
        for key, value in meta.get("counters", {}).items():
            if hasattr(trainer.counters, key):
                setattr(trainer.counters, key, value)
        print(f"Resumed from step {trainer.counters.step}")

    trainer.fit()


if __name__ == "__main__":
    main()
