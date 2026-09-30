from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import numpy as np
import torch

from tabwm.data.collate import CollateParams, HashEmbedder, collate
from tabwm.data.transition import TaskParams, sample_episode_pool, sample_task
from tabwm.model.tabwm import TabWM
from tabwm.world.generator import sample_world


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--world-seed", type=int, default=9_000_000)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model = TabWM.from_pretrained(args.model, device=args.device)
    world = sample_world(args.world_seed)
    rng = np.random.default_rng(args.world_seed)
    task_params = TaskParams()
    pool = sample_episode_pool(world, task_params)
    task = sample_task(world, rng, task_params, pool=pool, k=args.k)
    if task is None:
        raise RuntimeError("The sampled world produced no task")

    batch = collate(
        [task],
        HashEmbedder(model.config.rt.d_text),
        CollateParams(ctx_len=4096, d_text=model.config.rt.d_text),
        rng,
    )
    prediction = model.predict_deltas(batch, device=args.device, use_flex=False)[0]
    print(
        json.dumps(
            {
                "world_seed": args.world_seed,
                "k": args.k,
                "action": asdict(task.query_action),
                "prediction": asdict(prediction),
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()

