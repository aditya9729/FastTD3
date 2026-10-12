"""Small explicit train and frozen-evaluation commands."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--socket", required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--steps", type=int, default=1000)
    training.add_argument("--num-envs", type=int, default=1)
    training.add_argument("--batch-size", type=int, default=32768)
    training.add_argument("--buffer-size", type=int, default=51200)
    training.add_argument("--seed", type=int, default=1)
    training.add_argument("--layout-seed", type=int, default=9030)
    training.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    training.add_argument("--compile", action="store_true")
    training.add_argument("--amp", action="store_true")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--socket", required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--seeds", type=int, nargs="+", default=[20260511, 20260512, 20260513])
    evaluation.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    options = parser.parse_args()
    os.environ["WANDB_MODE"] = "disabled"
    if options.command == "train":
        from fast_td3.hyperparams import BaseArgs

        from .learner import remote_factory, train

        args = BaseArgs(
            env_name="ABC-PutBottles-v1",
            exp_name="FastTD3Scratch",
            seed=options.seed,
            cuda=options.device == "cuda",
            num_envs=options.num_envs,
            num_eval_envs=1,
            total_timesteps=options.steps,
            batch_size=options.batch_size,
            buffer_size=options.buffer_size,
            compile=options.compile,
            amp=options.amp,
            use_wandb=False,
            eval_interval=0,
            render_interval=0,
            save_interval=0,
        )
        train(args, remote_factory(options.socket, seed=options.layout_seed), options.output)
    else:
        from .environment import Endpoint, RemoteEnv
        from .learner import evaluate

        env = RemoteEnv(Endpoint(options.socket), device=options.device)
        try:
            evaluate(options.checkpoint, env, options.seeds, options.output)
        finally:
            env.close()


if __name__ == "__main__":
    main()
