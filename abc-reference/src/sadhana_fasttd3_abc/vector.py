"""Original FastTD3 training through an independently seeded ABC world pool."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from .environment import Endpoint, RemoteEnv

SCHEMA = "sadhana.fasttd3.abc.vector.v1"


class VectorRemoteEnv(RemoteEnv):
    """Keep original FastTD3 raw terminal/timeout semantics at the RPC boundary."""

    def __init__(self, endpoint: Endpoint, *, device: Any, profile: str = "training128"):
        super().__init__(endpoint, device=device, seed=200000)
        count = {"training128": 128, "cpu_acceptance2": 2}.get(profile)
        if count is None or self.description.get("profile") != profile:
            raise ValueError("ABC vector host has a different declared profile")
        if (self.num_envs, self.num_obs, self.num_actions, self.max_episode_steps) != (
            count,
            216,
            14,
            3540,
        ):
            raise ValueError("ABC vector host dimensions differ from the declared native profile")
        if (
            self.description.get("reset_options")
            != {
                "randomization": {
                    "bottle_count": 5,
                    "randomize_variants": True,
                    "randomize_scales": True,
                }
            }
            or self.description.get("carried_arm_reset") is not True
        ):
            raise ValueError("ABC vector reset distribution or carried arm semantics differ")
        if self.description.get("reset_seed_bank") != {
            "base": 200000,
            "rank_stride": 1,
            "episode_stride": 128,
            "max_episode_index": 780,
            "fallback_offset": 100000,
        }:
            raise ValueError("ABC vector training seed bank differs")
        self.requested_seeds: list[int] = []
        self.last_requested: list[int] = []
        self.transitions = 0

    def reset(self, *, seeds=None):
        if seeds is not None:
            raise ValueError("Training seeds belong to the independent native world bank")
        if self.closed:
            raise RuntimeError("ABC vector environment is closed")
        result = self.endpoint.call("reset")
        requested, actual = result.get("requested_seeds"), result.get("actual_seeds")
        if (
            type(requested) is not list
            or type(actual) is not list
            or len(requested) != self.num_envs
            or len(actual) != self.num_envs
        ):
            raise ValueError("ABC vector reset requires one seed pair per owned world")
        for rank, (seed, sampled) in enumerate(zip(requested, actual)):
            if (
                type(seed) is not int
                or type(sampled) is not int
                or seed < 200000
                or (seed - 200000 - rank) % 128
                or not 0 <= (seed - 200000 - rank) // 128 <= 780
            ):
                raise ValueError("ABC reset seed is outside its rank-specific training bank")
            if sampled not in (seed, seed + 100000):
                raise ValueError("ABC reset resampled outside the original failure fallback")
        self.requested_seeds.extend(requested)
        self.actual_seeds.extend(actual)
        self.last_requested = requested.copy()
        self.resets += 1
        return self._array(result.get("obs"), (self.num_envs, self.num_obs))

    def step(self, actions):
        import torch

        if len(self.last_requested) != self.num_envs:
            raise RuntimeError("ABC vector reset must precede stepping")
        result = super().step(actions)
        _, _, dones, infos = result
        resets = infos["autoresets"]
        if type(resets) is not list or len(resets) != self.num_envs:
            raise ValueError("ABC vector step requires one autoreset row or null per world")
        ended = set(torch.nonzero(dones, as_tuple=False).flatten().cpu().tolist())
        episodes = infos["episodes"]
        if (
            len(episodes) != len(ended)
            or any(
                type(row.get("rank")) is not int or row.get("completed") is not True
                for row in episodes
            )
            or {row.get("rank") for row in episodes} != ended
        ):
            raise ValueError("ABC completed episode evidence does not match native done flags")
        for rank, row in enumerate(resets):
            if rank not in ended:
                if row is not None:
                    raise ValueError("ABC autoreset occurred without an episode end")
                continue
            expected = self.last_requested[rank] + 128
            if (
                type(row) is not dict
                or row.get("rank") != rank
                or row.get("requested_seed") != expected
            ):
                raise ValueError("ABC autoreset did not advance its rank-specific seed bank")
            sampled = row.get("actual_seed")
            if type(sampled) is not int or sampled not in (expected, expected + 100000):
                raise ValueError("ABC autoreset actual seed differs from the declared fallback")
            self.requested_seeds.append(expected)
            self.actual_seeds.append(sampled)
            self.last_requested[rank] = expected
        self.transitions = self.controls * self.num_envs
        return result


def qualified_args(*, steps: int = 64, device: str = "cuda"):
    if type(steps) is not int or not 12 <= steps <= 64 or device not in {"cpu", "cuda"}:
        raise ValueError("Vector qualification requires12..64 steps and an explicit device")
    from fast_td3.hyperparams import BaseArgs

    return BaseArgs(
        env_name="ABC-FiveBottlesVector-v1",
        exp_name="FastTD3ScratchVector128",
        seed=1,
        cuda=device == "cuda",
        num_envs=128,
        num_eval_envs=1,
        total_timesteps=steps,
        use_wandb=False,
        eval_interval=0,
        render_interval=0,
        save_interval=0,
    )


def remote_factory(socket_path: str, *, timeout: float):
    def create(args, device):
        env = VectorRemoteEnv(Endpoint(socket_path, timeout=timeout, schema=SCHEMA), device=device)
        # Evaluation is disabled in this training qualification. Captured VAL
        # support waits for the independently published reset/camera API.
        return env, env, env

    return create


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--rpc-timeout", type=int, default=180)
    options = parser.parse_args()
    os.environ["WANDB_MODE"] = "disabled"
    from .learner import train

    train(
        qualified_args(steps=options.steps, device=options.device),
        remote_factory(options.socket, timeout=options.rpc_timeout),
        options.output,
    )


if __name__ == "__main__":
    main()
