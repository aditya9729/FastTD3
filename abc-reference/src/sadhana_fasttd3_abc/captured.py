"""Frozen original actor/normalizer evaluation on one authenticated VAL capture."""

from __future__ import annotations

import argparse
from pathlib import Path

from .environment import Endpoint, RemoteEnv

SCHEMA = "sadhana.fasttd3.abc.captured.v1"


class CapturedRemoteEnv(RemoteEnv):
    def __init__(self, endpoint: Endpoint, *, device):
        super().__init__(endpoint, device=device)
        if (self.num_envs, self.num_obs, self.num_actions, self.max_episode_steps) != (
            1,
            216,
            14,
            3540,
        ):
            raise ValueError(
                "Captured Fast evaluation requires one216-state/14-action full-horizon world"
            )
        reset = self.description.get("reset_policy", {})
        if (
            self.description.get("profile") != "captured_validation"
            or self.description.get("autoreset") is not False
            or self.description.get("action_profile") != "normalized14_affine_clip_v1"
            or reset.get("reset_verified") is not True
            or reset.get("profile") != "captured_demonstration_mjb_integration_v1"
            or type(reset.get("heldout_seed")) is not int
            or reset.get("allowed_seeds") != [reset["heldout_seed"]]
        ):
            raise ValueError("Captured host did not admit exact VAL reset/action semantics")
        self.seed = reset["heldout_seed"]

    def reset(self, *, seeds=None):
        if seeds != [self.seed]:
            raise ValueError("Frozen actor must request this exact captured VAL seed")
        return super().reset(seeds=seeds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--rpc-timeout", type=int, default=180)
    options = parser.parse_args()
    from .learner import evaluate, verify_source

    verify_source()
    env = CapturedRemoteEnv(
        Endpoint(options.socket, schema=SCHEMA, timeout=options.rpc_timeout), device=options.device
    )
    try:
        evaluate(options.checkpoint, env, [env.seed], options.output)
    finally:
        env.close()


if __name__ == "__main__":
    main()
