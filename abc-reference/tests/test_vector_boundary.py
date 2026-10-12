"""Installed vector proxy contracts; explicit CPU host doubles, no learner."""

import pytest
import torch

from sadhana_fasttd3_abc.vector import VectorRemoteEnv, qualified_args


class EndpointDouble:
    def __init__(self, *, terminated=True, truncated=False):
        self.terminated, self.truncated = terminated, truncated
        self.operations = []
        self.description = {
            "num_envs": 128,
            "num_obs": 216,
            "num_actions": 14,
            "max_episode_steps": 3540,
            "privileged_state": True,
            "profile": "training128",
            "reset_options": {
                "randomization": {
                    "bottle_count": 5,
                    "randomize_variants": True,
                    "randomize_scales": True,
                }
            },
            "carried_arm_reset": True,
            "reset_seed_bank": {
                "base": 200000,
                "rank_stride": 1,
                "episode_stride": 128,
                "max_episode_index": 780,
                "fallback_offset": 100000,
            },
        }

    def call(self, operation, **payload):
        self.operations.append((operation, payload))
        if operation == "describe":
            return self.description
        if operation == "reset":
            return {
                "obs": [[0.0] * 216 for _ in range(128)],
                "requested_seeds": list(range(200000, 200128)),
                "actual_seeds": list(range(200000, 200128)),
            }
        return {
            "obs": [[4.0] * 216 for _ in range(128)],
            "terminal_obs": [[3.0] * 216 for _ in range(128)],
            "rewards": [0.0] * 128,
            "terminated": [self.terminated] * 128,
            "truncated": [self.truncated] * 128,
            "episodes": [{"rank": rank, "completed": True} for rank in range(128)]
            if self.terminated or self.truncated
            else [],
            "autoresets": [
                {"rank": rank, "requested_seed": 200128 + rank, "actual_seed": 200128 + rank}
                if self.terminated or self.truncated
                else None
                for rank in range(128)
            ],
        }


def test_original128_recipe_defaults_without_learner_execution():
    args = qualified_args()
    assert (args.num_envs, args.batch_size, args.buffer_size) == (128, 32768, 51200)
    assert (args.num_steps, args.learning_starts, args.policy_frequency, args.num_updates) == (
        1,
        10,
        2,
        2,
    )
    assert args.compile and args.amp and args.amp_dtype == "bf16"
    assert (
        not args.use_wandb and args.eval_interval == args.render_interval == args.save_interval == 0
    )


@pytest.mark.parametrize(
    "terminated,truncated,bootstrap",
    [(True, False, False), (False, True, True), (False, False, True)],
)
def test_raw_terminal_timeout_and128_transition_accounting(terminated, truncated, bootstrap):
    endpoint = EndpointDouble(terminated=terminated, truncated=truncated)
    env = VectorRemoteEnv(endpoint, device="cpu")
    env.reset()
    obs, rewards, dones, info = env.step(torch.zeros((128, 14)))
    assert torch.equal(obs, torch.full((128, 216), 4.0))
    assert torch.equal(info["observations"]["raw"]["obs"], torch.full((128, 216), 3.0))
    mask = info["time_outs"] | ~dones  # Exact original update_main bootstrap expression.
    assert bool(mask[0]) == bootstrap
    assert env.controls == 1 and env.transitions == 128
    assert len(env.actual_seeds) == (256 if terminated or truncated else 128)
    assert endpoint.operations[1] == ("reset", {})


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_envs", 1),
        ("num_obs", 14),
        ("carried_arm_reset", False),
        ("profile", "cpu_acceptance2"),
    ],
)
def test_mismatched_original_profile_is_rejected(field, value):
    endpoint = EndpointDouble()
    endpoint.description[field] = value
    with pytest.raises(ValueError):
        VectorRemoteEnv(endpoint, device="cpu")


def test_no_evaluation_seed_override_in_training_proxy():
    env = VectorRemoteEnv(EndpointDouble(), device="cpu")
    with pytest.raises(ValueError, match="seed"):
        env.reset(seeds=[20260511])


def test_invalid_action_no_rpc_partial_advancement():
    endpoint = EndpointDouble()
    env = VectorRemoteEnv(endpoint, device="cpu")
    env.reset()
    count = len(endpoint.operations)
    action = torch.zeros((128, 14))
    action[-1, -1] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        env.step(action)
    assert len(endpoint.operations) == count
