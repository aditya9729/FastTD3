"""CPU contract checks for declared budgets and exact frozen consumer identity."""

from copy import deepcopy

import pytest
from fast_td3.hyperparams import BaseArgs

from sadhana_fasttd3_abc.captured import CapturedRemoteEnv
from sadhana_fasttd3_abc.vector import qualified_args, training_args


def test_fixed_training_preserves_all_original_learning_defaults():
    original = BaseArgs()
    args = training_args()
    for name in (
        "batch_size",
        "buffer_size",
        "num_updates",
        "policy_frequency",
        "actor_hidden_dim",
        "critic_hidden_dim",
        "actor_learning_rate",
        "critic_learning_rate",
        "gamma",
        "tau",
        "compile",
        "amp",
        "amp_dtype",
        "num_atoms",
        "v_min",
        "v_max",
        "obs_normalization",
        "reward_normalization",
        "std_min",
        "std_max",
        "learning_starts",
        "num_steps",
    ):
        assert getattr(args, name) == getattr(original, name), name
    assert (args.num_envs, args.total_timesteps) == (128, 4000)
    assert args.checkpoint_path is None
    assert args.exp_name != qualified_args().exp_name
    assert args.eval_interval == args.render_interval == args.save_interval == 0


@pytest.mark.parametrize("steps", [64, 3999, 4001, True])
def test_training_budget_cannot_silently_be_qualification_or_full_paper(steps):
    with pytest.raises(ValueError, match="4000"):
        training_args(steps=steps)


class EndpointDouble:
    def __init__(self, description):
        self.description = description
        self.calls = []

    def call(self, operation, **kwargs):
        self.calls.append((operation, kwargs))
        if operation == "describe":
            return self.description
        assert operation == "reset"
        return {"actual_seeds": [9100], "obs": [[0] * 216]}


@pytest.fixture
def description():
    return {
        "num_envs": 1,
        "num_obs": 216,
        "num_actions": 14,
        "max_episode_steps": 3540,
        "privileged_state": True,
        "profile": "captured_validation",
        "autoreset": False,
        "action_profile": "normalized14_affine_clip_v1",
        "reset_policy": {
            "reset_verified": True,
            "profile": "captured_demonstration_mjb_integration_v1",
            "heldout_seed": 9100,
            "allowed_seeds": [9100],
        },
    }


def test_frozen_client_accepts_only_the_captured_seed(description):
    endpoint = EndpointDouble(description)
    env = CapturedRemoteEnv(endpoint, device="cpu")
    assert tuple(env.reset(seeds=[9100]).shape) == (1, 216)
    with pytest.raises(ValueError, match="exact captured"):
        env.reset(seeds=[9101])
    env.close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_obs", 14),
        ("num_envs", 128),
        ("max_episode_steps", 64),
        ("autoreset", True),
        ("action_profile", "raw"),
        ("profile", "training128"),
    ],
)
def test_frozen_client_rejects_incompatible_host(description, field, value):
    bad = deepcopy(description)
    bad[field] = value
    with pytest.raises(ValueError):
        CapturedRemoteEnv(EndpointDouble(bad), device="cpu")


def test_frozen_client_rejects_unverified_or_other_reset(description):
    description["reset_policy"]["reset_verified"] = False
    with pytest.raises(ValueError):
        CapturedRemoteEnv(EndpointDouble(description), device="cpu")
