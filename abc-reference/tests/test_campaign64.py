"""Normal-installed CPU contracts for the explicit reduced-world campaign.

All observations/rewards and the author main callable are explicit doubles.
The last test executes the real learner wrapper and original Torch export;
it does not run native simulation, CUDA or author optimizer mathematics.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from fast_td3.fast_td3_utils import SimpleReplayBuffer, save_params
from fast_td3.hyperparams import BaseArgs
from tensordict import TensorDict

from sadhana_fasttd3_abc import campaign


def declaration(phase="training", count=64):
    return {
        "phase": phase,
        "profile": campaign.PROFILE64 if count == 64 else campaign.PROFILE,
        "num_envs": count,
        "vector_steps": campaign.PHASES[phase],
        "campaign": deepcopy(campaign.campaign_spec(phase)),
        "reward": {"profile": "abc_full_success_pbrs_v1", "gamma": 0.99},
    }


def episode(rank, index, *, success=False):
    count = 5 if success else 0
    return {
        "rank": rank,
        "episode_index": index,
        "completed": True,
        "success": success,
        "terminated": success,
        "truncated": not success,
        "controls": 17 if success else 3540,
        "final_native_task_eval": {
            "num_active_bottles": 5,
            "num_bottles_in_bin": count,
            "bottle_in_bin_mask": [success] * 5,
            "success": success,
            "reward": count / 5,
        },
        "reward_v2": {
            "spec": deepcopy(declaration()["reward"]),
            "peak_bottles": count,
            "peak_progress": {
                "phi": 0.02,
                "reach": 0.1,
                "clearance": 0.1,
                "lift_transport": 0.1,
            },
        },
    }


@pytest.mark.parametrize("count", [64, 128])
@pytest.mark.parametrize("phase", ["runtime_gate", "training"])
def test_declared_count_changes_only_parallelism_and_run_boundary(tmp_path, count, phase):
    config = declaration(phase, count)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    args = campaign.campaign_args(campaign.load_config(path), device="cpu")
    actual, original = asdict(args), asdict(BaseArgs())
    boundary_fields = {
        "env_name",
        "exp_name",
        "cuda",
        "num_envs",
        "num_eval_envs",
        "total_timesteps",
        "use_wandb",
        "eval_interval",
        "render_interval",
        "save_interval",
    }
    assert {k: v for k, v in actual.items() if k not in boundary_fields} == {
        k: v for k, v in original.items() if k not in boundary_fields
    }
    assert args.num_envs == count
    assert (args.batch_size, args.buffer_size, args.num_updates, args.policy_frequency) == (
        32768,
        51200,
        2,
        2,
    )
    assert args.total_timesteps == campaign.PHASES[phase]
    assert campaign.campaign_spec(phase)["improvement_thresholds"]["success_rate"] == 1 / 128
    if phase == "training" and count == 64:
        assert args.total_timesteps * count == 2718720


@pytest.mark.parametrize("phase", ["runtime_gate", "training"])
def test_historical128_config_without_added_field_stays_identical(tmp_path, phase):
    config = declaration(phase, 128)
    config.pop("num_envs")
    path = tmp_path / "historical.json"
    path.write_text(json.dumps(config))
    assert campaign.load_config(path) == config
    args = campaign.campaign_args(config, device="cpu")
    assert args.num_envs == 128
    assert args.exp_name == "FastTD3ABCRewardV2_" + phase
    monitor = campaign.TaskFutilityMonitor(config)
    assert len(monitor.counts) == 128
    assert monitor.evidence()["spec"] == config["campaign"]


@pytest.mark.parametrize("phase", ["runtime_gate", "training"])
def test_exact_saved_historical128_host_config_remains_compatible(phase):
    path = Path(__file__).parent / "fixtures" / f"fasttd3_reward_v2_{phase}128.json"
    config = campaign.load_config(path)
    assert config["profile"] == campaign.PROFILE and config["num_envs"] == 128
    assert config["campaign"] == campaign.campaign_spec(phase)
    args = campaign.campaign_args(config, device="cpu")
    assert args.num_envs == 128 and args.total_timesteps == campaign.PHASES[phase]
    assert args.exp_name == "FastTD3ABCRewardV2_" + phase


@pytest.mark.parametrize("value", [None, True, 64.0, 32, 128, "64"])
def test_reduced_profile_requires_exact_explicit_integer_count(value):
    config = declaration()
    config["num_envs"] = value
    with pytest.raises(ValueError, match="world count"):
        campaign.world_count(config)


@pytest.mark.parametrize("profile", [None, "reward_v2_training32", [], 64])
def test_unknown_profile_rejected(profile):
    config = declaration()
    config["profile"] = profile
    with pytest.raises(ValueError, match="declared"):
        campaign.world_count(config)


@pytest.mark.parametrize("count", [64, 128])
def test_rank_weighted_statistics_use_actual_worlds_with_unchanged_threshold(count):
    monitor = campaign.TaskFutilityMonitor(declaration(count=count))
    rows = [episode(rank, 0, success=rank == count - 1) for rank in range(count)]
    monitor.observe(3540, rows)
    metrics = monitor.windows[-1]["metrics"]
    assert metrics["success_rate"] == 1 / count
    assert metrics["final_bottles"] == 5 / count
    assert metrics["success_rate"] >= campaign.THRESHOLDS["success_rate"]


@pytest.mark.parametrize("count", [64, 128])
def test_original_replay_capacity_is_per_world_and_sampling_is_per_world(count):
    # Three slots per world keep this original-code probe below a few KiB.
    replay = SimpleReplayBuffer(count, 3, 1, 1, 1, device="cpu")
    assert replay.observations.shape == (count, 3, 1)
    rows = torch.arange(count, dtype=torch.float32).reshape(count, 1)
    replay.extend(
        TensorDict(
            {
                "observations": rows,
                "actions": torch.zeros((count, 1)),
                "next": {
                    "observations": rows + 1,
                    "rewards": torch.zeros(count),
                    "dones": torch.zeros(count, dtype=torch.long),
                    "truncations": torch.zeros(count, dtype=torch.long),
                },
            },
            batch_size=[count],
        )
    )
    per_world = 32768 // count
    batch = replay.sample(per_world)
    assert tuple(batch["observations"].shape) == (32768, 1)
    assert torch.bincount(batch["observations"][:, 0].long()).tolist() == [per_world] * count
    assert per_world == (512 if count == 64 else 256)


def test_reduced_monitor_requires_six_complete_episodes_for_every_rank_and_three_windows():
    monitor = campaign.TaskFutilityMonitor(declaration())
    for window in range(1, 10):
        rows = [episode(rank, monitor.counts[rank]) for rank in range(64)]
        decision = monitor.observe(window * 3540, rows)
        if window < 9:
            assert decision is None
        else:
            assert decision["reason"] == "no_observed_training_task_improvement"
            assert decision["controls"] == 31860
    assert monitor.windows[4]["eligible"] is False
    assert monitor.windows[5]["eligible"] is True
    assert monitor.windows[5]["stale_windows"] == 0
    assert monitor.counts == [9] * 64


def test_missing64th_world_cannot_drive_futility():
    monitor = campaign.TaskFutilityMonitor(declaration())
    for window in range(1, 13):
        rows = [episode(rank, monitor.counts[rank]) for rank in range(63)]
        decision = monitor.observe(window * 3540, rows)
        assert (
            decision is None if window < 12 else decision["reason"] == "declared_maximum_completed"
        )
    assert monitor.counts[-1] == 0
    assert not any(window["eligible"] for window in monitor.windows)


@pytest.mark.parametrize("rank", [64, 127, -1, True])
def test_reduced_monitor_rejects_outside_rank_inventory(rank):
    monitor = campaign.TaskFutilityMonitor(declaration())
    with pytest.raises(ValueError, match="rank/index"):
        monitor.observe(3540, [episode(rank, 0)])


class EndpointDouble:
    """Explicit64/128-world JSON tensor boundary with one injectable episode end."""

    def __init__(self, config, *, end_rank=None):
        self.count = campaign.world_count(config)
        self.end_rank = end_rank
        self.step_calls = 0
        self.seeds = list(range(200000, 200000 + self.count))
        self.description = {
            "num_envs": self.count,
            "num_obs": 216,
            "num_actions": 14,
            "max_episode_steps": 3540,
            "privileged_state": True,
            "profile": config["profile"],
            "reward": config["reward"],
            "campaign": config["campaign"],
            "carried_arm_reset": True,
            "reset_options": {
                "randomization": {
                    "bottle_count": 5,
                    "randomize_variants": True,
                    "randomize_scales": True,
                },
            },
            "reset_seed_bank": {
                "base": 200000,
                "rank_stride": 1,
                "episode_stride": 128,
                "max_episode_index": 780,
                "fallback_offset": 100000,
            },
        }

    def call(self, operation, **kwargs):
        if operation == "describe":
            return self.description
        if operation == "reset":
            return {
                "obs": [[0.0] * 216 for _ in range(self.count)],
                "requested_seeds": self.seeds.copy(),
                "actual_seeds": self.seeds.copy(),
            }
        assert operation == "step"
        assert len(kwargs["actions"]) == self.count
        self.step_calls += 1
        obs = [[0.0] * 216 for _ in range(self.count)]
        terminal = deepcopy(obs)
        ended = [False] * self.count
        episodes, resets = [], [None] * self.count
        if self.end_rank is not None and self.step_calls == 1:
            rank = self.end_rank
            ended[rank] = True
            terminal[rank][0] = 17.0
            episodes = [episode(rank, 0, success=True)]
            self.seeds[rank] += 128
            resets[rank] = {
                "rank": rank,
                "requested_seed": self.seeds[rank],
                "actual_seed": self.seeds[rank],
            }
        return {
            "obs": obs,
            "terminal_obs": terminal,
            "rewards": [0.0] * self.count,
            "terminated": ended,
            "truncated": [False] * self.count,
            "episodes": episodes,
            "autoresets": resets,
        }


def test64_client_keeps_original_seed_stride_and_terminal_observation():
    config = declaration("runtime_gate")
    endpoint = EndpointDouble(config, end_rank=63)
    args = campaign.campaign_args(config, device="cpu")
    env = campaign.CampaignVectorRemoteEnv(endpoint, settings=args, device="cpu", config=config)
    assert env.reset().shape == (64, 216)
    obs, _, done, info = env.step(torch.zeros((64, 14)))
    assert done.nonzero().flatten().tolist() == [63]
    assert obs[63, 0].item() == 0
    assert info["observations"]["raw"]["obs"][63, 0].item() == 17
    assert env.last_requested[63] == 200063 + 128
    assert env.transitions == 64
    assert env.monitor.counts == [0] * 63 + [1]
    env.close()


@pytest.mark.parametrize("field,value", [("num_envs", 128), ("profile", campaign.PROFILE)])
def test64_client_rejects128_host_description(field, value):
    config = declaration("runtime_gate")
    endpoint = EndpointDouble(config)
    endpoint.description[field] = value
    with pytest.raises(ValueError):
        campaign.CampaignVectorRemoteEnv(
            endpoint,
            settings=campaign.campaign_args(config, device="cpu"),
            device="cpu",
            config=config,
        )


def test64_client_rejects64_seed_stride():
    config = declaration("runtime_gate")
    endpoint = EndpointDouble(config)
    endpoint.description["reset_seed_bank"]["episode_stride"] = 64
    with pytest.raises(ValueError, match="reset_seed_bank"):
        campaign.CampaignVectorRemoteEnv(
            endpoint,
            settings=campaign.campaign_args(config, device="cpu"),
            device="cpu",
            config=config,
        )


def test_real_train_wrapper64_finishes_updates_then_original_export(monkeypatch, tmp_path):
    config = declaration("runtime_gate")
    endpoint = EndpointDouble(config)
    monkeypatch.setattr(campaign, "Endpoint", lambda *_, **__: endpoint)
    order = []

    def author_loop_double(*, args, env_factory, observer):
        env, eval_env, render_env = env_factory(args, torch.device("cpu"))
        assert env is eval_env is render_env
        env.reset()
        actor, critic = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
        target = deepcopy(critic)
        for model in (actor, critic):
            for parameter in model.parameters():
                parameter.grad = torch.ones_like(parameter)
        logs = {"qf_loss": torch.tensor(0.1), "actor_loss": torch.tensor(0.1)}
        step = 0
        while step < args.total_timesteps:
            env.step(torch.zeros((64, 14)))
            last = env.monitor.decision is not None
            if last:
                order.append("stop_at_completed_collection")
            if step > args.learning_starts:
                for update in range(args.num_updates):
                    observer("critic", critic, logs)
                    if last:
                        order.append("critic")
                    if update % args.policy_frequency == 1:
                        observer("actor", actor, logs)
                        if last:
                            order.append("actor")
                    if last:
                        order.append("soft_target")
            if last:
                order.append("schedulers_then_increment")
            step += 1
        order.append("final_export")
        normalizer = torch.nn.Identity()
        save_params(
            step,
            actor,
            critic,
            target,
            normalizer,
            normalizer,
            args,
            f"models/{args.env_name}__{args.exp_name}__{args.seed}_final.pt",
        )

    monkeypatch.setitem(sys.modules, "fast_td3.train", SimpleNamespace(main=author_loop_double))
    result = campaign.train_campaign(
        config, "/unused-double.sock", tmp_path / "train", device="cpu"
    )
    assert result["controls"] == 64 and result["transitions"] == 4096
    assert result["actual_updates"] == {"critic": 106, "actor": 53}
    assert result["parallelism"]["actual_rank_statistics_denominator"] == 64
    assert result["parallelism"]["maximum_transitions"] == 4096
    assert result["parallelism"]["replay_buffer_size_per_env"] == 51200
    assert result["parallelism"]["author_default_aggregate_replay_capacity"] == 6553600
    assert result["parallelism"]["actual_aggregate_replay_capacity"] == 3276800
    assert result["parallelism"]["critic_updates_per_vector_call"] == 2
    assert (
        result["parallelism"]["author_default_critic_updates_per_collected_transition"] == 0.015625
    )
    assert result["parallelism"]["actual_critic_updates_per_collected_transition"] == 0.03125
    assert order == [
        "stop_at_completed_collection",
        "critic",
        "soft_target",
        "critic",
        "actor",
        "soft_target",
        "schedulers_then_increment",
        "final_export",
    ]
    payload = torch.load(Path(result["checkpoint"]["path"]), map_location="cpu", weights_only=False)
    assert payload["global_step"] == 64 and payload["args"]["num_envs"] == 64
    assert result["stopping"]["decision"]["reason"] == "runtime_gate_completed"
