"""Installed CPU stop contracts; environment and train main are explicit doubles.

The integration case executes the real learner.train wrapper and original
save_params Torch export. It does not execute author learner math or a simulator.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from fast_td3.fast_td3_utils import save_params

from sadhana_fasttd3_abc import campaign
from sadhana_fasttd3_abc.vector import VectorRemoteEnv


def config(phase="training"):
    # Minimal declared reward metadata double; actual reward math has its own
    # independent ABC package tests and native admission.
    return {
        "phase": phase,
        "profile": campaign.PROFILE,
        "vector_steps": campaign.PHASES[phase],
        "campaign": deepcopy(campaign.campaign_spec(phase)),
        "reward": {"profile": "abc_full_success_pbrs_v1", "gamma": 0.99},
    }


def episode(rank, index, *, count=0, peak_bottles=None, progress=None, spec=None):
    success = count == 5
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
            "bottle_in_bin_mask": [slot < count for slot in range(5)],
            "success": success,
            "reward": count / 5,
            "ever_success": True,
            "max_bottles_in_bin_so_far": 5,
        },
        "reward_v2": {
            "spec": deepcopy(config()["reward"] if spec is None else spec),
            "peak_bottles": count if peak_bottles is None else peak_bottles,
            "peak_progress": {
                "phi": 0.02,
                "reach": 0.10,
                "clearance": 0.10,
                "lift_transport": 0.10,
                **(progress or {}),
            },
        },
        # Deliberately large loss changes must not drive this task monitor.
        "critic_loss": float(10000 / (index + 1)),
    }


def observe_window(monitor, window, **kwargs):
    return monitor.observe(
        3540 * window,
        [episode(rank, monitor.counts[rank], **kwargs) for rank in range(128)],
    )


def test_separate_finite_profiles_preserve_original128_learner_defaults(tmp_path):
    for phase, maximum in (("runtime_gate", 64), ("training", 42480)):
        declaration = config(phase)
        path = tmp_path / f"{phase}.json"
        path.write_text(json.dumps(declaration))
        loaded = campaign.load_config(path)
        args = campaign.campaign_args(loaded, device="cpu")
        assert (args.total_timesteps, args.num_envs, args.batch_size) == (maximum, 128, 32768)
        assert (
            args.buffer_size,
            args.learning_starts,
            args.num_updates,
            args.policy_frequency,
        ) == (
            51200,
            10,
            2,
            2,
        )
        assert args.gamma == 0.99
        assert args.amp and args.compile and args.amp_dtype == "bf16"
        assert not args.cuda and not args.use_wandb
        assert args.save_interval == args.eval_interval == args.render_interval == 0
    assert not config()["campaign"]["loss_based_stopping"]
    assert config()["campaign"]["validation_seeds_used_for_stopping"] == []


@pytest.mark.parametrize("field,value", [("vector_steps", 4000), ("vector_steps", True)])
def test_unreviewed_budgets_rejected(tmp_path, field, value):
    declaration = config()
    declaration[field] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(declaration))
    with pytest.raises(ValueError):
        campaign.load_config(path)


def test_first_minimum_exploration_window_then_three_complete_stale_windows():
    monitor = campaign.TaskFutilityMonitor(config())
    for window in range(1, 9):
        assert observe_window(monitor, window) is None
    assert monitor.windows[4]["eligible"] is False
    assert monitor.windows[5]["eligible"] is True
    assert monitor.windows[5]["stale_windows"] == 0
    decision = observe_window(monitor, 9)
    assert decision["reason"] == "no_observed_training_task_improvement"
    assert decision["controls"] == 31860
    assert min(decision["completed_episodes_per_rank"]) == 9
    assert decision["stale_windows"] == 3


def test_missing_world_or_underexplored_rank_cannot_trigger_futility():
    monitor = campaign.TaskFutilityMonitor(config())
    for window in range(1, 10):
        episodes = [episode(rank, monitor.counts[rank]) for rank in range(127)]
        assert monitor.observe(window * 3540, episodes) is None
    assert monitor.counts[-1] == 0
    assert not any(row["eligible"] for row in monitor.windows)
    assert monitor.stale == 0


@pytest.mark.parametrize("metric", tuple(campaign.THRESHOLDS))
def test_actual_native_or_geometric_threshold_improvement_resets_patience(metric):
    monitor = campaign.TaskFutilityMonitor(config())
    for window in range(1, 9):
        assert observe_window(monitor, window) is None
    assert monitor.stale == 2
    episodes = [episode(rank, monitor.counts[rank]) for rank in range(128)]
    if metric == "success_rate":
        episodes[0] = episode(0, monitor.counts[0], count=5)
    elif metric in ("final_bottles", "peak_bottles"):
        for rank in range(13):  # 13/128 exceeds the declared 0.1-bottle mean change.
            episodes[rank] = episode(
                rank,
                monitor.counts[rank],
                count=1 if metric == "final_bottles" else 0,
                peak_bottles=1,
            )
    else:
        component = metric.removeprefix("peak_")
        current = episodes[0]["reward_v2"]["peak_progress"][component]
        for row in episodes:
            row["reward_v2"]["peak_progress"][component] = current + campaign.THRESHOLDS[metric]
    assert monitor.observe(9 * 3540, episodes) is None
    assert metric in monitor.windows[-1]["meaningful_improvement"]
    assert monitor.stale == 0


def test_below_threshold_noise_and_loss_reduction_do_not_extend_patience():
    monitor = campaign.TaskFutilityMonitor(config())
    for window in range(1, 7):
        assert observe_window(monitor, window) is None
    for window in range(7, 9):
        assert observe_window(monitor, window, progress={"phi": 0.0249}) is None
    decision = observe_window(monitor, 9, progress={"phi": 0.0249})
    assert decision["stale_windows"] == 3


def test_equal_rank_weights_do_not_favor_a_world_with_more_success_episodes():
    monitor = campaign.TaskFutilityMonitor(config())
    rows = [episode(rank, 0) for rank in range(1, 128)]
    rows.extend(episode(0, index, count=5) for index in range(10))
    assert monitor.observe(3540, rows) is None
    metrics = monitor.windows[-1]["metrics"]
    assert metrics["success_rate"] == pytest.approx(1 / 128)
    assert metrics["final_bottles"] == pytest.approx(5 / 128)
    assert monitor.counts[0] == 10 and monitor.counts[-1] == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("completed", False),
        ("controls", 100),
        ("controls", True),
        ("rank", True),
        ("episode_index", False),
        ("episode_index", 0.0),
        ("success", True),
        ("terminated", True),
        ("truncated", False),
    ],
)
def test_incomplete_or_inconsistent_episode_cannot_drive_stopping(field, value):
    monitor = campaign.TaskFutilityMonitor(config())
    row = episode(0, 0)
    row[field] = value
    with pytest.raises(ValueError):
        monitor.observe(3540, [row])
    assert monitor.counts[0] == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_active_bottles", 4),
        ("num_bottles_in_bin", 0.0),
        ("num_bottles_in_bin", True),
        ("success", True),
        ("reward", 0.2),
        ("bottle_in_bin_mask", [0] * 5),
    ],
)
def test_typed_current_native_five_bottle_judge_is_required(field, value):
    monitor = campaign.TaskFutilityMonitor(config())
    row = episode(0, 0)
    row["final_native_task_eval"][field] = value
    with pytest.raises(ValueError):
        monitor.observe(3540, [row])


def test_nonfinite_progress_and_changed_reward_profile_rejected():
    monitor = campaign.TaskFutilityMonitor(config())
    row = episode(0, 0, progress={"phi": float("nan")})
    with pytest.raises(FloatingPointError):
        monitor.observe(3540, [row])
    row = episode(0, 0)
    row["reward_v2"]["spec"]["gamma"] = 0.9
    with pytest.raises(ValueError):
        monitor.observe(3540, [row])


def test_declared_maximum_has_priority_over_same_step_patience():
    monitor = campaign.TaskFutilityMonitor(config())
    monitor.stale = 2
    monitor.eligible_started = True
    monitor.best = campaign.TaskFutilityMonitor.episode_metrics(episode(0, 0))
    monitor.counts = [11] * 128
    decision = observe_window(monitor, 12)
    assert decision["reason"] == "declared_maximum_completed"
    assert decision["controls"] == 42480
    assert decision["stale_windows"] == 3


def test_runtime_gate_is_only_a64_call_budget_not_training_early_stopping():
    monitor = campaign.TaskFutilityMonitor(config("runtime_gate"))
    for control in range(1, 64):
        assert monitor.observe(control, []) is None
    decision = monitor.observe(64, [])
    assert decision["reason"] == "runtime_gate_completed"
    assert monitor.windows == []


@pytest.mark.parametrize("timeout", [0, -1, 601, float("inf")])
def test_rpc_deadline_has_declared_finite_maximum(timeout):
    with pytest.raises(ValueError):
        campaign.Endpoint("/unused-cpu-double.sock", timeout=timeout, schema=campaign.SCHEMA)


class EndpointDouble:
    def __init__(self, declaration):
        self.description = {
            "test_double": True,
            "num_envs": 128,
            "num_obs": 216,
            "num_actions": 14,
            "max_episode_steps": 3540,
            "privileged_state": True,
            "profile": campaign.PROFILE,
            "reward": declaration["reward"],
            "campaign": declaration["campaign"],
            "carried_arm_reset": True,
            "reset_options": {
                "randomization": {
                    "bottle_count": 5,
                    "randomize_variants": True,
                    "randomize_scales": True,
                }
            },
            "reset_seed_bank": {
                "base": 200000,
                "rank_stride": 1,
                "episode_stride": 128,
                "max_episode_index": 780,
                "fallback_offset": 100000,
            },
        }

    def call(self, operation, **_):
        if operation == "describe":
            return self.description
        if operation == "reset":
            return {
                "obs": [[0.0] * 216] * 128,
                "requested_seeds": list(range(200000, 200128)),
                "actual_seeds": list(range(200000, 200128)),
            }
        raise AssertionError("No native RPC is used by the explicit collection double")


def test_real_train_wrapper_finishes_current_updates_then_exports_natural_stop(
    monkeypatch, tmp_path
):
    declaration = config()
    endpoint = EndpointDouble(declaration)
    created, final_order, update_counts = [], [], {"critic": 0, "actor": 0}
    monkeypatch.setattr(campaign, "Endpoint", lambda *_, **__: endpoint)

    def collection_double(env, _actions):
        env.controls += 1
        env.transitions = env.controls * 128
        completed = (
            [episode(rank, env.monitor.counts[rank]) for rank in range(128)]
            if env.controls % 3540 == 0
            else []
        )
        env.episodes.extend(completed)
        return None, None, None, {"episodes": completed}

    monkeypatch.setattr(VectorRemoteEnv, "step", collection_double)

    def author_loop_double(*, args, env_factory, observer):
        env, eval_env, render_env = env_factory(args, torch.device("cpu"))
        assert env is eval_env is render_env
        created.append(env)
        env.reset()
        actor, critic = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
        target = deepcopy(critic)
        normalizer = torch.nn.Identity()
        for model in (actor, critic):
            for parameter in model.parameters():
                parameter.grad = torch.ones_like(parameter)
        logs = {"qf_loss": torch.tensor(0.1), "actor_loss": torch.tensor(0.1)}
        global_step = 0
        while global_step < args.total_timesteps:
            env.step(None)
            last = env.monitor.decision is not None
            if last:
                final_order.append("stop_requested_by_completed_collection")
            if global_step > args.learning_starts:
                for update in range(args.num_updates):
                    with torch.no_grad():
                        critic.weight.add_(1e-7)
                    observer("critic", critic, logs)
                    update_counts["critic"] += 1
                    if last:
                        final_order.append(f"critic{update}")
                    if update % args.policy_frequency == 1:
                        with torch.no_grad():
                            actor.weight.add_(1e-7)
                        observer("actor", actor, logs)
                        update_counts["actor"] += 1
                        if last:
                            final_order.append("actor")
                    with torch.no_grad():
                        target.weight.lerp_(critic.weight, args.tau)
                    if last:
                        final_order.append(f"soft_target{update}")
            # The original schedulers run before the completed outer-step increment.
            if last:
                final_order.extend(["schedulers", "outer_increment"])
            global_step += 1
        assert global_step == env.controls == args.total_timesteps
        final_order.append("final_export")
        save_params(
            global_step,
            actor,
            critic,
            target,
            normalizer,
            normalizer,
            args,
            f"models/{args.env_name}__{args.exp_name}__{args.seed}_final.pt",
        )

    # Real train wrapper still authenticates installed original file bytes. Only
    # its imported main callable is replaced by this disclosed CPU boundary double.
    monkeypatch.setitem(sys.modules, "fast_td3.train", SimpleNamespace(main=author_loop_double))
    result = campaign.train_campaign(
        declaration, "/unused-double.sock", tmp_path / "train", device="cpu"
    )
    assert result["controls"] == 31860
    assert result["transitions"] == 31860 * 128
    assert result["declared_maximum_args"]["total_timesteps"] == 42480
    assert result["effective_final_args"]["total_timesteps"] == 31860
    assert result["actual_updates"] == update_counts == {"critic": 63698, "actor": 31849}
    assert created[0].closed
    assert final_order == [
        "stop_requested_by_completed_collection",
        "critic0",
        "soft_target0",
        "critic1",
        "actor",
        "soft_target1",
        "schedulers",
        "outer_increment",
        "final_export",
    ]
    payload = torch.load(Path(result["checkpoint"]["path"]), map_location="cpu", weights_only=False)
    assert payload["global_step"] == payload["args"]["total_timesteps"] == 31860
    for name in ("actor_state_dict", "qnet_state_dict", "qnet_target_state_dict"):
        assert payload[name] and all(
            torch.isfinite(value).all() for value in payload[name].values()
        )
    receipt = json.loads((tmp_path / "train" / "receipt.json").read_text())
    assert receipt["status"] == "completed" and all(receipt["finite_gradient_samples"].values())
