"""CPU diagnostic contract tests; no Torch, native physics or policy execution."""

import ast
import inspect
import json
import sys
from pathlib import Path

import pytest

from sadhana_fasttd3_abc.learner import verify_source
from sadhana_fasttd3_abc.reach import ReachMonitor, reach_args
from sadhana_fasttd3_abc.reach_contract import REACH, validate_config

BUILD = Path("/home/user/aditya/RL/builds/abc-fasttd3-reach-diagnostic-20261013")


def config(phase="training"):
    return json.loads((BUILD / (phase + "-host.json")).read_text())


def episode(rank, index, *, initial=0.4, achieved=False):
    return {
        "rank": rank,
        "episode_index": index,
        "controls": 128,
        "completed": True,
        "terminated": False,
        "truncated": True,
        "reach_spec": REACH,
        "reach_diagnostic": {
            "initial_distance_m": initial,
            "minimum_distance_m": initial - 0.05,
            "final_distance_m": initial,
            "distance_gain_m": 0.05,
            "reward_sum": 40.0,
            "maximum_hold_controls": 5 if achieved else 0,
            "eligible_initial_distance": initial > 0.15,
            "reach_achieved": achieved,
            "qualified_reach_achieved": initial > 0.15 and achieved,
            "controls": 128,
        },
        "final_native_task_eval": {
            "num_active_bottles": 5,
            "num_bottles_in_bin": 0,
            "bottle_in_bin_mask": [False] * 5,
            "success": False,
        },
        "native_full_success": False,
        "physics": {
            key: {
                "time_finite": True,
                "finite": {"qpos": True, "qvel": True, "qacc": True, "ctrl": True},
                "warnings": {
                    name: {"number": 0}
                    for name in (
                        "INERTIA",
                        "CONTACTFULL",
                        "CNSTRFULL",
                        "BADQPOS",
                        "BADQVEL",
                        "BADQACC",
                        "BADCTRL",
                    )
                },
            }
            for key in ("initial", "final")
        },
    }


def test_author_defaults_and_candidate_explicit_override_source():
    # Normally installed author defaults are CPU-read in the existing normal
    # runtime. This check inspects candidate argument calls; it does not call
    # candidate BaseArgs construction without the unavailable Torch dependency.
    baseline = json.loads((BUILD / "AUTHOR_DEFAULTS.json").read_text())
    source = ast.parse(inspect.getsource(reach_args))
    calls = [
        node
        for node in ast.walk(source)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "BaseArgs"
    ]
    assert len(calls) == 1
    override_names = {keyword.arg for keyword in calls[0].keywords}
    assert override_names == {
        "env_name",
        "exp_name",
        "seed",
        "cuda",
        "num_envs",
        "num_eval_envs",
        "total_timesteps",
        "use_wandb",
        "eval_interval",
        "render_interval",
        "save_interval",
    }
    assert baseline["batch_size"] == 32768 and baseline["buffer_size"] == 51200
    assert baseline["learning_starts"] == 10 and baseline["num_atoms"] == 101
    assert baseline["v_min"] == -250 and baseline["v_max"] == 250
    assert baseline["amp"] is True and baseline["amp_dtype"] == "bf16"
    assert baseline["compile"] is True
    # Every literal override is checked against its declared value.
    literals = {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in calls[0].keywords
        if isinstance(keyword.value, ast.Constant)
    }
    assert literals == {
        "env_name": "ABC-FiveBottlesVector-v1",
        "seed": 1,
        "num_envs": 64,
        "num_eval_envs": 1,
        "use_wandb": False,
        "eval_interval": 0,
        "render_interval": 0,
        "save_interval": 0,
    }


def drive(monitor, *, initial=0.4, until=2048, achieved=False):
    decision = None
    for control in range(1, until + 1):
        rows = (
            [
                episode(rank, control // 128 - 1, initial=initial, achieved=achieved)
                for rank in range(64)
            ]
            if control % 128 == 0
            else []
        )
        decision = monitor.observe(control, rows)
        if decision is not None:
            break
    return decision


def test_plateau_stops_naturally_only_after_qualified_full_windows():
    monitor = ReachMonitor(config())
    decision = drive(monitor)
    assert decision == {
        "reason": "qualified_TRAIN_reach_progress_plateau",
        "controls": 1280,
        "stale_windows": 3,
    }
    assert monitor.counts == [10] * 64
    assert [window["eligible_for_stopping"] for window in monitor.windows] == [
        False,
        True,
        True,
        True,
        True,
    ]
    assert monitor.evidence()["matched_initial_final_deterministic_actor_comparison"] == "pending"


def test_already_near_resets_cannot_drive_qualified_stopping():
    monitor = ReachMonitor(config())
    decision = drive(monitor, initial=0.1, achieved=True)
    assert (
        decision["controls"] == 2048
        and decision["reason"] == "declared_diagnostic_maximum_completed"
    )
    assert all(not window["eligible_for_stopping"] for window in monitor.windows)
    assert monitor.best == {}


def test_new_runtime_gate_reaches_original_updates_without_score_claim():
    monitor = ReachMonitor(config("runtime_gate"))
    decision = drive(monitor, until=256)
    assert decision["controls"] == 256 and monitor.counts == [2] * 64
    assert all(not window["eligible_for_stopping"] for window in monitor.windows)


@pytest.mark.parametrize(
    "mutation",
    ["task_success_label", "partial_episode", "native_judge", "warning", "qualification"],
)
def test_malformed_episode_rejected(mutation):
    monitor = ReachMonitor(config())
    row = episode(0, 0)
    if mutation == "task_success_label":
        row["success"] = True
    if mutation == "partial_episode":
        row["controls"] = 127
    if mutation == "native_judge":
        row["final_native_task_eval"]["success"] = True
    if mutation == "warning":
        row["physics"]["final"]["warnings"]["CONTACTFULL"]["number"] = 1
    if mutation == "qualification":
        row["reach_diagnostic"]["qualified_reach_achieved"] = True
    with pytest.raises((ValueError, FloatingPointError)):
        monitor.observe(1, [row])


def test_monitor_rejects_nonsequential_controls():
    with pytest.raises(ValueError):
        ReachMonitor(config()).observe(2, [])


def test_protected_original_payloads_authenticated_and_not_imported():
    provenance = verify_source()
    assert len(provenance["files"]) == 12
    assert len(provenance["protected_update_functions"]) == 4
    assert "torch" not in sys.modules and "mujoco" not in sys.modules
    assert "fast_td3.train" not in sys.modules


def test_invalid_device_and_bounds_rejected():
    with pytest.raises(ValueError):
        reach_args(config(), device="gpu")
    value = config()
    value["vector_steps"] = 2049
    with pytest.raises(ValueError):
        validate_config(value)
