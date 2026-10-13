"""TRAIN-only reaching diagnostic; never a five-bottle task reward."""

from __future__ import annotations

from typing import Any

PROFILE = "reach_diagnostic64_v1"
HORIZON = 128
PHASES = {"runtime_gate": 256, "training": 2048}
REACH = {
    "schema": "sadhana.abc.reach-diagnostic/v1",
    "reward": "exp(-min_2_grasp_sites_by_5_bottle_COM_distance_m/0.15)",
    "scale_m": 0.15,
    "threshold_m": 0.12,
    "hold_controls": 5,
    "initial_eligibility_m": 0.15,
    "horizon_controls": HORIZON,
    "episode_end": "fixed_horizon_timeout_only",
    "native_five_bottle_judge": "measured_separately_unchanged",
    "comparison": "matched_initial_final_deterministic_TRAIN_actor_comparison_pending",
}
STOPPING = {
    "schema": "sadhana.abc.reach-diagnostic-stopping/v1",
    "window_calls": 256,
    "minimum_calls": 512,
    "patience_windows": 3,
    "qualified_reach_rate_improvement": 0.05,
    "distance_gain_improvement_m": 0.01,
    "eligibility": "each_rank_has_completed_episode_with_initial_distance_gt0.15_m",
    "scope": "TRAIN_progress_plateau_only_no_learning_or_convergence_claim",
    "validation_seeds": [],
}


def validate_config(value: Any) -> dict:
    fields = {
        "schema_version",
        "profile",
        "phase",
        "num_envs",
        "seed_base",
        "seed_stride",
        "max_episode_steps",
        "vector_steps",
        "reach",
        "stopping",
        "selection_evidence",
        "timeouts",
    }
    if type(value) is not dict or set(value) != fields:
        raise ValueError("Use the separate complete reach diagnostic schema1")
    for key, expected in {
        "schema_version": 1,
        "profile": PROFILE,
        "num_envs": 64,
        "seed_base": 200000,
        "seed_stride": 128,
        "max_episode_steps": HORIZON,
    }.items():
        if type(value[key]) is not type(expected) or value[key] != expected:
            raise ValueError("Reach diagnostic contract differs: " + key)
    phase = value["phase"]
    if type(phase) is not str or phase not in PHASES:
        raise ValueError("Use the separate runtime_gate or training phase")
    if type(value["vector_steps"]) is not int or value["vector_steps"] != PHASES[phase]:
        raise ValueError("Reach diagnostic call bound differs")
    if value["reach"] != REACH or value["stopping"] != STOPPING:
        raise ValueError("Reach reward, criterion or stopping specification differs")
    evidence = value["selection_evidence"]
    if type(evidence) is not dict or set(evidence) != {"path", "bytes", "sha256"}:
        raise ValueError("Pin the TRAIN-only selection evidence")
    limits = value["timeouts"]
    maxima = {"ready_s": 600, "step_s": 180, "shutdown_s": 30, "wall_s": 3540, "rpc_s": 600}
    if type(limits) is not dict or set(limits) != set(maxima):
        raise ValueError("Use every finite reach host watchdog")
    if any(
        type(limits[key]) is not int or not 1 <= limits[key] <= maximum
        for key, maximum in maxima.items()
    ):
        raise ValueError("Reach host watchdog exceeds its bound")
    if (
        limits["step_s"] > limits["rpc_s"]
        or limits["ready_s"] + limits["shutdown_s"] >= limits["wall_s"]
    ):
        raise ValueError("Reach host watchdogs do not fit")
    return value
