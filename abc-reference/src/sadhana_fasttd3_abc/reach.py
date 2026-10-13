"""Original FastTD3 learning on a separately named dense-reaching diagnostic."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .environment import Endpoint, RemoteEnv
from .learner import train, write_json
from .reach_contract import HORIZON, PROFILE, REACH, STOPPING, validate_config
from .vector import SCHEMA, VectorRemoteEnv


def reach_args(config: dict, *, device: str = "cuda"):
    validate_config(config)
    if device not in {"cpu", "cuda"}:
        raise ValueError("Choose an explicit learner device")
    from fast_td3.hyperparams import BaseArgs

    return BaseArgs(
        env_name="ABC-FiveBottlesVector-v1",
        exp_name="FastTD3ABCReachDiagnostic_" + config["phase"],
        seed=1,
        cuda=device == "cuda",
        num_envs=64,
        num_eval_envs=1,
        total_timesteps=config["vector_steps"],
        use_wandb=False,
        eval_interval=0,
        render_interval=0,
        save_interval=0,
    )


class ReachMonitor:
    """Stop only on declared budget or qualified TRAIN progress plateaus."""

    def __init__(self, config: dict):
        self.config = validate_config(config)
        self.counts = [0] * 64
        self.pending: list[dict] = []
        self.windows: list[dict] = []
        self.best: dict[str, float] = {}
        self.stale = 0
        self.decision = None
        self.last_controls = 0

    def observe(self, controls: int, episodes: list[dict]) -> dict | None:
        if type(controls) is not int or controls != self.last_controls + 1:
            raise ValueError("Reach monitor requires sequential vector controls")
        self.last_controls = controls
        for episode in episodes:
            rank = episode.get("rank")
            if (
                type(rank) is not int
                or not 0 <= rank < 64
                or episode.get("episode_index") != self.counts[rank]
            ):
                raise ValueError("Reach episode rank/index inventory differs")
            if (
                episode.get("completed") is not True
                or episode.get("controls") != HORIZON
                or episode.get("terminated") is not False
                or episode.get("truncated") is not True
            ):
                raise ValueError("Reach stopping requires completed fixed128-control timeouts")
            if episode.get("reach_spec") != REACH or "success" in episode:
                raise ValueError("Reach diagnostic must keep native task-success labels separate")
            reach, judge = episode["reach_diagnostic"], episode["final_native_task_eval"]
            count, mask = judge.get("num_bottles_in_bin"), judge.get("bottle_in_bin_mask")
            if (
                type(count) is not int
                or not 0 <= count <= 5
                or judge.get("num_active_bottles") != 5
                or type(mask) is not list
                or len(mask) != 5
                or any(type(item) is not bool for item in mask)
                or sum(mask) != count
                or type(judge.get("success")) is not bool
                or judge["success"] != (count == 5)
                or episode.get("native_full_success") is not judge["success"]
            ):
                raise ValueError("Native five-bottle judge inventory changed")
            for key in ("initial", "final"):
                physics = episode["physics"][key]
                if (
                    not physics["time_finite"]
                    or not all(physics["finite"].values())
                    or any(row["number"] for row in physics["warnings"].values())
                ):
                    raise FloatingPointError("Reach episode includes invalid native physics")
            initial, minimum = reach["initial_distance_m"], reach["minimum_distance_m"]
            values = [
                initial,
                minimum,
                reach["final_distance_m"],
                reach["distance_gain_m"],
                reach["reward_sum"],
            ]
            if (
                any(
                    type(value) not in (int, float) or not math.isfinite(value) or value < 0
                    for value in values
                )
                or minimum > initial
                or minimum > reach["final_distance_m"]
                or reach["reward_sum"] > HORIZON
                or not math.isclose(
                    reach["distance_gain_m"], initial - minimum, rel_tol=0, abs_tol=1e-12
                )
            ):
                raise ValueError("Reach distances/reward bounds differ")
            eligible = initial > REACH["initial_eligibility_m"]
            achieved = reach["maximum_hold_controls"] >= REACH["hold_controls"]
            if (
                reach["controls"] != HORIZON
                or reach["eligible_initial_distance"] is not eligible
                or reach["reach_achieved"] is not achieved
                or reach["qualified_reach_achieved"] is not (eligible and achieved)
            ):
                raise ValueError("Reach criterion/qualification labels differ")
            self.counts[rank] += 1
            self.pending.append(episode)
        if controls % STOPPING["window_calls"] == 0:
            rows = [
                [episode for episode in self.pending if episode["rank"] == rank]
                for rank in range(64)
            ]
            eligible = [
                [
                    episode
                    for episode in row
                    if episode["reach_diagnostic"]["eligible_initial_distance"]
                ]
                for row in rows
            ]
            complete = all(rows)
            qualified_complete = all(eligible)
            window = {
                "controls": controls,
                "rank_episode_counts": [len(row) for row in rows],
                "rank_qualified_episode_counts": [len(row) for row in eligible],
                "complete": complete,
                "qualified_complete": qualified_complete,
                "eligible_for_stopping": self.config["phase"] == "training"
                and controls >= STOPPING["minimum_calls"]
                and complete
                and qualified_complete,
            }
            if complete:
                window["mean_initial_distance_m"] = (
                    sum(
                        sum(ep["reach_diagnostic"]["initial_distance_m"] for ep in row) / len(row)
                        for row in rows
                    )
                    / 64
                )
                window["native_full_success_rate"] = (
                    sum(sum(ep["native_full_success"] for ep in row) / len(row) for row in rows)
                    / 64
                )
                window["mean_dense_reward"] = (
                    sum(
                        sum(ep["reach_diagnostic"]["reward_sum"] / HORIZON for ep in row) / len(row)
                        for row in rows
                    )
                    / 64
                )
            if window["eligible_for_stopping"]:
                metrics = {
                    "qualified_reach_rate": sum(
                        sum(ep["reach_diagnostic"]["qualified_reach_achieved"] for ep in row)
                        / len(row)
                        for row in eligible
                    )
                    / 64,
                    "distance_gain_m": sum(
                        sum(ep["reach_diagnostic"]["distance_gain_m"] for ep in row) / len(row)
                        for row in eligible
                    )
                    / 64,
                }
                thresholds = {
                    "qualified_reach_rate": STOPPING["qualified_reach_rate_improvement"],
                    "distance_gain_m": STOPPING["distance_gain_improvement_m"],
                }
                improved = [
                    key
                    for key, value in metrics.items()
                    if key not in self.best or value >= self.best[key] + thresholds[key]
                ]
                for key in improved:
                    self.best[key] = metrics[key]
                self.stale = 0 if improved else self.stale + 1
                window.update(metrics=metrics, meaningful_improvement=improved)
                if self.stale >= STOPPING["patience_windows"]:
                    self.decision = {
                        "reason": "qualified_TRAIN_reach_progress_plateau",
                        "controls": controls,
                        "stale_windows": self.stale,
                    }
            else:
                self.stale = 0
            window["stale_windows"] = self.stale
            self.windows.append(window)
            self.pending = []
        if controls == self.config["vector_steps"]:
            self.decision = {
                "reason": "declared_diagnostic_maximum_completed",
                "controls": controls,
                "stale_windows": self.stale,
            }
        return self.decision

    def evidence(self) -> dict:
        return {
            "spec": STOPPING,
            "windows": self.windows,
            "decision": self.decision,
            "completed_episodes_per_rank": self.counts,
            "interpretation": (
                "changing-policy TRAIN progress only; "
                "no isolated learning or held-out performance claim"
            ),
            "matched_initial_final_deterministic_actor_comparison": "pending",
        }


class ReachVectorRemoteEnv(VectorRemoteEnv):
    def __init__(self, endpoint: Any, *, settings: Any, device: Any, config: dict):
        validate_config(config)
        RemoteEnv.__init__(self, endpoint, device=device, seed=200000)
        expected = {
            "num_envs": 64,
            "num_obs": 216,
            "num_actions": 14,
            "max_episode_steps": HORIZON,
            "profile": PROFILE,
            "reach_diagnostic": REACH,
            "diagnostic_stopping": STOPPING,
            "native_terminate_on_success": False,
            "carried_arm_reset": True,
            "physics_dt": 0.002,
            "control_decimation": 17,
            "camera_backend": None,
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
            "physics_failure_policy": "fail_on_every_warning_nonfinite_or_internal_time_reset",
        }
        if settings.num_envs != 64 or any(
            type(self.description.get(key)) is not type(value) or self.description.get(key) != value
            for key, value in expected.items()
        ):
            raise ValueError("Reach host dimensions/reset/physics/diagnostic contract differs")
        self.requested_seeds, self.last_requested, self.transitions = [], [], 0
        self.settings, self.monitor = settings, ReachMonitor(config)

    def step(self, actions):
        result = super().step(actions)
        if self.monitor.observe(self.controls, result[3]["episodes"]) is not None:
            # Retain the original current replay insertion and all scheduled
            # updates. The next author while-bound exits and saves naturally.
            self.settings.total_timesteps = self.controls
        return result


def train_reach(
    config: dict, socket_path: str, output: Path, *, device: str = "cuda", timeout: float = 180
) -> dict:
    args = reach_args(config, device=device)
    initial = asdict(args)
    created = []

    def factory(settings, policy_device):
        env = ReachVectorRemoteEnv(
            Endpoint(socket_path, timeout=timeout, schema=SCHEMA),
            settings=settings,
            device=policy_device,
            config=config,
        )
        created.append(env)
        return env, env, env

    receipt = train(args, factory, output)
    if (
        len(created) != 1
        or created[0].monitor.decision is None
        or args.total_timesteps != created[0].controls
    ):
        raise RuntimeError("Original learner exited without a declared diagnostic stop")
    env = created[0]
    result = {
        "schema_version": 1,
        "status": "completed",
        "phase": config["phase"],
        "claim_scope": (
            "Dense-reaching diagnostic; no native five-bottle benchmark or isolated learning claim"
        ),
        "reach_spec": REACH,
        "controls": env.controls,
        "transitions": env.transitions,
        "declared_maximum_args": initial,
        "effective_final_args": asdict(args),
        "checkpoint": receipt["checkpoint"],
        "actual_updates": receipt["actual_updates"],
        "stopping": env.monitor.evidence(),
        "episodes": env.episodes,
        "parallelism": {
            "worlds": 64,
            "author_default_worlds": 128,
            "batch_size": args.batch_size,
            "replay_per_world": args.buffer_size,
            "aggregate_replay_capacity": 64 * args.buffer_size,
            "author_aggregate_replay_capacity": 128 * args.buffer_size,
            "critic_updates_per_transition_after_learning_start": args.num_updates / 64,
            "author_critic_updates_per_transition_after_learning_start": args.num_updates / 128,
            "seed_episode_stride": 128,
        },
        "resume_semantics": receipt["resume_semantics"],
    }
    write_json(output / "reach-diagnostic.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--rpc-timeout", type=int, default=180)
    options = parser.parse_args()
    train_reach(
        validate_config(json.loads(options.config.read_text())),
        options.socket,
        options.output,
        device=options.device,
        timeout=options.rpc_timeout,
    )


if __name__ == "__main__":
    main()
