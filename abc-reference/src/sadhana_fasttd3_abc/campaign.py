"""Declared reward-v2 task-futility stopping through the original learner."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .environment import Endpoint, RemoteEnv
from .learner import train, write_json
from .vector import SCHEMA, VectorRemoteEnv

PROFILE = "reward_v2_training128"
PHASES = {"runtime_gate": 64, "training": 42480}
THRESHOLDS = {
    "success_rate": 1 / 128,
    "final_bottles": 0.1,
    "peak_bottles": 0.1,
    "peak_phi": 0.005,
    "peak_reach": 0.02,
    "peak_clearance": 0.02,
    "peak_lift_transport": 0.02,
}


def campaign_spec(phase: str) -> dict:
    if phase not in PHASES:
        raise ValueError("Use the separate runtime_gate or training phase")
    return {
        "schema": "sadhana.fasttd3.abc.task-futility/v1",
        "phase": phase,
        "maximum_calls": PHASES[phase],
        "minimum_calls": 21240 if phase == "training" else None,
        "minimum_completed_episodes_per_rank": 6 if phase == "training" else None,
        "window_calls": 3540,
        "patience_windows": 3,
        "improvement_thresholds": THRESHOLDS,
        "rank_weighting": "equal_mean_of_per_rank_completed_episode_means",
        "stop_scope": "training_task_futility_not_validation_convergence_or_overfit",
        "validation_seeds_used_for_stopping": [],
        "loss_based_stopping": False,
        "checkpoint": "unchanged_author_final_export_after_natural_loop_exit",
    }


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    phase = config.get("phase")
    if phase not in PHASES or config.get("profile") != PROFILE:
        raise ValueError("Use a separate reward-v2 campaign phase")
    if (
        config.get("campaign") != campaign_spec(phase)
        or type(config.get("vector_steps")) is not int
        or config.get("vector_steps") != PHASES[phase]
    ):
        raise ValueError("Maximum, minimum exploration or stop thresholds differ")
    reward = config.get("reward", {})
    if (
        reward.get("profile") not in {"abc_full_success_sparse_v1", "abc_full_success_pbrs_v1"}
        or reward.get("gamma") != 0.99
    ):
        raise ValueError("Use the exact versioned full-success task reward and gamma0.99")
    return config


def campaign_args(config: dict, *, device: str = "cuda"):
    if device not in {"cpu", "cuda"}:
        raise ValueError("Choose an explicit learner device")
    from fast_td3.hyperparams import BaseArgs

    return BaseArgs(
        env_name="ABC-FiveBottlesVector-v1",
        exp_name="FastTD3ABCRewardV2_" + config["phase"],
        seed=1,
        cuda=device == "cuda",
        num_envs=128,
        num_eval_envs=1,
        total_timesteps=config["vector_steps"],
        use_wandb=False,
        eval_interval=0,
        render_interval=0,
        save_interval=0,
    )


class TaskFutilityMonitor:
    """Use completed native task episodes, with equal world-rank weights."""

    def __init__(self, config: dict):
        self.config = config
        self.counts = [0] * 128
        self.pending: list[dict] = []
        self.windows: list[dict] = []
        self.best: dict[str, float] = {}
        self.stale = 0
        self.eligible_started = False
        self.decision: dict | None = None

    @staticmethod
    def episode_metrics(episode: dict) -> dict:
        judge, reward = episode["final_native_task_eval"], episode["reward_v2"]
        peak = reward["peak_progress"]
        return {
            "success_rate": float(episode["success"]),
            "final_bottles": float(judge["num_bottles_in_bin"]),
            "peak_bottles": float(reward["peak_bottles"]),
            **{
                "peak_" + name: float(peak[name])
                for name in ("phi", "reach", "clearance", "lift_transport")
            },
        }

    def observe(self, controls: int, episodes: list[dict]) -> dict | None:
        for episode in episodes:
            rank = episode.get("rank")
            if (
                type(rank) is not int
                or not 0 <= rank < 128
                or type(episode.get("episode_index")) is not int
                or episode.get("episode_index") != self.counts[rank]
            ):
                raise ValueError("Completed episode rank/index inventory differs")
            if episode.get("completed") is not True or type(episode.get("success")) is not bool:
                raise ValueError("Task stopping requires completed native episodes")
            if episode.get("reward_v2", {}).get("spec") != self.config["reward"]:
                raise ValueError("Task episode reward specification differs")
            judge = episode.get("final_native_task_eval", {})
            count, mask = judge.get("num_bottles_in_bin"), judge.get("bottle_in_bin_mask")
            if (
                type(judge.get("num_active_bottles")) is not int
                or judge["num_active_bottles"] != 5
                or type(count) is not int
                or not 0 <= count <= 5
                or type(mask) is not list
                or len(mask) != 5
                or any(type(value) is not bool for value in mask)
                or sum(mask) != count
                or type(judge.get("success")) is not bool
                or judge["success"] != (count == 5)
                or episode.get("terminated") is not episode["success"]
                or episode.get("truncated") is not (not episode["success"])
                or episode["success"] != judge["success"]
                or type(judge.get("reward")) not in (int, float)
                or not math.isclose(judge["reward"], count / 5, rel_tol=0, abs_tol=1e-7)
                or type(episode["reward_v2"].get("peak_bottles")) is not int
            ):
                raise ValueError("Stopping success must match the current native five-bottle judge")
            length = episode.get("controls")
            if (
                type(length) is not int
                or not 1 <= length <= 3540
                or (not episode["success"] and length != 3540)
            ):
                raise ValueError("A partial failed native episode cannot drive stopping")
            values = self.episode_metrics(episode)
            if any(not math.isfinite(value) for value in values.values()):
                raise FloatingPointError("Task progress evidence is not finite")
            if (
                not 0 <= values["final_bottles"] <= values["peak_bottles"] <= 5
                or not 0 <= values["peak_phi"] <= 0.25
            ):
                raise ValueError("Native counts or potential bound differ")
            if any(
                not 0 <= values[name] <= 1
                for name in ("peak_reach", "peak_clearance", "peak_lift_transport")
            ):
                raise ValueError("Geometric reach/lift/transport progress exceeds bounds")
            self.counts[rank] += 1
            self.pending.append(episode)
        if self.config["phase"] == "training" and controls % 3540 == 0:
            rows = [
                [episode for episode in self.pending if episode["rank"] == rank]
                for rank in range(128)
            ]
            complete = all(rows)
            window = {
                "controls": controls,
                "rank_episode_counts": [len(row) for row in rows],
                "cumulative_rank_episode_counts": self.counts.copy(),
                "complete": complete,
                "eligible": controls >= 21240 and min(self.counts) >= 6 and complete,
            }
            if complete:
                metrics = {
                    name: math.fsum(
                        math.fsum(self.episode_metrics(episode)[name] for episode in row) / len(row)
                        for row in rows
                    )
                    / 128
                    for name in THRESHOLDS
                }
                improvement = [
                    name
                    for name, value in metrics.items()
                    if name not in self.best or value >= self.best[name] + THRESHOLDS[name]
                ]
                for name in improvement:
                    self.best[name] = metrics[name]
                window.update(metrics=metrics, meaningful_improvement=improvement)
                if window["eligible"]:
                    self.stale = 0 if improvement or not self.eligible_started else self.stale + 1
                    self.eligible_started = True
            else:
                self.stale = 0
            window["stale_windows"] = self.stale
            self.windows.append(window)
            self.pending = []
            if window["eligible"] and self.stale >= 3:
                self.decision = {
                    "reason": "no_observed_training_task_improvement",
                    "controls": controls,
                    "completed_episodes_per_rank": self.counts.copy(),
                    "stale_windows": self.stale,
                }
        if controls == self.config["vector_steps"]:
            self.decision = {
                "reason": "runtime_gate_completed"
                if self.config["phase"] == "runtime_gate"
                else "declared_maximum_completed",
                "controls": controls,
                "completed_episodes_per_rank": self.counts.copy(),
                "stale_windows": self.stale,
            }
        return self.decision

    def evidence(self) -> dict:
        return {
            "spec": self.config["campaign"],
            "windows": self.windows,
            "completed_episodes_per_rank": self.counts,
            "best_training_task_metrics": self.best,
            "decision": self.decision,
        }


class CampaignVectorRemoteEnv(VectorRemoteEnv):
    """Retain original transitions; request only a natural finite loop exit."""

    def __init__(self, endpoint: Any, *, settings: Any, device: Any, config: dict):
        RemoteEnv.__init__(self, endpoint, device=device, seed=200000)
        if (self.num_envs, self.num_obs, self.num_actions, self.max_episode_steps) != (
            128,
            216,
            14,
            3540,
        ):
            raise ValueError("Campaign state216/actions14/worlds128/horizon3540 differ")
        for key, expected in (
            ("profile", PROFILE),
            ("reward", config["reward"]),
            ("campaign", config["campaign"]),
            ("carried_arm_reset", True),
            (
                "reset_options",
                {
                    "randomization": {
                        "bottle_count": 5,
                        "randomize_variants": True,
                        "randomize_scales": True,
                    }
                },
            ),
            (
                "reset_seed_bank",
                {
                    "base": 200000,
                    "rank_stride": 1,
                    "episode_stride": 128,
                    "max_episode_index": 780,
                    "fallback_offset": 100000,
                },
            ),
        ):
            if self.description.get(key) != expected:
                raise ValueError(f"Campaign task/source boundary differs: {key}")
        self.requested_seeds, self.last_requested, self.transitions = [], [], 0
        self.settings, self.monitor = settings, TaskFutilityMonitor(config)

    def step(self, actions):
        result = super().step(actions)
        decision = self.monitor.observe(self.controls, result[3]["episodes"])
        if decision is not None:
            # The original current iteration still inserts replay and executes
            # all scheduled updates before its next while-bound test and save.
            self.settings.total_timesteps = self.controls
        return result


def train_campaign(
    config: dict, socket_path: str, output: Path, *, device: str = "cuda", timeout: float = 180
) -> dict:
    args = campaign_args(config, device=device)
    initial = asdict(args)
    created = []

    def factory(settings, policy_device):
        env = CampaignVectorRemoteEnv(
            Endpoint(socket_path, timeout=timeout, schema=SCHEMA),
            settings=settings,
            device=policy_device,
            config=config,
        )
        created.append(env)
        return env, env, env

    receipt = train(args, factory, output)
    if len(created) != 1:
        raise RuntimeError("Campaign must use exactly one original vector environment")
    env = created[0]
    decision = env.monitor.decision
    if (
        decision is None
        or decision["controls"] != env.controls
        or args.total_timesteps != env.controls
    ):
        raise RuntimeError("Original loop exited without a declared task/budget stop")
    campaign = {
        "schema_version": 1,
        "status": "completed",
        "phase": config["phase"],
        "reward": config["reward"],
        "declared_maximum_args": initial,
        "effective_final_args": asdict(args),
        "controls": env.controls,
        "transitions": env.transitions,
        "checkpoint": receipt["checkpoint"],
        "stopping": env.monitor.evidence(),
        "episodes": env.episodes,
        "actual_updates": receipt["actual_updates"],
        "resume_semantics": receipt["resume_semantics"],
    }
    write_json(output / "campaign.json", campaign)
    return campaign


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--rpc-timeout", type=int, default=180)
    options = parser.parse_args()
    train_campaign(
        load_config(options.config),
        options.socket,
        options.output,
        device=options.device,
        timeout=options.rpc_timeout,
    )


if __name__ == "__main__":
    main()
