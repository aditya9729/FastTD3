"""Matched initial/final deterministic TRAIN reach diagnostic; no learner calls."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .environment import Endpoint, RemoteEnv
from .learner import artifact, evaluate, verify_source, write_json
from .reach_contract import HORIZON, REACH

PROFILE = "reach_matched_TRAIN3_v1"
WARNINGS = {"INERTIA", "CONTACTFULL", "CNSTRFULL", "BADQPOS", "BADQVEL", "BADQACC", "BADCTRL"}


def checked_pin(pin: Any, maximum: int = 256 * 1024**2) -> Path:
    if type(pin) is not dict or not {"path", "bytes", "sha256"} <= set(pin):
        raise ValueError("Matched input requires path/bytes/SHA256")
    path = Path(pin["path"])
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("Matched input must be an absolute regular non-symlink file")
    if (
        type(pin["bytes"]) is not int
        or not 0 < pin["bytes"] <= maximum
        or artifact(path) != {key: pin[key] for key in ("path", "bytes", "sha256")}
    ):
        raise ValueError("Matched input differs from its bounded pin")
    return path


def checked_json(pin: Any, maximum: int = 1024**2) -> dict:
    value = json.loads(checked_pin(pin, maximum).read_text())
    if type(value) is not dict:
        raise ValueError("Matched JSON input must be an object")
    return value


def load_binding(path: Path) -> dict:
    """Consume the completed producer binding made by the ABC preparation API."""
    if path.is_symlink() or not 0 < path.stat().st_size <= 1024**2:
        raise ValueError("Matched binding must be a regular JSON file at most1MiB")
    binding = json.loads(path.read_text())
    fields = {
        "schema",
        "training_parent",
        "training_worker",
        "learner_receipt",
        "diagnostic_receipt",
        "initial_metadata",
        "initial_checkpoint",
        "final_checkpoint",
        "host_config",
    }
    if (
        type(binding) is not dict
        or set(binding) != fields
        or binding["schema"] != "sadhana.fasttd3.reach-matched-binding/v1"
    ):
        raise ValueError("Use the complete matched TRAIN checkpoint binding")
    parent, worker = (
        checked_json(binding["training_parent"]),
        checked_json(binding["training_worker"]),
    )
    learner = checked_json(binding["learner_receipt"])
    diagnostic = checked_json(binding["diagnostic_receipt"], 32 * 1024**2)
    initial = checked_json(binding["initial_metadata"])
    for key in ("initial_checkpoint", "final_checkpoint", "host_config"):
        checked_pin(binding[key])
    evidence = learner.get("initial_export", {})
    if (
        parent.get("status") != "completed"
        or parent.get("worker_exit_code") != 0
        or parent.get("worker_receipt_sha256") != binding["training_worker"]["sha256"]
        or worker.get("phase") != "fasttd3_reach_diagnostic_training"
        or worker.get("status") != "completed"
        or worker.get("child_exit_codes") != {"host": 0, "learner": 0}
        or worker.get("learner_receipt") != binding["learner_receipt"]
        or worker.get("diagnostic_receipt") != binding["diagnostic_receipt"]
        or worker.get("checkpoint") != binding["final_checkpoint"]
        or learner.get("status") != "completed"
        or diagnostic.get("status") != "completed"
        or diagnostic.get("phase") != "training"
        or learner.get("checkpoint") != binding["final_checkpoint"]
        or diagnostic.get("checkpoint") != binding["final_checkpoint"]
        or evidence != diagnostic.get("initial_export")
        or evidence.get("metadata") != binding["initial_metadata"]
        or initial != {key: value for key, value in evidence.items() if key != "metadata"}
        or initial.get("schema") != "sadhana.fasttd3.initial-scratch-export/v1"
        or initial.get("status") != "completed"
        or initial.get("checkpoint") != binding["initial_checkpoint"]
        or initial.get("source") != learner.get("source")
        or initial.get("source") != verify_source()
        or initial.get("before") != initial.get("after")
        or any(
            type(initial.get(key)) is not int or initial[key] != 0
            for key in ("global_step", "controls", "actor_updates", "critic_updates")
        )
        or any(
            type(learner.get("actual_updates", {}).get(role)) is not int
            or learner["actual_updates"][role] <= 0
            for role in ("actor", "critic")
        )
        or learner.get("actual_updates") != diagnostic.get("actual_updates")
        or diagnostic.get("controls") != learner.get("controls")
    ):
        raise ValueError(
            "Matched comparison requires true initial/final exports "
            "from one completed TRAIN producer"
        )
    return binding


class MatchedRemoteEnv(RemoteEnv):
    def __init__(self, endpoint: Endpoint, *, device: Any, host_config: dict):
        super().__init__(endpoint, device=device)
        expected = {
            "profile": PROFILE,
            "num_envs": 1,
            "num_obs": 216,
            "num_actions": 14,
            "max_episode_steps": HORIZON,
            "physics_dt": 0.002,
            "control_decimation": 17,
            "camera_backend": None,
            "native_terminate_on_success": False,
            "autoreset": False,
            "reach_spec": REACH,
        }
        if (
            host_config["profile"] != PROFILE
            or host_config["reach"] != REACH
            or any(
                type(self.description.get(key)) is not type(value) or self.description[key] != value
                for key, value in expected.items()
            )
        ):
            raise ValueError("Matched host physics/action/reward/horizon contract changed")
        self.host_config = host_config
        self.case_manifests = [checked_json(case["manifest"]) for case in host_config["cases"]]
        cases = [
            {"seed": row["seed"], "episode_id": row["episode_id"]} for row in self.case_manifests
        ]
        if self.description.get("cases") != cases or any(
            row["split"] != "train" for row in self.case_manifests
        ):
            raise ValueError("Matched host must expose the three pinned TRAIN captures")
        self.seeds = [row["seed"] for row in self.case_manifests]

    def reset(self, *, seeds=None):
        if seeds not in [[seed] for seed in self.seeds]:
            raise ValueError("Frozen diagnostic must request one exact declared TRAIN case")
        return super().reset(seeds=seeds)


def validate_episode(episode: dict) -> None:
    reach, proof, judge = (
        episode["reach_diagnostic"],
        episode["reset_proof"],
        episode["final_native_task_eval"],
    )
    initial = reach["initial_distance_m"]
    if (
        episode.get("completed") is not True
        or episode.get("controls") != HORIZON
        or episode.get("terminated") is not False
        or episode.get("truncated") is not True
        or episode.get("reach_spec") != REACH
        or proof.get("schema") != "sadhana.abc.reach-matched-TRAIN-reset/v1"
        or proof.get("split") != "train"
        or proof.get("seed") != episode.get("seed")
        or set(proof.get("checks", {}))
        != {"integration", "qpos", "qvel", "state14", "time", "compiled_model", "native_judge"}
        or any(value is not True for value in proof["checks"].values())
        or proof.get("geometry_preserves_integration") is not True
        or proof.get("geometry_matches_independent_FK") is not True
        or proof.get("initial_distance_m") != initial
        or type(initial) not in (int, float)
        or not math.isfinite(initial)
        or initial < 0
        or reach.get("eligible_initial_distance") is not (initial > REACH["initial_eligibility_m"])
        or proof.get("eligible_initial_distance") != reach["eligible_initial_distance"]
        or reach.get("controls") != HORIZON
        or reach.get("reach_achieved")
        is not (reach["maximum_hold_controls"] >= REACH["hold_controls"])
        or reach.get("qualified_reach_achieved")
        is not (reach["eligible_initial_distance"] and reach["reach_achieved"])
        or type(episode.get("success")) is not bool
        or episode["success"] is not judge.get("success")
        or episode.get("native_full_success") is not episode["success"]
        or judge.get("num_active_bottles") != 5
        or judge.get("success") is not (judge["num_bottles_in_bin"] == 5)
        or len(judge.get("bottle_in_bin_mask", [])) != 5
        or any(type(item) is not bool for item in judge["bottle_in_bin_mask"])
        or sum(judge["bottle_in_bin_mask"]) != judge["num_bottles_in_bin"]
    ):
        raise ValueError("Matched frozen episode reset/reach/native evidence differs")
    for physics in episode["physics"].values():
        if (
            physics.get("time_finite") is not True
            or not all(physics["finite"].values())
            or set(physics["warnings"]) != WARNINGS
            or any(row["number"] != 0 for row in physics["warnings"].values())
        ):
            raise FloatingPointError("Matched episode has invalid native physics or any warning")


def compare(initial: dict, final: dict) -> dict:
    """Validate matched receipts before computing separate diagnostic outcomes."""
    for receipt in (initial, final):
        if (
            receipt.get("status") != "completed"
            or receipt.get("actor_updates") != 0
            or receipt.get("critic_updates") != 0
            or receipt.get("action_selection") != "original_Actor.forward_no_explore_no_added_noise"
            or receipt.get("normalizer_update") is not False
            or receipt.get("policy_state_before") != receipt.get("policy_state_after")
            or receipt.get("policy_components_before") != receipt.get("policy_components_after")
            or set(receipt.get("policy_components_before", {})) != {"actor", "normalizer"}
            or receipt.get("rng_state_before") != receipt.get("rng_state_after")
            or len(receipt.get("episodes", [])) != 3
        ):
            raise ValueError(
                "Matched evaluation must preserve policy/normalizer/RNG "
                "with zero updates and no noise"
            )
        for row in receipt["episodes"]:
            validate_episode(row)
    if (
        initial["evaluation_seeds"] != final["evaluation_seeds"]
        or initial["environment"] != final["environment"]
        or initial["evaluation_precision"] != final["evaluation_precision"]
    ):
        raise ValueError("Initial/final evaluation seed, environment or precision differs")
    pairs = []
    for before, after in zip(initial["episodes"], final["episodes"]):
        if before["reset_proof"] != after["reset_proof"]:
            raise ValueError("Initial/final compiled model/full INTEGRATION/geometry reset differs")
        pairs.append(
            {
                "seed": before["seed"],
                "episode_id": before["reset_proof"]["episode_id"],
                "eligible_initial_distance": before["reach_diagnostic"][
                    "eligible_initial_distance"
                ],
                "initial_actor": {
                    "reach": before["reach_diagnostic"],
                    "native_full_success": before["success"],
                    "native_peak_bottles": before["peak_native_bottles"],
                },
                "final_actor": {
                    "reach": after["reach_diagnostic"],
                    "native_full_success": after["success"],
                    "native_peak_bottles": after["peak_native_bottles"],
                },
                "distance_gain_delta_m": after["reach_diagnostic"]["distance_gain_m"]
                - before["reach_diagnostic"]["distance_gain_m"],
            }
        )
    eligible = [row for row in pairs if row["eligible_initial_distance"]]
    return {
        "schema": "sadhana.fasttd3.reach-matched-comparison/v1",
        "status": "completed",
        "scope": (
            "TRAIN-only dense-reaching diagnostic; three captured cases; "
            "no full-task benchmark, held-out/generalization or paper reproduction claim"
        ),
        "pairs": pairs,
        "eligible_cases": len(eligible),
        "qualified_reach_counts": {
            role: sum(row[role]["reach"]["qualified_reach_achieved"] for row in eligible)
            for role in ("initial_actor", "final_actor")
        },
        "native_full_success_counts": {
            role: sum(row[role]["native_full_success"] for row in pairs)
            for role in ("initial_actor", "final_actor")
        },
        "policy_components": {
            "initial": initial["policy_components_before"],
            "final": final["policy_components_before"],
        },
        "updates_during_evaluation": {"actor": 0, "critic": 0},
    }


def run(
    binding_path: Path, socket_path: str, output: Path, *, device: str, timeout: int = 180
) -> dict:
    binding = load_binding(binding_path)
    import torch

    initial_path, final_path = (
        checked_pin(binding["initial_checkpoint"]),
        checked_pin(binding["final_checkpoint"]),
    )
    initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    final = torch.load(final_path, map_location="cpu", weights_only=False)
    diagnostic = checked_json(binding["diagnostic_receipt"], 32 * 1024**2)
    if (
        type(initial.get("global_step")) is not int
        or initial["global_step"] != 0
        or type(final.get("global_step")) is not int
        or final["global_step"] <= 10
        or final["global_step"] != diagnostic["controls"]
    ):
        raise ValueError("Pair must contain a true step0 export and a post-update final export")
    keys = set(initial["args"]) | set(final["args"])
    if any(
        initial["args"].get(key) != final["args"].get(key) for key in keys - {"total_timesteps"}
    ):
        raise ValueError(
            "Initial/final scratch learner architecture/precision/configuration differs"
        )
    output.mkdir(parents=True, exist_ok=False)
    env = None
    receipt = {"status": "running", "binding": artifact(binding_path)}
    try:
        env = MatchedRemoteEnv(
            Endpoint(socket_path, timeout=timeout),
            device=device,
            host_config=checked_json(binding["host_config"]),
        )
        before = evaluate(initial_path, env, env.seeds, output / "initial")
        saved_initial = checked_json(binding["initial_metadata"])["before"]["states"]
        if before["policy_components_before"] != {
            "actor": saved_initial["actor"],
            "normalizer": saved_initial["obs_normalizer"],
        }:
            raise ValueError(
                "Evaluated initial actor/normalizer differs from the true pre-loop export"
            )
        after = evaluate(final_path, env, env.seeds, output / "final")
        result = compare(before, after)
        result["binding"] = artifact(binding_path)
        result["evaluation_receipts"] = {
            role: artifact(output / role / "receipt.json") for role in ("initial", "final")
        }
        write_json(output / "comparison.json", result)
        receipt.update(status="completed", comparison=artifact(output / "comparison.json"))
        return result
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if env is not None:
            env.close()
        write_json(output / "receipt.json", receipt)


def verify_output(binding_path: Path, output: Path, verification: Path) -> dict:
    """Recheck actual public frozen receipts without constructing a policy."""
    binding = load_binding(binding_path)
    before, after = (
        json.loads((output / role / "receipt.json").read_text()) for role in ("initial", "final")
    )
    if (
        before["checkpoint"] != binding["initial_checkpoint"]
        or after["checkpoint"] != binding["final_checkpoint"]
    ):
        raise ValueError("Matched receipt checkpoint differs from producer export")
    states = checked_json(binding["initial_metadata"])["before"]["states"]
    if before["policy_components_before"] != {
        "actor": states["actor"],
        "normalizer": states["obs_normalizer"],
    }:
        raise ValueError("Frozen initial actor/normalizer differs from the true scratch export")
    result = compare(before, after)
    result["binding"] = artifact(binding_path)
    result["evaluation_receipts"] = {
        role: artifact(output / role / "receipt.json") for role in ("initial", "final")
    }
    if result != json.loads((output / "comparison.json").read_text()):
        raise ValueError("Matched summary differs from recomputed frozen receipts")
    evidence = {
        "status": "completed",
        "comparison": artifact(output / "comparison.json"),
        "recomputed_from_frozen_receipts": True,
        "zero_optimizer_calls_reported": True,
    }
    if verification.exists():
        raise FileExistsError("Refuse an existing verification result")
    write_json(verification, evidence)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--socket")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--verify-output", type=Path)
    args = parser.parse_args()
    if args.verify_output is not None:
        if args.socket is not None or args.device is not None:
            parser.error("Output verification is read-only; omit --socket/--device")
        verify_output(args.binding, args.verify_output, args.output)
    else:
        if args.socket is None or args.device is None:
            parser.error("Matched actor execution requires --socket and --device")
        run(args.binding, args.socket, args.output, device=args.device)


if __name__ == "__main__":
    main()
