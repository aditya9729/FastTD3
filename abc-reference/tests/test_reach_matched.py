"""Installed CPU boundary checks; all tensor/checkpoint writers below are explicit doubles."""

import ast
import hashlib
import importlib.metadata
import inspect
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sadhana_fasttd3_abc import initial_export as export
from sadhana_fasttd3_abc import reach_matched as matched
from sadhana_fasttd3_abc.learner import evaluate, verify_source
from sadhana_fasttd3_abc.reach_contract import REACH

BUILD = Path("/home/user/aditya/RL/builds/abc-fasttd3-reach-matched-20261013")
BASE = Path("/home/user/aditya/RL/builds/abc-fasttd3-reach-diagnostic-20261013/FastTD3")


class ArrayTensorDouble:
    def __init__(self, array):
        self.array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def numpy(self):
        return self.array


class ModelDouble:
    def __init__(self):
        self.array = np.zeros(3, dtype=np.float32)
        self.training = True

    def state_dict(self):
        return {"fixture.weight": ArrayTensorDouble(self.array)}

    def modules(self):
        return [self]


@pytest.fixture
def boundary(monkeypatch):
    # Torch is deliberately absent in this light prefix. Its RNG boundary is
    # an explicit mutable fixture; Python/NumPy states are read for real.
    torch_fixture = {"state": "unchanged"}

    def rng_double():
        return {
            "python": repr(random.getstate()),
            "numpy": np.random.get_state()[1].tobytes().hex(),
            "torch_fixture": dict(torch_fixture),
        }

    monkeypatch.setattr(export, "rng_fingerprint", rng_double)
    monkeypatch.setattr(export, "verify_source", lambda: {"explicit_CPU_fixture": True})
    models = {
        name: ModelDouble()
        for name in ("actor", "qnet", "qnet_target", "obs_normalizer", "critic_obs_normalizer")
    }
    args = SimpleNamespace(checkpoint_path=None, total_timesteps=2048)

    def writer_double(path, global_step, models, args):
        assert global_step == 0
        path.write_bytes(b"explicit fixture export; not a Torch checkpoint")

    monkeypatch.setattr(export, "save_initial", writer_double)
    return models, args, torch_fixture, writer_double


def test_true_initial_export_once_with_unchanged_boundary(tmp_path, boundary):
    models, args, _, _ = boundary
    capture = export.InitialExport(tmp_path / "initial.pt")
    capture(global_step=0, **models, args=args)
    assert capture.evidence["status"] == "completed"
    assert capture.evidence["before"] == capture.evidence["after"]
    assert capture.evidence["global_step"] == capture.evidence["controls"] == 0
    assert capture.evidence["resume_semantics"].startswith("Policy export only")
    assert (
        matched.checked_json(capture.evidence["metadata"])["checkpoint"]
        == capture.evidence["checkpoint"]
    )
    with pytest.raises(ValueError, match="exactly one"):
        capture(global_step=0, **models, args=args)


@pytest.mark.parametrize(
    "mutation",
    ["actor", "obs_normalizer", "mode", "arguments", "python_rng", "numpy_rng", "torch_rng"],
)
def test_export_fails_and_records_partial_state_on_any_mutation(
    tmp_path, monkeypatch, boundary, mutation
):
    models, args, torch_fixture, writer = boundary

    def corrupting_double(path, global_step, models, args):
        writer(path, global_step, models, args)
        if mutation in {"actor", "obs_normalizer"}:
            models[mutation].array[0] = 1
        elif mutation == "mode":
            models["actor"].training = False
        elif mutation == "arguments":
            args.total_timesteps = 1
        elif mutation == "python_rng":
            random.random()
        elif mutation == "numpy_rng":
            np.random.random()
        else:
            torch_fixture["state"] = "changed"

    monkeypatch.setattr(export, "save_initial", corrupting_double)
    capture = export.InitialExport(tmp_path / "initial.pt")
    with pytest.raises(RuntimeError, match="changed model"):
        capture(global_step=0, **models, args=args)
    assert json.loads((tmp_path / "initial.json").read_text())["status"] == "failed"
    assert capture.evidence["checkpoint"] is None


@pytest.mark.parametrize("invalid", ["step1", "resume", "existing"])
def test_after_first_control_resume_and_existing_export_rejected(tmp_path, boundary, invalid):
    models, args, _, _ = boundary
    path = tmp_path / "initial.pt"
    if invalid == "resume":
        args.checkpoint_path = "prior.pt"
    if invalid == "existing":
        path.write_bytes(b"old")
    capture = export.InitialExport(path)
    with pytest.raises(ValueError):
        capture(global_step=1 if invalid == "step1" else 0, **models, args=args)
    assert capture.calls == 0


def test_original_source_and_default_absence_are_exact_except_declared_boundary():
    provenance = verify_source()
    distribution = importlib.metadata.distribution("fast-td3")
    before = ast.parse((BASE / "fast_td3/train.py").read_text())
    after = ast.parse(Path(distribution.locate_file("fast_td3/train.py")).read_text())
    main = next(
        node for node in after.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    assert main.args.kwonlyargs[-1].arg == "initial_observer"
    assert (
        isinstance(main.args.kw_defaults[-1], ast.Constant)
        and main.args.kw_defaults[-1].value is None
    )
    hooks = [
        (i, node)
        for i, node in enumerate(main.body)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "initial_observer is not None"
    ]
    assert len(hooks) == 1
    index, hook = hooks[0]
    assert "global_step = 0" in ast.unparse(main.body[index - 1])
    assert (
        isinstance(main.body[index + 1], ast.Assign)
        and ast.unparse(main.body[index + 1]) == "dones = None"
    )
    assert "initial_observer(" in ast.unparse(hook) and "save_params" not in ast.unparse(hook)
    main.body.remove(hook)
    main.args.kwonlyargs.pop()
    main.args.kw_defaults.pop()
    assert ast.dump(before, include_attributes=False) == ast.dump(after, include_attributes=False)
    for row in provenance["files"]:
        if row["path"] != "fast_td3/train.py":
            assert hashlib.sha256((BASE / row["path"]).read_bytes()).hexdigest() == row["sha256"]
    original = ast.parse(Path(distribution.locate_file("fast_td3/train.py")).read_text())
    for name, expected in provenance["protected_update_functions"].items():
        node = next(
            node
            for node in ast.walk(original)
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        assert (
            hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
            == expected
        )
    assert "torch" not in sys.modules and "mujoco" not in sys.modules


def episode(seed=9030, achieved=False, initial=0.4, native=False):
    return {
        "seed": seed,
        "completed": True,
        "controls": 128,
        "terminated": False,
        "truncated": True,
        "success": native,
        "native_full_success": native,
        "peak_native_bottles": 5 if native else 0,
        "reach_spec": REACH,
        "reach_diagnostic": {
            "initial_distance_m": initial,
            "minimum_distance_m": 0.1,
            "final_distance_m": 0.1,
            "distance_gain_m": max(0, initial - 0.1),
            "eligible_initial_distance": initial > 0.15,
            "reach_achieved": achieved,
            "qualified_reach_achieved": initial > 0.15 and achieved,
            "maximum_hold_controls": 5 if achieved else 0,
            "controls": 128,
        },
        "reset_proof": {
            "schema": "sadhana.abc.reach-matched-TRAIN-reset/v1",
            "seed": seed,
            "episode_id": f"fixture-train-{seed}",
            "split": "train",
            "compiled_model": "fixture-model-pin",
            "initial_integration": "fixture-full-state-pin",
            "restored_integration_sha256": "fixture-state-hash",
            "observation_float32_sha256": "fixture-obs-hash",
            "initial_distance_m": initial,
            "eligible_initial_distance": initial > 0.15,
            "checks": {
                key: True
                for key in (
                    "integration",
                    "qpos",
                    "qvel",
                    "state14",
                    "time",
                    "compiled_model",
                    "native_judge",
                )
            },
            "geometry_preserves_integration": True,
            "geometry_matches_independent_FK": True,
        },
        "final_native_task_eval": {
            "success": native,
            "num_active_bottles": 5,
            "num_bottles_in_bin": 5 if native else 0,
            "bottle_in_bin_mask": [native] * 5,
        },
        "physics": {
            key: {
                "time_finite": True,
                "finite": {"qpos": True},
                "warnings": {warning: {"number": 0} for warning in matched.WARNINGS},
            }
            for key in ("initial", "final")
        },
    }


def receipt(achieved=False):
    return {
        "status": "completed",
        "actor_updates": 0,
        "critic_updates": 0,
        "action_selection": "original_Actor.forward_no_explore_no_added_noise",
        "normalizer_update": False,
        "policy_state_before": "fixture-policy-hash",
        "policy_state_after": "fixture-policy-hash",
        "policy_components_before": {
            "actor": "fixture-actor-hash",
            "normalizer": "fixture-normalizer-hash",
        },
        "policy_components_after": {
            "actor": "fixture-actor-hash",
            "normalizer": "fixture-normalizer-hash",
        },
        "rng_state_before": "fixture-rng",
        "rng_state_after": "fixture-rng",
        "evaluation_seeds": [9030, 9031, 9032],
        "environment": {"fixture": True},
        "evaluation_precision": {"fixture": True},
        "episodes": [episode(seed, achieved=achieved) for seed in (9030, 9031, 9032)],
    }


def test_reach_gain_does_not_become_native_five_bottle_success():
    result = matched.compare(receipt(), receipt(True))
    assert result["qualified_reach_counts"] == {"initial_actor": 0, "final_actor": 3}
    assert result["native_full_success_counts"] == {"initial_actor": 0, "final_actor": 0}
    assert result["eligible_cases"] == 3
    assert result["updates_during_evaluation"] == {"actor": 0, "critic": 0}


def test_already_near_pair_excluded_from_qualified_denominator():
    before, after = receipt(), receipt(True)
    before["episodes"][0] = episode(9030, initial=0.1)
    after["episodes"][0] = episode(9030, achieved=True, initial=0.1)
    result = matched.compare(before, after)
    assert result["eligible_cases"] == 2 and result["qualified_reach_counts"]["final_actor"] == 2


@pytest.mark.parametrize(
    "invalid",
    [
        "model",
        "integration",
        "obs",
        "actor",
        "normalizer",
        "updates",
        "exploration",
        "normalizer_update",
        "rng",
        "warning",
        "partial",
        "native_label",
        "VAL",
        "geometry",
    ],
)
def test_unmatched_or_unfrozen_episode_rejected(invalid):
    before, after = receipt(), receipt(True)
    row = after["episodes"][0]
    if invalid in {"model", "integration", "obs"}:
        key = {
            "model": "compiled_model",
            "integration": "initial_integration",
            "obs": "observation_float32_sha256",
        }[invalid]
        row["reset_proof"][key] = "changed"
    elif invalid in {"actor", "normalizer"}:
        after["policy_components_after"][invalid] = "changed"
    elif invalid == "updates":
        after["actor_updates"] = 1
    elif invalid == "exploration":
        after["action_selection"] = "Actor.explore"
    elif invalid == "normalizer_update":
        after["normalizer_update"] = True
    elif invalid == "rng":
        after["rng_state_after"] = "changed"
    elif invalid == "warning":
        row["physics"]["initial"]["warnings"]["CONTACTFULL"]["number"] = 1
    elif invalid == "partial":
        row["controls"] = 127
    elif invalid == "native_label":
        row["success"] = True
    elif invalid == "VAL":
        row["reset_proof"]["split"] = "validation"
    else:
        row["reset_proof"]["geometry_preserves_integration"] = False
    with pytest.raises((ValueError, FloatingPointError)):
        matched.compare(before, after)


def test_frozen_actor_selection_body_preserved_except_read_only_hash_evidence():
    before = ast.parse((BASE / "abc-reference/src/sadhana_fasttd3_abc/learner.py").read_text())
    previous = next(
        node
        for node in before.body
        if isinstance(node, ast.FunctionDef) and node.name == "evaluate"
    )
    current = ast.parse(inspect.getsource(evaluate)).body[0]
    assert ast.dump(
        next(
            node
            for node in ast.walk(previous)
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "seed"
        ),
        include_attributes=False,
    ) == ast.dump(
        next(
            node
            for node in ast.walk(current)
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "seed"
        ),
        include_attributes=False,
    )


def test_incomplete_binding_cannot_be_promoted(tmp_path):
    value = {"schema": "sadhana.fasttd3.reach-matched-binding/v1"}
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="complete matched"):
        matched.load_binding(path)


def test_changed_or_symlink_pin_rejected(tmp_path):
    file = tmp_path / "plain.json"
    file.write_text("{}")
    pin = matched.artifact(file)
    file.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="differs"):
        matched.checked_json(pin)
    link = tmp_path / "link.json"
    link.symlink_to(file)
    pin = {**matched.artifact(file), "path": str(link)}
    with pytest.raises(ValueError, match="non-symlink"):
        matched.checked_json(pin)
