"""Installed CPU admission checks; no native environment or learner loop runs."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from sadhana_fasttd3_hb import stage
from sadhana_fasttd3_hb.qualification import qualified_args, verify_learner_source

from sadhana_fasttd3_abc.learner import verify_source

EXPECTED = os.environ["FAST_FACTORY_RELEASE_EXPECTATION"]
PROFILE = Path(os.environ["FAST_FACTORY_RELEASE_PROFILE"])


def test_installed_guards_bind_the_real_train_payload():
    assert EXPECTED in {"stale", "released"}
    if EXPECTED == "stale":
        with pytest.raises(
            RuntimeError, match="Installed original FastTD3 payload differs: fast_td3/train.py"
        ):
            verify_source()
        with pytest.raises(
            RuntimeError, match="Installed FastTD3 source differs: fast_td3/train.py"
        ):
            verify_learner_source()
    else:
        abc = verify_source()
        hb = verify_learner_source()
        assert abc["explicit_environment_factory_types"] == ["abc", "humanoid_bench"]
        assert len(abc["files"]) == len(hb["learner_files"]) == 12
        assert (
            next(row for row in abc["files"] if row["path"] == "fast_td3/train.py")["sha256"]
            == next(row for row in hb["learner_files"] if row["path"] == "fast_td3/train.py")[
                "sha256"
            ]
        )


def test_actual_stage_preflight_precedes_any_native_child(tmp_path, monkeypatch):
    # The actual -I preflight subprocess reads candidate installed files and
    # metadata. Only canonical lease and native-child construction are doubles.
    # Popen is stopped before a host, model, worker or learner can be created.
    config = json.loads(PROFILE.read_text())
    config["learner_python"] = sys.executable
    root = tmp_path / "owned"
    (root / "builds").mkdir(parents=True)
    output = tmp_path / "worker"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-EXPLICIT-CPU-LEASE-DOUBLE")
    lease = SimpleNamespace(require_subset_lease=Mock())
    native_child = Mock(side_effect=RuntimeError("explicit native construction stop"))
    real_popen = subprocess.Popen

    def launch(command, *args, **kwargs):
        # subprocess.run itself uses Popen. Delegate that real preflight child;
        # stop only the subsequent native host at its process boundary.
        if "--preflight" in command:
            return real_popen(command, *args, **kwargs)
        return native_child(command, *args, **kwargs)

    with (
        patch.dict(sys.modules, {"abc_bench.subset": lease}),
        patch.object(stage, "ROOT", root),
        patch.object(stage, "load_config", return_value=config),
        patch.object(stage.subprocess, "Popen", side_effect=launch),
    ):
        if EXPECTED == "stale":
            with pytest.raises(subprocess.CalledProcessError):
                stage.run(PROFILE, output)
            native_child.assert_not_called()
            assert not (output / "learner_preflight.json").exists()
        else:
            with pytest.raises(RuntimeError, match="explicit native construction stop"):
                stage.run(PROFILE, output)
            native_child.assert_called_once()
            preflight = json.loads((output / "learner_preflight.json").read_text())
            assert len(preflight["learner_files"]) == 12
            assert "sadhana_fasttd3_hb.host" in native_child.call_args.args[0]
    assert not any((root / "builds").iterdir())
    assert json.loads((output / "receipt.json").read_text())["status"] == "failed"
    assert "humanoid_bench" not in sys.modules
    assert "sadhana_fasttd3_hb._author.humanoid_bench_env" not in sys.modules


def test_exact_h1_stand_profile_admitted_without_native_imports():
    config = stage.load_config(PROFILE)
    args = qualified_args()
    assert config["qualification"] == dict(
        env_name="h1hand-stand-v0", num_envs=128, vector_steps=64, seed=1
    )
    assert (args.env_name, args.num_envs, args.total_timesteps, args.seed) == (
        "h1hand-stand-v0",
        128,
        64,
        1,
    )
    assert (args.batch_size, args.buffer_size) == (32768, 51200)
    assert args.compile is True and args.amp is True and args.amp_dtype == "bf16"
    assert (args.use_wandb, args.eval_interval, args.render_interval, args.save_interval) == (
        False,
        0,
        0,
        0,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("env_name", "h1hand-walk-v0"),
        ("num_envs", 1),
        ("vector_steps", 63),
        ("seed", 2),
        ("seed", True),
    ],
)
def test_other_qualification_profiles_are_rejected(tmp_path, field, value):
    config = json.loads(PROFILE.read_text())
    config["qualification"][field] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Only original Stand128world/64vector-step"):
        stage.load_config(path)


@pytest.mark.skipif(EXPECTED != "released", reason="Old train intentionally has no type dispatch")
@pytest.mark.parametrize("environment_type", ["abc", "humanoid_bench"])
def test_actual_main_dispatches_declared_type_before_replay(environment_type):
    import torch
    from fast_td3.train import main

    args = qualified_args()
    args.cuda = False  # Explicit test-only CPU boundary; no training follows.
    factory = Mock(side_effect=RuntimeError("explicit factory boundary stop"))
    with pytest.raises(RuntimeError, match="explicit factory boundary stop"):
        main(args=args, env_factory=factory, environment_type=environment_type)
    settings, device = factory.call_args.args
    assert settings is args and device.type == "cpu"
    assert not torch.cuda.is_initialized()


@pytest.mark.skipif(EXPECTED != "released", reason="Old train intentionally has no type dispatch")
@pytest.mark.parametrize("environment_type", ["isaaclab", "humanoidbench", None])
def test_actual_main_rejects_undeclared_factory_type(environment_type):
    from fast_td3.train import main

    args = qualified_args()
    args.cuda = False
    factory = Mock()
    with pytest.raises(ValueError, match="Unsupported explicit environment factory type"):
        main(args=args, env_factory=factory, environment_type=environment_type)
    factory.assert_not_called()


@pytest.mark.skipif(EXPECTED != "released", reason="Guard intentionally rejects old payloads")
def test_all_installed_original_functions_match_author_ast_seal():
    import importlib.metadata

    provenance = verify_source()
    distribution = importlib.metadata.distribution("fast-td3")
    tree = ast.parse(Path(distribution.locate_file("fast_td3/train.py")).read_text())
    expected = provenance["protected_update_functions"]
    actual = {
        node.name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in expected
    }
    assert actual == expected
