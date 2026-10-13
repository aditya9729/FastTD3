"""Read-only policy export at the original learner's optional pre-loop boundary."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from .learner import artifact, state_digest, verify_source, write_json


def rng_fingerprint() -> dict[str, Any]:
    """Read current RNG states without initializing a CUDA device."""
    import numpy as np
    import torch

    numpy_state = np.random.get_state()
    numpy_digest = hashlib.sha256(numpy_state[1].tobytes())
    numpy_digest.update(repr((numpy_state[0], *numpy_state[2:])).encode())
    states = {"cpu": torch.random.get_rng_state()}
    cuda_initialized = torch.cuda.is_initialized()
    if cuda_initialized:
        states.update(
            {f"cuda:{i}": state for i, state in enumerate(torch.cuda.get_rng_state_all())}
        )
    return {
        "python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
        "numpy": numpy_digest.hexdigest(),
        "torch": {
            name: hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest()
            for name, state in states.items()
        },
        "cuda_initialized": cuda_initialized,
    }


def boundary_snapshot(models: dict[str, Any], args: Any) -> dict[str, Any]:
    return {
        "states": {name: state_digest(model) for name, model in models.items()},
        "modes": {
            name: [module.training for module in model.modules()] for name, model in models.items()
        },
        "arguments": json.loads(json.dumps(vars(args), allow_nan=False)),
        "rng": rng_fingerprint(),
    }


def save_initial(path: Path, global_step: int, models: dict[str, Any], args: Any) -> None:
    """Use the author's public export format, without copying learner setup."""
    from fast_td3.fast_td3_utils import save_params

    save_params(global_step, **models, args=args, save_path=str(path))


class InitialExport:
    """Called once before the first normalize, action, transition or update."""

    def __init__(self, checkpoint: Path):
        self.checkpoint = checkpoint.resolve()
        self.calls = 0
        self.evidence: dict[str, Any] | None = None

    def __call__(
        self,
        *,
        global_step: int,
        actor: Any,
        qnet: Any,
        qnet_target: Any,
        obs_normalizer: Any,
        critic_obs_normalizer: Any,
        args: Any,
    ) -> None:
        if self.calls or type(global_step) is not int or global_step != 0:
            raise ValueError("Initial export requires exactly one scratch pre-loop call at step0")
        if args.checkpoint_path is not None:
            raise ValueError("Initial scratch export must not load a checkpoint")
        if (
            self.checkpoint.exists()
            or self.checkpoint.is_symlink()
            or self.checkpoint.with_suffix(".json").exists()
        ):
            raise ValueError("Refuse a pre-existing initial export")
        models = {
            "actor": actor,
            "qnet": qnet,
            "qnet_target": qnet_target,
            "obs_normalizer": obs_normalizer,
            "critic_obs_normalizer": critic_obs_normalizer,
        }
        self.calls += 1
        before = boundary_snapshot(models, args)
        self.evidence = {
            "schema": "sadhana.fasttd3.initial-scratch-export/v1",
            "status": "running",
            "global_step": 0,
            "controls": 0,
            "actor_updates": 0,
            "critic_updates": 0,
            "boundary": "after_original_initial_reset_before_first_training_loop",
            "resume_semantics": "Policy export only; no optimizer/replay/RNG continuation state",
            "before": before,
            "checkpoint": None,
        }
        metadata = self.checkpoint.with_suffix(".json")
        try:
            save_initial(self.checkpoint, global_step, models, args)
            after = boundary_snapshot(models, args)
            self.evidence["after"] = after
            if before != after:
                raise RuntimeError("Initial export changed model, normalizer, modes, args or RNG")
            self.evidence["checkpoint"] = artifact(self.checkpoint)
            self.evidence["source"] = verify_source()
            self.evidence["status"] = "completed"
        except BaseException as error:
            self.evidence.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            write_json(metadata, self.evidence)
            self.evidence["metadata"] = artifact(metadata)
