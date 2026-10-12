"""Run the original FastTD3 loop and evaluate exported policies without updates."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.resources
import json
import math
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .environment import Endpoint, RemoteEnv


def verify_source() -> dict[str, Any]:
    """Authenticate normal installed author payloads against the frozen source seal."""
    provenance = json.loads(
        importlib.resources.files("sadhana_fasttd3_abc").joinpath("provenance.json").read_text()
    )
    distribution = importlib.metadata.distribution("fast-td3")
    for row in provenance["files"]:
        path = Path(distribution.locate_file(row["path"]))
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise RuntimeError(f"Installed original FastTD3 payload differs: {row['path']}")
    return provenance


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def artifact(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def state_digest(*models: Any) -> str:
    """Fingerprint persistent policy and normalization tensors, including buffers."""
    digest = hashlib.sha256()
    for index, model in enumerate(models):
        for name, tensor in sorted(model.state_dict().items()):
            array = tensor.detach().cpu().contiguous().numpy()
            digest.update(f"{index}:{name}:{array.dtype}:{array.shape}".encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


class UpdateObserver:
    """Read-only measurements after the original optimizer updates."""

    def __init__(self, *, check_interval: int = 50) -> None:
        if check_interval < 1:
            raise ValueError("Gradient check interval must be positive")
        self.check_interval = check_interval
        self.counts = {"critic": 0, "actor": 0}
        self.gradient_samples = {"critic": 0, "actor": 0}
        self.first_loss: dict[str, float] = {}
        self.last_loss: dict[str, float] = {}

    def __call__(self, role: str, model: Any, logs: Any) -> None:
        import torch

        self.counts[role] += 1
        loss = float(logs["qf_loss" if role == "critic" else "actor_loss"].detach().cpu())
        if not math.isfinite(loss):
            raise FloatingPointError(f"Original FastTD3 {role} loss is not finite")
        self.last_loss[role] = loss
        count = self.counts[role]
        if count == 1:
            self.first_loss[role] = loss
        if count == 1 or count % self.check_interval == 0:
            gradients = [
                parameter.grad for parameter in model.parameters() if parameter.grad is not None
            ]
            if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
                raise FloatingPointError(
                    f"Original FastTD3 {role} gradients are absent or not finite"
                )
            self.gradient_samples[role] += 1

    def evidence(self) -> dict[str, Any]:
        return {
            "actual_updates": dict(self.counts),
            "finite_gradient_samples": dict(self.gradient_samples),
            "gradient_check_interval": self.check_interval,
            "last_loss": dict(self.last_loss),
            "first_loss": dict(self.first_loss),
        }


def validate_args(args: Any) -> None:
    if args.agent != "fasttd3" or args.sim_type:
        raise ValueError("This benchmark uses original FastTD3 without SimbaV2 or SimNorm")
    if args.checkpoint_path is not None:
        raise ValueError("Scratch training must not load an initialization checkpoint")
    if args.eval_interval != 0 or args.render_interval != 0 or args.use_wandb:
        raise ValueError(
            "Use separate frozen evaluation; in-loop evaluation/render/tracking are disabled"
        )
    if args.num_envs < 1 or args.batch_size < args.num_envs or args.batch_size % args.num_envs:
        raise ValueError("Batch size must be a positive multiple of the environment count")
    if args.total_timesteps < args.learning_starts + 2 or args.buffer_size <= args.num_steps:
        raise ValueError(
            "Training must reach actual learner updates and have a valid replay capacity"
        )


def train(args: Any, env_factory: Any, output: Path, *, check_interval: int = 50) -> dict[str, Any]:
    """Call the normal installed original training module; do not copy its math."""
    validate_args(args)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "resolved_args.json", asdict(args))
    observer = UpdateObserver(check_interval=check_interval)
    created: list[Any] = []

    def tracked_factory(settings: Any, device: Any):
        environments = env_factory(settings, device)
        for env in environments:
            if not any(env is other for other in created):
                created.append(env)
        if environments[0].num_envs != args.num_envs:
            raise ValueError("Host environment count differs from the original FastTD3 recipe")
        return environments

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "condition": "original_fasttd3_scratch_privileged_state_abc",
        "status": "running",
        "python": platform.python_version(),
        "runtime": {
            name: importlib.metadata.version(name) for name in ("torch", "numpy", "tensordict")
        },
        "configuration": asdict(args),
        "source": verify_source(),
        "checkpoint": None,
        "resume_semantics": "Original policy export only; optimizer, replay and RNG are not saved.",
    }
    before = Path.cwd()
    start = time.monotonic()
    try:
        # Import only after validation. The original module imports no ABC simulator.
        from fast_td3.train import main

        os.chdir(output)
        main(args=args, env_factory=tracked_factory, observer=observer)
        checkpoint = output / "models" / f"{args.env_name}__{args.exp_name}__{args.seed}_final.pt"
        if not checkpoint.is_file() or not all(observer.counts.values()):
            raise RuntimeError(
                "Original FastTD3 completed without its checkpoint or actor/critic updates"
            )
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        for name in (
            "actor_state_dict",
            "qnet_state_dict",
            "qnet_target_state_dict",
            "obs_normalizer_state",
            "critic_obs_normalizer_state",
        ):
            state = payload.get(name)
            if state is not None and not all(
                torch.isfinite(tensor).all() for tensor in state.values()
            ):
                raise FloatingPointError(f"Original FastTD3 export contains nonfinite {name}")
        receipt["checkpoint"] = artifact(checkpoint)
        receipt["status"] = "completed"
    except BaseException as exc:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        os.chdir(before)
        for env in reversed(created):
            env.close()
        receipt["seconds"] = time.monotonic() - start
        receipt.update(observer.evidence())
        if created:
            receipt["environment"] = getattr(created[0], "description", {"test_double": True})
            receipt["controls"] = getattr(created[0], "controls", None)
            receipt["actual_reset_seeds"] = getattr(created[0], "actual_seeds", [])
        write_json(output / "receipt.json", receipt)
    return receipt


def remote_factory(socket_path: str, *, seed: int = 9030):
    def create(args: Any, device: Any):
        env = RemoteEnv(Endpoint(socket_path), device=device, seed=seed)
        # Separate frozen evaluation is the only supported evaluation mode.
        return env, env, env

    return create


def evaluate(checkpoint: Path, env: Any, seeds: list[int], output: Path) -> dict[str, Any]:
    """Original evaluation autocast; policy, normalization and RNG stay unchanged."""
    import torch
    from fast_td3.fast_td3 import Actor
    from fast_td3.fast_td3_utils import EmpiricalNormalization
    from torch.amp import autocast

    source = verify_source()
    if env.num_envs != 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Frozen evaluation requires one world and distinct explicit seeds")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    payload = torch.load(checkpoint, map_location=env.device, weights_only=False)
    args = payload["args"]
    if args["agent"] != "fasttd3" or args.get("sim_type"):
        raise ValueError("Checkpoint is not original FastTD3")
    # Match train.main's precision setup and nested evaluate autocast exactly.
    amp_enabled = args["amp"] and args["cuda"] and torch.cuda.is_available()
    amp_device_type = (
        "cuda"
        if args["cuda"] and torch.cuda.is_available()
        else "mps" if args["cuda"] and torch.backends.mps.is_available() else "cpu"
    )
    amp_dtype = torch.bfloat16 if args["amp_dtype"] == "bf16" else torch.float16
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.deterministic = args["torch_deterministic"]

    def rng_state():
        states = {"torch_cpu": torch.random.get_rng_state()}
        if env.device.type == "cuda":
            states["torch_policy_device"] = torch.cuda.get_rng_state(env.device)
        return {
            name: hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest()
            for name, state in states.items()
        }

    rng_before = rng_state()
    # Construction draws unused random initial weights/noise buffers before loading.
    # Keep those draws separate from the caller's deterministic evaluation stream.
    devices = (
        [env.device.index if env.device.index is not None else torch.cuda.current_device()]
        if env.device.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=devices):
        actor = Actor(
            n_obs=env.num_obs,
            n_act=env.num_actions,
            num_envs=args["num_envs"],
            init_scale=args["init_scale"],
            hidden_dim=args["actor_hidden_dim"],
            std_min=args["std_min"],
            std_max=args["std_max"],
            device=env.device,
        )
    actor.load_state_dict(payload["actor_state_dict"])
    actor.eval()
    if args["obs_normalization"]:
        normalizer = EmpiricalNormalization(shape=env.num_obs, device=env.device)
        normalizer.load_state_dict(payload["obs_normalizer_state"])
    else:
        normalizer = torch.nn.Identity()
    normalizer.eval()
    normalize_obs = (
        torch.compile(normalizer.forward, mode=None) if args["compile"] else normalizer.forward
    )
    if not all(
        torch.isfinite(tensor).all()
        for model in (actor, normalizer)
        for tensor in model.state_dict().values()
    ):
        raise ValueError("Checkpoint policy or normalization contains nonfinite tensors")
    frozen = state_digest(actor, normalizer)
    start = time.monotonic()
    records = []
    action_dtypes = set()
    for seed in seeds:
        obs = env.reset(seeds=[seed])
        reward_sum, final = 0.0, None
        for control in range(1, env.max_episode_steps + 1):
            with torch.no_grad(), autocast(
                device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
            ):
                normalized = normalize_obs(obs, update=False) if args["obs_normalization"] else obs
                actions = actor(normalized)
            action_dtypes.add(str(actions.dtype))
            obs, rewards, dones, infos = env.step(actions.float())
            reward_sum += float(rewards[0].cpu())
            if bool(dones[0]):
                episodes = infos["episodes"]
                if len(episodes) != 1 or episodes[0].get("seed") != seed:
                    raise ValueError("Frozen episode evidence is missing or has a different seed")
                final = dict(episodes[0])
                break
        if (
            final is None
            or type(final.get("success")) is not bool
            or final.get("completed") is not True
        ):
            raise ValueError("Frozen evaluation ended without complete success evidence")
        final.update({"controls": control, "return": reward_sum})
        records.append(final)
        write_json(output / f"episode-{seed}.json", final)
    after = state_digest(actor, normalizer)
    if frozen != after:
        raise RuntimeError("Frozen evaluation mutated actor or observation normalization")
    rng_after = rng_state()
    if rng_before != rng_after:
        raise RuntimeError("Frozen evaluation mutated the policy RNG state")
    receipt = {
        "schema_version": 1,
        "condition": "original_fasttd3_scratch_privileged_state_abc",
        "status": "completed",
        "checkpoint": artifact(checkpoint),
        "evaluation_seeds": seeds,
        "episodes": records,
        "successes": sum(record["success"] for record in records),
        "actor_updates": 0,
        "critic_updates": 0,
        "policy_state_before": frozen,
        "policy_state_after": after,
        "rng_state_before": rng_before,
        "rng_state_after": rng_after,
        "evaluation_precision": {
            "autocast_enabled": bool(amp_enabled),
            "autocast_device_type": amp_device_type,
            "configured_amp": args["amp"],
            "configured_cuda": args["cuda"],
            "configured_compile": args["compile"],
            "compiled_normalizer": args["compile"],
            "normalizer_compile_mode": None,
            "actor_compiled": False,
            "amp_dtype": args["amp_dtype"],
            "autocast_dtype": str(amp_dtype),
            "policy_parameter_dtype": sorted({str(p.dtype) for p in actor.parameters()}),
            "policy_action_dtype": sorted(action_dtypes),
            "environment_action_dtype": "torch.float32",
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
        "environment": env.description,
        "seconds": time.monotonic() - start,
        "source": source,
    }
    write_json(output / "receipt.json", receipt)
    return receipt
