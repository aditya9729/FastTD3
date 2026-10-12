"""Bounded Unix RPC for physical ABC state; imports no simulator."""

from __future__ import annotations

import json
import socket
import time
from pathlib import Path
from typing import Any

SCHEMA = "sadhana.fasttd3.abc.v1"
MAX_FRAME_BYTES = 4 * 1024**2


class Endpoint:
    """One connection per request. State remains in the task-owned host."""

    def __init__(self, path: str | Path, *, timeout: float = 60.0, schema: str = SCHEMA) -> None:
        if type(timeout) not in (int, float) or not 0 < timeout <= 600:
            raise ValueError("RPC timeout must be positive and at most 600 seconds")
        if schema not in (SCHEMA, "sadhana.fasttd3.abc.vector.v1"):
            raise ValueError("Unsupported FastTD3 ABC RPC schema")
        self.path, self.timeout, self.request_id, self.schema = str(path), timeout, 0, schema

    def call(self, operation: str, **payload: Any) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout
        self.request_id += 1
        body = (
            json.dumps(
                {"schema": self.schema, "id": self.request_id, "operation": operation, **payload},
                allow_nan=False,
                separators=(",", ":"),
            ).encode()
            + b"\n"
        )
        if len(body) > MAX_FRAME_BYTES:
            raise ValueError("RPC request exceeds the 4 MiB frame limit")

        def remaining() -> float:
            seconds = deadline - time.monotonic()
            if seconds <= 0:
                raise TimeoutError("ABC RPC exceeded its total deadline")
            return seconds

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(remaining())
            connection.connect(self.path)
            connection.settimeout(remaining())
            connection.sendall(body)
            frame = bytearray()
            while True:
                connection.settimeout(remaining())
                chunk = connection.recv(min(65536, MAX_FRAME_BYTES + 1 - len(frame)))
                if not chunk:
                    raise RuntimeError("ABC host closed before a complete response")
                frame.extend(chunk)
                if len(frame) > MAX_FRAME_BYTES:
                    raise ValueError("RPC response exceeds the 4 MiB frame limit")
                if b"\n" in frame:
                    if frame.count(b"\n") != 1 or not frame.endswith(b"\n"):
                        raise ValueError("ABC host must send one JSON frame per connection")
                    break
        response = json.loads(frame)
        if not isinstance(response, dict) or response.get("schema") != self.schema:
            raise ValueError("ABC host response has an incompatible schema")
        if response.get("id") != self.request_id:
            raise ValueError("ABC host response request identity differs")
        if response.get("ok") is not True:
            raise RuntimeError(f"ABC host rejected {operation}: {response.get('error')}")
        result = response.get("result")
        if not isinstance(result, dict):
            raise ValueError("ABC host result must be an object")
        remaining()
        return result


class RemoteEnv:
    """RSL-RL-style boundary required by the original FastTD3 training loop."""

    asymmetric_obs = False

    def __init__(self, endpoint: Endpoint, *, device: Any, seed: int = 9030) -> None:
        import torch

        self.endpoint, self.device, self.next_seed = endpoint, torch.device(device), seed
        description = endpoint.call("describe")
        for key in ("num_envs", "num_obs", "num_actions", "max_episode_steps"):
            value = description.get(key)
            if type(value) is not int or value < 1:
                raise ValueError(f"ABC host {key} must be a positive integer")
            setattr(self, key, value)
        if self.num_actions != 14 or self.max_episode_steps > 10000:
            raise ValueError("ABC host must provide 14 actions and a finite horizon")
        if description.get("privileged_state") is not True:
            raise ValueError("FastTD3 scratch requires an explicit privileged-state observation")
        self.description = description
        self.closed, self.controls, self.resets = False, 0, 0
        self.episodes: list[dict[str, Any]] = []
        self.actual_seeds: list[int] = []

    def _array(self, value: Any, shape: tuple[int, ...], *, boolean: bool = False):
        import numpy as np
        import torch

        array = np.asarray(value)
        if array.shape != shape:
            raise ValueError(f"ABC host expected array shape {shape}, got {array.shape}")
        if boolean:
            if array.dtype.kind != "b":
                raise ValueError("ABC termination flags must be JSON booleans")
            dtype = torch.bool
        else:
            if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
                raise ValueError("ABC host state/reward must be finite numeric data")
            array = array.astype(np.float32)
            if not np.isfinite(array).all():
                raise ValueError("ABC host data exceeds float32 limits")
            dtype = torch.float32
        return torch.as_tensor(array, device=self.device, dtype=dtype)

    def reset(self, *, seeds: list[int] | None = None):
        if self.closed:
            raise RuntimeError("ABC environment is closed")
        if seeds is None:
            seeds = list(range(self.next_seed, self.next_seed + self.num_envs))
            self.next_seed += self.num_envs
        if len(seeds) != self.num_envs or any(type(seed) is not int or seed < 0 for seed in seeds):
            raise ValueError("Reset requires one nonnegative integer seed per environment")
        result = self.endpoint.call("reset", seeds=seeds)
        actual = result.get("actual_seeds")
        if actual != seeds:
            raise ValueError("ABC reset resampled a requested layout; matched seeds are required")
        self.actual_seeds.extend(actual)
        self.resets += 1
        return self._array(result.get("obs"), (self.num_envs, self.num_obs))

    def step(self, actions):
        import torch

        if self.closed:
            raise RuntimeError("ABC environment is closed")
        if not isinstance(actions, torch.Tensor) or tuple(actions.shape) != (self.num_envs, 14):
            raise ValueError("ABC action must be a Tensor shaped (num_envs,14)")
        if not torch.isfinite(actions).all():
            raise ValueError("ABC action must be finite")
        result = self.endpoint.call("step", actions=actions.detach().cpu().tolist())
        obs = self._array(result.get("obs"), (self.num_envs, self.num_obs))
        terminal = self._array(result.get("terminal_obs"), (self.num_envs, self.num_obs))
        rewards = self._array(result.get("rewards"), (self.num_envs,))
        terminated = self._array(result.get("terminated"), (self.num_envs,), boolean=True)
        truncated = self._array(result.get("truncated"), (self.num_envs,), boolean=True)
        if torch.any(terminated & truncated):
            raise ValueError("ABC success termination cannot also be a timeout")
        episodes = result.get("episodes", [])
        if not isinstance(episodes, list) or any(not isinstance(item, dict) for item in episodes):
            raise ValueError("ABC episode evidence must be a list of objects")
        self.episodes.extend(episodes)
        self.controls += 1
        return (
            obs,
            rewards,
            terminated | truncated,
            {
                "time_outs": truncated,
                "observations": {"raw": {"obs": terminal}},
                "episodes": episodes,
                "autoresets": result.get("autoresets", []),
            },
        )

    def close(self) -> None:
        # Host lifetime belongs to root. Closing a learner must not kill its host.
        self.closed = True
