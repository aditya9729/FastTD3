"""CPU RPC and terminal-state boundary checks; no simulator is involved."""

from __future__ import annotations

import json
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path

import torch

from sadhana_fasttd3_abc.environment import SCHEMA, Endpoint, RemoteEnv


class ExplicitEndpointFixture:
    def __init__(self):
        self.calls = []
        self.resample = False
        self.overlap = False

    def call(self, operation, **payload):
        self.calls.append((operation, payload))
        if operation == "describe":
            return {
                "num_envs": 1,
                "num_obs": 24,
                "num_actions": 14,
                "max_episode_steps": 1000,
                "privileged_state": True,
            }
        if operation == "reset":
            seeds = payload["seeds"]
            return {"actual_seeds": [seeds[0] + 1] if self.resample else seeds, "obs": [[0.0] * 24]}
        if operation == "step":
            return {
                "obs": [[-99.0] * 24],
                "terminal_obs": [[7.0] * 24],
                "rewards": [0.2],
                "terminated": [self.overlap],
                "truncated": [True],
                "episodes": [{"seed": 20260511, "success": False, "completed": True}],
            }
        raise AssertionError(operation)


class RemoteBoundaryTests(unittest.TestCase):
    def test_captured_consumer_uses_actual_unix_endpoint(self):
        from sadhana_fasttd3_abc.captured import SCHEMA as captured_schema
        from sadhana_fasttd3_abc.captured import CapturedRemoteEnv

        description = {
            "num_envs": 1,
            "num_obs": 216,
            "num_actions": 14,
            "max_episode_steps": 3540,
            "privileged_state": True,
            "profile": "captured_validation",
            "autoreset": False,
            "action_profile": "normalized14_affine_clip_v1",
            "reset_policy": {
                "reset_verified": True,
                "profile": "captured_demonstration_mjb_integration_v1",
                "heldout_seed": 9100,
                "allowed_seeds": [9100],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "captured.sock"
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.settimeout(2)
            server.bind(str(path))
            server.listen(1)
            failures, operations = [], []

            def handle():
                try:
                    with server:
                        for _ in range(2):
                            connection, _ = server.accept()
                            with connection:
                                connection.settimeout(2)
                                request = json.loads(connection.recv(4096))
                                self.assertEqual(request["schema"], captured_schema)
                                operations.append(request["operation"])
                                if request["operation"] == "describe":
                                    result = description
                                else:
                                    self.assertEqual(request["operation"], "reset")
                                    self.assertEqual(request["seeds"], [9100])
                                    result = {"actual_seeds": [9100], "obs": [[0.0] * 216]}
                                response = {
                                    "schema": captured_schema,
                                    "id": request["id"],
                                    "ok": True,
                                    "result": result,
                                }
                                connection.sendall(json.dumps(response).encode() + b"\n")
                except Exception as error:
                    failures.append(error)

            thread = threading.Thread(target=handle, daemon=True)
            thread.start()
            try:
                env = CapturedRemoteEnv(
                    Endpoint(path, timeout=2, schema=captured_schema), device="cpu"
                )
                self.assertEqual(tuple(env.reset(seeds=[9100]).shape), (1, 216))
                env.close()
            finally:
                thread.join(timeout=3)
                server.close()
            self.assertFalse(thread.is_alive())
            self.assertEqual(failures, [])
            self.assertEqual(operations, ["describe", "reset"])

    def test_actual_unix_slow_response_cannot_extend_total_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "slow.sock"
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(str(path))
            server.listen(1)
            failures = []

            def handle():
                try:
                    with server:
                        connection, _ = server.accept()
                        with connection:
                            connection.settimeout(1)
                            request = json.loads(connection.recv(4096))
                            response = (
                                json.dumps(
                                    {
                                        "schema": SCHEMA,
                                        "id": request["id"],
                                        "ok": True,
                                        "result": {"slow_response_fixture": True},
                                    }
                                ).encode()
                                + b"\n"
                            )
                            for offset in range(0, len(response), 20):
                                connection.sendall(response[offset : offset + 20])
                                time.sleep(0.04)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # Expected: the deadline closes only this owned connection.
                except (OSError, ValueError) as error:
                    failures.append(error)

            thread = threading.Thread(target=handle, daemon=True)
            thread.start()
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                Endpoint(path, timeout=0.06).call("describe")
            elapsed = time.monotonic() - started
            thread.join(timeout=1)
            self.assertFalse(thread.is_alive())
            self.assertEqual(failures, [])
            self.assertLess(elapsed, 0.15)

    def test_terminal_state_and_timeout_survive_autoreset(self):
        env = RemoteEnv(ExplicitEndpointFixture(), device="cpu")
        env.reset(seeds=[20260511])
        obs, reward, done, infos = env.step(torch.zeros(1, 14))
        self.assertTrue(torch.all(obs == -99))
        self.assertTrue(torch.all(infos["observations"]["raw"]["obs"] == 7))
        self.assertTrue(bool(infos["time_outs"][0]))
        self.assertTrue(bool(done[0]))
        self.assertAlmostEqual(float(reward[0]), 0.2, places=6)
        env.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            env.step(torch.zeros(1, 14))

    def test_reset_resampling_is_rejected(self):
        endpoint = ExplicitEndpointFixture()
        endpoint.resample = True
        env = RemoteEnv(endpoint, device="cpu")
        with self.assertRaisesRegex(ValueError, "resampled"):
            env.reset(seeds=[20260511])

    def test_invalid_actions_never_reach_host_and_overlap_is_rejected(self):
        endpoint = ExplicitEndpointFixture()
        env = RemoteEnv(endpoint, device="cpu")
        count = len(endpoint.calls)
        with self.assertRaisesRegex(ValueError, "finite"):
            env.step(torch.full((1, 14), float("nan")))
        self.assertEqual(count, len(endpoint.calls))
        endpoint.overlap = True
        with self.assertRaisesRegex(ValueError, "timeout"):
            env.step(torch.zeros(1, 14))

    def test_actual_unix_framing_and_request_correlation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.sock"
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(str(path))
            server.listen(1)

            def handle():
                with server:
                    connection, _ = server.accept()
                    with connection:
                        raw = bytearray()
                        while not raw.endswith(b"\n"):
                            raw.extend(connection.recv(1024))
                        request = json.loads(raw)
                        response = (
                            json.dumps(
                                {
                                    "schema": SCHEMA,
                                    "id": request["id"],
                                    "ok": True,
                                    "result": {"real_unix_fixture": True},
                                }
                            ).encode()
                            + b"\n"
                        )
                        connection.sendall(response[:5])
                        connection.sendall(response[5:])

            thread = threading.Thread(target=handle, daemon=True)
            thread.start()
            result = Endpoint(path, timeout=2.0).call("describe")
            thread.join(timeout=2.0)
            self.assertFalse(thread.is_alive())
            self.assertTrue(result["real_unix_fixture"])


if __name__ == "__main__":
    unittest.main()
