"""CPU RPC and terminal-state boundary checks; no simulator is involved."""

from __future__ import annotations

import json
import socket
import tempfile
import threading
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
