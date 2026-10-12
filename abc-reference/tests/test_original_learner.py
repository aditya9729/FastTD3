"""CPU numerical qualification. All environments here are explicit test doubles."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from fast_td3.fast_td3 import Actor, Critic
from fast_td3.fast_td3_utils import EmpiricalNormalization, SimpleReplayBuffer
from fast_td3.hyperparams import BaseArgs
from tensordict import TensorDict

from sadhana_fasttd3_abc.learner import evaluate, state_digest, train

torch.set_num_threads(1)


class SyntheticEnv:
    """Reset and terminal observation fixture; it is not ABC or an RL benchmark."""

    asymmetric_obs = False
    num_obs, num_actions, max_episode_steps = 20, 14, 5

    def __init__(self, count=4, device="cpu"):
        self.num_envs, self.device = count, torch.device(device)
        self.description = {
            "test_double": True,
            "purpose": "original learner numerical qualification",
        }
        self.controls, self.actual_seeds, self.closed = 0, [], False
        self.step_in_episode = 0
        self.seed = 0

    def reset(self, *, seeds=None):
        self.step_in_episode = 0
        self.seed = seeds[0] if seeds else 0
        self.actual_seeds.extend(seeds or [0] * self.num_envs)
        return torch.zeros((self.num_envs, self.num_obs), device=self.device)

    def step(self, actions):
        self.controls += 1
        self.step_in_episode += 1
        observation = torch.full(
            (self.num_envs, self.num_obs), float(self.step_in_episode), device=self.device
        )
        observation[:, :14] += actions
        reward = 0.2 - actions.square().mean(dim=1)
        done = torch.full((self.num_envs,), self.step_in_episode == 5, device=self.device)
        raw = observation.clone()
        episodes = []
        if done.all():
            observation.zero_()
            self.step_in_episode = 0
            if self.num_envs == 1:
                episodes = [{"seed": self.seed, "success": False, "completed": True}]
        return (
            observation,
            reward,
            done,
            {
                "time_outs": done,
                "observations": {"raw": {"obs": raw}},
                "episodes": episodes,
            },
        )

    def close(self):
        self.closed = True


class OriginalLearnerTests(unittest.TestCase):
    def test_terminal_distribution_projection_clamps_and_conserves_mass(self):
        critic = Critic(4, 2, 5, 0.0, 4.0, 32, "", 64, 8, device=torch.device("cpu"))
        rewards = torch.tensor([0.0, 1.0, 2.0, 4.0, 10.0, -10.0])
        first, second = critic.projection(
            torch.zeros(6, 4),
            torch.zeros(6, 2),
            rewards,
            torch.zeros(6),
            torch.full((6,), 0.99),
        )
        expected = torch.nn.functional.one_hot(torch.tensor([0, 1, 2, 4, 4, 0]), 5).float()
        for projected in (first, second):
            self.assertTrue(torch.allclose(projected, expected, atol=1e-6))
            self.assertTrue(torch.allclose(projected.sum(dim=1), torch.ones(6), atol=1e-6))
            self.assertTrue(torch.all(projected >= 0))

    def test_n_step_replay_stops_at_episode_boundary(self):
        replay = SimpleReplayBuffer(1, 16, 2, 1, 2, n_steps=3, gamma=0.5, device="cpu")
        for index in range(6):
            replay.extend(
                TensorDict(
                    {
                        "observations": torch.tensor([[float(index), 0.0]]),
                        "actions": torch.zeros(1, 1),
                        "next": {
                            "observations": torch.tensor([[float(index + 1), 99.0]]),
                            "rewards": torch.tensor([float(index + 1)]),
                            "dones": torch.tensor([index == 1]),
                            "truncations": torch.tensor([index == 1]),
                        },
                    },
                    batch_size=(1,),
                )
            )
        torch.manual_seed(7)
        sample = replay.sample(128)
        starts = sample["observations"][:, 0]
        start_zero = starts == 0
        self.assertTrue(start_zero.any())
        self.assertTrue(torch.all(sample["next"]["effective_n_steps"][start_zero] == 2))
        self.assertTrue(
            torch.allclose(
                sample["next"]["rewards"][start_zero],
                torch.full_like(sample["next"]["rewards"][start_zero], 2.0),
            )
        )
        self.assertTrue(torch.all(sample["next"]["observations"][start_zero, 0] == 2))
        self.assertTrue(torch.all(sample["next"]["truncations"][start_zero] == 1))

    def test_normalization_freeze_preserves_state(self):
        normalizer = EmpiricalNormalization(4, "cpu")
        normalizer(torch.tensor([[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]]))
        before = state_digest(normalizer)
        result = normalizer(torch.full((2, 4), 100.0), update=False)
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(before, state_digest(normalizer))

    def test_original_loop_updates_both_networks_and_frozen_export_reloads(self):
        args = BaseArgs(
            env_name="CPU-fixture",
            exp_name="OriginalLoop",
            seed=7,
            cuda=False,
            num_envs=4,
            total_timesteps=32,
            batch_size=64,
            buffer_size=64,
            actor_hidden_dim=32,
            critic_hidden_dim=64,
            compile=False,
            amp=False,
            use_wandb=False,
            eval_interval=0,
            render_interval=0,
            save_interval=0,
        )
        environments = []

        def factory(settings, device):
            env = SyntheticEnv(settings.num_envs, device)
            environments.append(env)
            return env, env, env

        with tempfile.TemporaryDirectory() as directory:
            receipt = train(args, factory, Path(directory) / "train", check_interval=1)
            self.assertEqual(receipt["actual_updates"], {"critic": 42, "actor": 21})
            self.assertEqual(receipt["finite_gradient_samples"], {"critic": 42, "actor": 21})
            self.assertGreater(receipt["first_loss"]["critic"], 5.0)
            self.assertLess(receipt["first_loss"]["critic"], 15.0)
            self.assertTrue(environments[0].closed)
            checkpoint = Path(receipt["checkpoint"]["path"])
            payload = torch.load(checkpoint, weights_only=False)
            torch.manual_seed(7)
            initial = Actor(20, 14, 4, args.init_scale, 32, device=torch.device("cpu"))
            trained = payload["actor_state_dict"]
            self.assertTrue(
                any(
                    not torch.equal(value, trained[name])
                    for name, value in initial.state_dict().items()
                    if name.startswith("net.")
                )
            )
            frozen_env = SyntheticEnv(1)
            measured = evaluate(checkpoint, frozen_env, [9011, 9012], Path(directory) / "eval")
            self.assertEqual(measured["successes"], 0)
            self.assertEqual(measured["actor_updates"], 0)
            self.assertEqual(measured["policy_state_before"], measured["policy_state_after"])
            self.assertTrue(all(item["controls"] == 5 for item in measured["episodes"]))
            self.assertEqual(measured["checkpoint"]["sha256"], receipt["checkpoint"]["sha256"])


if __name__ == "__main__":
    unittest.main()
