"""CPU numerical qualification. All environments here are explicit test doubles."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

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
    def test_checkpoint_amp_matches_author_context_and_keeps_rng_frozen(self):
        # Explicit CPU fixture: fake CUDA availability routes the author's context
        # to real CPU autocast. No CUDA device or native simulator is initialized.
        class RecordingEnv(SyntheticEnv):
            def __init__(self):
                super().__init__(1)
                self.applied = []

            def reset(self, *, seeds=None):
                self.current_obs = super().reset(seeds=seeds)
                return self.current_obs

            def step(self, actions):
                with torch.no_grad(), real_autocast(
                    "cpu", dtype=expected_dtype, enabled=expected_enabled
                ):
                    expected = original_forward(
                        actor, normalizer(self.current_obs, update=False)
                    ).float()
                torch.testing.assert_close(actions, expected, rtol=0, atol=0)
                self.applied.append(actions.clone())
                result = super().step(actions)
                self.current_obs = result[0]
                return result

        real_autocast = torch.amp.autocast
        for amp, cuda, expected_enabled, amp_dtype in (
            (True, True, True, "bf16"),
            (True, True, True, "fp16"),
            (False, True, False, "bf16"),
            (True, False, False, "bf16"),
        ):
            with self.subTest(amp=amp, cuda=cuda, amp_dtype=amp_dtype):
                args = BaseArgs(
                    num_envs=1, actor_hidden_dim=32, amp=amp, cuda=cuda,
                    amp_dtype=amp_dtype, obs_normalization=True, torch_deterministic=True,
                )
                actor = Actor(20, 14, 1, args.init_scale, 32, device=torch.device("cpu"))
                normalizer = EmpiricalNormalization(20, "cpu")
                normalizer(torch.arange(40, dtype=torch.float32).reshape(2, 20))
                normalizer.eval()
                expected_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
                context_calls, actor_dtypes = [], []
                original_forward = Actor.forward

                def cpu_context(*, device_type, dtype, enabled):
                    context_calls.append((device_type, dtype, enabled))
                    return real_autocast("cpu", dtype=dtype, enabled=enabled)

                def observed_forward(model, obs):
                    self.assertEqual(torch.is_autocast_enabled("cpu"), expected_enabled)
                    self.assertFalse(torch.is_grad_enabled())
                    result = original_forward(model, obs)
                    actor_dtypes.append(result.dtype)
                    return result

                with tempfile.TemporaryDirectory() as directory:
                    checkpoint = Path(directory) / "policy.pt"
                    torch.save({
                        "args": asdict(args), "actor_state_dict": actor.state_dict(),
                        "obs_normalizer_state": normalizer.state_dict(),
                    }, checkpoint)
                    before = torch.random.get_rng_state().clone()
                    env = RecordingEnv()
                    with (
                        patch("torch.cuda.is_available", return_value=True),
                        patch("torch.amp.autocast", side_effect=cpu_context),
                        patch.object(Actor, "forward", observed_forward),
                    ):
                        receipt = evaluate(checkpoint, env, [9030], Path(directory) / "eval")
                    self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
                    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
                    self.assertEqual(context_calls, [
                        ("cuda" if cuda else "cpu", dtype, expected_enabled)
                    ] * 5)
                    self.assertEqual(
                        actor_dtypes, [dtype if expected_enabled else torch.float32] * 5
                    )
                    self.assertTrue(all(action.dtype == torch.float32 for action in env.applied))
                    self.assertEqual(receipt["policy_state_before"], receipt["policy_state_after"])
                    self.assertEqual(receipt["rng_state_before"], receipt["rng_state_after"])
                    precision = receipt["evaluation_precision"]
                    self.assertEqual(precision["autocast_enabled"], expected_enabled)
                    self.assertEqual(precision["policy_action_dtype"], [str(actor_dtypes[0])])
                    self.assertEqual(precision["policy_parameter_dtype"], ["torch.float32"])
                    self.assertEqual(precision["environment_action_dtype"], "torch.float32")
                    self.assertEqual(precision["float32_matmul_precision"], "high")
                    self.assertTrue(precision["cudnn_deterministic"])
                    self.assertEqual((receipt["actor_updates"], receipt["critic_updates"]), (0, 0))

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
