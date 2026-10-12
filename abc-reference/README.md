# Original FastTD3 from scratch in ABC-Sim

This package connects the original FastTD3 learner to a separately hosted ABC
simulator. The actor uses privileged physical scene state. It does not use a
VLA, images, pretrained weights or residual actions.

The upstream pin is `229ed59bbf43ea2f7a2d5d90d1076314839944d7`.
The original architecture, C51 distributional critic, replay, normalization,
optimizers and update function bodies are preserved. `fast_td3/train.py` changes
package imports and adds explicit arguments, an environment factory and a
read-only observer. The observer records real update counts and finite samples.

Adapter `0.1.5` seals all 12 original Python files, including the four environment
wrappers. Packaging now includes those wrapper files without changing their bytes.
The explicit factory type accepts `abc` or `humanoid_bench`. The separate
`sadhana_fasttd3_hb` stage admits only the reviewed H1 Stand 128-world, 64-step
qualification. That GPU qualification remains pending. Its native CPU host uses
a separate runtime; this learner runtime does not install HumanoidBench or JAX.

Use the separate Python 3.10 / Torch 2.6.0+cu124 runtime.
Install the normal FastTD3 and adapter wheels. Optional upstream simulator
backends, JAX, Isaac, TorchRL, Gymnasium and media/tracking clients are not used.
WandB is an original import dependency; external tracking stays disabled.

Root starts the original ABC host under the serial GPU 0 lease.
The host supplies robot and object position, velocity and task state.
It maps normalized `[-1,1]` actions to documented physical limits.
It saves the final state before automatic episode reset.
The client applies one total RPC deadline across connect, send and all response
reads. A server cannot extend this deadline by sending small response fragments.

```sh
python -I -u -m sadhana_fasttd3_abc.cli train \
  --socket /absolute/private/host.sock --output /absolute/new/train \
  --steps 1000 --num-envs 1 --batch-size 32768 --buffer-size 51200 \
  --device cuda --amp

python -I -u -m sadhana_fasttd3_abc.cli evaluate \
  --socket /absolute/private/host.sock --output /absolute/new/eval \
  --checkpoint /absolute/train/models/ABC-PutBottles-v1__FastTD3Scratch__1_final.pt \
  --seeds 20260511 20260512 20260513 --device cuda
```

These are finite diagnostic settings. One world differs from the author's
128-world HumanoidBench recipe. `--steps` counts vector calls. Individual
transitions equal vector calls times the world count. Replay capacity is per
world. `--compile` and `--amp` are explicit. Compilation stays off by default.

Training and frozen evaluation use separate commands. Evaluation uses the
original deterministic actor and frozen observation normalization. It checks
policy state and policy RNG before and after evaluation. It uses the saved
`amp`, `cuda` and `amp_dtype` settings with the author's evaluation autocast.
The receipt records the actual autocast profile, parameter/action dtypes and
original `high` float32 matrix multiplication precision. Original actions are
converted to float32 before simulator dispatch. When saved `compile` is true,
the normalizer is compiled with `mode=None`, as in the author's nested evaluator.
The actor stays eager in that original evaluation path. Missing or incomplete episode
evidence fails evaluation; it does not become a zero score.

The original checkpoint exports actor, critics, normalizers and arguments.
It does not save optimizers, replay or random state. Exact continuation is not
supported or claimed. Checkpoints and native recordings stay outside Git.

CPU tests use explicit synthetic environments. They qualify the original
numeric path; they are not native ABC scores. Root alone launches GPU and
native runs. Keep Leela jobs unchanged.

The explanation uses ASD-STE100 guidance. A full compliance check was not done.
