# Setup and fidelity

The original source comes from [FastTD3](https://github.com/younggyoseo/FastTD3)
at `229ed59bbf43ea2f7a2d5d90d1076314839944d7`.
The [paper](https://arxiv.org/abs/2505.22642v3) uses parallel simulation,
large batches and distributional value learning.

The current condition is original FastTD3 from scratch with privileged ABC
state. It does not adapt ABC-VLA. It does not add a residual action.
Its actor starts with random weights. This differs from post-training a VLA.

The original actor has hidden widths 512, 256 and 128.
Each original critic has hidden widths 1024, 512 and 256.
Each critic predicts 101 categorical atoms on `[-250,250]`.
The actor uses current critic values. There is no target actor in this recipe.

The original settings remain `gamma=0.99`, `tau=0.1`, two critic updates per
vector call, one delayed actor update, AdamW learning rates `3e-4`, weight
decay `0.1`, target noise `0.001`, noise clip `0.5` and per-episode exploration
scales in `[0.001,0.4]`. Learning starts after vector call 10.
The observer records actual updates. It does not calculate them from budget.

The original replay stores `buffer_size` rows for each environment.
The original sampler draws approximately `batch_size` rows in total.
`total_timesteps` counts vector calls. Multiply by `num_envs` for transitions.
Keep these units separate when you compare wall time or sample counts.

The initial ABC condition has one world. The author HumanoidBench recipe has
128 worlds. One world changes data diversity and exploration. It does not
reproduce the author's performance experiment. The short budget tests the
installed learner and simulator boundary. It cannot establish method ranking.

Only `fast_td3/train.py` changes among the 25 original tracked files.
It uses package-relative imports and an explicit caller boundary.
The boundary accepts arguments, an environment factory and a read-only observer.
The original `update_main`, `update_pol`, `soft_update` and `evaluate` bodies
have the same normalized AST hashes as upstream. Architecture, projection,
replay, normalization, optimizer steps and update cadence stay unchanged.
The upstream warmup scheduler warning is preserved.

The normal runtime uses Python 3.10.16, Torch 2.6.0+cu124, TensorDict 0.7.2
and NumPy 1.26.4. `runtime/pyproject.toml` and `runtime/uv.lock` define the
dependency prefix. Build and install the two source wheels separately.
Their hashes and source checks identify the accepted installed packages.
No editable install or source-path injection is used.

The optional author simulator backends are absent. The ABC host owns simulation.
Gymnasium, TorchRL, JAX, IsaacLab, TorchVision, TorchAudio and media packages
are not required by this execution path. WandB remains an original import
dependency. External tracking is disabled. No dataset or media upload occurs.

Root runs the original sequential ABC MuJoCo physics to match its new baseline.
The host supplies full physical `qpos`, `qvel` and fixed scene metadata.
The model uses privileged state. Images are not policy inputs.
The host clips normalized actions to `[-1,1]`, then applies physical limits.
The original learner stores its sampled normalized action, as upstream does.
The host records clipping and the physical action schema.

The host saves terminal state before automatic reset. It separates success
termination from time-limit truncation. The original critic bootstraps across
timeouts. It does not bootstrap across success termination.
Matched evaluation rejects layout resampling and incomplete evidence.

Training and evaluation are separate. Frozen evaluation calls the original
deterministic actor. It does not update observation normalization.
It compares persistent policy tensors before and after all evaluation episodes.
The output must contain one complete episode for each requested seed.

Original checkpoints contain actors, critics, normalizers and arguments.
They omit optimizer, replay, scheduler and random state.
They support policy export and approximate restart. Exact continuation is absent.

Eight installed CPU tests passed. Four tests check original numerical behavior.
They include 42 critic updates and 21 actor updates with finite gradient samples.
Four tests check the RPC boundary, terminal state, seed integrity and failures.
The synthetic environments are test doubles. They are not ABC benchmark results.

All 60 installed distribution RECORDs passed. They cover 17,048 hashed payloads.
Both normal wheels match the installed payloads and current source.
The CLI help command runs without importing the simulator or learner.
Ruff and diff checks passed. Root owns native qualification and benchmark runs.

The FastTD3 agent owns source, runtime and numeric qualification.
The ABC host agent owns the simulator boundary and physical state schema.
Root owns the GPU 0 lease, execution budget, frozen comparison and viewer.
No agent changes Leela jobs or other GPUs.

[Architecture source](ARCHITECTURE.mmd) shows these boundaries.
Native connections require root's actual execution evidence.
The explanation uses ASD-STE100 guidance. A full compliance check was not done.
