# Original FastTD3 ABC integration

Read `/home/user/aditya/AGENTS.md` and the coordination board.

Upstream is `https://github.com/younggyoseo/FastTD3` at
`229ed59bbf43ea2f7a2d5d90d1076314839944d7`.
Work on `codex/fasttd3-abc-original`. Never push to the author's remote.

Keep the original network, replay and normalization modules unchanged.
The narrow `train.py` boundary uses package-relative imports, an explicit
environment factory, caller-supplied arguments and a read-only update observer.
Protect the original update function bodies with normalized AST hashes.

The adapter is a normal package in `abc-reference`.
Install both normal wheels into the new `RL/runtimes/fasttd3-author-abc-py310` prefix.
Do not use source-path injection or edit other runtime prefixes.
The locked runtime has original Torch 2.6.0+cu124, TensorDict 0.7.2 and NumPy 1.26.4.
Optional author simulator backends are absent.

Root owns all native and GPU execution under the ABC coordinator GPU 0 lease.
Do not launch native jobs, use other GPUs or disturb Leela.
CPU qualification must hide CUDA and disable external tracking.
Record actual checks and known recipe differences. Do not claim full paper reproduction.

## Isolated R1 Lite dispatch preparation — 14 October 2026

This worktree uses branch `codex/r1lite-fasttd3-dispatch-20261014` for the
explicitly authorized R1 Lite task. It starts from `0bb4ab9488eae3fcdfce1699f6ca44f49ef42f33`.
Only the named `leela_r1lite` factory type, RGB render dispatch and the existing
post-evaluation reset predicate change. Evaluation reuses the training pool,
as the original HumanoidBench path does.
Keep all learner, network, replay, normalization and update settings unchanged.
The Sadhana task package owns Leela observations, actions, reward and resets.
Root owns normal installation, native acceptance and canonical GPU admission.
This dispatch preparation is not an author-supported R1 Lite benchmark result.
