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
