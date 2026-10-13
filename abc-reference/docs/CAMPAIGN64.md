# Reduced-world FastTD3 ABC campaign

This release adds `reward_v2_training64`. It also retains
`reward_v2_training128` and the existing campaign settings. The 64-world profile
must declare `num_envs: 64`. A missing count is allowed only for historical Fast
128-world configurations; the ABC host continues to require an explicit count.

The original FastTD3 learner, models, replay, normalization, optimizers and
updates have no changes in this release. Both profiles retain batch size32768,
replay size51200 per world, two critic updates per vector call, one actor update,
gamma0.99, and the same versioned ABC reward/action/reset settings. The separate
runtime gate has64 calls. The training limit is42480 calls, with no task-futility
stop before21240 calls and six complete episodes per rank. Three complete stale
windows may then request a natural exit after the current updates. These are
TRAIN task measures, not validation scores or proof of convergence.

The reduced world count changes the effective data regime:

| Quantity | 128 worlds | 64 worlds |
| --- | ---: | ---: |
| Maximum training transitions | 5437440 | 2718720 |
| Aggregate replay capacity | 6553600 | 3276800 |
| Batch samples per world per critic update | 256 | 512 |
| Critic updates per collected transition after warmup | 0.015625 | 0.03125 |
| Total critic batch rows per new transition after warmup | 512 | 1024 |

The last row counts sampled rows, including repeated samples. It does not count
unique replay transitions. The unchanged author code allocates
`[num_envs, buffer_size, ...]` in `fast_td3_utils.py` and samples
`batch_size // num_envs` rows per world in `train.py`. Thus this campaign is an
explicit reduced-world ABC adaptation, not an exact128-world paper recipe.

Actual task statistics use64 equal rank weights. The declared success-improvement
threshold remains1/128. One successful episode per64-rank window gives a rate of
1/64 when every rank contributes one episode. The seed bank still advances by128
per rank and episode, so the first64 rank scene sequences match the prior profile.
The original arm carry, placement retry and fallback remain unchanged.

The Fast adapter receives observations, rewards, terminal states and reset
evidence from the ABC host. It requests a clean loop exit. It does not change
learner updates. The parent owns memory admission, the GPU lease, host cleanup
and native jobs. The ABC stage peer owns the paired host and stage. This agent
owns the small Fast campaign boundary and its CPU contracts.

CPU tests use explicit environment and author-loop doubles. A small test also
uses the original replay implementation, and another uses the unchanged Torch
checkpoint export. These tests do not prove simulator task performance. A new
64-world native gate must pass before a64-world training run. The prior128-world
gate is not a64-world native result. A final export does not contain optimizer,
replay or environment/RNG state for exact continuation.

See [the architecture](CAMPAIGN64.mmd). Historical configs and receipts remain
intact. A new launch must pin its new paired source/wheel/config artifacts.
