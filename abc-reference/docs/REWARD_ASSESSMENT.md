# ABC scratch reward assessment

The 2026-10-12 diagnostic exposes a task objective problem. It does not establish
that the original FastTD3 algorithm fails.

Run `author-subset-20261012T214336-5f3c65` used the original author learner with
128 native ABC worlds, 4,000 vector steps, batch 32,768 and discount 0.99. It
collected 512,000 transitions and completed 128 training episodes, with no full
success. Frozen VAL9100 completed 3,540 controls, with zero bottles placed and
zero successes in one episode. VAL9101 was cancelled before completion and is
excluded from the completed score.

## Reward objective

The ABC host passes the native current bottle fraction to the learner on every
control. One bottle yields 0.2; all five yield 1. The environment terminates at
full success. The unchanged original critic stops bootstrapping on this true
termination and bootstraps on time-limit truncation. Reward normalization is off.

An illustrative reward sequence shows the incentive problem:

```text
one bottle retained for 100 controls:
  0.2 * sum(0.99**t for t in range(100)) = 12.679353174535404

immediate full success, followed by termination:
  1.0
```

This objective can prefer sustained partial placement to immediate completion.
The counterexample describes reward sequences. It does not prove that the
measured actor deliberately exploits this incentive. There is no reaching,
grasping, lifting or action-smoothness term in this task reward.

The native fraction is useful as an evaluation score. Repeating it as an
episodic RL reward requires a separate objective design. Preserve the native
success judge when introducing an explicitly versioned training reward. No
reward correction has been applied in the reported runs.

## Comparison scope

| Property | Current FastTD3 diagnostic | Planned ordinary EXPO-FT |
| --- | --- | --- |
| Initial policy | Random state policy | Pretrained, task-adapted pi0.5 |
| Demonstrations | None | Three complete TRAIN episodes |
| Inputs | Privileged state216 | Three RGB views and state14 |
| Learner reward | Current bottle fraction each control | Binary full-task success |
| Action interface | Normalized14 mapped to physical joint commands | Raw physical14, eight-control chunk prefix |

These settings compare different learning strategies. They cannot isolate an
algorithm advantage. The Fast schedule is 2.667% of the author's default
150,000 vector steps and supplies approximately 1.13 episode horizons per world.
That default is not an author-provided ABC recipe. ABC is our environment adapter.

FastTD3 already learns twin distributional value functions. The simulator must
provide the reward; no supplied value model is missing. More training alone
does not repair the reward objective. The [original paper](https://arxiv.org/html/2505.22642v2#S3)
also discusses reward design and different reward requirements across algorithms.

For the deployed-VLA question, a FastTD3 residual comparison would require a
separate, disclosed adapter and a common initial VLA. The original scratch
learner remains a separate comparison.

## Evidence

Exact executed arguments, source hashes and the mathematical counterexample are
in `/home/user/aditya/RL/builds/abc-fasttd3-training-stage-v1/REWARD_OBJECTIVE_ASSESSMENT.json`.
Canonical training and frozen evaluation receipts remain unchanged. Actual
videos and comparison limits are at `http://127.0.0.1:8770/original-methods/`.

Further scratch training needs a corrected task reward and a declared learning
budget. Native task success must remain the common evaluation measurement.
