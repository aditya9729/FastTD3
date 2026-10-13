# Initial actor export and matched TRAIN reach diagnostic

The original update observer runs after an optimizer update. It cannot export
the true initial actor. The new optional `initial_observer` boundary runs after
the original initial environment reset and step selection, before the first
training loop. It rejects a resumed checkpoint. When the observer is absent,
the original learner body remains identical after removal of that boundary.

`InitialExport` receives the actual actor, critics and observation normalizers.
It calls the public author `save_params`; it does not create a second learner.
Before and after that call, it compares all five persistent state dictionaries,
all module training modes, the arguments, and the Python, NumPy and Torch RNG
states. It reads CUDA RNG states only if CUDA is already initialized. A change
fails the job. The initial file is a policy export, not resume state. Optimizer,
replay, scheduler, environment and complete continuation state are not saved.

The `reach` entry point writes the initial export and binds it into its normal
learner receipt and diagnostic receipt. A fresh runtime gate and training job
must use this candidate. A previous actor after one or more updates cannot be
used as the initial actor. Original network, replay, normalizer, update math,
learning starts, critic support, batch, AMP and compilation settings stay the
same. Eleven original modules and four original function ASTs stay exact. The
only new original-source change is the optional train boundary.

The public `reach_matched` consumer calls the existing frozen `learner.evaluate`
twice. It uses `Actor.forward`, no exploration or added noise, no optimizer
calls, and `normalizer(update=False)`. Each receipt includes separate actor and
normalizer hashes before and after evaluation. The initial receipt must match
the states measured at the pre-loop export. Evaluation also preserves the
policy RNG. Initial and final normalization are each frozen at their own
exported values: this comparison measures the actor and its normalizer together.

The separate ABC matched host restores the same three admitted TRAIN compiled
models and full 531-value INTEGRATION states for both actors. It measures reset
distance before an action. Both actors use the same dense reward and fixed 128
controls. The native five-bottle judge stays active and is recorded separately.
The existing full-task campaign and VAL-only capture interfaces do not change.

Installed CPU fixtures check these boundaries. Torch checkpoint writing,
candidate argument construction, native restoration, actor execution and the
actual initial/final comparison need a released normal runtime and root-owned
canonical jobs. Fixture checks are not native or learning evidence.

The preparation agent owns the candidate code and CPU checks. Root owns source
review, normal release, GPU0 admission, actual jobs and results. No external
review service is part of this candidate.

This explanation follows ASD-STE100 guidance. A full compliance check is not
available. See the [architecture](INITIAL_ACTOR_EXPORT.mmd).
