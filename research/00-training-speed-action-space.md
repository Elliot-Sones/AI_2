# Training speed: possible directions before choosing an engine rewrite

Research date: 2026-09-05. Scope: compare options and choose informative next
experiments. This document is a research recommendation, not an approved
implementation plan or a measured speedup for any proposed architecture.

## Question and success criteria

What is the best next investment after animation caching and observation reuse:
more CPU optimization, native CPU code, GPU simulation, a different training
pipeline, or a learning change?

There are two distinct objectives:

- **Execute the same training workload sooner:** measure complete PPO collection
  and updates at a fixed transition budget, with comparable settings and behavior.
- **Reach useful playing strength sooner:** measure elapsed time to a declared
  navigation or combat evaluation score across multiple seeds. More transitions
  per second do not establish this by themselves.

The first is the primary comparison for infrastructure changes. The second is
required for model, curriculum, action-frequency, algorithm, or sampling changes.
Engineering effort and expected future training volume also affect whether an
engine rewrite pays for itself. No development-time estimate or port speedup has
been measured here.

## Current repository evidence

1. **Our latest optimization has CPU evidence only.** Observation reuse reduced
   median fixed-workload elapsed time by 15.2% for the configured network in a
   one-environment CPU benchmark, with substantial variation across three seeds.
   The unchanged optimizer also varied. This was neither the full 48-environment
   production setup nor a measurement of time to a target score.
   [Results](../baselines/observation-reuse-20260905/README.md).

2. **Batching has already mattered on this game.** After animation caching,
   matched runs on the same RTX 4090 host achieved median 2,165 transitions/s
   with 8 environments and 2,946 with 16: 36.1% higher aggregate throughput.
   These runs predate observation reuse. Sixteen was the best count tested,
   not a demonstrated optimum.
   [Matched measurements](../baselines/animation-cache-20260904/README.md).

3. **The bottleneck changes with the setup.** The earlier 16-environment CUDA
   diagnostic attributed about 60% to the environment/coordination/frame-stack
   interval and 19% to policy forward. A newer one-environment CUDA diagnostic
   on a different CPU host attributed about 33% and 45%, respectively, with
   another 9% in rollout storage and 3% in PPO updates. Both are short,
   instrumented, pre-observation-reuse runs. Timings are CPU-observed intervals,
   not kernel-only GPU measurements. They cannot establish a causal host speedup
   or a stable current percentage split.
   [16-env diagnostic](../baselines/pipeline-profile-20260904/README.md),
   [1-env rerun](../baselines/rerun-20260905/README.md),
   [1-env stage data](../baselines/rerun-20260905/artifacts/natural_1env/pipeline_profile.json).

4. **Physics is already native.** Pymunk calls a C physics engine. A native port
   would principally target our surrounding Python game logic, observation and
   reward construction, and the batch interface. Compiling the game alone would
   not fix policy launch overhead or SB3 minibatch preparation.
   [Existing code/profile analysis](../baselines/pipeline-profile-20260904/README.md).

5. **There is still concrete avoidable CPU work.** The effective `ConstantAgent`
   samples an action and discards it to return zeros. `OpponentHistoryWrapper`
   rebuilds a deque-derived array each step, including all-zero navigation
   history. Removing the sampling affects action-space RNG advancement and must
   be checked; buffer reuse must preserve independently owned observations.
   [Agent implementation](../UTMIST-AI2-main/environment/agent.py),
   [History wrapper](../train.py).

6. **The input design is a separate experimental lever.** Configuration appends
   60 x 10 history values to 64 game values and stacks four frames: 2,656 input
   values. Navigation uses zero history, making 2,400 of these values zero. Cheap
   construction can preserve the existing interface; shortening history or
   changing network width changes the model and checkpoint/phase-transfer
   contract. It requires learning evaluation.
   [Configuration](../config.yaml), [Factories and wrappers](../train.py).

7. **Navigation does not cover combat costs.** Current performance experiments
   use navigation and a frozen opponent. Combat configures a different opponent
   mix, including self-play; the environment calls the opponent policy in its
   step. Grouping neural-opponent inference is a candidate, but actual self-play
   loading, input compatibility, device use, and costs need a combat baseline
   before any speed claim.
   [Combat factory](../train.py), [Opponent step](../UTMIST-AI2-main/environment/agent.py).

## Axes of choice

Language, hardware, and execution layout are separate choices. Native C/C++ can
be used for CPU batching or GPU kernels. Python-authored array programs can be
compiled for an accelerator. Neither merely changing a language nor selecting
CUDA automatically batches this object-based Pymunk game.

The full practical space is broader than two alternatives:

| Direction | What changes | Main uncertainty | Relative scope |
| --- | --- | --- | --- |
| Remove redundant work | Zero actions, history packing, allocations, repeated queries | Remaining share after the latest optimization; RNG and ownership | Small, targeted |
| Tune existing execution | Environment count, process count, Torch threads, CPU allocation, inference device | Best settings depend on actual CPU/GPU and workload | Small experiments |
| Batch CPU games and transport | Several games per worker, preallocated arrays, shared CPU memory | Fewer messages can help, but more serial work per worker can hurt | Moderate prototype |
| Improve policy and rollout handling | Larger action batches, fewer tensor copies, contiguous/GPU rollout storage, compiled inference | GPU launch/data handling versus actual computation | Moderate; some options alter the trainer |
| Overlap independent work | Double-buffer sampling, partial batches, simulation/inference overlap | Synchronization, stragglers, policy versions | Moderate to large |
| Compile selected CPU hotspots | Typed loops/arrays around existing physics | Python bytecode versus already-native or waiting costs | Targeted to moderate |
| Build a native batched CPU simulator | Game state and stepping designed for many arenas | Porting effort, exact behavior, remaining trainer bottleneck | Large |
| Build a GPU simulator | Batched state, collision/physics, rules, rewards and resets on GPU | Physics equivalence, sufficient batch size, complete training integration | Large |
| Reduce network/history work | Smaller network, shorter or differently encoded memory | Playing strength, checkpoint transfer and partial observability | Learning experiment |
| Reduce experience needed | PPO tuning, curriculum/opponents, pretrained initialization | Sample efficiency, robustness and evaluation quality | Learning experiment |
| Use both players' experience | Train from both perspectives when both actions come from the current trainable policy | Changes self-play/opponent distribution; fixed or older opponents do not supply ordinary on-policy learner data | Learning and collector experiment |
| Change decision/simulation frequency | Repeat actions; separately consider a different physics tick | Control quality, discount/time-limit semantics and combat timing | Changes learning or game semantics |
| Add machines or accelerators | More CPU capacity, distributed sampling, multi-GPU learner | Coordination cost and whether the current limiter scales | Later, after single-machine evidence |

These options can be combined. Their gains should not be multiplied: after one
stage becomes cheaper, another stage can limit the complete pipeline.

## What the papers add

| Reference | Evidence relevant to this decision | What it does not establish |
| --- | --- | --- |
| [EnvPool, NeurIPS 2022](https://arxiv.org/abs/2206.10558) | Native CPU environment pools combine batching and execution scheduling. | Our Python game gains the published speed simply by installing the library. |
| [PufferLib 2.0, RLJ/RLC 2025](https://rlj.cs.umass.edu/2025/papers/Paper151.html) | Shared arrays and grouping environments per worker are useful alternatives to a full simulator port. | The first-party environments' headline rate applies to this game. |
| [Sample Factory, ICML 2020](https://proceedings.mlr.press/v119/petrenko20a.html) | Separating simulation, batched inference, and learning can improve utilization. | Its asynchronous results transfer unchanged to our synchronous PPO. |
| [SEED RL, ICLR 2020](https://arxiv.org/abs/1910.06591) | Central inference is a useful architectural reference, especially for many policy requests. | Our current single learner policy is unbatched; SB3 already batches those requests. |
| [Kinetix/Jax2D, ICLR 2025](https://arxiv.org/abs/2410.23208) | GPU physics for many small 2D worlds is a credible direction. | Compatibility with Pymunk dynamics or our fighting mechanics. |
| [Madrona, SIGGRAPH 2023](https://madrona-engine.github.io/) | Custom C++ game logic can be organized for large GPU batches. | A conversion of existing Python game files into an equivalent GPU simulator. |
| [What Matters for On-Policy Deep Actor-Critic Methods?, ICLR 2021](https://research.google/pubs/what-matters-for-on-policy-deep-actor-critic-methods-a-large-scale-study/) | Implementation and optimization choices can substantially affect learning. | A winning hyperparameter recipe for this game. |

The studies support design patterns and experiments. They do not compare these
alternatives on this repository. The ordering below is our inference from the
papers, current code, and measurement gaps.

Detailed notes: [CPU options](01-cpu-simulation-and-batching.md),
[GPU options](02-gpu-simulation-options.md),
[training and sample efficiency](03-training-pipeline-and-sample-efficiency.md).

## Recommendation and decision rules

**First architectural experiment: a synchronous batched CPU runner with efficient
policy/data handling. Confidence: medium.** Keep current game physics, test fewer
processes with several games per worker, and use preallocated or shared buffers.
SB3 already batches the learner's actions across its vector environments; the
experiment is to improve batch size, scheduling, packing, and storage around that
existing behavior. Fewer workers can also increase serial stepping time, so the
runner is a candidate to measure, not a promised improvement.

**Native CPU implementation remains a conditional next step.** If repeated
profiles show Python game logic still controls throughput after small fixes and
runner changes, compile one substantial hot path first. A full native batched
simulator becomes more attractive if that experiment demonstrates a meaningful
end-to-end gain and reusing the existing physics keeps porting costs manageable.

**GPU simulation is the more ambitious research branch. Confidence in feasibility:
medium; confidence in a repo-specific speedup: low until measured.** For our 2D
game, investigate a JAX/Jax2D-style prototype; Madrona is an alternative if C++
and PyTorch integration better suit the eventual design. The prototype should
include representative contact, attack, knockback, pickup and reset mechanics.
If many simultaneous games cannot preserve useful learning horizons or fit in
memory, large-batch simulator results will not translate into faster training.
The rollout/training interface must also keep data on the accelerator to test
the complete architecture; the existing NumPy-based SB3 path would need adaptation.

**Learning efficiency deserves a separate experiment.** History/model ablations
and curriculum/opponent selection may reduce the number of transitions needed.
They should compete on time to a fixed evaluation score. A smaller input during
navigation needs an explicit strategy for resuming existing models or transferring
to combat; removing zero-valued dimensions is not automatically a compatible
checkpoint change.

An additional repo-derived idea is two-sided self-play: the raw game already
returns both perspectives, while the training wrapper exposes one learner. A
collector could use both when both players are controlled by the current policy.
This is a hypothesis, not a demonstrated twofold speedup: it increases policy and
update work, changes the opponent curriculum, and requires evaluation against a
fixed opponent suite. Count physics advances and player training samples
separately so a reporting change cannot masquerade as faster simulation.

Defer multi-machine or multi-GPU scaling until there is evidence that local CPU
capacity or learning computation is the limiter. Additional GPU capacity alone
does not speed up the current CPU game workers. More CPU cores are useful only
while the measured runner continues to scale.

## Smallest informative experiments

1. **Refresh the baseline on the intended GPU-training host.** Use the latest
   observation-reuse source. Compare 1, 8 and 16 environments; add higher counts
   only when available CPU resources justify them. Keep total rollout capacity,
   update budget and network comparable. Use a capacity divisible by every tested
   environment count. Repeat isolated runs across at least three seeds, separating
   warmup and profiling from headline throughput. Include a representative combat
   run once opponent loading/input/device behavior is verified. The historical
   worker timer instruments `observe`; update it to include `_observe_all` before
   trusting a new worker-level breakdown.

2. **Test the middle path at one fixed environment count.** Compare today's runner
   with two or four games per worker, then independently test preallocation/shared
   arrays. Keep policy weights fixed throughout each collection phase. Measure
   full training time, CPU use, memory, batch gathering, policy forward, and the
   distribution of environment-round latency. Validate per-environment fixed-action
   replays, observation ownership, final observations, truncation bootstrapping,
   resets, rewards and episode accounting.

3. **Select the next prototype from the measured residual.** Dominant Python game
   logic suggests a selective native implementation. Dominant policy/buffer work
   suggests batching, data layout or inference-device experiments. Persistent
   simulation limits plus substantial future training volume justify comparing a
   representative GPU prototype with the CPU reference over increasing batch sizes.
   Measure both raw simulation and complete PPO; include compilation/setup costs
   separately. Treat a simplified-game result only as feasibility evidence.

Sampling and learning can be separated carefully: overlapping CPU stepping with
inference for another batch can retain fixed policy weights. Collecting experience
while those weights are updated introduces policy lag and requires a separate
algorithm/learning evaluation. Faster-first collection can also change which
environments contribute data. It needs sample accounting even with fixed weights.

Changing environment count while holding total rollout capacity fixed changes
the per-environment rollout length. Such a throughput comparison does not alone
establish unchanged learning behavior. Any final deployment setting must also
pass the score-based evaluation.

## Research tasks

- [x] Frame the question using current code and benchmark provenance.
- [x] Check scope: explore possibilities first; implementation is a later task.
- [x] Compare CPU, GPU, and training/learning papers and primary documentation.
- [x] Rank the approaches and define the smallest useful experiments.
- [x] Verify source support and document the recommendation.
