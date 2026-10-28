# Curriculum learning: reading guide for AI_2

Research date: 2026-09-13.

## Research question and scope

Which papers and comparable game projects explain how to construct useful reinforcement-learning curricula, transfer skills between lessons, and diagnose apparent curriculum failures in this fighting-game project?

This review uses original papers, author project pages, official code repositories, and the locally preserved September 7–8 curriculum/comparison reports. It does not establish the state of a live training job. The user's exact current symptom was not specified; the recommendations therefore prioritize the strongest available local evidence. The newer contact curriculum differs substantially from the older root `config.yaml` / `train.py` phase system.

## Foundation

A curriculum chooses the experiences an agent practices so that it learns the final task more effectively. Three separate decisions matter: which practice tasks to create, when/how often to train on each, and what knowledge to carry forward. Success on an intermediate exercise is useful when it helps performance on the target task. The relevant comparison includes the cost of learning the earlier exercises. See Narvekar et al., [Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey](https://jmlr.org/papers/v21/20-212.html), especially Sections 2.2–2.3, 3, and 4.1.

## Recommended reading order

Start with the closest physical-game example, then use the survey to organize the different design decisions. The remaining papers each address a specific question raised by the saved AI_2 results.

| Order | Paper | Focus while reading |
| --- | --- | --- |
| 1 | Bansal et al., [Emergent Complexity via Multi-Agent Competition](https://arxiv.org/abs/1710.03748), ICLR 2018 | Sections 4.1–4.2 and 5.3–5.5: early movement rewards, their removal, and opponent sampling. |
| 2 | Narvekar et al., [Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey](https://jmlr.org/papers/v21/20-212.html), JMLR 2020 | Sections 3.1–3.3 and 4.1–4.2: distinguish task design, sequencing, transfer, and evaluation. |
| 3 | Ng, Harada, and Russell, [Policy Invariance under Reward Transformations](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf), ICML 1999 | Theorem 1 and Section 5: when does an additional reward preserve the original objective? |
| 4 | Florensa et al., [Reverse Curriculum Generation for Reinforcement Learning](https://proceedings.mlr.press/v78/florensa17a.html), CoRL 2017 | Methodology, Algorithm 1, and the “good starts” experiment: design resets from which the current agent can sometimes finish the skill. |
| 5 | Liu et al., [From Motor Control to Team Play in Simulated Humanoid Football](https://arxiv.org/abs/2105.12196), Science Robotics 2022 | Sections 3.3, 3.4.2, and 7 of the preprint: reusable drills, transfer mechanisms, and ablations. |
| 6 | Jiang et al., [Prioritized Level Replay](https://proceedings.mlr.press/v139/jiang21b.html), ICML 2021 | Section 3 and its algorithms: choose which previously seen scenarios to practice again. |

For PLR, “replay” means revisiting scenarios and collecting fresh on-policy experience; it does not mean training ordinary PPO on a buffer of old transitions. Reverse curricula assume useful reset access and known goal states. Neither paper establishes an optimal success threshold or sampling ratio for AI_2.

[Method notes and optional follow-up papers](05-curriculum-methods-and-papers.md) cover reward-shaping assumptions, teacher/student scheduling, ALP-GMM, and ACCEL.

## Comparable projects to read

| Project / paper | Read first | Question for AI_2 |
| --- | --- | --- |
| Bansal et al., [Emergent Complexity via Multi-Agent Competition](https://arxiv.org/abs/1710.03748), ICLR 2018 | Sections 4.1–4.2 and 5.3–5.5 | When should early reward aids fade, and how should opponent variety grow? |
| Liu et al., [From Motor Control to Team Play in Simulated Humanoid Football](https://arxiv.org/abs/2105.12196), 2021 preprint / Science Robotics 2022 | Sections 3.3–3.4.2 and 7 in the arXiv version | How can a later task use movement and contact skills deliberately? |
| Firoiu et al., [Beating the World's Best at Super Smash Bros. with Deep Reinforcement Learning](https://arxiv.org/abs/1702.06230), 2017 preprint | Sections 4.1–4.4 | How do narrow fighting-game training setups become brittle against other opponents? |

The official [OpenAI competition code](https://github.com/openai/multiagent-competition) and [Phillip code](https://github.com/vladfi1/phillip) provide concrete implementations to inspect. Both are historical references. The football study uses motion capture and reusable behavior priors; it is evidence about skill transfer, not a requirement to adopt its architecture. See [the project notes](06-curriculum-comparable-game-projects.md) for mechanisms, source limits, and implementation availability.

## What the preserved project evidence establishes

### 1. Navigation checkpoints were not trained fighting policies

In the newer navigation setup, light/heavy attack and pickup/throw/taunt outputs had no gameplay consequences. Those controls became active in combat evaluation. In the larger checkpoint's 160-match panel, 36 of 106 wins involved zero damage dealt. These wins count under the evaluation rules, but they do not demonstrate purposeful attacking.

Evidence: [combat comparison README](../baselines/combat-comparison-session12/README.md), lines 7–42. These were development matches against four fixed saved policies, not a human evaluation or a study of independent training seeds.

**Interpretation:** Treat navigation as movement initialization. Check whether each new attack skill is actually experienced and learned before inferring combat mastery from navigation or aggregate win rate. This is a transfer/measurement concern, not proof that navigation pretraining was harmful.

### 2. The planned contact-training comparison was not completed

The saved contact run collected exactly 32,768 new transitions for a light-attack calibration. The subsequent paired joint-training arms did not start. The plan included contact practice interleaved with navigation maintenance, so it would be inaccurate to say the new design lacks retention practice.

Evidence: [September 8 results](../baselines/combat-curriculum-session13/fresh-run/approved-resume/RESULTS.md), lines 3–8 and 61–83; [approved contact plan](../baselines/combat-curriculum-session13/contact-upload/artifacts/provenance/approved-plan.md), lines 47–55.

**Interpretation:** This run cannot establish a plateau, failure of the complete curriculum, or the effect of the planned joint-training schedule.

### 3. Attack selection and safe completion are separate problems

| Post-calibration evaluation | Without action sampling | With action sampling |
| --- | ---: | ---: |
| Episodes | 20 | 20 |
| Approached the target | 20 | 20 |
| Episodes with an attributed hit | 0 | 20 |
| Safe completions | 0 | 7 |

The highest deterministic raw light-attack output was 0.02688; the game button requires a value greater than 0.5. All 13 sampled failures first involved the target crossing its floor edge after damage. The learner also crossed subsequently in 11 cases. Zero learner stock losses before termination does not certify safety because the episode may end earlier for another failure.

Evidence: [September 8 results](../baselines/combat-curriculum-session13/fresh-run/approved-resume/RESULTS.md), lines 12–38.

**Interpretation:** Audit the conversion from policy output to executed game buttons and define which action-selection mode the skill must work under. Sampling can expose successful attacks before the unsampled policy reliably requests them. These observations do not prove that increasing entropy, lowering the threshold, or switching action distributions would solve the problem.

### 4. The reported navigation alarm was not observed forgetting

All 280 matched parent/calibrated navigation episode pairs had identical executed trajectories. Both versions already failed the same absolute wide-navigation standard. The saved report corrects the use of this later-stage standard as a mandatory stop for the earlier contact lesson: nearby retention governed the earlier stop condition and passed.

Evidence: [September 8 results](../baselines/combat-curriculum-session13/fresh-run/approved-resume/RESULTS.md), lines 45–83.

**Interpretation:** Distinguish loss of an existing skill from failure to meet a stronger future skill requirement. This calibration did not demonstrate movement forgetting. It also cannot establish that later joint training would preserve movement.

### 5. The reward needs to be interpreted against the lesson's success definition

The contact lesson awards +1 for the first hit and +2 for safe completion. Own stock loss costs −5, timeout costs −1, and each tick costs −0.001. Target stock loss and unsafe floor support can end an episode without an additional corresponding penalty; earlier hit credit remains. The lesson intentionally requires both players to remain supported after the hit, whereas knocking the opponent off stage is desirable in the final game.

Evidence: [contact environment](../baselines/combat-curriculum-session13/contact-upload/combat_curriculum.py), lines 493–527; [contact specification](../baselines/combat-curriculum-session13/contact-upload/artifacts/provenance/approved-plan.md), lines 19–29.

The pre-update collection contained 358 completed episodes with hits, but only 90 safe completions, while the completed-episode reward total remained positive. This is a rollout subset, not a balanced held-out evaluation. Do not compare its 90/358 directly with the separate post-update 7/20 as a measured learning gain.

Evidence: [collection diagnostic](../baselines/combat-curriculum-session13/fresh-run/approved-resume/analysis-calibration-full/diagnostic.md), lines 49–67.

**Interpretation:** Check which failed behaviors still earn useful return and whether the intermediate safety objective transfers to fighting. Positive reward on a failed episode does not by itself prove reward exploitation: successful episodes can still be preferable, and partial credit can aid learning. The drill was deliberately designed and scripted qualification found successful trajectories; its transfer value remains an experimental question.

## Questions to bring back from the reading

1. What concrete final-game behavior should this lesson teach, and what measurement demonstrates that behavior?
2. Does the learner experience enough successful examples of the complete lesson, rather than only its first step?
3. Does unsampled execution perform the skill, or does it depend on exploratory action samples?
4. Are reward, termination, and promotion criteria consistent about what counts as success?
5. Does a failed safety test indicate regression from the parent, or an existing limitation that belongs to a later lesson?
6. Have the intended training arms actually run before their effectiveness is judged?
7. Does the curriculum improve final-task learning compared with a simpler alternative under comparable total training cost?

## Evaluation reading

Agarwal, Schwarzer, Castro, Courville, and Bellemare, **Deep Reinforcement Learning at the Edge of the Statistical Precipice** (NeurIPS 2021): [paper](https://papers.nips.cc/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf), [author explanation](https://agarwl.github.io/rliable/), [official code](https://github.com/google-research/rliable).

Read Section 4, “Recommendations and Tools for Reliable Evaluation.” Its lesson is to keep uncertainty and evaluation protocols visible when comparing RL methods. For this project, distinguish independent training runs from many episodes of one checkpoint; preserve per-opponent and per-action-mode outcomes. A bootstrap cannot turn one trained checkpoint into independent training replications. This paper is an evaluation reference, not a curriculum algorithm.

## Evidence and limits

- Confidence is high in the quoted local counts and code paths, based on the preserved reports and inspected source. The reports' replay/hash checks were read, not rerun here.
- Confidence is lower in causal explanations. No new policy inference, training, reward change, or controlled comparison was performed for this research.
- No numerical promotion threshold, task-mixture ratio, or replacement architecture is established as optimal by this reading exercise.
- Historical project code is a reference for mechanisms; compatibility with this repository has not been tested.

## Relevance

HIGH: the research question directly concerns the project's movement-to-contact-to-combat transfer and the interpretation of its latest preserved curriculum attempt.
