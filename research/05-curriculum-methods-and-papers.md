# Curriculum Learning Methods and Failure Modes

## Research Question

Which curriculum-learning papers should we read to understand why an RL curriculum can appear to work on proxy tasks but fail after transfer into combat, and how should we schedule tasks so PPO sees learnable examples of the actual behavior we care about?

Scope: curriculum foundations, task selection, start-state/level scheduling, and reward-shaping invariance for RL. Project-specific interpretations rely on the inspected local findings in the [main guide](04-curriculum-learning-reading-guide.md). The latest preserved contact evaluation recorded zero hit episodes without sampling, 20/20 with sampling, and 7/20 sampled safe completions. Only attack calibration ran; the planned joint-training comparison did not start.

Discovery note: Semantic Scholar API discovery was attempted first as requested, but the API returned HTTP 429 rate-limit responses. I continued with primary publisher pages, arXiv, author project pages, and author/official GitHub repositories.

## Key Findings

### 1. Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey (Narvekar et al., JMLR 2020)

- **Source:** JMLR 21(181):1-50, 2020. Authors: Sanmit Narvekar, Bei Peng, Matteo Leonetti, Jivko Sinapov, Matthew E. Taylor, Peter Stone.
- **URLs:** [JMLR page](https://www.jmlr.org/papers/v21/20-212.html), [PDF](https://www.jmlr.org/papers/volume21/20-212/20-212.pdf).
- **Read:** Sections 3.1 "Curricula," 3.2 "Curriculum Learning," 3.3 "Evaluating Curricula," 4.1 "Task Generation," 4.2 "Sequencing," and 6 "Open Questions."
- **What it establishes:** A curriculum is not merely a linear list of phases. Narvekar et al. define curricula as graphs over task/sample experience and split curriculum learning into task generation, sequencing, and transfer learning.
- **Useful result:** Evaluation should compare target-task learning with and without the curriculum, accounting for the cost of intermediate training. The survey also emphasizes that intermediate tasks need to be relevant to the final task and avoid negative transfer.
- **Limit:** It is a survey and framework, not a recipe for PPO combat training.
- **Relevance:** Highest-priority foundation. It gives the vocabulary to separate three different questions that often get mixed together: "Are our intermediate tasks good?", "Are we switching/sampling them well?", and "Is checkpoint transfer preserving the behavior we need?"
- **Code:** None expected; survey paper.

### 2. Reverse Curriculum Generation for Reinforcement Learning (Florensa et al., CoRL 2017)

- **Source:** Proceedings of the 1st Annual Conference on Robot Learning, PMLR 78:482-495, 2017. Authors: Carlos Florensa, David Held, Markus Wulfmeier, Michael Zhang, Pieter Abbeel.
- **URLs:** [PMLR page](https://proceedings.mlr.press/v78/florensa17a.html), [PDF](http://proceedings.mlr.press/v78/florensa17a/florensa17a.pdf), [author project page](https://sites.google.com/view/reversecurriculum), [project code link advertised by authors](https://sites.google.com/view/reversecurriculum).
- **Read:** Assumptions for reverse curriculum generation (p. 4), Methodology and Algorithm 1 (pp. 4-5), "Policy Optimization with modified start state distribution" (p. 5), "Effect of good starts" (p. 7), and Conclusions (p. 8).
- **What it establishes:** In sparse-goal RL, train from states where the agent succeeds sometimes but not always, then expand outward. Their "good starts" criterion is explicitly performance-gated by success probability bounds.
- **Useful result:** The ablation shows that sampling around "good starts" is better than expanding from all previous starts, because the curriculum stays near a learnable frontier.
- **Limit:** Assumes reset access to useful start states and at least one successful goal state. It is most natural for goal-reaching/navigation, less direct for adversarial combat unless you can create combat states near successful hits.
- **Relevance:** Very high for fixing proxy curricula. If a phase teaches "approach a frozen target" but not "enter a state where an actual hit sometimes happens," the curriculum may graduate the wrong skill. For combat, the analogous task is likely start-state generation around near-hit, punish, recovery, spacing, and safety states, not just easier maps or easier opponents.
- **Code:** The paper's bit.ly code link resolves to the project page; the page advertises code, but the crawler did not expose a direct GitHub URL.

### 3. Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping (Ng, Harada, Russell, ICML 1999)

- **Source:** International Conference on Machine Learning, 1999. Authors: Andrew Y. Ng, Daishi Harada, Stuart J. Russell.
- **URLs:** [author-hosted PDF](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf), [ACM record](https://dl.acm.org/doi/10.5555/645528.657613).
- **Read:** Theorem 1 and surrounding discussion (pp. 4-5), Section 4 examples/potential construction (pp. 7-8), Section 5 "Discussion and Conclusions" (pp. 8-9).
- **What it establishes:** Potential-based shaping is the classic sufficient and, under their assumptions, necessary form of reward shaping that preserves optimal policies. Arbitrary dense rewards can change what policy is optimal.
- **Useful result:** Reward shaping can speed learning, but if the shaping reward is not tied to a potential difference, it may train a policy that is good under the proxy reward and bad under the true objective.
- **Limit:** This is an optimal-policy result for a fixed MDP under stated assumptions, not a guarantee of PPO convergence or successful transfer when tasks, opponents, or terminations change. A non-potential reward is not automatically harmful; it lacks this general invariance guarantee.
- **Relevance:** Very high. If approach reward, safety reward, hit reward, and win reward are reweighted across curriculum phases, success in an early phase may not mean the checkpoint is close to the later objective. This is an evidence-backed concern about objective mismatch, not a proof that any local reward is wrong.
- **Code:** None.

### 4. Prioritized Level Replay (Jiang, Grefenstette, Rocktaschel, ICML 2021)

- **Source:** Proceedings of the 38th International Conference on Machine Learning, PMLR 139:4940-4950, 2021. Authors: Minqi Jiang, Edward Grefenstette, Tim Rocktaschel.
- **URLs:** [PMLR page](https://proceedings.mlr.press/v139/jiang21b.html), [PDF](http://proceedings.mlr.press/v139/jiang21b/jiang21b.pdf), [official code](https://github.com/facebookresearch/level-replay).
- **Read:** Section 3 "Prioritized Level Replay" (pp. 3-5), Algorithm 1 and 2 (pp. 3-4), Section 3.1 "Scoring Levels for Learning Potential" (p. 4), Section 3.2 "Staleness-Aware Prioritization" (p. 5), Section 5 "Results and Discussion" (p. 6).
- **What it establishes:** In procedurally generated environments, the next level should depend on the current policy. PLR samples previously seen levels by estimated learning potential, using TD-error/GAE-style scores and a staleness term so stale levels get revisited.
- **Useful result:** Gains depend on both score-based prioritization and staleness-based revisits; the paper reports that omitting staleness can produce inconsistent results.
- **Limit:** Requires replayable, identifiable levels; it does not generate tasks itself. In its on-policy setting, replay means collecting fresh trajectories on a revisited level, not reusing old transitions in ordinary PPO. Prediction error is an estimate of learning potential, not proof that a difficult or noisy level is learnable.
- **Relevance:** High if AI_2 can revisit scenario seeds or state templates. Compare selective revisiting against a simpler fixed mixture before adding a scheduler.
- **Code:** `facebookresearch/level-replay` is the official code link advertised by the paper. GitHub metadata checked 2026-09-13: public Python repo, archived, last push 2021-06-11.

## Read Next If Needed

### Teacher-Student Curriculum Learning (Matiisen, Oliver, Cohen, Schulman; 2017 preprint, IEEE TNNLS 2020)

- **URLs:** [arXiv page](https://arxiv.org/abs/1707.00183), [PDF](https://arxiv.org/pdf/1707.00183), [journal DOI](https://doi.org/10.1109/TNNLS.2019.2934906).
- **Read:** Section 3 "Algorithms" (pp. 3-4), Appendix A algorithms (p. 11), Minecraft setup and training details (pp. 6 and 12), Conclusion (p. 10).
- **Why optional:** Read this when task scheduling becomes the main question. The teacher uses estimated learning progress to select tasks, including revisiting deteriorating skills. The journal version appeared in IEEE TNNLS 31(9):3732–3740 in 2020, following online publication in 2019.
- **Use when:** You already have a fixed list of curriculum tasks and need a small teacher/scheduler before building a continuous parameterized task space.

### Teacher Algorithms for Curriculum Learning of Deep RL in Continuously Parameterized Environments / ALP-GMM (Portelas et al., CoRL 2019 / PMLR 2020)

- **URLs:** [PMLR page](https://proceedings.mlr.press/v100/portelas20a.html), [PDF](http://proceedings.mlr.press/v100/portelas20a/portelas20a.pdf), [official code](https://github.com/flowersteam/teachDeepRL).
- **Read:** Section 4.1 and Algorithm 1 (pp. 3-4), Discussion and Conclusion (p. 8).
- **Why optional:** This is the best next paper if the curriculum exposes continuous knobs such as target speed, starting distance, opponent handicap, hit timing window, or platform layout. It samples task parameters by absolute learning progress, with random exploration retained.

### Evolving Curricula with Regret-Based Environment Design / ACCEL (Parker-Holder et al., ICML 2022)

- **URLs:** [PMLR page](https://proceedings.mlr.press/v162/parker-holder22a.html), [PDF](https://proceedings.mlr.press/v162/parker-holder22a/parker-holder22a.pdf), [interactive paper](https://accelagent.github.io), [official code](https://github.com/facebookresearch/dcd).
- **Read:** Section 3 "ACCEL" (pp. 4-5), Algorithm 1 (p. 5), Section 4.4 "Discussion and Limitations" (p. 9).
- **Why optional:** It is a stronger environment-design paper than an immediate first fix. It becomes relevant if AI_2 grows toward procedural scenario generation and wants tasks at the "simplest unsolved" frontier.

## Standalone Synthesis

The strongest through-line is that a useful RL curriculum is not just "train on easy task, then harder task." It is a mechanism for keeping the learner near examples where it can get the real training signal: states, tasks, or levels where the current policy sometimes succeeds, sometimes fails, and can improve.

Narvekar et al. give the map: curriculum learning has task generation, sequencing, and transfer. A failure in any one of these can look like "curriculum learning is broken." Florensa et al. make the key operational rule concrete with "good starts": train around states with intermediate success probability, then expand. Jiang et al. turns that into a replay rule: revisit levels with learning potential and stale estimates. Ng et al. add the reward-design warning: if intermediate reward weights change the objective, a checkpoint can optimize the proxy and still be poorly aligned with hits, safety, or winning.

For AI_2, calibration did not demonstrate navigation forgetting: all matched executed trajectories were unchanged. The later joint-training arms never ran, so their effectiveness is unknown. The observed attack-selection gap and incomplete safe completions motivate questions about useful practice and rewards, but do not establish that the full curriculum failed. The saved report identifies a separate gate-scope error: an existing later-stage navigation weakness was used to stop earlier training.

Use the papers to frame controlled comparisons, rather than assuming a redesign is already justified:

- Compare current resets with starts from which the learner sometimes completes the intended skill.
- Evaluate the lesson's actual behavior in the intended execution mode. Separate parent-relative retention from future mastery requirements; do not demand a later skill before permitting training of the earlier one.
- Measure whether maintenance practice helps retention; the existing contact plan already includes navigation maintenance.
- Audit reward and termination incentives against the lesson and final-game objectives.
- Compare adaptive sampling with a fixed mixture under a comparable total training budget.

## Relevance

HIGH. The papers directly address sparse rewards, fixed-versus-adaptive curricula, task/frontier selection, reward-shaping mismatch, and replaying stale or high-learning-potential tasks. They do not prove any specific bug in AI_2; they provide the concepts and methods needed to test whether phase gates, reward proxies, and scenario sampling are training the intended combat behavior.

## Evidence Vs. Inference

- **Published evidence:** Narvekar defines curriculum components and evaluation criteria; Florensa uses success-probability-bounded start states; Ng et al. prove potential-based shaping preserves optimal policies under their assumptions; Jiang samples levels by learning potential plus staleness. Portelas/ALP-GMM and ACCEL are included as optional extensions.
- **Project-specific inference:** Approach-without-hit behavior suggests proxy success may not match the target combat objective. Reward-weight changes may create objective mismatch. These are plausible hypotheses to test, not conclusions from the papers.

## Recommended Reading Order

1. Narvekar et al. 2020, Sections 3.1-3.3 and 4.1-4.2, to name the problem correctly.
2. Florensa et al. 2017, Methodology and "good starts," to design learnable near-success combat starts.
3. Ng et al. 1999, Theorem 1 and Discussion, to understand when shaped rewards change the target.
4. Jiang et al. 2021, PLR, if the curriculum can replay seeds/states/levels and score learning potential.
5. Matiisen et al. 2017, TSCL, if you want the simplest fixed-task teacher scheduler.

## Proposed methodology note

No `METHODOLOGY.md` exists in this checkout. A useful evaluation principle is to measure final-task behavior and total time to a defined target, while testing whether a more elaborate curriculum improves on a simpler schedule.
