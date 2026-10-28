# Curriculum Learning Case Studies For Physical Competitive Games

## Research Question

Which game/RL projects best inform a `move -> contact -> hit -> safe-combat` curriculum for a 2D platform-fighting PPO agent?

## Request Type

Comprehensive research: primary papers plus official implementation links, scoped to comparable competitive or physical games.

## Search Path

Paper facts come from arXiv, Science/Science Robotics metadata, official repositories, and project pages. Repo READMEs are used only for code/version status.

## 1. Emergent Complexity via Multi-Agent Competition, Bansal et al. (ICLR 2018)

- **Paper:** https://arxiv.org/abs/1710.03748
- **Official code:** https://github.com/openai/multiagent-competition
- **Verified status:** arXiv lists v1 on 2017-10-10, v3 on 2018-03-14, and ICLR 2018 publication. The repo is archived and uses Python 3.6/Gym 0.9.1/MuJoCo 1.31/TensorFlow 1.1.0, so treat it as reference code.
- **Read:** Sections 4.1, 4.2, 5.3, 5.4, 5.5, Appendix A.

**Paper facts:** OpenAI trained PPO agents in competitive MuJoCo tasks including run-to-goal, you-shall-not-pass, sumo, and kick-and-defend. Section 4.1 adds an early dense "exploration reward" for prerequisite motor behavior, then linearly anneals it toward zero so the sparse competition reward dominates most of training. Section 5.3 reports that keeping dense reward throughout training produced proxy behaviors such as standing or carrying the ball rather than solving the competition. Sections 4.2 and 5.4 study opponent sampling; latest-opponent training was less stable than sampling older opponent checkpoints. Section 5.5 adds gradual world randomization and policy pools to reduce narrow overfitting.

**AI_2 inference:** Treat movement/contact/hit rewards as scaffolding. Promote only when the target event is common enough for the next stage to learn from, then reduce the old reward weight. Training opponents can include current and older checkpoints or a training pool, but reserve separate evaluation opponents and seeds that are never used for policy updates.

## 2. From Motor Control to Team Play in Simulated Humanoid Football, Liu et al. (Science Robotics 2022)

- **Paper:** https://arxiv.org/abs/2105.12196
- **Journal metadata:** https://www.science.org/doi/10.1126/scirobotics.abo0235
- **Verified status:** Science Robotics, Vol. 7, Issue 69, 2022-08-31, DOI `10.1126/scirobotics.abo0235`. I found no official full training-code release.
- **Read:** Sections 3, 3.3, 3.4.2, 7, Appendix B.5, Appendix C.2.

**Paper facts:** DeepMind uses a three-stage training scheme: low-level humanoid movement from motion capture, mid-level football drills, then full multi-agent football with population-based self-play. Section 3.3 trains drills such as follow, dribble, shoot, and kick-to-target; Section 3.4.2 distills drill experts into behavior priors and regularizes the full-game policy toward a mixture of those priors. The paper evaluates against fixed agents and behavior metrics, not only training return. Section 7 reports that sparse full-game reward alone performed poorly, training without drill priors produced degenerate locomotion or ball-control local optima, and adding extra coordination shaping rewards could reduce performance through conflicting incentives.

**AI_2 inference:** Use drills that isolate prerequisites without changing embodiment: stage traversal/recovery, approach spacing, contact, hit-confirm, and safe disengage. Retention gates should be stage-local regression checks: a safe-combat checkpoint should still pass movement/contact/hit minimums, but the gates verify prerequisites rather than mastery of every earlier lesson under all later-stage conditions.

## 3. Beating the World's Best at Super Smash Bros. with Deep Reinforcement Learning / Phillip, Firoiu et al. (arXiv 2017; MIT artifact)

- **Paper:** https://arxiv.org/abs/1702.06230
- **MIT DSpace record:** https://dspace.mit.edu/entities/publication/bb6fbc26-f70d-4004-862a-3aac3fc7e9ed
- **Original code:** https://github.com/vladfi1/phillip
- **Verified status:** arXiv lists v1 on 2017-02-21 and v3 on 2017-05-08; MIT DSpace lists 2017. The repo says it is inactive and subject to bit rot, so treat it as historical reference.
- **Read:** Sections 2, 4.1, 4.2, 4.3, 4.4, 5.1.

**Paper facts:** Phillip is the closest fighting-game case study. Section 4.1 fixes Captain Falcon on Battlefield to simplify training and compare curves. The reward includes KO/game outcome plus damage dealt and taken, because KOs are sparse and damage is meaningful progress toward them. Section 4.2 moves from the in-game AI to self-play against old versions after fixed opponents produced weak or exploitable agents. It also reports brittleness from narrow self-play: a human found a degenerate edge-crouch weakness, and the trained agent performed worse when character or stage changed. Section 4.3 uses multiple character-specific agents as opponents and benchmarks; Section 5.1 discusses nonstationary-opponent difficulty.

**AI_2 inference:** Start narrow enough that the agent actually lands hits, then widen stage starts, stage geometry, weapons, and opponent styles after hit/safe-exit events are frequent. Use training opponents for learning and a separate benchmark set for evaluation. Damage/contact rewards can be useful, but they should remain aligned with eventual KO/survival outcomes.

## Cross-Project Takeaway

These projects suggest a useful design question: does each practice setup expose behavior that helps the final game? Early rewards, drills, fixed starts, and fixed opponents can serve different purposes. The appropriate transition or transfer mechanism needs evaluation; these studies do not establish one required success rate or schedule for AI_2.

For AI_2, measure movement, contact, hits, and safe completion separately; distinguish prerequisite regression from future mastery; and reserve evaluation opponents and scenarios outside the training sampler. If return improves without the intended behavior improving, inspect the reward components before interpreting that as successful skill learning.
