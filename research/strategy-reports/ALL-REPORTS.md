AI_2 TRAINING STRATEGY SERIES - combined plain-text edition for feeding to AI tools
Generated 2026-09-15 from research/strategy-reports/*.html. Facts verified against primary sources; 'reported' marks single-source numbers.



================================================================================
FILE: index.html
================================================================================

AI_2 Training Strategy Series

-

AI_2 · UTMIST platform fighter · reinforcement learning

# Training Strategy Series

Eight reports, researched and written 2026-09-14/15. Sources: the original papers and blogs of each system, verified by web research, plus the repo's own curriculum research (research/04–06) and preserved September baselines.

The question behind this series: AI_2's fighter is not improving the way we want, so are the curriculum and training strategy wrong? These reports work through how the landmark game-playing systems were actually trained, one design decision at a time, and end with a concrete, ordered experiment plan for this project. Each report assumes no prior knowledge and defines every term it uses.

## Read in this order

Report 1
AlphaGo to KataGo
Self-play from zero to superhuman: the loop, the 55% promotion gate, the data window, and KataGo's cheap-training tricks. What survives the trip from turn-based Go to a 30fps fighter: auto-curriculum, gating, auxiliary prediction targets.

Report 2
OpenAI Five and Hide-and-Seek
PPO, your algorithm, at world-champion scale: the 80/20 opponent mix, reward annealing, discrete action heads, surgery. Hide-and-seek shows what pure competition discovers and what it costs, and ADR shows how to automate difficulty. Includes the scale reality check for one GPU.

Report 3
AlphaStar and League Training
Why plain self-play chases its tail in games with style cycles, the spinning-top picture, the league of main agents and exploiters, PFSP opponent weighting, and the minimal league-lite a 1v1 fighter needs.

Report 4
Fighting-Game AI Case Studies
Seven systems that fight in real time: Blade & Soul's pro-beating recipe and time-penalty aggression dial, FightLadder's benchmark, Phillip and slippi-ai for Smash, GT Sophy, JueWu. Ends with the consensus table, and the consensus is unanimous about discrete actions.

Report 5
Self-Play by Design
The theory that explains the practice: Nash, best response, exploitability, why training win rate lies. The algorithm ladder from naive self-play to PSRO, and a source-by-source audit of AI_2's current opponent mix and 7-checkpoint pool.

Report 6
Curriculum Engineering
The operational layer on top of the repo's earlier curriculum research: how real systems ran promotion gates, faded training-wheel rewards, and carried skills across stages without forgetting. Maps directly onto phases 0a–0e and 1–4.

Report 7 · the measured blocker
The Attack-Button Problem
Why the agent hits 20/20 when sampling actions and 0/20 when acting deterministically: the Gaussian mean sits at 0.03, the button fires at 0.5, and nothing in the training signal moves the mean. Mechanism, picture, and six fixes ranked by evidence.

Report 8 · the plan
The AI_2 Playbook
The synthesis: the curriculum's shape is right; four mechanics are the likely performance sinks. An ordered experiment ladder, E0 reward-wiring audit through E7 auxiliary heads, each with cost, success metric, and kill criterion.

If you only read two
Read report 7, because it explains the one failure the project has actually measured, then report 8, because it turns all six evidence reports into the next month of experiments. The others are the supporting evidence, in teaching order.

## Glossary

Policy
The trained function that maps what the agent sees to what it does. "The model" and "the policy" are the same thing here.

Checkpoint
A saved snapshot of the policy's weights at some moment in training.

Self-play
Training where the opponent is the agent itself, live or as a past checkpoint.

Opponent pool
The set of past checkpoints and bots that training opponents are drawn from.

Curriculum
Any deliberate ordering of practice: easier tasks first, rewards that guide early learning, opponents that get harder.

Reward shaping
Extra rewards added to guide learning (distance, damage) beyond the real goal (winning). Helpful early, distorting if left in place.

Deterministic vs. sampled
Two ways to act: take the policy's single most-preferred action, or draw randomly from its preferences. They can behave very differently (report 7).

Entropy
A measure of how random the policy is. Training adds a small bonus for staying random so exploration does not die out.

Best response
The most effective counter-strategy against one specific opponent.

Nash equilibrium
Strategies with no profitable counter. The theoretical end state of self-play.

Exploitability
How badly the best counter-strategy beats you. The honest score of a self-play agent, measured by training a fresh opponent against your frozen policy.

Non-transitive
A game where styles beat each other in circles, like rock-paper-scissors. Fighting games are; Go mostly is not.

PFSP
Prioritized fictitious self-play: pick pool opponents with probability weighted toward the ones you currently lose to.

League
AlphaStar's structure: a main agent, plus exploiter agents whose only job is to find its weaknesses, plus a growing pool of frozen snapshots.

Kickstarting / distillation
Teaching one network to imitate another, used to carry skills between training stages or from demonstrations.

Transition
One step of experience for one player: what it saw, did, and the reward that followed. The unit all training budgets are measured in.

## Provenance and honesty rules

Facts about external systems were gathered 2026-09-14/15 by six research passes over the primary papers, official blogs, and repositories, and cross-checked against the repo's existing research notes. Numbers that could only be confirmed in one secondary source are marked "reported" in the text. Every claim about AI_2 itself cites a preserved baseline or config file. No statement in this series is a new measured AI_2 result: the audits and experiment designs in reports 7 and 8 are hypotheses with proposed tests, in keeping with the project's evidence discipline. Local reading copies live at research/strategy-reports/ in the main AI_2 checkout; the earlier foundations are research/04, research/05, and research/06.


================================================================================
FILE: 01-alphago-lineage.html
================================================================================

AlphaGo to KataGo

-

AI_2 · Training Strategy Series · Report 1 of 8

# AlphaGo to KataGo

Researched 2026-09-15 against the original papers and official pages. Numbers that come from a single secondary source are marked "reported".

The AlphaGo family is the cleanest demonstration that an agent playing against itself can climb from random play to superhuman skill. The search machinery does not fit a real-time fighting game, but four of its training ideas do, and one late descendant, KataGo, is the best practical recipe for cheap self-play training ever published.

## The family in one table

|
| System | Year | Human data? | Self-play | Promotion gate | Big idea

| AlphaGo | 2016 | Yes, 30M positions | Against random past checkpoints | None formal | Bootstrap from humans, then improve by self-play

| AlphaGo Zero | 2017 | None | Best checkpoint generates all games | Beat champion 55% over 400 games | Search is the teacher; the network imitates its own search

| AlphaZero | 2018 | None | Latest network, always | Removed | Same recipe works for Go, chess, and shogi

| MuZero | 2020 | None | Same as AlphaZero | None | Learns its own model of the game; no rules needed

| KataGo | 2019+ | None | Latest network | None | Same strength for roughly 50x less compute

Terms used on this page. A policy is the function that picks moves. A checkpoint is a saved copy of the network at some point in training. Self-play means the agent's training opponent is a copy of itself. MCTS (Monte Carlo Tree Search) simulates many possible futures before committing to a move, like a chess player reading out lines.

## AlphaGo: why they started from human games

Go gives one reward, win or lose, after roughly 200 moves. An agent starting from random play almost never wins for a meaningful reason, so there is nothing to learn from. AlphaGo solved this cold-start problem by copying humans first.

- Supervised policy. A network learned to predict the next move of strong amateur players from 30 million board positions. It reached about 57% prediction accuracy. This is imitation, not reinforcement learning.

- Self-play policy. That network then played games against randomly selected older copies of itself and was updated by policy gradient toward moves that led to wins. Playing against a pool of past versions, rather than only the newest one, was already there in 2016 to keep the opponent varied.

- Value network. A separate network learned to look at a position and guess who wins, trained on 30 million positions from the self-play games.

- Search at match time. MCTS combined all of the above to pick actual moves. It beat Lee Sedol 4 to 1 in March 2016.

The lesson that survives: when the useful event is too rare for a fresh agent to ever produce, seed the policy with imitation and let reinforcement learning refine it. AlphaStar, and the modern Smash Bros. agent slippi-ai, both reuse this move (reports 3 and 4).

## AlphaGo Zero: the loop that needs no humans

A year later the same team removed the human data entirely. AlphaGo Zero starts from random weights and runs one loop:

- The current best network plays games against itself. For each move it runs MCTS with 1,600 simulations. Search produces a better move choice than the raw network would make alone.

- Every position, the search's move preferences, and the final game result go into a buffer holding the most recent 500,000 games. Older games are thrown away, because they were played by a weaker agent and would teach stale lessons.

- The network trains to match two targets at once: predict the search's move preferences (policy head) and predict the game winner (value head). One network, two output heads.

- The gate. Every so often the freshly trained candidate plays 400 games against the current best network. Only if it wins at least 55% does it become the new best and take over generating self-play games. If it loses, training continues and the old best keeps its job.

Two exploration details matter. For the first 30 moves of every self-play game, moves are sampled in proportion to search visit counts rather than played greedily, so openings stay varied. And random Dirichlet noise is mixed into the search's starting preferences, so every legal move gets tried occasionally.

The core idea, in plain terms
Search acts as a teacher. On every move, thinking ahead produces a slightly better decision than the network's instinct. Training makes the instinct match the thought-out decision. Next iteration, search starts from a better instinct and produces even better decisions. The agent bootstraps itself upward with no outside teacher.

This exact mechanism does not transfer to AI_2, because at 30 frames per second there is no time to simulate futures before each button press, and your PPO setup has no search component. What transfers is everything around it: the data window, the gate, the sampled-vs-greedy distinction, and injected exploration noise.

## AlphaZero: they removed the gate

AlphaZero simplified the loop: no evaluation arena, no "best" copy. The single latest network generates all self-play games and is updated continuously. The paper states this directly, and the same recipe reached superhuman play in Go, chess, and shogi within hours on thousands of TPUs. Search dropped to 800 simulations per move, and the exploration noise was scaled to each game's move count.

What to take from the gate story
The field's two strongest systems disagree about gating: AlphaGo Zero gated at 55%, AlphaZero removed the gate and was fine. KataGo also runs gateless. The honest reading is that gating buys safety against a broken checkpoint poisoning your training data, at the cost of machinery and delay. For AI_2 the analogous decision is which checkpoints are allowed into the self-play opponent pool. A cheap arena check before pool admission, some fixed win-rate over a few hundred games against a pool sample, is the AlphaGo Zero position. Admitting every 500k-step snapshot blindly, as the current config does, is the AlphaZero position. Both are defensible; the difference only matters if a bad snapshot can enter the pool.

## MuZero, in one paragraph

MuZero kept the whole AlphaZero loop but replaced the real game rules inside the search with a small learned model: one function encodes the observation into a hidden state, another predicts what the hidden state becomes after an action, a third predicts policy, value, and reward from it. Search runs inside the learned model. This mattered because it extended the recipe to Atari, where the rules are not available as code you can query. It does not change the training-strategy picture for AI_2.

## KataGo: the budget version, and the most useful one here

KataGo is an open-source AlphaZero-style Go program by David Wu. Its claim to fame is efficiency: it surpassed ELF OpenGo, a system trained with roughly 74 GPU-years, using about 1.4 GPU-years. The paper ablates where the roughly 50x saving came from, and the biggest items are training-signal ideas, not engineering.

|
| Technique | What it is | Reported gain

| Auxiliary targets | Besides win/lose, the network also predicts which player owns each board point and the final score margin | ~1.65x

| Global pooling architecture | Lets the net see whole-board context cheaply | ~1.6x

| Game-specific input features | Hand-built features like liberties and ladders | ~1.55x

| Playout cap randomization | 25% of moves get deep search (600 simulations), 75% get shallow (100) | ~1.37x

| Auxiliary policy target | Also predict the opponent's next move | ~1.3x

Why auxiliary targets work. Win/lose is one bit of signal per game. Ownership and score are hundreds of numbers per position, and they explain why a position is winning. The gradient from these extra predictions shapes the same shared network body that the policy uses, so the policy learns from a far denser signal at almost no extra cost. This is the single most transferable idea in the lineage.

The built-in curriculum. KataGo trains on a mix of board sizes at once, weighted toward small boards early (reported: 19x19 starts at ~37.5% of games and rises to ~50%). Small boards give short, cheap, informative games; the full board dominates later. It also randomizes komi (the scoring handicap) and sometimes starts games from handicap positions. Difficulty and variety are baked into the data distribution rather than scheduled in stages.

## What transfers to AI_2, and what does not

|
| Does not transfer | Why

| MCTS at decision time | At 30 fps there is no budget to simulate futures per button press; your policy must answer in one forward pass

| Search-as-teacher training | The whole improvement operator assumes search; PPO improves by gradient on advantages instead

| Perfect information, turn-based structure | The fighting game is simultaneous-move and continuous-time; nothing in AlphaZero addresses reading a live opponent

|
| Transfers | Concrete AI_2 form

| Self-play as automatic curriculum | Your opponent pool grows with your agent; this is already partly in place and is the subject of report 5

| Data/pool freshness window | They discard games from >500k-games ago; your pool keeps 7 snapshots. The principle: old data teaches stale lessons, but some depth is needed for variety. Report 5 covers pool depth

| Arena gating | Optional quality check before a checkpoint joins the opponent pool

| Auxiliary prediction targets | Add heads predicting near-future damage dealt/taken, KO within N steps, or the opponent's next action, trained alongside PPO's value loss. Hypothesis for AI_2, backed by KataGo's ~1.65x ablation

| Sampled vs. greedy action selection as separate modes | AlphaGo Zero explicitly samples early moves and plays greedily later. Your project's deterministic-vs-sampled attack gap (report 7) is the same distinction, currently unmanaged

| Domain randomization as curriculum | Board sizes and komi map to your spawn distances, stage geometry, and target counts (phases 0a–0e knobs)

## Since 2022

Two continuations worth knowing about, neither urgent for AI_2. Gumbel MuZero (2022) fixed search so the loop still improves with tiny simulation budgets, as low as 2 to 16 simulations per move, which matters when compute is scarce. Student of Games (2023) merged the AlphaZero line with counterfactual regret minimization so one algorithm handles both perfect-information games and hidden-information games like poker; it is the pointer to follow if you ever care about opponent-mind-reading formally.

## Primary sources

- Silver et al., Mastering the game of Go with deep neural networks and tree search, Nature 2016

- Silver et al., Mastering the game of Go without human knowledge, Nature 2017 — nature.com/articles/nature24270

- Silver et al., AlphaZero, Science 2018 — arXiv:1712.01815

- Schrittwieser et al., MuZero, Nature 2020

- Wu, Accelerating Self-Play Learning in Go (KataGo) — arXiv:1902.10565

- Danihelka et al., Policy improvement by planning with Gumbel, ICLR 2022

- Schmid et al., Student of Games, Science Advances 2023 — doi:10.1126/sciadv.adg3256

Series index
Next: OpenAI Five and Hide-and-Seek →


================================================================================
FILE: 02-openai-agents.html
================================================================================

OpenAI Five and Hide-and-Seek

-

AI_2 · Training Strategy Series · Report 2 of 8

# OpenAI Five and Hide-and-Seek

Researched 2026-09-15 against arXiv:1912.06680, arXiv:1909.07528, arXiv:1910.07113, arXiv:2206.11795 and the OpenAI blog. Single-source numbers marked "reported".

These are the most relevant systems in this series for one blunt reason: they used your algorithm. OpenAI Five beat the Dota 2 world champions with PPO and GAE, the same method Stable-Baselines3 implements. Everything they did is a scaled-up version of choices you face in config.yaml: how to mix opponents, how to shape rewards, how to anneal exploration, and how to keep training a policy while the game around it changes.

## OpenAI Five: PPO at planetary scale

OpenAI Five played 5-versus-5 Dota 2, one LSTM-based policy shared by five heroes. It trained continuously for about ten months, consuming roughly 180 years of gameplay per day across hundreds of GPUs and tens of thousands of CPU cores. In April 2019 it beat OG, the reigning world champions, 2 to 0, then won 99.4% of 7,215 public games.

### The settings that map onto yours

|
| Knob | OpenAI Five | AI_2 today

| Algorithm | PPO + GAE | PPO + GAE (SB3)

| Discount gamma | 0.9993 (~3-minute horizon at their tick rate) | 0.99

| GAE lambda | 0.95 | 0.95

| Entropy coefficient | 0.01 early, annealed to 0.001 late | 0.01 fixed (0.05 in navigation)

| Batch per optimizer step | on the order of a million transitions | 8,192 (16,384 in the native benchmark)

| Memory | LSTM, 4096 units | Frame stack of 4 + opponent-action history buffer

| Action space | Discrete heads with masking (see below) | 10-dim continuous vector with 0.5 button thresholds

| Reaction time | 217 ms average | every step (33 ms)

### Discrete action heads, not one big output

Dota's raw action space is astronomically large. Instead of one output covering every possibility, the network has separate heads: one picks the action type, one picks the delay, one picks the target unit, one picks a map position. Invalid combinations are masked out before sampling, so the policy never wastes probability on illegal moves. Every serious game-playing system in this series factored its actions into discrete heads like this. Your continuous 10-dim vector with a 0.5 threshold per button is the outlier design, and report 7 traces your measured attack-initiation failure directly to it.

### The opponent mix: 80% latest, 20% past

Each training game, the opponent was the current newest policy 80% of the time and a past checkpoint 20% of the time. Past checkpoints were not sampled uniformly: each pool member carried a quality score, updated from results, so opponents that still posed a challenge were picked more often, and pushovers faded out. The paper describes this as a lightweight relative of fictitious play. Two things to notice against your config: their self-play share was 80%, yours peaks at 32% in phase 4; and their pool sampling was difficulty-weighted, yours is uniform over 7 snapshots. Report 5 works through what the literature supports changing.

### Reward shaping, and how they weaned the agent off it

Five never trained on win/loss alone. It got dense shaped rewards: experience, gold, health, last-hits, kills, tower damage, minus deaths. Two mechanisms kept shaping from taking over:

- Team spirit annealing. Each hero's reward was blended with the team average: reward_i = tau * mean(team) + (1 - tau) * own_i. Tau started near 0.3, so early heroes learned selfish basics, and was annealed toward 1.0, so late training optimized pure team outcome. This is a reward schedule, the same species as your phase-by-phase reward weight changes, but continuous instead of stepped.

- Time decay (reported). Shaped rewards were scaled down as the game clock advanced, concentrating shaping on the early game where it is least likely to distort the outcome.

Their own caveat
The paper says the restrictions and shaping were "not introduced gradually in an effort to build a perfect curriculum," but added incrementally as ordinary engineering. The lesson is not that OpenAI found the optimal curriculum. It is that a decent shaped-reward schedule plus overwhelming self-play was enough, and they spent their care on keeping training continuous instead.

### Surgery: never restart training

Dota patched itself every couple of weeks, changing the game under the agent. Instead of retraining, OpenAI applied "surgery": offline transformations that map the old trained weights into a new architecture or observation layout so the new network behaves identically at first, then training resumes. Over ten months they did this roughly every two weeks. Your project already lives this way, resuming checkpoints across phases, and the fullgame port kept checkpoint compatibility for the same reason. The transferable discipline is: treat trained weights as capital, and make every environment change checkpoint-preserving.

## Hide-and-seek: curriculum for free, at a price

The 2019 emergent tool use paper is the purest demonstration of an autocurriculum: put two teams in competition, give only +1/-1 for the hide-and-seek outcome, and every time one side learns a trick, the other side faces a new problem. No phases, no reward schedule. Six strategy generations emerged on their own:

|
| Phase | Behavior | Onset (episodes, approx.)

| 1 | Chasing and running | 0

| 2 | Hiders build box forts | ~25M

| 3 | Seekers use ramps to breach forts | ~100M

| 4 | Hiders lock the ramps away first | ~110M

| 5 | Seekers "box surf" over walls | ~380M

| 6 | Hiders lock every box in advance | ~400M+

Read the right-hand column carefully. Each new behavior took tens to hundreds of millions of episodes, tens of billions of environment steps, and the paper reports that below very large batch sizes the later phases never emerged at all. Phase 5, box surfing, is also the series' best cautionary tale: the seekers exploited a physics glitch, because competition optimizes against your simulator, not against your intent.

What this means for AI_2's design choice
Your project chose hand-designed phases (navigation drills, then staged combat) over "just let self-play discover everything." At your compute scale that choice is correct, and this paper is the evidence: pure emergence needed 10^10 steps and enormous batches. The staged-drill approach is what DeepMind's humanoid football used at comparable scale (see research/06 in the repo). Keep the curriculum; use self-play as one component inside it, not as the whole plan.

## Automatic Domain Randomization: a thermostat for difficulty

From the Rubik's Cube robot-hand work. ADR is the cleanest published mechanism for automating "when do we make training harder," which your config currently does by hand when you switch phases 0a through 0e.

- Every environment parameter (in your case: spawn distance, fall penalty, time limit, target count) has a range [low, high]. Training samples uniformly inside the ranges. All ranges start at a single easy value.

- A fraction of episodes are boundary probes: one parameter is pinned at its current low or high edge while the others sample normally.

- Probe results accumulate in a buffer per boundary (they used buffers of 240 episodes). When the buffer fills, average performance is compared to two thresholds: above the high threshold, push that boundary outward by a step; below the low threshold, pull it back in.

The design has two properties worth copying. Gates are local: an easy parameter keeps widening even while a hard one is stuck, unlike a phase switch that gates everything on one metric. And difficulty is reversible: if the agent regresses, the range contracts instead of leaving the agent drowning. Your 0a→0e ladder is exactly one hand-run ADR on the distance parameter: 0a is [1,3], 0d is [1,15]. ADR is the same idea running unattended, with a measured widening rule instead of a human deciding when to promote.

## VPT: when to bootstrap from demonstrations

Video PreTraining (2022) trained a Minecraft agent by behavior-cloning 70,000 hours of pseudo-labeled YouTube video, then fine-tuning with RL. It exists because some behaviors, like crafting a diamond pickaxe through 20+ sequential steps, are simply never discovered by exploration from scratch. The decision rule it supports: use imitation when the target behavior has near-zero probability under any exploratory policy. Your attack-initiation problem is a miniature of this. The agent already lands hits when sampling, so full VPT-style pretraining is oversized; but behavior-cloning a handful of its own successful sampled trajectories to move the deterministic policy is the same principle at the right scale, and it appears in the ranked fixes of report 7.

## Scale reality check

|
| System | Approximate experience | Hardware

| OpenAI Five | ~10^13 frames | hundreds of GPUs, tens of thousands of CPU cores, 10 months

| Hide-and-seek, phase 6 | ~10^10–10^11 frames | large Rapid cluster

| Hide-and-seek, phase 2 | ~10^9–10^10 frames | same, hours to days

| AI_2 native pipeline | ~10^9 per GPU-hour | one RTX 4090 at ~278k transitions/s (raw benchmark conditions)

The right conclusion is not despair. At 278k transitions per second, one billion transitions is about an hour of loop time. That puts you in hide-and-seek phase 1–2 territory per day of training, which is exactly the regime where shaped rewards, drills, and a managed opponent pool, rather than raw emergence, carry the load. It also means opponent-pool experiments and ADR-style range sweeps, which looked expensive under the old 2k-transitions-per-second Python pipeline, now cost afternoon money.

### What transfers at your scale

- Directly: entropy annealing (0.01 → 0.001 across a run), the 80/20 opponent split with difficulty-weighted pool sampling, discrete action heads, reward normalization, checkpoint-preserving environment changes, ADR-style automatic ranges.

- Downscaled: shaped rewards that fade over game time; team spirit does not apply (1v1).

- Does not transfer: counting on late-stage emergent strategy discovery; it needs one to two orders of magnitude more experience than a single GPU delivers in reasonable time.

## Primary sources

- Berner et al., Dota 2 with Large Scale Deep Reinforcement Learning — arXiv:1912.06680

- Baker et al., Emergent Tool Use From Multi-Agent Autocurricula — arXiv:1909.07528

- OpenAI et al., Solving Rubik's Cube with a Robot Hand (ADR) — arXiv:1910.07113

- Baker et al., Video PreTraining — arXiv:2206.11795

- OpenAI Five blog — openai.com/index/openai-five; Emergent tool use blog — openai.com/index/emergent-tool-use

← AlphaGo to KataGo
Series index
Next: AlphaStar and League Training →


================================================================================
FILE: 03-alphastar-and-leagues.html
================================================================================

AlphaStar and League Training

-

AI_2 · Training Strategy Series · Report 3 of 8

# AlphaStar and League Training

Researched 2026-09-15 against the Nature 2019 paper, arXiv:2004.09468, arXiv:2206.15378 and arXiv:2406.02081. Single-source numbers marked "reported".

Fighting games have styles that beat other styles in circles, like rock-paper-scissors. In games shaped like that, plain self-play does not steadily improve; it chases its own tail. AlphaStar is the system that named this problem in public and shipped the industrial fix, the league. This report explains why the cycling happens, how the league works, and what the smallest useful version looks like for a 1v1 fighter.

## The problem: strategy spaces have cycles

Call a game transitive if "A beats B" and "B beats C" together imply "A beats C". In a transitive game, self-play works beautifully: any improvement against yourself is a real improvement against everyone. Go is close to transitive, which is why the AlphaGo line (report 1) never needed anything fancier.

Now suppose your fighter has three viable styles: rushdown pressures zoning, zoning walls out grappling, grappling punishes rushdown. An agent training only against its current self can circle forever: it learns rushdown, its opponent-self is rushdown so it learns grappling to counter, then zoning to counter that, then rushdown again. Every step is an improvement against the current opponent and none of it accumulates. Published self-play surveys describe exactly this signature: checkpoints that beat their immediate predecessor and lose to their own past and future selves.

> Figure: The "spinning top" picture from Czarnecki et al. (2020). Width at each height is how many genuinely different strategies survive at that skill level. The theory result that matters: to keep improving through a level, your training population must cover the strategy cycle at that level. One agent, or a pool of near-identical snapshots of one agent, does not.

The same paper estimates that real games like StarCraft are non-transitive on an enormous scale, and proves that fictitious-play-style training converges only when the population is large enough to cover the cycle at the current level. That is the theoretical justification for everything else on this page.

## AlphaStar's three-part answer

AlphaStar reached Grandmaster on the real StarCraft II ladder, above 99.8% of human players, on all three races. Its training had three pillars.

### 1. Start from humans, stay anchored to humans

It first imitated 971,000 replays of strong players (reported thresholds: above-average ranked players for the bulk, roughly top-grandmaster games for fine-tuning). During the entire RL phase, a penalty kept the policy from drifting too far from the imitation policy (a KL term, a measure of how different two policies are). On top of that, each game was conditioned on a variable z encoding an opening build order sampled from human games, with a bonus for following it. Purpose: keep a diversity of openings alive instead of letting RL collapse everything into one style.

### 2. The league: three kinds of agents with different jobs

> Figure: The league. Only the main agent is meant to become strong. The exploiters exist to be sacrificed: they specialize in beating the current champion or the current pool, their discoveries force the main agent to patch its weaknesses, and then they are reset and start over.

PFSP, prioritized fictitious self-play, is the sampling rule for picking opponents from the pool. Instead of uniform choice, opponent i is drawn with weight proportional to a function of your current win rate p against it. Two published shapes: (1 - p)^x focuses on opponents you lose to; p(1 - p) focuses on even matchups where a game carries the most information. Either way the effect is the same: no training time is wasted farming opponents you already beat 95% of the time, which is exactly what uniform sampling over a stale pool does.

### 3. Results, and what the ablation showed

The paper's ablation, run on a simplified setup, is the key evidence table of this whole subject: plain self-play cycled; fictitious self-play (uniform over all past) improved but stayed exploitable; PFSP beat uniform; and the full league with exploiters was both strongest and least exploitable. Removing the main exploiters left the champion with lasting blind spots.

One caveat for honesty: AlphaStar did not use PPO. It used an off-policy actor-critic (V-trace, plus a self-imitation term called UPGO) because its infrastructure demanded it. The league is a who-do-I-train-against mechanism and does not care which policy-gradient algorithm sits underneath; FightLadder (below) ran a league on PPO.

## The 2022 counterpoint: fix the dynamics instead

DeepNash (Science 2022) reached expert level in Stratego, a huge hidden-information game, with no league and no pool. Its method, Regularized Nash Dynamics, changes the learning rule itself: each round, the reward is modified by a penalty for straying from a reference policy, the learner converges to a fixed point, and the fixed point becomes the next reference. The sequence of fixed points spirals into an equilibrium instead of orbiting it. It won 84% of games against human experts on the Gravon platform (reported). The takeaway for you is not to implement R-NaD; it is that the field has two answers to cycling, populations (AlphaStar) and regularized dynamics (DeepNash), and populations are the practical one at hobby scale.

## League training measured on actual fighting games

FightLadder (ICML 2024) built a benchmark on real fighting games (Street Fighter II and III, Mortal Kombat, King of Fighters '97, Fatal Fury 2) and ran the alternatives head to head, all on PPO (numbers as reported in the paper, Street Fighter II mirror match):

|
| Training scheme | Elo of best agent | Note

| League (AlphaStar-style) | 1682 | strongest, most compute

| PSRO (population + game-theory solver, report 5) | 1517 |

| Plain independent PPO self-play | 1164 | cheapest

| Fictitious self-play, uniform | 1150 |

The humbling part
Every one of those agents, including the league champion, remained massively exploitable: a fresh PPO agent trained specifically against them reached roughly 94% win rate (reported), and human players found a simple defensive-crouch-plus-low-kick pattern that beat all of them. Leagues buy relative strength and robustness against known styles. Nobody has demonstrated an unexploitable fighting-game agent. Set your goal accordingly: robust against a wide suite, not theoretically unbeatable.

## A league-lite for AI_2

Your current setup already has pieces of this: a pool (last 7 snapshots, uniform sampling) and a cast of scripted bots that act like fixed styles. The evidence above supports a specific minimal upgrade path, each step independently testable:

|
| Step | Change | Evidence

| 1 | Replace uniform pool sampling with PFSP weighting, (1 - winrate)^0.5, using win rates you already log per opponent | AlphaStar ablation: PFSP > uniform; costs almost nothing to implement

| 2 | Deepen the pool: keep 20+ snapshots at spaced intervals rather than the last 7 | Spinning-top coverage argument; OpenAI Five and AlphaStar both kept long histories

| 3 | Add one exploiter: periodically train a short-budget fresh agent purely against the current main checkpoint, then add it to the pool and reset it | AlphaStar ablation: exploiters remove lasting blind spots; also doubles as your exploitability meter

| 4 | Keep the scripted bots as an evaluation suite and as a small early-training share, not as the majority training diet late | Phillip's overfitting finding (report 4); FightLadder's exploitability warning cuts both ways

Everything here is a hypothesis for AI_2 until measured; the series playbook (report 8) turns these into concrete experiments with budgets and kill criteria.

## Primary sources

- Vinyals et al., Grandmaster level in StarCraft II using multi-agent reinforcement learning, Nature 2019 — nature.com/articles/s41586-019-1724-z

- Czarnecki et al., Real World Games Look Like Spinning Tops — arXiv:2004.09468

- Perolat et al., Mastering Stratego (DeepNash) — arXiv:2206.15378

- Li et al., FightLadder, ICML 2024 — arXiv:2406.02081

- Survey of self-play methods — arXiv:2408.01072

← OpenAI Five and Hide-and-Seek
Series index
Next: Fighting-Game Case Studies →


================================================================================
FILE: 04-fighting-games.html
================================================================================

Fighting-Game AI Case Studies

-

AI_2 · Training Strategy Series · Report 4 of 8

# Fighting-Game AI Case Studies

Researched 2026-09-15 against arXiv:1904.03821, arXiv:2406.02081, arXiv:2210.10595, arXiv:1702.06230, arXiv:1810.07286, the slippi-ai repository, Nature (GT Sophy) and arXiv:1912.09729. Single-source numbers marked "reported".

This is the closest evidence to your actual problem: systems trained to fight in real time against another player. Across seven of them, the design choices converge hard on a few answers, and the strongest convergence is on the one place AI_2 currently differs from everyone: the action space.

## Blade & Soul: a commercial fighter, beaten pros, published recipe

NCSOFT trained agents for the arena mode of Blade & Soul, a real-time 1v1 fighting game, and fielded them in blind matches against professional players at their 2018 world championship. The paper reports pro-level performance; secondary accounts of the exact match record vary, so treat the win-rate as order-of-magnitude. Three design choices carry the value:

- Discrete everything. The agent picks among 44 skills plus movement and targeting directions, about 144 combinations, once every 0.1 seconds, through separate output heads for skill, movement, and target (an actor-critic LSTM). No continuous outputs, no thresholds.

- The time penalty is the aggression dial. They shipped three distinct fighting styles from one codebase purely by re-weighting rewards. The aggressive style has a large per-step time penalty, so stalling is expensive and the agent is forced to engage. The defensive style weights its own remaining health. The balanced style sits between. Same environment, same algorithm, three personalities.

- A pool of styles, not a pool of snapshots. All three styles train concurrently and serve as opponents for each other. This covers a slice of the rock-paper-scissors space (report 3) deliberately rather than hoping history provides it.

Direct AI_2 hook
Your contact lesson has a per-step cost of 0.001, which over a 300-step episode totals 0.3, small next to the +1 hit and +2 completion rewards. Blade & Soul used the time cost as a primary lever to force engagement. An agent that approaches but never commits to an attack is precisely the behavior a meaningful time penalty attacks. Worth an experiment; also note your scripted bots (edge-guard, weapon-hunter, recovery-punish) already are a small "pool of styles", which is a genuinely good feature of the current setup.

## FightLadder: the benchmark that ran the bake-off

Covered in report 3 for its league results. What matters here is its environment design, because it encodes the community's consensus about fighting-game action spaces. It offers two: a 12-way binary space mirroring the physical joystick and buttons, and a "transformed" categorical space of 8 motions times 6 attacks where each special move is a single action. The authors chose to hard-code special-move motions as single discrete actions rather than ask RL to discover quarter-circle inputs. Abstraction of button sequences into atomic actions is considered part of the environment, not cheating. Its other finding worth repeating: the built-in CPU opponents of these 1990s games still beat most freshly trained agents, and every trained agent stayed exploitable by a simple human-found pattern.

## DIAMBRA: the hobby-scale baseline

DIAMBRA Arena wraps real fighting games (Street Fighter III, Tekken Tag, more) for RL through an emulator, with discrete or multi-discrete action modes only. The default reward is the health-bar difference: damage dealt positive, damage taken negative, no win-only sparse mode by default. Their published PPO baseline learns visible competence, such as withholding attacks when out of range, within about 25 million steps on a desktop (reported). Useful as a sanity anchor: at your throughput, 25M transitions is a few minutes of native-pipeline time. Basic attack competence in a fighting game is not a large-sample problem.

## Phillip and the reaction-time follow-up: Smash Bros. specifically

Phillip (2017) reached pro-competitive play in Super Smash Bros. Melee. The repo notes in your research/06 file already cover its curriculum story: training against the in-game AI produced exploitable agents; self-play against old versions fixed it; narrow training (one character, one stage) made it brittle. Facts to add here:

- Action space: 54 discrete actions, factored as 9 stick positions times a few button states. The GameCube controller is analog hardware, and they still discretized it.

- Reward: damage shaping plus KO. Plus/minus per KO with a smaller weight on damage dealt and taken, because KOs alone are too sparse.

- Reaction time. The original agent acted within about 33 ms, three to six times faster than a human, and was criticized for it. The 2018 follow-up ("At Human Speed") trained with an enforced input delay and found performance degrades sharply past 6 to 10 frames of delay, and that agents must be trained with the delay to cope with it. If AI_2 ever plays humans and fairness matters, delay is a training-time decision, not an evaluation-time patch.

## slippi-ai: the modern Smash agent

The successor project by the same author (vladfi1), actively maintained. Pipeline: behavior-clone a large corpus of human Slippi replay files, then fine-tune with PPO self-play. It has beaten a top-30-ranked professional in an exhibition set (reported: 10-3 in a Fox mirror against Moky) and holds its own against other pros. It is the strongest existing evidence that for a platform fighter at accessible compute, imitation bootstrap plus self-play is the winning formula, echoing AlphaGo and AlphaStar. AI_2 has no human replay corpus, but the same mechanism applies at small scale: behavior-clone your own successful sampled episodes (self-imitation), then continue RL. Report 7 ranks this among the fixes for the attack problem.

## GT Sophy: reward engineering under a deadline

Sony's Gran Turismo agent (Nature 2022) is racing, not fighting, but it is the best documented case of reward design in a shipped competitive game. Its reward summed eight terms: track progress, passing, several collision penalties, off-course, wall contact, and steering-smoothness costs. Two lessons travel:

- Scenario tables. Training episodes were drawn from a curated mixture: alone on track, one opponent nearby, packs of three, packs of seven, each with its own replay buffer so no scenario starved. That is a curriculum expressed as a data mixture, the same pattern as your phase opponent mixes, and it argues for keeping explicit per-scenario coverage rather than one blended stream.

- Reward hacking under pressure. A week before the public event, Sophy was still deliberately ramming opponents because the collision penalties underpriced the racing advantage. Re-weighting the penalty terms fixed it in days. Behavior follows the actual numbers in the reward, not the designer's intent, and re-weighting is cheap once you can measure the behavior.

## JueWu 1v1: action factoring at industrial scale

Tencent's Honor of Kings 1v1 system (the "Solo" line) beat top players in thousands of public matches (reported 99.8%). Relevant mechanics: the action is a factored discrete triple, what to do, whom to target, where to aim, with an action mask removing illegal choices before sampling; rewards are a weighted sum over health, gold, experience, farming, kills, deaths, tower damage (the paper's table lists health weighted highest); and the agent acts at a 133 ms interval, deliberately matched to strong human reaction time. Every one of those choices rhymes with OpenAI Five's, made independently by a different lab. That convergence is the point.

## The consensus table

|
| Question | What the seven systems say

| Discrete or continuous actions? | Discrete, unanimously, for every fighting game. Blade & Soul 144 discrete combos; FightLadder binary or categorical; DIAMBRA discrete/multi-discrete only; Phillip 54 discrete; slippi-ai discrete controller inputs; JueWu factored discrete with masking. The only continuous-control system, GT Sophy, is a driving game where the physical controls really are continuous. AI_2's 10-dim continuous vector with 0.5 thresholds has no precedent in this list.

| Damage-shaped or win-only reward? | Damage shaping everywhere. No system trains on win/loss alone; all use health or damage deltas as the dense signal with win/KO on top. Your phase 2 to 4 damage weights follow the standard pattern.

| Time penalty to force engagement? | Proven decisive in Blade & Soul, present in FightLadder's annealed aggression coefficient, absent elsewhere. Use when the observed failure is passivity, which is AI_2's observed failure.

| Imitation bootstrap or from scratch? | Academic systems go from scratch; the two strongest human-competitive agents in this list against skilled humans, slippi-ai and AlphaStar, both bootstrap from imitation. FightLadder's authors conjecture human data is needed to escape exploitability. At minimum: self-imitation of your own successful episodes is cheap and directionally supported.

| Reaction-time handicap? | Train at full speed for capability work; add delay during training, not just evaluation, if fairness against humans becomes a goal (Phillip follow-up, JueWu's 133 ms).

Reading this against AI_2's symptom
The project's measured blocker, an agent that approaches but never presses attack deterministically, sits at the intersection of the two strongest consensus rows: an unprecedented continuous-threshold action design, and an engagement failure that the one commercial fighter fixed with a time penalty. Report 7 gives the mechanism; report 8 sequences the fixes.

## Primary sources

- Oh et al., Creating Pro-Level AI for a Real-Time Fighting Game Using Deep RL — arXiv:1904.03821

- Li et al., FightLadder — arXiv:2406.02081

- DIAMBRA Arena — arXiv:2210.10595, docs.diambra.ai

- Firoiu et al., Beating the World's Best at Super Smash Bros. — arXiv:1702.06230; At Human Speed — arXiv:1810.07286

- slippi-ai — github.com/vladfi1/slippi-ai

- Wurman et al., GT Sophy, Nature 2022 — nature.com/articles/s41586-021-04357-7

- Ye et al., Mastering Complex Control in MOBA Games — arXiv:1912.09729

← AlphaStar and League Training
Series index
Next: Self-Play by Design →


================================================================================
FILE: 05-self-play-design.html
================================================================================

Self-Play by Design

-

AI_2 · Training Strategy Series · Report 5 of 8

# Self-Play by Design

Researched 2026-09-15 against arXiv:1603.01121, arXiv:1710.03748, arXiv:1711.00832, arXiv:2004.09468, arXiv:2408.01072 and the systems papers from reports 1–4. Single-source numbers marked "reported".

In a two-player zero-sum game, the opponent mix is not a detail of training. It is the curriculum, the difficulty dial, and the thing your reward is measured against, all at once. This report gives you the small amount of theory that explains why, the ladder of algorithms the field built on top of it, and a source-by-source audit of AI_2's current opponent configuration.

## Four ideas, in plain terms

Zero-sum. One player's win is the other's loss. All the reward your agent can get comes from outplaying whoever it trains against. Change the opponent and you have changed the objective.

Best response. The single most effective strategy against one specific opponent. Best responses tend to be narrow specialists: the perfect anti-rushdown policy may be helpless against anything else.

Nash equilibrium. A pair of strategies where neither side can gain by changing alone. In practice: a policy with no cheap counter. This is the theoretical target of all self-play training.

Exploitability. How badly the best possible counter-strategy beats your policy. This is the honest score of a self-play agent. It is measured by actually training a fresh opponent against your frozen policy and seeing how well it does.

The trap every self-play project falls into once
Training win rate tells you almost nothing. When both sides of self-play adapt to each other, the win rate hovers near 50% whether the policies are brilliant or have collapsed into a private meta only they understand. A 2026 study documented a 51.9-point drop in generalization win rate that was completely invisible in the training curves (reported). Progress must be measured against something that does not move: a frozen benchmark suite, a round-robin of past checkpoints scored by Elo, or a trained exploiter probe. AI_2's held-out scripted evaluation panels already follow this principle; keep them sacred and never train on them.

## The algorithm ladder

Each rung fixes the failure mode of the one below it.

|
| Rung | Opponent is... | Fails when

| Naive self-play | the current policy itself | the game has style cycles: training chases its tail (report 3). Works in near-transitive games like Go

| Past-checkpoint sampling (Bansal et al. 2018) | a random past snapshot from a recent window | the window is shallow or sampling is uniform: the pool fills with pushovers and near-clones. Their finding: sampling old checkpoints beat latest-only for stability

| Fictitious self-play (NFSP, 2016) | effectively the average of all past policies | averaging is slow and dilutes; but the theory is why pools work at all: the time-average of self-play converges toward equilibrium in zero-sum games

| PFSP (AlphaStar) | pool members weighted toward the ones you lose to | the pool lacks diversity to begin with; weighting cannot conjure styles that are not there

| PSRO (Lanctot et al. 2017) | a game-theoretically chosen mixture: build a win-rate matrix among all pool members, solve for the equilibrium mixture, train a best response to it, add it to the pool, repeat | cost: the pairwise matrix is N² matches and grows every round. The principled endpoint of the pool idea

| League (AlphaStar) | pool + dedicated exploiter agents whose only job is finding blind spots | compute-hungry; see report 3 for the small version

| R-NaD (DeepNash, 2022) | the current self, but the learning rule itself is regularized to spiral into equilibrium | research-grade; the population-free alternative

## What the big systems actually shipped

|
| System | Recipe

| OpenAI Five | 80% latest self, 20% past pool; pool members carry difficulty scores that steer sampling

| AlphaStar | full league; mains sample by PFSP weight (1 - winrate)^0.5 (reported)

| Bansal et al. | uniform over a recent window of snapshots; explicitly better than latest-only

| KataGo | latest network only; viable because Go is close to transitive and search stabilizes play

| Blade & Soul | a live pool of three deliberately different styles training against each other

The pattern: nobody strong trains majority-scripted, everybody manages a pool, and the two systems facing style-cycling games (AlphaStar, Blade & Soul) engineered diversity into the pool on purpose.

## Scripted opponents: anchor early, poison late

Rule-based bots have a real job at the start of training. Two random agents flailing at each other generate almost no signal; a scripted opponent that walks forward and attacks provides stable, learnable structure, and your curriculum uses constant and random agents this way in the navigation and early combat phases. That use is supported.

The documented failure is leaving them in charge. Phillip's authors found that training against the built-in Smash AI produced an agent that exploited that AI's specific patterns and collapsed against anything else, and they moved to self-play. A deterministic script is a fixed lock; RL manufactures its key and stops there. The general finding repeats across the fighting-game literature: agents trained mostly on rule-based opponents become quasi-open-loop pattern players.

The supportable schedule: scripted share high early, decaying toward a small residual (10–15%) once the agent reliably beats every script, with the freed share going to the self-play pool. Keep a separate, never-trained-on scripted suite for evaluation. AI_2's config already separates these roles in spirit; the shares are the issue, below.

## Audit of AI_2's current opponent configuration

Phase 3 mix from config.yaml: random 0.20, based 0.18, constant 0.08, clockwork 0.06, self-play 0.28, edge-guard 0.06, weapon-hunter 0.06, recovery-punish 0.04, noisy-based 0.04. Phase 4: self-play 0.32, based 0.28, others smaller. Pool: last 7 checkpoints, saved every 500k steps, sampled uniformly. Each row below is a hypothesis for AI_2, grounded in a named source, not a measured AI_2 result.

|
| Setting | Concern | Supported alternative

| Self-play share peaks at 0.32 | 68–72% of late-phase experience comes from scripted bots. Every strong system inverts this ratio: Five at 80% self, AlphaStar at effectively 100% pool. Majority-scripted training risks the Phillip failure at exactly the phase where opponent quality matters most | Ramp self-play to 0.6–0.8 across phases 3–4 as scripts become reliably beaten; retain ~0.1–0.15 scripted as anchor (OpenAI Five; AlphaStar; Phillip)

| Pool of 7 recent snapshots | Seven consecutive snapshots of one training run are near-clones at one skill level: a thin slice of the spinning top. Coverage of style cycles needs history depth and, ideally, distinct styles | Keep 20+ snapshots at spaced intervals; consider admitting differently-rewarded variants (an aggressive-reward run, a defensive one) Blade-&-Soul-style (Czarnecki 2020; Oh et al.)

| Uniform pool sampling | Equal time on a stale 500k-step checkpoint and on the one opponent that still wins. Uniform was the baseline every serious system upgraded away from | PFSP weighting (1 - winrate)^0.5 from per-opponent win rates you already log (AlphaStar ablation: PFSP beats uniform)

| No exploiter | Nothing in the loop actively searches for the champion's blind spots, so robustness is only ever tested by accident | Periodic short-budget exploiter trained against the frozen main; its final win rate is your exploitability meter, and its policy joins the pool (Lanctot 2017; AlphaStar; FightLadder)

| Style bots (edge-guard, weapon-hunter, recovery-punish) | None. This is the good part: hand-built style diversity, the same idea as Blade & Soul's style pool | Keep them, but shift their weight from training diet toward the evaluation suite as self-play ramps

## How to evaluate whatever you change

- Fixed benchmark suite: the scripted cast plus a handful of frozen historical checkpoints, never trained against, same seeds, both action-selection modes. Your repo's evaluation-discipline notes (research/04, the rliable reference) already say this; it bears repeating because opponent-mix changes are precisely where training curves lie.

- Round-robin Elo across checkpoints: cheap with the native pipeline, and it detects the cycling signature directly: if checkpoint N beats N-1 but loses to N-3, you are watching the rock-paper-scissors chase in your own data.

- Exploiter probe: the strongest single number. Budget X million steps of fresh PPO against your frozen best; report its final win rate over time. FightLadder's sobering 94% (reported) is the reminder that this number will be ugly at first, and that trend, not level, is the signal.

## Primary sources

- Heinrich & Silver, NFSP — arXiv:1603.01121

- Bansal et al., Emergent Complexity via Multi-Agent Competition — arXiv:1710.03748

- Lanctot et al., PSRO — arXiv:1711.00832

- Czarnecki et al., Spinning Tops — arXiv:2004.09468

- Vinyals et al., AlphaStar, Nature 2019; Berner et al., OpenAI Five — arXiv:1912.06680

- Perolat et al., DeepNash — arXiv:2206.15378; NeuPL — arXiv:2202.07415

- Survey of self-play methods — arXiv:2408.01072; Firoiu et al. — arXiv:1702.06230

← Fighting-Game Case Studies
Series index
Next: Curriculum Engineering →


================================================================================
FILE: 06-curriculum-engineering.html
================================================================================

Curriculum Engineering

-

AI_2 · Training Strategy Series · Report 6 of 8

# Curriculum Engineering

Researched 2026-09-15. Builds on the repo's own curriculum research (research/04 and research/05, dated 2026-09-13), which covers the foundations: the Narvekar survey, reverse curriculum, reward-shaping invariance, and Prioritized Level Replay. This report adds the operational layer: how the big systems ran their gates, fades, and skill hand-offs.

AI_2 already has a curriculum: navigation drills 0a–0e, then combat phases 1–4 with shifting reward weights and opponent mixes, promotion gated on success metrics. The design matches what the literature recommends. Where projects live or die is in the mechanics: when exactly to advance, how to remove training-wheel rewards, and how to carry a skill into the next phase without losing it. That operational layer is what the big systems documented, and it is where your measured navigation-to-combat gap sits.

## Promotion gates: what the systems actually used

|
| System | Gate | Scope

| AlphaGo Zero | New checkpoint must win ≥55% of 400 arena games against the incumbent | Per checkpoint, head-to-head against its predecessor

| AlphaZero / KataGo | No gate; latest network always used | Safety traded for simplicity; worked in near-transitive Go

| AlphaStar | Exploiters retire when they reach ~70% win rate against their target, or on timeout | Per agent, against its assigned opponents

| ADR (Rubik's) | Per-parameter: widen a difficulty range when boundary-episode performance beats a high threshold, shrink when below a low one, decided over a 240-episode buffer | Local to each parameter, reversible

| XLand | No fixed thresholds; tasks filtered to the frontier (not-too-easy, not-too-hard) per agent, with the filter's own thresholds tuned by population-based training | Relative, self-adjusting

Three properties recur. Gates compare against something local (the predecessor, the current range boundary), not against a distant end goal. Gates are decided over a buffer of episodes, never a handful. And difficulty is reversible where possible: ADR shrinks ranges on regression rather than leaving the agent stuck.

The gate-scope error, which this project already met
Gating an early lesson on a later lesson's standard stalls training with no information gained. The September 8 contact run's saved report caught precisely this: an absolute wide-navigation standard was used as a mandatory stop for the earlier contact lesson, and the corrected analysis notes nearby-retention was the right gate and it passed (research/04, finding 4). The literature's version of the rule: gate each phase on the behavior that phase teaches, measured in the mode that matters. For phase 1 of combat, that is attack-initiation events per encounter, not win rate, and explicitly in the action-selection mode the skill must work under, deterministic or sampled (report 7).

## Fading the training wheels

Shaped rewards, approach bonuses, distance rewards, exist to make early learning possible. Ng's invariance theorem (in the repo's research/05 notes) says almost any shaping that is not potential-based changes what the optimal policy is, so a shaped reward left in place permanently trains a different game. The operational question is the removal schedule, and there are two published patterns:

- Anneal within a phase. Bansal et al. gave their competing MuJoCo agents a dense exploration reward and annealed it linearly to zero over the early fraction of training, after which only the sparse competition reward remained. Kept constant instead, the agents learned proxy behaviors: standing still in sumo, carrying the ball instead of kicking it. This is the classic evidence that dense aids must fade.

- Anneal a blend continuously. OpenAI Five's team-spirit parameter slid from selfish to team-average reward over training, a smooth version of the same move.

AI_2's phase table steps reward weights between phases (distance 2.0 in phase 1, 0.5 in phase 2, 0.1 in phase 3, 0.0 in phase 4), which is a coarse staircase version of the anneal. The literature pattern suggests one refinement: fade within a phase too, so the last third of each phase is already training mostly on the next phase's objective, and the phase switch stops being a cliff.

## Keeping old skills: the four published mechanisms

|
| Mechanism | How it works | Who used it

| Freeze the skill | Lock the earlier skill into a frozen module (a low-level controller); later training acts through it and cannot corrupt it | DeepMind humanoid football: locomotion frozen as a low-level controller, football trained on top

| Regularize toward it | Keep a penalty for drifting away from a reference policy (a KL anchor) | AlphaStar (anchor to the human-imitation policy); football (anchor to drill experts)

| Rehearse it | Mix earlier-phase episodes into later training at some ratio | Your own approved contact plan (navigation maintenance interleaved); general continual-learning practice; published game systems rarely state ratios

| Distill and restart | Train the next stage's network fresh, but initialize it by cloning the previous stage's behavior (kickstarting) | XLand between generations; kickstarting reports ~10x fewer steps to match, then surpassing, the teacher (reported)

Worth stating clearly because the September experiments checked it: AI_2 has so far not observed forgetting. All 280 matched navigation episode pairs were identical after contact calibration (research/04, finding 4). The retention problem is currently hypothetical; the transition problem is the measured one. Spend accordingly.

## The transition problem, which is the measured one

The symptom, from the September 8 runs: after navigation training, the agent approaches its combat target on every episode but requested zero light attacks in deterministic mode; under sampling it hits in 20 of 20 episodes. Three mechanisms from this literature aim directly at a stuck skill hand-off, ordered by fit:

- Reverse-curriculum starts. Florensa's rule (repo research/05): train from start states where the current agent sometimes succeeds, then expand outward. The contact lesson already spawns players close together, which is this idea. The refinement is to gate expansion on measured success probability staying in a band, rather than a fixed spawn recipe.

- Kickstart the mean, not just the buffer. The agent already produces successful attack episodes when sampling. Behavior-clone those episodes (its own successes, no human data needed) into the policy before continuing RL. This moves the deterministic policy toward the demonstrated behavior directly, which matters because the failure is specifically in the deterministic output (report 7 explains why the gradient alone moves it so slowly).

- Gate on the behavior, decided over a buffer. Promote out of the contact lesson on attack-initiation rate and safe-completion rate over a few hundred episodes, in the declared action mode, ADR-style. The 80% navigation gates were well-formed; the combat gates need the same specificity.

## Automating the difficulty knobs you already have

Phases 0a–0e are one hand-run curriculum over four numbers: spawn distance range, fall penalty, time limit, target count. 0a is distances 1–3; 0d is 1–15 with three targets. ADR (mechanism in report 2) is the same ladder run automatically: make each number a range, probe the edges, widen what the agent handles, shrink what it fails. Two payoffs beyond convenience: the difficulty always sits at the agent's frontier instead of waiting for a human to notice a plateau, and regression pulls difficulty back instead of wasting samples. At the native pipeline's throughput, the probe episodes ADR spends are cheap. The combat phases have their own ADR-able knobs: opponent mix shares, spawn separation, stock counts, and, if adopted from report 4, the time-penalty weight.

## The 2023–2026 fringe, briefly

Three lines are worth knowing about, none load-bearing for AI_2 today. EUREKA uses an LLM to write and iteratively improve reward functions, judged by training outcomes; its headline was solving pen-spinning via a two-stage curriculum of generated rewards. CurricuLLM has an LLM propose the subtask sequence itself, with explicit instructions to keep earlier-task terms in later rewards to fight forgetting. XLand-style open-ended task generation remains the reference for full automation. The honest summary: these automate the authoring of curricula, while your bottleneck is the mechanics of one specific hand-off, so they are future conveniences, not current fixes.

## What this means for the 9-phase design

- The architecture of the curriculum, drills before combat, shaping that decreases across phases, gated promotion, diverse scripted styles, matches the published playbook. The premise of the question "is our curriculum wrong?" deserves a precise answer: the shape is right, and the evidence points at four specific mechanics instead: the action-space/evaluation-mode contract (report 7), the opponent shares late in training (report 5), gate definitions per phase (this report), and reward wiring correctness (report 8's first experiment, the audit).

- Adopt buffered, behavior-specific gates for combat phases; keep the corrected retention gates stage-local.

- Fade shaping within phases, not only between them.

- Treat ADR as the eventual automation of 0a–0e and of combat difficulty, once the native port of the curriculum makes probe episodes nearly free.

## Primary sources

- OpenAI et al., Solving Rubik's Cube (ADR) — arXiv:1910.07113

- DeepMind, Open-Ended Learning (XLand) — arXiv:2107.12808

- Bansal et al. — arXiv:1710.03748; Berner et al. — arXiv:1912.06680

- Liu et al., humanoid football — arXiv:2105.12196

- Schmitt et al., Kickstarting Deep RL — arXiv:1803.03835

- EUREKA — arXiv:2310.12931; CurricuLLM — arXiv:2409.18382

- Repo foundations: research/04-curriculum-learning-reading-guide.md and research/05-curriculum-methods-and-papers.md (Narvekar, Florensa, Ng, PLR, TSCL, ALP-GMM, ACCEL)

← Self-Play by Design
Series index
Next: The Attack-Button Problem →


================================================================================
FILE: 07-attack-button.html
================================================================================

The Attack-Button Problem

-

AI_2 · Training Strategy Series · Report 7 of 8

# The Attack-Button Problem

Researched 2026-09-15. AI_2 measurements are from the preserved September 8 contact-calibration results (baselines/combat-curriculum-session13). External findings verified against the cited papers; two 2026 preprints are single-source and marked "reported".

This is the one report about a measured AI_2 failure rather than someone else's system. After contact calibration, the agent approached its target in 20 of 20 episodes and pressed the attack button in none of them, when actions were taken deterministically. Under sampling, it landed hits in 20 of 20. The highest deterministic attack output was 0.0269 against a button threshold of 0.5. This page explains the mechanism, then ranks the fixes by evidence.

## How your policy chooses actions

The policy used in these experiments is a Gaussian over a 10-dimensional continuous vector. For each action dimension the network outputs a mean, and the policy keeps a learned spread (standard deviation) per dimension. Two ways to act:

- Sampled: draw a random value from the bell curve around the mean. Used during training so the agent explores.

- Deterministic: just take the mean. Used by default during evaluation in Stable-Baselines3.

The game turns dimension values into buttons with a threshold: the light-attack button is pressed when its dimension exceeds 0.5. Now look at what the measurement says: the attack dimension's mean sits near 0.03, and the spread is wide enough that random draws exceed 0.5 fairly often.

> Figure: The measured situation on the attack dimension. Deterministic play takes the center of the bell, 0.03, and never presses the button. Sampling reaches the shaded tail on roughly one draw in seven (with a spread around 0.45), which at 30 decisions per second is several presses per second of play, enough to land hits in every sampled episode. Both observations, 0 of 20 deterministic and 20 of 20 sampled, come from one policy whose mean simply never moved.

## Why training never moved the mean

Four forces, each documented, combine here:

- The mean starts at zero and the reward is on the far side of a threshold. Standard PPO initialization gives every action dimension a mean near 0. Reward for attacking only arrives when a sample crosses 0.5. Learning must drag the mean rightward using only those rare tail events.

- The pull from successful tail samples is diluted and partially cancelled. The gradient moves the mean toward actions that went better than expected and away from ones that went worse. Attack presses are a small fraction of steps; the surrounding thousands of no-attack steps carry their own noisy advantages, and PPO's clipping caps how far any single update can move. Analyses of sparse or flat-reward regions describe exactly this: gradient estimates dominated by noise and cancelling contributions (reported, 2026 preprint on zero-reward collapse).

- Entropy regularization does not help the mean at all. The entropy bonus that keeps exploration alive has zero gradient with respect to the mean of a Gaussian; it only widens or narrows the spread. A 2026 study of clipped-Gaussian PPO (reported) works this out and measures the consequence: means wander wherever weak advantage signals leave them, with no force centering them into the useful range. Raising ent_coef, which the navigation config does, widens the bell but does not move its center.

- Clipping distorts what the boundary samples teach. Actions are clipped into [0,1] before the game sees them, while the policy's probability math uses the unclipped values. Fujita & Maeda showed this mismatch adds variance to the gradient exactly when mass piles up at a boundary. A secondary effect here, but part of why continuous-plus-threshold is a hostile design for buttons.

The repo's own framing, answered
The September curriculum review (research/04) said: audit the conversion from policy output to executed game buttons, and decide which action-selection mode the skill must work under. This report is that audit's mechanism half. The answer to "why does sampling work when deterministic play does not" is not a mystery of learning; it is geometry. The policy's knowledge lives partly in its spread, and deterministic evaluation throws the spread away.

## What everyone else does

Report 4's consensus table carries the weight here: Blade & Soul, FightLadder, DIAMBRA, Phillip, slippi-ai, JueWu, and OpenAI Five all use discrete action heads for button-like decisions. A discrete head outputs a probability for "press" directly. When pressing earns reward, the gradient raises that probability, and the deterministic action becomes "press" the moment the probability crosses one half. Learning and greedy execution stop being different worlds. There is no published fighting-game system that drives buttons through thresholded continuous Gaussians.

## The fixes, ranked by evidence

|
| Rank | Fix | What the literature predicts | Cost

| 1 | Make button dimensions discrete (Bernoulli/MultiBinary heads, or a MultiDiscrete factoring like move × attack) | Removes the threshold mismatch at the root; every attack reward directly raises press probability; deterministic and sampled behavior converge. Backed by the unanimous practice of report 4 and by bounded-action analyses | Changes policy and env interface; a contained A/B on the contact lesson; checkpoint transfer needs care

| 2 | Declare sampled action selection as the skill's execution mode, now | Zero training cost: the current policy already succeeds 20/20 sampled. Precedent: AlphaGo Zero samples early moves; OpenAI Five and AlphaStar act stochastically in real matches. Makes evaluation honest while structural fixes land | A decision plus an evaluation-config line

| 3 | Self-imitation warm start: behavior-clone the agent's own successful sampled episodes, then resume PPO | Directly moves the deterministic mean into the attacking region, exactly the quantity that is stuck; kickstarting literature and sparse-reward BC-then-RL studies support it, and slippi-ai is the domain precedent | A few hours: collect sampled successes, one BC pass, continue training

| 4 | Beta-distribution policy for bounded dimensions | Removes boundary bias and clip variance (Chou et al.); still does not by itself create the missing gradient if attack events stay rare. A better continuous policy, not a cure | Custom SB3 distribution; moderate

| 5 | Engagement-shaped rewards: meaningful per-step time cost, or a small bonus per attack initiation, annealed later | Attacks the incentive side rather than the geometry; Blade & Soul's time penalty demonstrably forced aggression. Complements any of the above; beware rewarding button-mashing, gate on hits | Config-level; needs the fade discipline of report 6

| 6 | Entropy measured after clipping / executed-action entropy | A 2026 preprint (reported) shows this adds a centering force on means; would push the attack mean inward toward the playable range. Least proven of the list | Fork of SB3's entropy term

Before changing anything: one audit
The repo has two action-space stories. config.yaml's settings block says action_space: "discrete" for the classic train.py path, while the contact and native benchmarks demonstrably run the 10-dim continuous Box with 0.5 thresholds. First establish, in code, which representation each training path actually uses end to end, from policy output through wrapper to the game's button check. The fix menu above assumes the continuous-threshold path, which is the one the September measurements came from. If a discrete path already exists in the classic trainer, fix 1 becomes "port that to the contact/native path" rather than new design work.

## How to test it

- Instrument first (no retraining). Log the attack dimension's mean and spread per rollout to W&B for existing and future runs. The diagnosis predicts: mean flat near 0 across contact training while sampled hit-rate climbs. One chart settles the mechanism.

- A/B on the contact lesson. Same seeds, same budget (the 524k-transition blocks the native pipeline already runs in ~25 seconds make this cheap once the curriculum port lands): continuous-threshold baseline versus Bernoulli attack head, versus baseline-plus-self-imitation. Success metric, per the corrected gate design of report 6: attack-initiation rate and hit rate in deterministic mode, plus no regression in sampled mode and navigation retention.

- Decide the contract. Whatever wins, write down the execution mode the project evaluates under from now on, and use it everywhere: gates, panels, videos, and eventual human matches. The deterministic-vs-sampled ambiguity is how this failure stayed invisible until a 20-episode panel caught it.

## Primary sources

- AI_2 measurements: baselines/combat-curriculum-session13 results, summarized in research/04 (finding 3)

- Chou et al., Beta-distribution policy gradients, ICML 2017 — proceedings.mlr.press/v70/chou17a

- Fujita & Maeda, Clipped Action Policy Gradient — arXiv:1802.07564

- Seyde et al., Is Bang-Bang Control All You Need? — arXiv:2111.02552

- Where Entropy Is Measured Matters (2026 preprint, reported) — arXiv:2608.24488; zero-reward collapse analysis (2026 preprint, reported) — arXiv:2605.30896

- Plappert et al., Parameter Space Noise — arXiv:1706.01905; Schmitt et al., Kickstarting — arXiv:1803.03835

- Practice consensus: report 4's seven systems; SB3 documentation on deterministic evaluation and MultiBinary/MultiDiscrete support

← Curriculum Engineering
Series index
Next: The AI_2 Playbook →


================================================================================
FILE: 08-ai2-playbook.html
================================================================================

The AI_2 Playbook

-

AI_2 · Training Strategy Series · Report 8 of 8

# The AI_2 Playbook

Written 2026-09-15. Everything here is a proposed experiment with a success metric and a kill criterion, in the project's own evidence style. Nothing below is a measured AI_2 result yet, except where explicitly cited to the baselines.

The question that started this series was: are we training with the wrong curriculum and strategy? Short answer from six reports of evidence: the curriculum's shape is right and matches what worked elsewhere, and the performance problem most likely lives in four specific mechanics, each cheap to test now that the native pipeline runs at 278k transitions per second. This page is the ordered plan.

## The verdict on the current strategy

|
| Design choice in AI_2 today | Verdict from the literature

| Staged curriculum: movement drills, then escalating combat phases | Right call at single-GPU scale. Matches DeepMind's football pipeline and the case studies; pure emergence needs 10-100x your samples (reports 2, 6)

| Damage-shaped rewards with win/KO on top, weights shifting across phases | Standard practice everywhere; the refinement is fading within phases, not only between them (reports 4, 6)

| Scripted style bots (edge-guard, weapon-hunter, recovery-punish) | Genuinely good idea, same family as Blade & Soul's style pool; their late-phase share is the problem, not their existence (reports 4, 5)

| Promotion gates on success metrics with retention checks | Right mechanism; gates need to name the specific behavior and mode, and the project already fixed one gate-scope error (report 6)

| Continuous 10-dim actions with 0.5 button thresholds | The outlier. No fighting-game precedent, and the measured attack-initiation failure has a direct mechanistic explanation in it (reports 4, 7)

| Self-play at 28-32% with a uniform pool of 7 recent snapshots | Below every reference system late in training; uniform sampling and shallow pool are the two weakest links (reports 3, 5)

## The experiment ladder

Ordered by information per dollar. Each experiment stands alone; run them in sequence and stop whenever the fighter is good enough.

### E0. Reward and action wiring audit — one session, no GPU

Before any strategy change, verify the plumbing, because README_BEST_MODEL_PLAN in the repo already flags three suspected wiring bugs, and a sign error dominates any curriculum question:

- Win reward sign: the plan file says current logic flips win/loss attribution. Confirm in the live train path.

- Phase 1 knockout weight is negative in config.yaml (-3.0) while the plan says the reward manager treats knockout as "agent got KO'd", so the sign may be backwards for the intended penalty.

- Aggression reward: the plan says it checks a state_machine attribute that does not exist, making the weight dead. Either wire it or delete the weight.

- Also audit: the navigation entropy comment says "lower entropy" while nav_ent_coef: 0.05 is five times the base 0.01; and confirm which action representation each train path really uses (report 7's audit box).

Success: a one-page audit note stating, with file and line references, which of these are live bugs, already fixed, or misunderstandings. Kill criterion: none; this is reconnaissance and it caps a day.

### E1. Instrument the attack dimension and declare the execution mode — no retraining

Add per-rollout logging of the attack dimension's policy mean and spread (W&B is already wired). Re-run the existing 20-scenario contact panels for the preserved checkpoints in both action modes, which the harness already supports. Then make one decision and write it into the docs: which action-selection mode is the contract for gates and evaluations, and if it is deterministic, E2 becomes mandatory rather than optional.

Success: a chart of mean/spread over past training confirming or refuting the stuck-mean mechanism of report 7, and a written mode contract. Kill criterion: if the attack mean turns out to move healthily and the failure lies elsewhere, report 7's ranking re-orders and E2's design changes; the instrumentation still pays for itself.

### E2. Fix attack initiation on the contact lesson — the first real A/B

Three arms, matched seeds and budget, on the B1 contact lesson: current continuous-threshold policy as control; discrete button head (Bernoulli or MultiDiscrete) arm; self-imitation arm (behavior-clone the agent's own successful sampled episodes, then continue PPO). Small time-penalty variant optional as a fourth arm if budget allows (report 4's Blade & Soul lever).

Success: deterministic attack-initiation in ≥80% of encounters and deterministic hit rate at or above the current sampled rate, with navigation retention panels unchanged, over ≥3 seeds. Kill criterion: an arm that wins on attacks but loses navigation retention or safe-completion by more than the seed spread is not promoted. Cost note: at native-pipeline speed a 5M-transition arm is minutes of GPU loop time; the real cost is the curriculum port (E6) or running on the existing slower Python path, which is still only hours.

### E3. Modernize the opponent mix — matched-budget A/B in phase 2 or 3

Control: current mix. Treatment, from reports 3 and 5: self-play share raised toward 0.6-0.7, pool deepened to 20+ spaced snapshots, PFSP sampling (1 - winrate)^0.5 replacing uniform, scripted bots kept at ~0.15 for anchoring. Evaluate both arms on the frozen benchmark suite (all scripted styles plus fixed historical checkpoints), never on training win rate.

Success: higher win rate on the held-out suite and no cycling signature in a round-robin of the arm's own checkpoints. Kill criterion: if the treatment arm's held-out performance is flat after an agreed budget, keep the simpler mix and bank the negative result.

### E4. Gate and fade mechanics — cheap config discipline alongside E3

Rewrite combat-phase gates as behavior-specific, buffered, mode-explicit checks (report 6): attack-initiation rate, hit rate, safe-completion, each over a few hundred episodes. Add within-phase annealing of the shaping weights that currently step between phases. These are config and harness changes; measure them as part of E3's arms rather than separately.

### E5. Standing exploitability probe — the honest scoreboard

Once per milestone checkpoint: train a fresh short-budget PPO exploiter against the frozen champion and record its final win rate. Trend is the metric; FightLadder says the level will be humbling (report 3). The probe's product doubles as a pool member for E3's treatment arm and, if it finds a degenerate counter-style, as a bug report on the game or reward.

### E6. Port the curriculum onto the native pipeline — the multiplier

Already the stated next step in the fullgame results, and it is what turns E2-E5 from overnight jobs into coffee-break jobs. The 3.98x-scaled configuration collects 16.7M transitions per minute of loop time; every experiment above is sized in tens of millions.

### E7. Optional, later: auxiliary prediction heads

KataGo's densest-signal trick (report 1): add heads predicting near-future damage events and KO probability, trained alongside the value loss. A clean single-variable experiment once the pipeline is stable, with the usual multi-seed, held-out-suite evaluation.

## Two training-health items that ride along

- The KL early stop. The September 9 profile found every measured update stopped after one optimizer step of a possible sixteen on the large checkpoint, on target_kl 0.01. That report explicitly declined to explain it and warned against deleting the safeguard for speed. It belongs on the strategy list too: if updates routinely take one step, the effective learning rate schedule is not what the config says. Diagnose with per-minibatch approx-KL logging in E1's instrumentation pass; candidate causes to test are advantage normalization, learning rate, and the threshold itself, one at a time.

- Entropy schedule. OpenAI Five annealed entropy 0.01 to 0.001 across training (report 2). AI_2 holds 0.01 flat (0.05 in navigation, with a comment contradicting the value; see E0). After E2 lands, an entropy anneal in late combat phases is a one-line experiment.

## What not to change yet

Model size, frame stacking, gamma, GAE lambda, and the PPO core match reference systems at your scale and are not implicated by any measurement. The opponent-history buffer (60x10 appended values, mostly zeros in navigation) is a measured compute topic in the repo's speed notes, not a strategy topic. One variable at a time is what made the September baselines trustworthy; keep that discipline as the experiments get more interesting.

The one-paragraph version
Keep the curriculum. Audit the reward wiring first because a sign bug beats any strategy. Fix the attack button by making buttons discrete or by adopting sampled execution, with self-imitation as the bridge. Then shift the late-game diet from scripted bots to a deeper, difficulty-weighted self-play pool with an occasional exploiter, gate phases on the specific behavior they teach, and keep scoring everything against frozen benchmarks. Every reference system in this series is one of these moves at bigger scale, and the native pipeline has made all of them affordable.

## Where each claim lives

- Curriculum verdicts: reports 2, 4, 6; repo research/04-06

- Action space and attack failure: report 7 (mechanism), report 4 (practice consensus)

- Opponent mix: reports 3 and 5; OpenAI Five, AlphaStar, Phillip, Czarnecki

- Gates and fading: report 6; ADR, Bansal, AlphaGo Zero

- AI_2 measurements cited: baselines/combat-curriculum-session13 (attack panels), baselines/contact-collection-profile-20260909 (KL early stop), baselines/fullgame-20260908 and research/update-profile-20260913 (throughput), README_BEST_MODEL_PLAN (wiring flags)

← The Attack-Button Problem
Series index