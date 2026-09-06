# Research: GPU-native simulation options for a custom 2D Pymunk fighting game

## Request Type

Comprehensive research: external papers and official project documentation for GPU-native simulation approaches, with emphasis on 2D physics/JAX evidence and applicability to a Python + Pymunk + SB3 PPO fighting-game repo.

## Synthesis

The strongest answer is that "C vs GPU" is the wrong first split. There are two separate axes:

1. **Authoring language / simulator implementation:** Python, C, C++, CUDA C, JAX, Warp kernels, etc.
2. **Execution placement and batching model:** ordinary CPU execution, CPU vectorized/batched execution, GPU kernels/tensor programs, or mixed CPU/GPU.

Writing the current game in C or C++ could make CPU stepping cheaper, but C/C++ by itself does not put the game engine on the GPU. C++ can also be GPU code when it is written as CUDA/device code or lowered by a framework such as Madrona. Conversely, Python-authored JAX or NVIDIA Warp code can execute on GPU only because those systems trace/JIT-compile restricted Python functions into accelerator programs. Ordinary Python loops, object mutation, Pymunk callbacks, process pipes, and Gym wrappers do not "move to GPU" just because the learner uses CUDA.

For this repo, the GPU-lane hypothesis is **a representative, bounded simulator prototype**. Its purpose would be to test whether the game's behavior and complete training workload benefit from a GPU implementation. No port or repo-specific GPU simulation speedup has been demonstrated.

The practical action space looks like this:

1. **CPU C/C++ batched simulator.** Keep semantics closest to current game by reusing existing native physics where possible, while rewriting Python game logic around a batch of worlds. This attacks Python overhead and worker coordination. It is the more conservative path if exact environment fidelity matters, because it does not require changing physics engines as the first move.
2. **JAX/Jax2D-style simulator.** Rewrite the game as fixed-shape arrays with `jax.jit` and `jax.vmap`, using a 2D rigid-body engine or custom simplified physics. This can target CPU or GPU and integrates cleanly with large-batch JAX PPO. It is the most relevant GPU research path for this game because the domain is 2D.
3. **Madrona C++/GPU ECS simulator.** Rewrite game state into an entity-component-system layout in C++; use Madrona to run thousands of worlds on GPU, exporting tensors to PyTorch. This is the strongest high-throughput game-engine style option, but it is a larger rewrite and currently requires core environment logic in C++.
4. **CUDA/WarpDrive or NVIDIA Warp kernels.** Write only the step/reset kernels in CUDA C, Numba, or Warp-style Python kernels. This gives fine GPU control and an explicit CPU/GPU kernel boundary. It is evidence that custom GPU simulation can be built from Python-facing tools, but it is not an out-of-the-box 2D fighting-game physics solution.
5. **Hybrid "simple navigation first" prototype.** Port only movement, walls/floors, pickups, and shaped rewards to JAX/Jax2D or Madrona, then compare against the current CPU game. This is useful as a de-risking experiment, but success on navigation is not proof for full combat, animation timing, collision fidelity, knockback, weapon logic, or opponent history.

My strongest recommendation for the GPU lane is: **do not jump straight to a full GPU engine port. First test a minimal comparable batched prototype with the smallest action/state subset that still looks like this game.** If JAX/Jax2D can run thousands of tiny 2D worlds and keep the policy update on the same device, it could be the most promising GPU direction. If the prototype needs so much simplification that it loses fidelity, a CPU batched C/C++ path using the current native physics is likely the better engineering bet.

## Official Docs Evidence

- [Kinetix, ICLR 2025](https://arxiv.org/abs/2410.23208), [full text, Appendix B](https://arxiv.org/html/2410.23208): JAX-based 2D physics and PPO provide the closest domain analogue in this survey. Best-case rates were 824K steps/s including training and 9.049M engine-only, at 32,768 and 16,384 environments. Hardware was one L40S with two 64-core EPYC 9554 CPUs. Scenes differed: Box2D used 3 polygons/2 joints; Jax2D used 6 polygons/3 circles/2 joints/2 thrusters. Training also differed: SB3 versus PureJaxRL-style code. Consequently the roughly 30x training comparison does not isolate GPU simulation. Raw Box2D won below 1,024 environments in that setup; this is not a universal crossover.

- Jax2D official repository: https://github.com/MichaelTMatthews/Jax2D  
  Establishes that Jax2D is a 2D rigid-body physics engine written entirely in JAX, based on Box2D, with dynamic scene configuration and `vmap` parallelization. Its own README cautions that runtime is O(n^2) in entity count because it calculates full pairwise collision resolution, and says it is best for lots of small diverse scenes rather than scenes with more than about 100 entities.

- Kinetix official repository: https://github.com/FLAIROx/Kinetix  
  Establishes the practical package context: reinforcement learning on 2D rigid-body physics worlds in JAX, ICLR 2025 Oral, with PPO, SFL/PLR experiments, editors, docs, and an errata note that the published paper behavior depends on a pinned Kinetix/Jax2D version for reproducibility.

- JAX `jit` documentation: https://docs.jax.dev/en/latest/201/jit.html  
  Establishes the key distinction for Python-authored GPU code: `jax.jit` compiles a Python function into an optimized computation specialized for CPU, GPU, or TPU. Without JIT, operations dispatch one by one and Python/runtime overhead remains.

- JAX `vmap` documentation: https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html  
  Establishes the batching mechanism: `vmap` maps a function over array axes to create a batched/vectorized function. This is the tool that makes a single-world function become a many-world function when the program is written in JAX-compatible array style.

- Madrona Engine paper/project page: https://madrona-engine.github.io/  
  Establishes Madrona as a research game engine for high-throughput batched learning environments, using a GPU-accelerated ECS architecture. The SIGGRAPH 2023 paper claims two to three orders of magnitude over open-source CPU baselines and 5x to 33x over strong 32-thread CPU baselines, with OpenAI Hide and Seek above 1.9M env steps/s on one GPU.

- Madrona FAQ: https://madrona-engine.github.io/  
  Establishes adoption constraints. Madrona is a framework for creating custom high-throughput simulators, not a drop-in RL environment. Current custom core game logic must be written in C++ and reorganized into ECS state. Reward functions can remain in PyTorch tensor operations after the step if exported ECS state is enough.

- Madrona Escape Room example: https://github.com/shacklettbp/madrona_escape_room  
  Establishes the integration pattern: a C++ simulator with CPU or CUDA execution modes, Python bindings, exported PyTorch tensors, built-in rigid-body physics/rendering, and a PPO training loop. The example recommends `--gpu-sim` and thousands of worlds for high-end GPU training.

- WarpDrive JMLR 2022 paper: https://www.jmlr.org/papers/v23/22-0185.html  
  Establishes end-to-end GPU MARL with thousands of simulations and agents, eliminating CPU/GPU copying, and reports 2.9M env steps/s for a 2D Tag simulation with 2000 environments and 1000 agents.

- WarpDrive official repository: https://github.com/salesforce/warp-drive  
  Establishes practical status and limits: the repository was archived May 1, 2025; its environment backend is CUDA C and Numba, training backend PyTorch, and it advertises consistency checking between CPU and GPU versions. This is useful evidence for kernel-style env development, but less attractive as a new dependency because it is archived.

- NVIDIA Warp docs: https://nvidia.github.io/warp/index.html  
  Establishes another route for Python-authored kernel code: Warp JIT-compiles Python functions into efficient CPU or GPU kernel code and provides primitives for simulation, robotics, geometry, PyTorch/JAX/Paddle integration, and differentiability.

- NVIDIA CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-programming-guide/index.html  
  Establishes that CUDA is a parallel computing platform and programming model for GPUs. The C++ section explains that GPU functions are kernels, compiled with `nvcc`, and run by many parallel threads. This is the authoritative source for "C++ can be GPU code, but only when written/compiled as CUDA/device code."

## Version Note

- Kinetix paper version referenced is arXiv v2, 2025-03-03, listed as ICLR 2025 Oral.  
- Exact JAX package behavior can vary by installed version and accelerator backend.  
- Any future adoption should pin and validate the selected package/backend versions.  
- WarpDrive is not a current active dependency candidate without extra caution because the official GitHub repository is archived as of 2025-05-01.

## Applicability to This Repo

The main project-specific uncertainty is fidelity. We would need explicit tests for current collision behavior, attack timing, pickups, knockback, resets and rewards. Compiled control flow and masked fixed-capacity entity arrays can represent branches and changing populations. The engineering question is whether converting our callbacks, mutable objects and containers preserves the game at acceptable complexity and cost.

Madrona is the strongest "game engine on GPU" option if the team wants a long-term custom simulator with C++ state ownership and PyTorch tensor integration. It cleanly separates language from placement: Madrona logic is C++, but can run CPU or CUDA depending on backend. Its speedups come from thousands of worlds in one ECS simulator, not just from translating Python files to C++.

WarpDrive and NVIDIA Warp are useful as lower-level patterns. WarpDrive demonstrates that end-to-end GPU training can be huge for 2D multi-agent simulations when envs and agents scale into the thousands, but it is archived and its examples are simpler than full rigid-body combat. NVIDIA Warp is alive and more general, but it is a kernel programming model, not a ready fighting-game simulator.

## Minimal Comparable Prototype

A useful prototype should test the bottleneck that papers say matters: many worlds sharing one compiled/batched step. It should not try to prove full gameplay immediately.

Recommended prototype shape:

1. Implement a fixed-size 2-player navigation/combat-lite state in JAX or Jax2D: positions, velocities, facing, floor/walls, one pickup, simple hitbox/contact reward, reset, terminal flags.
2. Run `jit(vmap(step))` for batch sizes 128, 512, 2048, 8192 on CPU and one NVIDIA GPU.
3. Keep observations shaped like the current symbolic observations where possible, including the history/frame-stack cost if the training loop will keep it.
4. Compare raw engine-only steps/s and PPO end-to-end steps/s separately.
5. Add one branchy mechanic at a time: pickup state, attack cooldown, hitstun/knockback, dynamic collision rule, then reset behavior.
6. Stop early if fidelity, batch shape, compilation cost, or integration cost makes the prototype a different game or removes the throughput advantage.

Success criteria should be qualitative and quantitative: the prototype must show at least a large raw stepping gap at thousands of worlds, keep policy inference/update on device, and preserve enough mechanics that the result is still a meaningful proxy for the actual game. A fast simplified toy that drops the hard mechanics is interesting, but it should not justify a full port by itself.

## Caveats / Ambiguity Flags

- GPU simulation wins in the papers rely on large batches. If this repo's PPO setup cannot use thousands of environments without hurting learning dynamics or memory, headline steps/s will not transfer directly.
- Full Pymunk equivalence is unproven. Jax2D and Madrona could produce different contact stability, friction, tunneling, collision ordering, and reset behavior.
- Differentiable physics is not required for PPO. JAX/Warp differentiability is useful tooling, but the main speed benefit here is compilation, batching, and avoiding CPU/GPU transfer.
- Current local evidence says observation reuse was CPU-validated only; it does not measure latest-code GPU end-to-end performance. Historical 16-env GPU timing is not enough to justify exact GPU-port speedup claims.

## Reusable Takeaway

The proposed GPU experiment must measure the whole pipeline and retain representative mechanics. A fast simplified simulator alone would not justify replacing the current game. The overall recommendation and ordering against CPU alternatives are in [00-training-speed-action-space.md](00-training-speed-action-space.md).
