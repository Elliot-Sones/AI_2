# Kernel playground

Write CUDA-shaped kernels and test them on this Mac. No GPU needed.

    make test

`cuda_shim.h` fakes the CUDA launch on a CPU: `LAUNCH(kernel, blocks, threads, ...)`
loops over every block and thread and sets `blockIdx`, `blockDim`, `threadIdx`
the way the hardware numbers them. Results are identical to a real GPU.
Speed is not: one core does all the threads one after another.

The shim also simulates warps. Call `BRANCH("name")` inside a branch and the
test output tells you how many passes each warp of 32 threads would need.

## Files

- `kernels.h` — your kernels. Five exercises, first one done.
- `test_kernels.c` — one test per exercise. Run `make test` after each edit.
- `cuda_shim.h` — the fake CUDA. Read it once; it is short.
- `Makefile` — `make test` here, `make cuda` on a machine with nvcc.

## How this relates to the real project

The real native code in `AI_2-native-update-profile-20260913/native/` uses the
same idea: the `AI2N_HD` macro compiles each kernel body for both CPU and GPU,
and `game_lanes.c` loops where `game_lanes.cu` launches. Exercise 3 is the
exact pointer arithmetic from `ai2g_lanes_step_kernel`. Exercise 4 is why the
game step kernel diverges.

## When you do rent a GPU

Copy this folder over and run `make cuda`. Same files, same tests, real
hardware. That is the only way to see timing differences.
