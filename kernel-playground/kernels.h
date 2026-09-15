/* kernels.h  -  YOUR KERNELS GO HERE.
 *
 * Rules of the game:
 *   1. A kernel is a function marked __global__. It returns void.
 *   2. It runs once per thread. Compute your index from blockIdx/blockDim/threadIdx.
 *   3. Always guard: if (i >= n) return;   (the launch may create extra threads)
 *   4. Everything the kernel touches must be a pointer you were handed. No globals.
 *
 * Exercises are numbered. Make each test in test_kernels.c pass, then invent your own.
 */
#ifndef KERNELS_H
#define KERNELS_H
#include "cuda_shim.h"
#include <math.h>

/* ---------------------------------------------------------------------------
 * Exercise 1  (done for you)  -  one thread, one element.
 * PPO's  policy_loss_1 = advantages * ratio
 * ------------------------------------------------------------------------- */
__global__ void multiply_elements(const float *advantages, const float *ratios, float *products, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    products[i] = advantages[i] * ratios[i];
}

/* ---------------------------------------------------------------------------
 * Exercise 2  -  clamp, like PPO's  clamp(ratio, 1-eps, 1+eps)
 * TODO: fill in the body. Each thread clamps one element into [lo, hi].
 * ------------------------------------------------------------------------- */
__global__ void clamp_elements(const float *in, float *out, float lo, float hi, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    /* TODO */
    out[i] = in[i];
}

/* ---------------------------------------------------------------------------
 * Exercise 3  -  one thread per GAME, several values per game.
 * Each game has 2 players x 10 action floats = 20 floats, laid out
 * game-major: actions[game*20 + player*10 + k].
 * Write the sum of player 0's actions into out[game*2 + 0] and player 1's
 * into out[game*2 + 1]. This is the pointer arithmetic from game_lanes.cu.
 * ------------------------------------------------------------------------- */
__global__ void sum_actions_per_player(const float *actions, float *out, int games) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= games) return;
    const float *mine = actions + (size_t)g * 20;
    /* TODO: two sums, two writes */
    out[g * 2 + 0] = 0.0f;
    out[g * 2 + 1] = 0.0f;
    (void)mine;
}

/* ---------------------------------------------------------------------------
 * Exercise 4  -  divergence. A tiny "game step".
 * Each thread owns one fighter with a state:  0 idle, 1 run, 2 jump, 3 attack.
 * Apply one frame of rules:
 *     idle   -> x unchanged
 *     run    -> x += 1
 *     jump   -> y += 2, and state becomes idle
 *     attack -> hits += 1
 * Call BRANCH("idle") / BRANCH("run") / ... inside each branch so the shim can
 * count how many passes each warp needs. Then look at the test output:
 * sorted states diverge far less than shuffled ones. Same work, same code.
 * ------------------------------------------------------------------------- */
__global__ void fighter_step(int *state, float *x, float *y, int *hits, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    switch (state[i]) {
        case 0: BRANCH("idle");   /* TODO */ break;
        case 1: BRANCH("run");    /* TODO */ break;
        case 2: BRANCH("jump");   /* TODO */ break;
        case 3: BRANCH("attack"); /* TODO */ break;
    }
}

/* ---------------------------------------------------------------------------
 * Exercise 5  -  a reduction. Sum n floats into out[0].
 * The naive version below is WRONG on a real GPU: thousands of threads writing
 * out[0] at once is a race. It happens to work in the CPU shim because threads
 * run one at a time. Fix it so it would be correct on hardware too:
 *   option A: use atomicAdd(out, in[i])  (define a CPU fallback below)
 *   option B: one thread per block sums its block's slice, writes partial[blockIdx.x]
 * ------------------------------------------------------------------------- */
#ifndef __CUDACC__
static inline float atomicAdd(float *addr, float v) { float old = *addr; *addr += v; return old; }
#endif
__global__ void sum_naive(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[0] += in[i];   /* race on hardware. TODO: make it correct. */
}

#endif /* KERNELS_H */
