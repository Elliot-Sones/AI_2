#include "navigation_episode.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/*
 * CUDA driver for the closed-loop navigation tick. One lane = one complete
 * episode (contact world, both state machines, platform, spawners, wrapper)
 * resident in device memory, stepped by the shared header code compiled with
 * -fmad=false so the arithmetic matches the unfused CPU reference.
 *
 * Parity entry points keep every lane identical: the same action and the same
 * recorded draws go to all lanes (each lane has its own replay cursor), the
 * compare kernel checks every lane against lane zero byte for byte, and lane
 * zero is copied back to the host for the field-level comparison with the CPU
 * reference.  The benchmark entry point runs generator-mode lanes with random
 * actions and in-kernel resets, which is the whole environment step without a
 * policy: not environment SPS with a policy, not PPO SPS.
 */

static AI2NEpisode *g_eps = 0;
static AI2NRng *g_rngs = 0;
static AI2NDraw *g_draws = 0;
static AI2NPlatformTable *g_table = 0;
static AI2NStepResult *g_results = 0;
static double *g_action = 0;
static int *g_errors = 0;
static int *g_flag = 0;
static unsigned long long *g_counts = 0;
static int g_lanes = 0;
static int g_draw_capacity = 0;

__global__ void ai2n_replicate_kernel(AI2NEpisode *eps, int lanes) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane <= 0 || lane >= lanes) return;
    const unsigned long long *src = (const unsigned long long *)&eps[0];
    unsigned long long *dst = (unsigned long long *)&eps[lane];
    const int words = (int)(sizeof(AI2NEpisode) / sizeof(unsigned long long));
    for (int i = 0; i < words; ++i) dst[i] = src[i];
}

__global__ void ai2n_rng_replay_kernel(AI2NRng *rngs, const AI2NDraw *draws, int count, int lanes) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= lanes) return;
    ai2n_rng_replay(&rngs[lane], draws, count);
}

__global__ void ai2n_reset_kernel(AI2NEpisode *eps, AI2NConfig cfg, AI2NRng *rngs, int lanes, int *errors) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= lanes) return;
    errors[lane] = ai2n_reset(&eps[lane], &cfg, &rngs[lane]);
}

__global__ void ai2n_step_kernel(AI2NEpisode *eps, const double *action, AI2NRng *rngs, const AI2NPlatformTable *table, AI2NStepResult *results, int lanes, int *errors) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= lanes) return;
    errors[lane] = ai2n_step(&eps[lane], action, &rngs[lane], table, &results[lane]);
}

__global__ void ai2n_compare_kernel(const AI2NEpisode *eps, const AI2NStepResult *results, const AI2NRng *rngs, int lanes, int *mismatch) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane <= 0 || lane >= lanes) return;
    const unsigned long long *a = (const unsigned long long *)&eps[0];
    const unsigned long long *b = (const unsigned long long *)&eps[lane];
    const int words = (int)(sizeof(AI2NEpisode) / sizeof(unsigned long long));
    for (int i = 0; i < words; ++i) {
        if (a[i] != b[i]) { atomicOr(mismatch, 1); return; }
    }
    const unsigned long long *ra = (const unsigned long long *)&results[0];
    const unsigned long long *rb = (const unsigned long long *)&results[lane];
    const int rwords = (int)(sizeof(AI2NStepResult) / sizeof(unsigned long long));
    for (int i = 0; i < rwords; ++i) {
        if (ra[i] != rb[i]) { atomicOr(mismatch, 2); return; }
    }
    if (rngs[0].cursor != rngs[lane].cursor || rngs[0].error != rngs[lane].error) atomicOr(mismatch, 4);
}

/* Generator-mode lanes: random actions from the lane stream, reset in-kernel when an episode ends. */
__global__ void ai2n_bench_kernel(AI2NEpisode *eps, AI2NConfig cfg, AI2NRng *rngs, const AI2NPlatformTable *table, int lanes, int iterations, unsigned long long seed,
                                  int *errors, unsigned long long *episodes_done) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= lanes) return;
    AI2NRng *rng = &rngs[lane];
    AI2NEpisode *ep = &eps[lane];
    AI2NStepResult out;
    double action[AI2N_ACTION_WIDTH];
    unsigned long long done = 0;
    int rc;
    ai2n_rng_seed(rng, seed, (unsigned long long)lane);
    rc = ai2n_reset(ep, &cfg, rng);
    if (rc) { errors[lane] = rc; return; }
    for (int it = 0; it < iterations; ++it) {
        const unsigned long long bits = ai2n_rng_next(rng);
        for (int k = 0; k < AI2N_ACTION_WIDTH; ++k) action[k] = ((bits >> (k * 3)) & 7ull) < 2ull ? 1.0 : 0.0;
        rc = ai2n_step(ep, action, rng, table, &out);
        if (rc) { errors[lane] = rc; return; }
        if (out.terminated || out.truncated) {
            done += 1;
            rc = ai2n_reset(ep, &cfg, rng);
            if (rc) { errors[lane] = rc; return; }
        }
    }
    errors[lane] = 0;
    episodes_done[lane] = done;
}

static int ai2n_cuda_ensure(int lanes, int draw_count) {
    if (lanes <= 0) return AI2N_ERR_CONFIG;
    if (g_eps == 0 || g_lanes < lanes) {
        if (g_eps) { cudaFree(g_eps); cudaFree(g_rngs); cudaFree(g_results); cudaFree(g_errors); cudaFree(g_counts); g_eps = 0; }
        if (cudaMalloc((void **)&g_eps, sizeof(AI2NEpisode) * (size_t)lanes) != cudaSuccess) return AI2N_ERR_CAPACITY;
        if (cudaMalloc((void **)&g_rngs, sizeof(AI2NRng) * (size_t)lanes) != cudaSuccess) return AI2N_ERR_CAPACITY;
        if (cudaMalloc((void **)&g_results, sizeof(AI2NStepResult) * (size_t)lanes) != cudaSuccess) return AI2N_ERR_CAPACITY;
        if (cudaMalloc((void **)&g_errors, sizeof(int) * (size_t)lanes) != cudaSuccess) return AI2N_ERR_CAPACITY;
        if (cudaMalloc((void **)&g_counts, sizeof(unsigned long long) * (size_t)lanes) != cudaSuccess) return AI2N_ERR_CAPACITY;
        g_lanes = lanes;
    }
    if (g_flag == 0 && cudaMalloc((void **)&g_flag, sizeof(int)) != cudaSuccess) return AI2N_ERR_CAPACITY;
    if (g_action == 0 && cudaMalloc((void **)&g_action, sizeof(double) * AI2N_ACTION_WIDTH) != cudaSuccess) return AI2N_ERR_CAPACITY;
    if (draw_count > g_draw_capacity) {
        if (g_draws) cudaFree(g_draws);
        g_draw_capacity = draw_count < 64 ? 64 : draw_count;
        if (cudaMalloc((void **)&g_draws, sizeof(AI2NDraw) * (size_t)g_draw_capacity) != cudaSuccess) { g_draws = 0; g_draw_capacity = 0; return AI2N_ERR_CAPACITY; }
    }
    /* Recursive tree helpers plus the state-machine locals need more stack than the default. */
    cudaDeviceSetLimit(cudaLimitStackSize, 32768);
    return AI2N_ERR_NONE;
}

static int ai2n_cuda_load_draws(const AI2NDraw *draws, int count, int lanes) {
    if (count > 0 && cudaMemcpy(g_draws, draws, sizeof(AI2NDraw) * (size_t)count, cudaMemcpyHostToDevice) != cudaSuccess) return AI2N_ERR_CAPACITY;
    const int threads = 128;
    const int blocks = (lanes + threads - 1) / threads;
    ai2n_rng_replay_kernel<<<blocks, threads>>>(g_rngs, g_draws, count, lanes);
    return cudaGetLastError() == cudaSuccess ? AI2N_ERR_NONE : AI2N_ERR_CONFIG;
}

static int ai2n_cuda_collect(int lanes, AI2NEpisode *host_out, AI2NStepResult *host_result, int *all_equal, int *lane_error, int *rng_error, int *rng_cursor) {
    const int threads = 128;
    const int blocks = (lanes + threads - 1) / threads;
    int zero = 0;
    int flag = 0;
    AI2NRng rng0;
    if (cudaMemcpy(g_flag, &zero, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) return AI2N_ERR_CAPACITY;
    ai2n_compare_kernel<<<blocks, threads>>>(g_eps, g_results, g_rngs, lanes, g_flag);
    if (cudaDeviceSynchronize() != cudaSuccess) return AI2N_ERR_CONFIG;
    if (cudaMemcpy(&flag, g_flag, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) return AI2N_ERR_CAPACITY;
    *all_equal = flag == 0;
    *lane_error = 0;
    {
        int *errors = (int *)malloc(sizeof(int) * (size_t)lanes);
        if (errors == 0) return AI2N_ERR_CAPACITY;
        if (cudaMemcpy(errors, g_errors, sizeof(int) * (size_t)lanes, cudaMemcpyDeviceToHost) != cudaSuccess) { free(errors); return AI2N_ERR_CAPACITY; }
        for (int i = 0; i < lanes; ++i) { if (errors[i] != 0) { *lane_error = errors[i]; break; } }
        free(errors);
    }
    if (host_out && cudaMemcpy(host_out, g_eps, sizeof(AI2NEpisode), cudaMemcpyDeviceToHost) != cudaSuccess) return AI2N_ERR_CAPACITY;
    if (host_result && cudaMemcpy(host_result, g_results, sizeof(AI2NStepResult), cudaMemcpyDeviceToHost) != cudaSuccess) return AI2N_ERR_CAPACITY;
    if (cudaMemcpy(&rng0, g_rngs, sizeof(AI2NRng), cudaMemcpyDeviceToHost) != cudaSuccess) return AI2N_ERR_CAPACITY;
    if (rng_error) *rng_error = rng0.error;
    if (rng_cursor) *rng_cursor = rng0.cursor;
    return AI2N_ERR_NONE;
}

extern "C" int ai2n_cuda_episode_size(void) { return (int)sizeof(AI2NEpisode); }

extern "C" int ai2n_cuda_table_size(void) { return (int)sizeof(AI2NPlatformTable); }

/* Upload the host-built platform table once; kernels read it from global memory. */
extern "C" int ai2n_cuda_set_table(const AI2NPlatformTable *table) {
    if (g_table == 0 && cudaMalloc((void **)&g_table, sizeof(AI2NPlatformTable)) != cudaSuccess) { g_table = 0; return AI2N_ERR_CAPACITY; }
    if (cudaMemcpy(g_table, table, sizeof(AI2NPlatformTable), cudaMemcpyHostToDevice) != cudaSuccess) return AI2N_ERR_CAPACITY;
    return AI2N_ERR_NONE;
}

extern "C" int ai2n_cuda_release(void) {
    if (g_eps) cudaFree(g_eps);
    if (g_rngs) cudaFree(g_rngs);
    if (g_results) cudaFree(g_results);
    if (g_errors) cudaFree(g_errors);
    if (g_counts) cudaFree(g_counts);
    if (g_flag) cudaFree(g_flag);
    if (g_action) cudaFree(g_action);
    if (g_draws) cudaFree(g_draws);
    if (g_table) cudaFree(g_table);
    g_table = 0;
    g_eps = 0; g_rngs = 0; g_results = 0; g_errors = 0; g_counts = 0; g_flag = 0; g_action = 0; g_draws = 0;
    g_lanes = 0; g_draw_capacity = 0;
    return AI2N_ERR_NONE;
}

/* Upload one host episode into lane zero and replicate it to every lane (the CPU reference state). */
extern "C" int ai2n_cuda_upload(const AI2NEpisode *host_ep, int lanes) {
    int rc = ai2n_cuda_ensure(lanes, 0);
    if (rc) return rc;
    if (cudaMemcpy(g_eps, host_ep, sizeof(AI2NEpisode), cudaMemcpyHostToDevice) != cudaSuccess) return AI2N_ERR_CAPACITY;
    const int threads = 128;
    const int blocks = (lanes + threads - 1) / threads;
    ai2n_replicate_kernel<<<blocks, threads>>>(g_eps, lanes);
    if (cudaDeviceSynchronize() != cudaSuccess) return AI2N_ERR_CONFIG;
    if (cudaMemset(g_results, 0, sizeof(AI2NStepResult) * (size_t)lanes) != cudaSuccess) return AI2N_ERR_CAPACITY;
    return AI2N_ERR_NONE;
}

/* Reset every lane on the device from the same recorded reset draws. */
extern "C" int ai2n_cuda_reset_replicated(const AI2NConfig *cfg, const AI2NDraw *draws, int count, int lanes, AI2NEpisode *host_out,
                                          int *all_equal, int *lane_error, int *rng_error, int *rng_cursor) {
    int rc = ai2n_cuda_ensure(lanes, count);
    if (rc) return rc;
    rc = ai2n_cuda_load_draws(draws, count, lanes);
    if (rc) return rc;
    const int threads = 128;
    const int blocks = (lanes + threads - 1) / threads;
    if (cudaMemset(g_results, 0, sizeof(AI2NStepResult) * (size_t)lanes) != cudaSuccess) return AI2N_ERR_CAPACITY;
    ai2n_reset_kernel<<<blocks, threads>>>(g_eps, *cfg, g_rngs, lanes, g_errors);
    if (cudaDeviceSynchronize() != cudaSuccess) return AI2N_ERR_CONFIG;
    return ai2n_cuda_collect(lanes, host_out, 0, all_equal, lane_error, rng_error, rng_cursor);
}

/* Step every resident lane with the same action and the same recorded draws. */
extern "C" int ai2n_cuda_step(const double *action, const AI2NDraw *draws, int count, int lanes, AI2NEpisode *host_out, AI2NStepResult *host_result,
                              int *all_equal, int *lane_error, int *rng_error, int *rng_cursor) {
    int rc = ai2n_cuda_ensure(lanes, count);
    if (rc) return rc;
    rc = ai2n_cuda_load_draws(draws, count, lanes);
    if (rc) return rc;
    if (cudaMemcpy(g_action, action, sizeof(double) * AI2N_ACTION_WIDTH, cudaMemcpyHostToDevice) != cudaSuccess) return AI2N_ERR_CAPACITY;
    const int threads = 128;
    const int blocks = (lanes + threads - 1) / threads;
    if (g_table == 0) return AI2N_ERR_PLATFORM;
    ai2n_step_kernel<<<blocks, threads>>>(g_eps, g_action, g_rngs, g_table, g_results, lanes, g_errors);
    if (cudaDeviceSynchronize() != cudaSuccess) return AI2N_ERR_CONFIG;
    return ai2n_cuda_collect(lanes, host_out, host_result, all_equal, lane_error, rng_error, rng_cursor);
}

/* Whole-tick throughput with generator-mode lanes and random actions; no policy, no host transfer per step. */
extern "C" int ai2n_cuda_bench(const AI2NConfig *cfg, int lanes, int iterations, unsigned long long seed, double *elapsed_ms,
                               unsigned long long *episodes_done, int *lane_error) {
    int rc = ai2n_cuda_ensure(lanes, 0);
    if (rc) return rc;
    cudaEvent_t start, stop;
    const int threads = 128;
    const int blocks = (lanes + threads - 1) / threads;
    float ms = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    if (g_table == 0) return AI2N_ERR_PLATFORM;
    ai2n_bench_kernel<<<blocks, threads>>>(g_eps, *cfg, g_rngs, g_table, lanes, iterations, seed, g_errors, g_counts);
    cudaEventRecord(stop);
    if (cudaDeviceSynchronize() != cudaSuccess) return AI2N_ERR_CONFIG;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    *elapsed_ms = (double)ms;
    *lane_error = 0;
    {
        int *errors = (int *)malloc(sizeof(int) * (size_t)lanes);
        unsigned long long *counts = (unsigned long long *)malloc(sizeof(unsigned long long) * (size_t)lanes);
        unsigned long long total = 0;
        if (errors == 0 || counts == 0) { free(errors); free(counts); return AI2N_ERR_CAPACITY; }
        if (cudaMemcpy(errors, g_errors, sizeof(int) * (size_t)lanes, cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(counts, g_counts, sizeof(unsigned long long) * (size_t)lanes, cudaMemcpyDeviceToHost) != cudaSuccess) {
            free(errors); free(counts); return AI2N_ERR_CAPACITY;
        }
        for (int i = 0; i < lanes; ++i) { if (errors[i] != 0 && *lane_error == 0) *lane_error = errors[i]; total += counts[i]; }
        *episodes_done = total;
        free(errors);
        free(counts);
    }
    return AI2N_ERR_NONE;
}
