/* cuda_shim.h
 * Lets you write CUDA-shaped kernels and run them on a CPU.
 *
 * Under nvcc (a real GPU host) everything here disappears and the real
 * CUDA syntax is used. Under clang/gcc on a Mac, LAUNCH() loops over every
 * block and thread, setting blockIdx / blockDim / threadIdx for each one,
 * exactly the way the hardware would number them. Same numbering, same
 * results. Not the same speed: this is one CPU core doing all the threads
 * one after another.
 *
 * It also simulates warps: every 32 consecutive threads form a warp. If you
 * call BRANCH("name") inside a kernel, the shim records how many distinct
 * branch names each warp took. That count is how many passes a real warp
 * would need (divergence). Print it with shim_report().
 */
#ifndef CUDA_SHIM_H
#define CUDA_SHIM_H

#ifdef __CUDACC__
/* ---------------- real CUDA ---------------- */
#define HD __host__ __device__
#define LAUNCH(kernel, blocks, threads, ...) \
    do { kernel<<<(blocks), (threads)>>>(__VA_ARGS__); cudaDeviceSynchronize(); } while (0)
#define BRANCH(name) ((void)0)
static inline void shim_report(const char *label) { (void)label; }
static inline void shim_reset(void) {}
#define DEVICE_ALLOC(ptr, bytes) cudaMalloc((void**)&(ptr), (bytes))
#define DEVICE_FREE(ptr) cudaFree(ptr)
#define TO_DEVICE(dst, src, bytes) cudaMemcpy((dst), (src), (bytes), cudaMemcpyHostToDevice)
#define TO_HOST(dst, src, bytes) cudaMemcpy((dst), (src), (bytes), cudaMemcpyDeviceToHost)

#else
/* ---------------- CPU simulation ---------------- */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define __global__
#define __device__
#define __host__
#define __shared__ static
#define HD
#define __syncthreads() ((void)0)   /* threads run one at a time here, so a barrier is a no-op */

typedef struct { unsigned x, y, z; } dim3_t;
static dim3_t blockIdx, blockDim, threadIdx, gridDim;

/* warp divergence bookkeeping */
#define SHIM_MAX_WARPS 4096
#define SHIM_MAX_TAGS 16
static struct { const char *tags[SHIM_MAX_TAGS]; int n; } shim_warp[SHIM_MAX_WARPS];
static int shim_warps_seen = 0;

static inline void shim_reset(void) { memset(shim_warp, 0, sizeof(shim_warp)); shim_warps_seen = 0; }

static inline void shim_branch(const char *name) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned w = tid / 32;
    if (w >= SHIM_MAX_WARPS) return;
    if ((int)w + 1 > shim_warps_seen) shim_warps_seen = (int)w + 1;
    for (int i = 0; i < shim_warp[w].n; ++i) if (shim_warp[w].tags[i] == name || strcmp(shim_warp[w].tags[i], name) == 0) return;
    if (shim_warp[w].n < SHIM_MAX_TAGS) shim_warp[w].tags[shim_warp[w].n++] = name;
}
#define BRANCH(name) shim_branch(name)

static inline void shim_report(const char *label) {
    int total = 0, worst = 0;
    for (int w = 0; w < shim_warps_seen; ++w) { total += shim_warp[w].n; if (shim_warp[w].n > worst) worst = shim_warp[w].n; }
    printf("  [%s] warps: %d   mean branch passes per warp: %.2f   worst warp: %d passes\n",
           label, shim_warps_seen, shim_warps_seen ? (double)total / shim_warps_seen : 0.0, worst);
}

/* LAUNCH: run kernel for every (block, thread), in the order the hardware numbers them. */
#define LAUNCH(kernel, blocks, threads, ...)                                   \
    do {                                                                       \
        gridDim.x = (unsigned)(blocks); blockDim.x = (unsigned)(threads);      \
        gridDim.y = gridDim.z = blockDim.y = blockDim.z = 1;                   \
        for (blockIdx.x = 0; blockIdx.x < gridDim.x; ++blockIdx.x)             \
            for (threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x)     \
                kernel(__VA_ARGS__);                                           \
    } while (0)

#define DEVICE_ALLOC(ptr, bytes) ((ptr) = malloc(bytes))
#define DEVICE_FREE(ptr) free(ptr)
#define TO_DEVICE(dst, src, bytes) memcpy((dst), (src), (bytes))
#define TO_HOST(dst, src, bytes) memcpy((dst), (src), (bytes))
#endif

#endif /* CUDA_SHIM_H */
