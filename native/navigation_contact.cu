#include "navigation_contact.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

/*
 * CUDA driver for the navigation contact step. One lane = one complete world
 * (bodies, shapes, broadphase tree, arbiter cache) resident in device memory.
 * The parity entry point replicates a host world across all lanes, steps them
 * with the shared header code, checks that every lane is byte-identical to lane
 * zero, and returns lane zero to the host. Compile with -fmad=false so the
 * arithmetic matches the unfused CPU reference bit for bit.
 */

static AI2CWorld *g_worlds = 0;
static int *g_errors = 0;
static int *g_flag = 0;
static int g_lanes = 0;

__global__ void ai2c_replicate_kernel(AI2CWorld *worlds, int lanes) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane <= 0 || lane >= lanes) return;
    const unsigned long long *src = (const unsigned long long *)&worlds[0];
    unsigned long long *dst = (unsigned long long *)&worlds[lane];
    const int words = (int)(sizeof(AI2CWorld) / sizeof(unsigned long long));
    for (int i = 0; i < words; ++i) dst[i] = src[i];
}

__global__ void ai2c_step_kernel(AI2CWorld *worlds, int lanes, double dt, double bias_coef, double damping_dt, int *errors) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= lanes) return;
    errors[lane] = ai2c_world_step(&worlds[lane], dt, bias_coef, damping_dt);
}

__global__ void ai2c_compare_kernel(const AI2CWorld *worlds, int lanes, int *mismatch) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane <= 0 || lane >= lanes) return;
    const unsigned long long *a = (const unsigned long long *)&worlds[0];
    const unsigned long long *b = (const unsigned long long *)&worlds[lane];
    const int words = (int)(sizeof(AI2CWorld) / sizeof(unsigned long long));
    for (int i = 0; i < words; ++i) {
        if (a[i] != b[i]) { atomicOr(mismatch, 1); return; }
    }
}

static int ai2c_cuda_ensure(int lanes) {
    if (lanes <= 0) return AI2C_ERR_INVALID;
    if (g_worlds != 0 && g_lanes >= lanes) return AI2C_OK;
    if (g_worlds) { cudaFree(g_worlds); cudaFree(g_errors); cudaFree(g_flag); g_worlds = 0; g_errors = 0; g_flag = 0; }
    if (cudaMalloc((void **)&g_worlds, sizeof(AI2CWorld) * (size_t)lanes) != cudaSuccess) return AI2C_ERR_CAPACITY;
    if (cudaMalloc((void **)&g_errors, sizeof(int) * (size_t)lanes) != cudaSuccess) return AI2C_ERR_CAPACITY;
    if (cudaMalloc((void **)&g_flag, sizeof(int)) != cudaSuccess) return AI2C_ERR_CAPACITY;
    g_lanes = lanes;
    /* Recursive tree helpers need a little more stack than the default. */
    cudaDeviceSetLimit(cudaLimitStackSize, 16384);
    return AI2C_OK;
}

extern "C" int ai2c_cuda_release(void) {
    if (g_worlds) { cudaFree(g_worlds); cudaFree(g_errors); cudaFree(g_flag); }
    g_worlds = 0; g_errors = 0; g_flag = 0; g_lanes = 0;
    return AI2C_OK;
}

extern "C" int ai2c_cuda_world_size(void) { return (int)sizeof(AI2CWorld); }

/* Replicate host_in across `lanes` lanes, step every lane once, verify all lanes agree,
 * and return lane zero in host_out. lane_error receives the max per-lane error code. */
extern "C" int ai2c_cuda_step_replicated(const AI2CWorld *host_in, AI2CWorld *host_out, int lanes,
                                         double dt, double bias_coef, double damping_dt,
                                         int *all_equal, int *lane_error) {
    int rc = ai2c_cuda_ensure(lanes);
    int zero = 0;
    int flag = 0;
    int *errors_host;
    int i;
    const int block = 128;
    const int grid = (lanes + block - 1) / block;
    if (rc != AI2C_OK) return rc;
    if (host_in == 0 || host_out == 0 || all_equal == 0 || lane_error == 0) return AI2C_ERR_NULL;
    if (cudaMemcpy(g_worlds, host_in, sizeof(AI2CWorld), cudaMemcpyHostToDevice) != cudaSuccess) return AI2C_ERR_INVALID;
    ai2c_replicate_kernel<<<grid, block>>>(g_worlds, lanes);
    ai2c_step_kernel<<<grid, block>>>(g_worlds, lanes, dt, bias_coef, damping_dt, g_errors);
    cudaMemcpy(g_flag, &zero, sizeof(int), cudaMemcpyHostToDevice);
    ai2c_compare_kernel<<<grid, block>>>(g_worlds, lanes, g_flag);
    if (cudaDeviceSynchronize() != cudaSuccess) return AI2C_ERR_INVALID;
    if (cudaMemcpy(host_out, g_worlds, sizeof(AI2CWorld), cudaMemcpyDeviceToHost) != cudaSuccess) return AI2C_ERR_INVALID;
    cudaMemcpy(&flag, g_flag, sizeof(int), cudaMemcpyDeviceToHost);
    *all_equal = (flag == 0);
    errors_host = (int *)malloc(sizeof(int) * (size_t)lanes);
    if (errors_host == 0) return AI2C_ERR_CAPACITY;
    cudaMemcpy(errors_host, g_errors, sizeof(int) * (size_t)lanes, cudaMemcpyDeviceToHost);
    *lane_error = 0;
    for (i = 0; i < lanes; ++i) if (errors_host[i] > *lane_error) *lane_error = errors_host[i];
    free(errors_host);
    return AI2C_OK;
}

/* Resident microbenchmark: upload one world, replicate, then run `iterations` steps
 * back to back without host synchronisation. Returns elapsed milliseconds. */
extern "C" int ai2c_cuda_bench_resident(const AI2CWorld *host_in, int lanes, int iterations,
                                        double dt, double bias_coef, double damping_dt, double *elapsed_ms) {
    int rc = ai2c_cuda_ensure(lanes);
    cudaEvent_t start, stop;
    float ms = 0.0f;
    const int block = 128;
    const int grid = (lanes + block - 1) / block;
    int i;
    if (rc != AI2C_OK) return rc;
    if (host_in == 0 || elapsed_ms == 0 || iterations <= 0) return AI2C_ERR_NULL;
    if (cudaMemcpy(g_worlds, host_in, sizeof(AI2CWorld), cudaMemcpyHostToDevice) != cudaSuccess) return AI2C_ERR_INVALID;
    ai2c_replicate_kernel<<<grid, block>>>(g_worlds, lanes);
    cudaDeviceSynchronize();
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (i = 0; i < iterations; ++i) {
        ai2c_step_kernel<<<grid, block>>>(g_worlds, lanes, dt, bias_coef, damping_dt, g_errors);
    }
    cudaEventRecord(stop);
    if (cudaEventSynchronize(stop) != cudaSuccess) return AI2C_ERR_INVALID;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    *elapsed_ms = (double)ms;
    return AI2C_OK;
}
