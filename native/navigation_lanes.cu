/*
 * CUDA runtime for the navigation training lanes.  One thread per lane, the
 * lane state resident in device memory, actions read from and observations,
 * rewards and done flags written to device buffers owned by the collector
 * (torch tensors), on the collector's stream.  No host transfer per step;
 * completed-episode records and the error flag are read at rollout
 * boundaries.  Compiled with -fmad=false so the arithmetic matches the
 * unfused host reference in navigation_lanes.c.
 */
#include "navigation_lanes.h"

#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>

struct AI2NLanesCuda {
    AI2NConfig cfg;
    int lanes;
    int record_capacity;
    int last_cuda_error;
    AI2NLane *d_lanes;
    AI2NPlatformTable *d_table;
    AI2NEpisodeRecord *d_records;
    int *d_counters; /* [0] record count, [1] record overflow, [2] max lane error */
    unsigned long long *d_seeds;
};

#define AI2N_LANES_THREADS 128

__global__ void ai2n_lanes_reset_kernel(AI2NLane *lanes, AI2NConfig cfg, const unsigned long long *seeds, float *obs, int *counters, int n) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= n) return;
    const int rc = ai2n_lane_reset(&lanes[lane], &cfg, seeds[lane], (unsigned long long)lane, obs + (size_t)lane * AI2N_OBS_WIDTH);
    if (rc) atomicMax(&counters[2], rc);
}

__global__ void ai2n_lanes_step_kernel(AI2NLane *lanes, AI2NConfig cfg, const AI2NPlatformTable *table, const float *actions, float *obs, float *terminal,
                                       float *rewards, uint8_t *terminated, uint8_t *truncated, AI2NEpisodeRecord *records, int record_capacity,
                                       int *counters, int n) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= n) return;
    const int rc = ai2n_lane_step(&lanes[lane], lane, &cfg, table, actions + (size_t)lane * AI2N_ACTION_WIDTH, obs + (size_t)lane * AI2N_OBS_WIDTH,
                                  terminal + (size_t)lane * AI2N_OBS_WIDTH, rewards + lane, terminated + lane, truncated + lane, records, record_capacity,
                                  &counters[0], &counters[1]);
    if (rc) atomicMax(&counters[2], rc);
}

static int ai2n_lanes_cuda_fail(AI2NLanesCuda *h, cudaError_t err, int code) {
    if (h) h->last_cuda_error = (int)err;
    return code;
}

extern "C" int ai2n_lanes_cuda_lane_size(void) { return (int)sizeof(AI2NLane); }
extern "C" int ai2n_lanes_cuda_record_size(void) { return (int)sizeof(AI2NEpisodeRecord); }

extern "C" void *ai2n_lanes_cuda_create(const AI2NConfig *cfg, const AI2NPlatformTable *table, int lanes, int record_capacity, int *rc) {
    AI2NLanesCuda *h;
    cudaError_t err;
    if (rc) *rc = AI2N_ERR_NONE;
    if (cfg == 0 || table == 0 || lanes <= 0 || record_capacity <= 0) { if (rc) *rc = AI2N_ERR_CONFIG; return 0; }
    h = (AI2NLanesCuda *)calloc(1, sizeof(*h));
    if (h == 0) { if (rc) *rc = AI2N_ERR_CAPACITY; return 0; }
    memcpy(&h->cfg, cfg, sizeof(*cfg));
    h->lanes = lanes;
    h->record_capacity = record_capacity;
    /* Recursive tree helpers plus the state-machine locals need more stack than the default. */
    cudaDeviceSetLimit(cudaLimitStackSize, 32768);
    if ((err = cudaMalloc((void **)&h->d_lanes, sizeof(AI2NLane) * (size_t)lanes)) != cudaSuccess ||
        (err = cudaMalloc((void **)&h->d_table, sizeof(AI2NPlatformTable))) != cudaSuccess ||
        (err = cudaMalloc((void **)&h->d_records, sizeof(AI2NEpisodeRecord) * (size_t)record_capacity)) != cudaSuccess ||
        (err = cudaMalloc((void **)&h->d_counters, sizeof(int) * 3)) != cudaSuccess ||
        (err = cudaMalloc((void **)&h->d_seeds, sizeof(unsigned long long) * (size_t)lanes)) != cudaSuccess) {
        if (rc) *rc = AI2N_ERR_CAPACITY;
        h->last_cuda_error = (int)err;
        cudaFree(h->d_lanes); cudaFree(h->d_table); cudaFree(h->d_records); cudaFree(h->d_counters); cudaFree(h->d_seeds);
        free(h);
        return 0;
    }
    if ((err = cudaMemcpy(h->d_table, table, sizeof(AI2NPlatformTable), cudaMemcpyHostToDevice)) != cudaSuccess ||
        (err = cudaMemset(h->d_lanes, 0, sizeof(AI2NLane) * (size_t)lanes)) != cudaSuccess ||
        (err = cudaMemset(h->d_counters, 0, sizeof(int) * 3)) != cudaSuccess ||
        /* Creation uses the legacy default stream. Complete its initialization
           before reset/step can run on a nonblocking PyTorch current stream.
           This is startup-only; the per-step path remains asynchronous. */
        (err = cudaDeviceSynchronize()) != cudaSuccess) {
        if (rc) *rc = AI2N_ERR_CAPACITY;
        h->last_cuda_error = (int)err;
        cudaFree(h->d_lanes); cudaFree(h->d_table); cudaFree(h->d_records); cudaFree(h->d_counters); cudaFree(h->d_seeds);
        free(h);
        return 0;
    }
    return h;
}

extern "C" int ai2n_lanes_cuda_destroy(void *handle) {
    AI2NLanesCuda *h = (AI2NLanesCuda *)handle;
    if (h == 0) return AI2N_ERR_NULL;
    cudaFree(h->d_lanes); cudaFree(h->d_table); cudaFree(h->d_records); cudaFree(h->d_counters); cudaFree(h->d_seeds);
    free(h);
    return AI2N_ERR_NONE;
}

extern "C" int ai2n_lanes_cuda_last_cuda_error(void *handle) {
    AI2NLanesCuda *h = (AI2NLanesCuda *)handle;
    return h ? h->last_cuda_error : -1;
}

/* Seed every lane and reset it; `obs` is a device buffer of lanes * 64 floats. */
extern "C" int ai2n_lanes_cuda_reset(void *handle, const unsigned long long *seeds, float *obs, void *stream) {
    AI2NLanesCuda *h = (AI2NLanesCuda *)handle;
    cudaStream_t s = (cudaStream_t)stream;
    cudaError_t err;
    if (h == 0 || seeds == 0 || obs == 0) return AI2N_ERR_NULL;
    if ((err = cudaMemcpyAsync(h->d_seeds, seeds, sizeof(unsigned long long) * (size_t)h->lanes, cudaMemcpyHostToDevice, s)) != cudaSuccess) return ai2n_lanes_cuda_fail(h, err, AI2N_ERR_CAPACITY);
    if ((err = cudaMemsetAsync(h->d_counters, 0, sizeof(int) * 3, s)) != cudaSuccess) return ai2n_lanes_cuda_fail(h, err, AI2N_ERR_CAPACITY);
    {
        const int blocks = (h->lanes + AI2N_LANES_THREADS - 1) / AI2N_LANES_THREADS;
        ai2n_lanes_reset_kernel<<<blocks, AI2N_LANES_THREADS, 0, s>>>(h->d_lanes, h->cfg, h->d_seeds, obs, h->d_counters, h->lanes);
    }
    if ((err = cudaGetLastError()) != cudaSuccess) return ai2n_lanes_cuda_fail(h, err, AI2N_ERR_CONFIG);
    /* The seed buffer is read by the kernel on this stream; wait before the caller may reuse its host array. */
    if ((err = cudaStreamSynchronize(s)) != cudaSuccess) return ai2n_lanes_cuda_fail(h, err, AI2N_ERR_CONFIG);
    return AI2N_ERR_NONE;
}

/* Queue one step for every lane on `stream`; returns as soon as the kernel is launched. */
extern "C" int ai2n_lanes_cuda_step(void *handle, const float *actions, float *obs, float *terminal, float *rewards, uint8_t *terminated,
                                    uint8_t *truncated, void *stream) {
    AI2NLanesCuda *h = (AI2NLanesCuda *)handle;
    cudaStream_t s = (cudaStream_t)stream;
    cudaError_t err;
    if (h == 0 || actions == 0 || obs == 0 || terminal == 0 || rewards == 0 || terminated == 0 || truncated == 0) return AI2N_ERR_NULL;
    {
        const int blocks = (h->lanes + AI2N_LANES_THREADS - 1) / AI2N_LANES_THREADS;
        ai2n_lanes_step_kernel<<<blocks, AI2N_LANES_THREADS, 0, s>>>(h->d_lanes, h->cfg, h->d_table, actions, obs, terminal, rewards, terminated, truncated,
                                                                      h->d_records, h->record_capacity, h->d_counters, h->lanes);
    }
    if ((err = cudaGetLastError()) != cudaSuccess) return ai2n_lanes_cuda_fail(h, err, AI2N_ERR_CONFIG);
    return AI2N_ERR_NONE;
}

/* Synchronize the stream, copy the completed-episode records to the host and clear them. */
extern "C" int ai2n_lanes_cuda_drain(void *handle, AI2NEpisodeRecord *out, int capacity, int *count, int *overflow, void *stream) {
    AI2NLanesCuda *h = (AI2NLanesCuda *)handle;
    cudaStream_t s = (cudaStream_t)stream;
    cudaError_t err;
    int counters[3] = {0, 0, 0};
    int stored;
    if (h == 0 || count == 0 || overflow == 0) return AI2N_ERR_NULL;
    if ((err = cudaStreamSynchronize(s)) != cudaSuccess) return ai2n_lanes_cuda_fail(h, err, AI2N_ERR_CONFIG);
    if ((err = cudaMemcpy(counters, h->d_counters, sizeof(int) * 3, cudaMemcpyDeviceToHost)) != cudaSuccess) return ai2n_lanes_cuda_fail(h, err, AI2N_ERR_CAPACITY);
    stored = counters[0] < h->record_capacity ? counters[0] : h->record_capacity;
    if (stored > capacity) return AI2N_ERR_CAPACITY;
    if (stored > 0 && out == 0) return AI2N_ERR_NULL;
    if (stored > 0 && (err = cudaMemcpy(out, h->d_records, sizeof(AI2NEpisodeRecord) * (size_t)stored, cudaMemcpyDeviceToHost)) != cudaSuccess) return ai2n_lanes_cuda_fail(h, err, AI2N_ERR_CAPACITY);
    *count = stored;
    *overflow = counters[1];
    if ((err = cudaMemset(h->d_counters, 0, sizeof(int) * 2)) != cudaSuccess) return ai2n_lanes_cuda_fail(h, err, AI2N_ERR_CAPACITY);
    return AI2N_ERR_NONE;
}

/* Synchronize the stream and report the largest lane error code seen since the last reset (0 when none). */
extern "C" int ai2n_lanes_cuda_error(void *handle, int *error, void *stream) {
    AI2NLanesCuda *h = (AI2NLanesCuda *)handle;
    cudaStream_t s = (cudaStream_t)stream;
    cudaError_t err;
    int counters[3] = {0, 0, 0};
    if (h == 0 || error == 0) return AI2N_ERR_NULL;
    if ((err = cudaStreamSynchronize(s)) != cudaSuccess) return ai2n_lanes_cuda_fail(h, err, AI2N_ERR_CONFIG);
    if ((err = cudaMemcpy(counters, h->d_counters, sizeof(int) * 3, cudaMemcpyDeviceToHost)) != cudaSuccess) return ai2n_lanes_cuda_fail(h, err, AI2N_ERR_CAPACITY);
    *error = counters[2];
    return AI2N_ERR_NONE;
}

extern "C" int ai2n_lanes_cuda_synchronize(void *handle, void *stream) {
    AI2NLanesCuda *h = (AI2NLanesCuda *)handle;
    cudaError_t err;
    if (h == 0) return AI2N_ERR_NULL;
    if ((err = cudaStreamSynchronize((cudaStream_t)stream)) != cudaSuccess) return ai2n_lanes_cuda_fail(h, err, AI2N_ERR_CONFIG);
    return AI2N_ERR_NONE;
}

extern "C" int ai2n_lanes_cuda_lane_get(void *handle, int lane, AI2NLane *out, void *stream) {
    AI2NLanesCuda *h = (AI2NLanesCuda *)handle;
    cudaError_t err;
    if (h == 0 || out == 0) return AI2N_ERR_NULL;
    if (lane < 0 || lane >= h->lanes) return AI2N_ERR_CONFIG;
    if ((err = cudaStreamSynchronize((cudaStream_t)stream)) != cudaSuccess) return ai2n_lanes_cuda_fail(h, err, AI2N_ERR_CONFIG);
    if ((err = cudaMemcpy(out, h->d_lanes + lane, sizeof(AI2NLane), cudaMemcpyDeviceToHost)) != cudaSuccess) return ai2n_lanes_cuda_fail(h, err, AI2N_ERR_CAPACITY);
    return AI2N_ERR_NONE;
}

extern "C" int ai2n_lanes_cuda_lane_set(void *handle, int lane, const AI2NLane *in, void *stream) {
    AI2NLanesCuda *h = (AI2NLanesCuda *)handle;
    cudaError_t err;
    if (h == 0 || in == 0) return AI2N_ERR_NULL;
    if (lane < 0 || lane >= h->lanes) return AI2N_ERR_CONFIG;
    if ((err = cudaStreamSynchronize((cudaStream_t)stream)) != cudaSuccess) return ai2n_lanes_cuda_fail(h, err, AI2N_ERR_CONFIG);
    if ((err = cudaMemcpy(h->d_lanes + lane, in, sizeof(AI2NLane), cudaMemcpyHostToDevice)) != cudaSuccess) return ai2n_lanes_cuda_fail(h, err, AI2N_ERR_CAPACITY);
    return AI2N_ERR_NONE;
}
