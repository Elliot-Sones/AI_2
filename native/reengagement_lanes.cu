/* Device-resident B2 reengagement episodes, one learner/thread. Only setup and
 * explicit diagnostics transfer host data. Host calls per context serialize.
 */
#include <cuda_runtime.h>
#include "reengagement_lanes.h"
#include <stdlib.h>

struct AI2B2LanesCuda {
    AI2NConfig cfg;
    double scale;
    int lanes, capacity, episodes_per_lane, device, last_cuda_error;
    AI2B2Lane *state;
    AI2GPlatformTable *table;
    AI2B2EpisodeRecord *records;
    AI2B2Scenario *scenarios;
    int *counters;
    int32_t *errors, *reset_errors;
    uint64_t *valid_steps;
    uint8_t *mask;
    cudaEvent_t ready;
};

#define AI2B2_LANES_THREADS 64

__global__ void ai2b2_lanes_initialize_kernel(AI2B2Lane *state, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) state[i].scenario_index = -1;
}

__global__ void ai2b2_lanes_reset_kernel(AI2B2Lane *state, AI2NConfig cfg, double scale,
    const AI2B2Scenario *scenarios, int episodes_per_lane, const uint8_t *mask,
    float *obs, int32_t *errors, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (mask[i]) ai2b2_lane_begin(state + i, i, &cfg, scale, scenarios, episodes_per_lane, obs + (size_t)i * 64);
    else ai2b2_lane_obs(state + i, obs + (size_t)i * 64);
    errors[i] = state[i].error;
}

__global__ void ai2b2_lanes_step_kernel(AI2B2Lane *state, AI2NConfig cfg,
    const AI2GPlatformTable *table, double scale, const AI2B2Scenario *scenarios,
    int episodes_per_lane, const float *actions, float *obs, float *terminal,
    float *rewards, uint8_t *terminated, uint8_t *truncated,
    AI2B2EpisodeRecord *records, int capacity, int *counters, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ai2b2_lane_step(state + i, i, &cfg, table, scale, scenarios, episodes_per_lane,
        actions + (size_t)i * 10, obs + (size_t)i * 64, terminal + (size_t)i * 64,
        rewards + i, terminated + i, truncated + i, records, capacity, counters, counters + 1);
}

__global__ void ai2b2_lanes_status_kernel(const AI2B2Lane *state, int32_t *errors,
    uint64_t *steps, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    errors[i] = state[i].error; steps[i] = state[i].valid_steps;
}

static int ai2b2_cuda_fail(AI2B2LanesCuda *h, cudaError_t error) {
    if (h && !h->last_cuda_error) h->last_cuda_error = (int)error;
    return AI2B2_LANE_ERR_CUDA;
}

static int ai2b2_cuda_owner(AI2B2LanesCuda *h) {
    int device;
    cudaError_t error;
    if (!h) return AI2N_ERR_NULL;
    error = cudaGetDevice(&device);
    if (error != cudaSuccess) return ai2b2_cuda_fail(h, error);
    return device == h->device ? 0 : AI2B2_LANE_ERR_DEVICE;
}

static int ai2b2_cuda_begin(AI2B2LanesCuda *h, cudaStream_t stream) {
    const int rc = ai2b2_cuda_owner(h);
    cudaError_t error;
    if (rc) return rc;
    if (h->last_cuda_error) return AI2B2_LANE_ERR_CUDA;
    error = cudaStreamWaitEvent(stream, h->ready, 0);
    return error == cudaSuccess ? 0 : ai2b2_cuda_fail(h, error);
}

static int ai2b2_cuda_end(AI2B2LanesCuda *h, cudaStream_t stream, int synchronize) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return ai2b2_cuda_fail(h, error);
    error = cudaEventRecord(h->ready, stream);
    if (error != cudaSuccess) return ai2b2_cuda_fail(h, error);
    if (synchronize && (error = cudaStreamSynchronize(stream)) != cudaSuccess) return ai2b2_cuda_fail(h, error);
    return 0;
}

static cudaError_t ai2b2_cuda_release(AI2B2LanesCuda *h) {
    cudaError_t first = cudaSuccess, error;
#define AI2B2_FREE(pointer) do { if (pointer) { error = cudaFree(pointer); if (first == cudaSuccess && error != cudaSuccess) first = error; } } while (0)
    AI2B2_FREE(h->state); AI2B2_FREE(h->table); AI2B2_FREE(h->records);
    AI2B2_FREE(h->scenarios); AI2B2_FREE(h->counters); AI2B2_FREE(h->errors);
    AI2B2_FREE(h->valid_steps); AI2B2_FREE(h->mask);
#undef AI2B2_FREE
    if (h->ready) {
        error = cudaEventDestroy(h->ready);
        if (first == cudaSuccess && error != cudaSuccess) first = error;
    }
    free(h->reset_errors);
    return first;
}

extern "C" size_t ai2b2_lanes_cuda_lane_size(void) { return sizeof(AI2B2Lane); }
extern "C" size_t ai2b2_lanes_cuda_record_size(void) { return sizeof(AI2B2EpisodeRecord); }

#define AI2B2_CUDA_FIELD_TABLE(prefix) \
    extern "C" size_t ai2b2_lanes_cuda_##prefix##_field_count(void) { \
        return sizeof(ai2b2_lanes_##prefix##_offsets) / sizeof(ai2b2_lanes_##prefix##_offsets[0]); \
    } \
    extern "C" size_t ai2b2_lanes_cuda_##prefix##_field_offset(size_t index) { \
        return index < ai2b2_lanes_cuda_##prefix##_field_count() ? ai2b2_lanes_##prefix##_offsets[index] : (size_t)-1; \
    }

static const size_t ai2b2_lanes_lane_offsets[] = {
    offsetof(AI2B2Lane, episode),
    offsetof(AI2B2Lane, rng),
    offsetof(AI2B2Lane, ret),
    offsetof(AI2B2Lane, last_obs),
    offsetof(AI2B2Lane, valid_steps),
    offsetof(AI2B2Lane, length),
    offsetof(AI2B2Lane, episodes),
    offsetof(AI2B2Lane, error),
    offsetof(AI2B2Lane, active),
    offsetof(AI2B2Lane, scenario_index)
};
AI2B2_CUDA_FIELD_TABLE(lane)

static const size_t ai2b2_lanes_record_offsets[] = {
    offsetof(AI2B2EpisodeRecord, lane),
    offsetof(AI2B2EpisodeRecord, length),
    offsetof(AI2B2EpisodeRecord, terminated),
    offsetof(AI2B2EpisodeRecord, truncated),
    offsetof(AI2B2EpisodeRecord, episode_index),
    offsetof(AI2B2EpisodeRecord, scenario_index),
    offsetof(AI2B2EpisodeRecord, ret),
    offsetof(AI2B2EpisodeRecord, diagnostics)
};
AI2B2_CUDA_FIELD_TABLE(record)

extern "C" int ai2b2_lanes_cuda_device(void *handle) {
    return handle ? ((AI2B2LanesCuda*)handle)->device : -1;
}
extern "C" int ai2b2_lanes_cuda_last_cuda_error(void *handle) {
    return handle ? ((AI2B2LanesCuda*)handle)->last_cuda_error : -1;
}

extern "C" void *ai2b2_lanes_cuda_create(const AI2NConfig *cfg, const AI2GPlatformTable *table,
    int lanes, int capacity, double scale, const AI2B2Scenario *scenarios,
    int episodes_per_lane, int *rc) {
    AI2B2LanesCuda *h;
    cudaError_t error = cudaSuccess;
    size_t stack = 0;
    const unsigned blocks = lanes > 0 ? ((unsigned)lanes + AI2B2_LANES_THREADS - 1) / AI2B2_LANES_THREADS : 0;
    if (rc) *rc = AI2N_ERR_CONFIG;
    if (!ai2b2_lanes_config_valid(cfg, table, lanes, capacity, scale, scenarios, episodes_per_lane)) return NULL;
    if (rc) *rc = AI2N_ERR_CAPACITY;
    h = (AI2B2LanesCuda*)calloc(1, sizeof(*h));
    if (!h) return NULL;
    h->cfg = *cfg; h->scale = scale; h->lanes = lanes;
    h->capacity = capacity; h->episodes_per_lane = episodes_per_lane;
    h->reset_errors = (int32_t*)calloc((size_t)lanes, sizeof(int32_t));
    if (!h->reset_errors) { free(h); return NULL; }
#define AI2B2_CREATE_CHECK(expr) do { error = (expr); if (error != cudaSuccess) goto fail; } while (0)
    AI2B2_CREATE_CHECK(cudaGetDevice(&h->device));
    AI2B2_CREATE_CHECK(cudaDeviceGetLimit(&stack, cudaLimitStackSize));
    if (stack < 32768) AI2B2_CREATE_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 32768));
    AI2B2_CREATE_CHECK(cudaMalloc((void**)&h->state, (size_t)lanes * sizeof(AI2B2Lane)));
    AI2B2_CREATE_CHECK(cudaMalloc((void**)&h->table, sizeof(AI2GPlatformTable)));
    AI2B2_CREATE_CHECK(cudaMalloc((void**)&h->records, (size_t)capacity * sizeof(AI2B2EpisodeRecord)));
    AI2B2_CREATE_CHECK(cudaMalloc((void**)&h->scenarios, (size_t)lanes * episodes_per_lane * sizeof(AI2B2Scenario)));
    AI2B2_CREATE_CHECK(cudaMalloc((void**)&h->counters, 2 * sizeof(int)));
    AI2B2_CREATE_CHECK(cudaMalloc((void**)&h->errors, (size_t)lanes * sizeof(int32_t)));
    AI2B2_CREATE_CHECK(cudaMalloc((void**)&h->valid_steps, (size_t)lanes * sizeof(uint64_t)));
    AI2B2_CREATE_CHECK(cudaMalloc((void**)&h->mask, (size_t)lanes));
    AI2B2_CREATE_CHECK(cudaEventCreateWithFlags(&h->ready, cudaEventDisableTiming));
    AI2B2_CREATE_CHECK(cudaMemcpy(h->table, table, sizeof(*table), cudaMemcpyHostToDevice));
    AI2B2_CREATE_CHECK(cudaMemcpy(h->scenarios, scenarios,
        (size_t)lanes * episodes_per_lane * sizeof(AI2B2Scenario), cudaMemcpyHostToDevice));
    AI2B2_CREATE_CHECK(cudaMemset(h->state, 0, (size_t)lanes * sizeof(AI2B2Lane)));
    AI2B2_CREATE_CHECK(cudaMemset(h->counters, 0, 2 * sizeof(int)));
    ai2b2_lanes_initialize_kernel<<<blocks, AI2B2_LANES_THREADS>>>(h->state, lanes);
    AI2B2_CREATE_CHECK(cudaGetLastError());
    AI2B2_CREATE_CHECK(cudaEventRecord(h->ready, 0));
    AI2B2_CREATE_CHECK(cudaEventSynchronize(h->ready));
    if (rc) *rc = 0;
    return h;
fail:
    h->last_cuda_error = (int)error;
    if (ai2b2_cuda_release(h) != cudaSuccess && rc) *rc = AI2B2_LANE_ERR_CUDA;
    if (rc) *rc = AI2B2_LANE_ERR_CUDA;
    free(h); return NULL;
#undef AI2B2_CREATE_CHECK
}

extern "C" int ai2b2_lanes_cuda_destroy(void *handle) {
    AI2B2LanesCuda *h = (AI2B2LanesCuda*)handle;
    int rc = ai2b2_cuda_owner(h);
    cudaError_t first, release;
    if (rc) return rc;
    first = cudaEventSynchronize(h->ready);
    release = ai2b2_cuda_release(h);
    free(h);
    return first != cudaSuccess || release != cudaSuccess ? AI2B2_LANE_ERR_CUDA : 0;
}

extern "C" int ai2b2_lanes_cuda_reset(void *handle, const uint8_t *mask, float *obs, void *stream) {
    AI2B2LanesCuda *h = (AI2B2LanesCuda*)handle;
    cudaStream_t s = (cudaStream_t)stream;
    cudaError_t error;
    int rc, i;
    if (!h || !obs) return AI2N_ERR_NULL;
    if ((rc = ai2b2_cuda_begin(h, s))) return rc;
    error = mask ? cudaMemcpyAsync(h->mask, mask, (size_t)h->lanes, cudaMemcpyHostToDevice, s) :
        cudaMemsetAsync(h->mask, 1, (size_t)h->lanes, s);
    if (error != cudaSuccess) return ai2b2_cuda_fail(h, error);
    const unsigned blocks = ((unsigned)h->lanes + AI2B2_LANES_THREADS - 1) / AI2B2_LANES_THREADS;
    ai2b2_lanes_reset_kernel<<<blocks, AI2B2_LANES_THREADS, 0, s>>>(h->state, h->cfg,
        h->scale, h->scenarios, h->episodes_per_lane, h->mask, obs, h->errors, h->lanes);
    if ((error = cudaGetLastError()) != cudaSuccess) return ai2b2_cuda_fail(h, error);
    error = cudaMemcpyAsync(h->reset_errors, h->errors, (size_t)h->lanes * sizeof(int32_t), cudaMemcpyDeviceToHost, s);
    if (error != cudaSuccess) return ai2b2_cuda_fail(h, error);
    if ((rc = ai2b2_cuda_end(h, s, 1))) return rc;
    for (i = 0; i < h->lanes; ++i) if ((!mask || mask[i]) && h->reset_errors[i]) return h->reset_errors[i];
    return 0;
}

extern "C" int ai2b2_lanes_cuda_step(void *handle, const float *actions, float *obs,
    float *terminal, float *rewards, uint8_t *terminated, uint8_t *truncated, void *stream) {
    AI2B2LanesCuda *h = (AI2B2LanesCuda*)handle;
    cudaStream_t s = (cudaStream_t)stream;
    int rc;
    if (!h || !actions || !obs || !terminal || !rewards || !terminated || !truncated) return AI2N_ERR_NULL;
    if ((rc = ai2b2_cuda_begin(h, s))) return rc;
    const unsigned blocks = ((unsigned)h->lanes + AI2B2_LANES_THREADS - 1) / AI2B2_LANES_THREADS;
    ai2b2_lanes_step_kernel<<<blocks, AI2B2_LANES_THREADS, 0, s>>>(h->state, h->cfg,
        h->table, h->scale, h->scenarios, h->episodes_per_lane, actions, obs, terminal,
        rewards, terminated, truncated, h->records, h->capacity, h->counters, h->lanes);
    return ai2b2_cuda_end(h, s, 0);
}

extern "C" int ai2b2_lanes_cuda_synchronize(void *handle, void *stream) {
    AI2B2LanesCuda *h = (AI2B2LanesCuda*)handle;
    const int rc = ai2b2_cuda_begin(h, (cudaStream_t)stream);
    return rc ? rc : ai2b2_cuda_end(h, (cudaStream_t)stream, 1);
}

extern "C" int ai2b2_lanes_cuda_drain(void *handle, AI2B2EpisodeRecord *out, int capacity,
    int *count, int *overflow, int32_t *errors, uint64_t *valid_steps, void *stream) {
    AI2B2LanesCuda *h = (AI2B2LanesCuda*)handle;
    cudaStream_t s = (cudaStream_t)stream;
    cudaError_t error;
    int counters[2], rc;
    if (!h || !count || !overflow || !errors || !valid_steps) return AI2N_ERR_NULL;
    if ((rc = ai2b2_cuda_begin(h, s))) return rc;
    error = cudaMemcpyAsync(counters, h->counters, sizeof(counters), cudaMemcpyDeviceToHost, s);
    if (error != cudaSuccess) return ai2b2_cuda_fail(h, error);
    if ((error = cudaStreamSynchronize(s)) != cudaSuccess) return ai2b2_cuda_fail(h, error);
    if (capacity < counters[0]) return AI2N_ERR_CAPACITY;
    if (counters[0] && !out) return AI2N_ERR_NULL;
    const unsigned blocks = ((unsigned)h->lanes + AI2B2_LANES_THREADS - 1) / AI2B2_LANES_THREADS;
    ai2b2_lanes_status_kernel<<<blocks, AI2B2_LANES_THREADS, 0, s>>>(h->state, h->errors, h->valid_steps, h->lanes);
    if ((error = cudaGetLastError()) != cudaSuccess) return ai2b2_cuda_fail(h, error);
    if (counters[0] && (error = cudaMemcpyAsync(out, h->records, (size_t)counters[0] * sizeof(*out), cudaMemcpyDeviceToHost, s)) != cudaSuccess)
        return ai2b2_cuda_fail(h, error);
    if ((error = cudaMemcpyAsync(errors, h->errors, (size_t)h->lanes * sizeof(int32_t), cudaMemcpyDeviceToHost, s)) != cudaSuccess ||
        (error = cudaMemcpyAsync(valid_steps, h->valid_steps, (size_t)h->lanes * sizeof(uint64_t), cudaMemcpyDeviceToHost, s)) != cudaSuccess)
        return ai2b2_cuda_fail(h, error);
    if ((error = cudaStreamSynchronize(s)) != cudaSuccess) return ai2b2_cuda_fail(h, error);
    if ((error = cudaMemsetAsync(h->counters, 0, 2 * sizeof(int), s)) != cudaSuccess) return ai2b2_cuda_fail(h, error);
    if ((rc = ai2b2_cuda_end(h, s, 1))) return rc;
    *count = counters[0]; *overflow = counters[1]; return 0;
}

extern "C" int ai2b2_lanes_cuda_lane_get(void *handle, int index, AI2B2Lane *out, void *stream) {
    AI2B2LanesCuda *h = (AI2B2LanesCuda*)handle;
    cudaStream_t s = (cudaStream_t)stream;
    int rc;
    cudaError_t error;
    if (!h || !out) return AI2N_ERR_NULL;
    if (index < 0 || index >= h->lanes) return AI2N_ERR_CONFIG;
    if ((rc = ai2b2_cuda_begin(h, s))) return rc;
    error = cudaMemcpyAsync(out, h->state + index, sizeof(*out), cudaMemcpyDeviceToHost, s);
    return error == cudaSuccess ? ai2b2_cuda_end(h, s, 1) : ai2b2_cuda_fail(h, error);
}

extern "C" int ai2b2_lanes_cuda_lane_set(void *handle, int index, const AI2B2Lane *in, void *stream) {
    AI2B2LanesCuda *h = (AI2B2LanesCuda*)handle;
    cudaStream_t s = (cudaStream_t)stream;
    int rc, current_index;
    AI2B2Scenario expected;
    cudaError_t error;
    if (!h || !in) return AI2N_ERR_NULL;
    if (index < 0 || index >= h->lanes || !ai2b2_lane_import_valid(in, h->episodes_per_lane)) return AI2N_ERR_CONFIG;
    if ((rc = ai2b2_cuda_begin(h, s))) return rc;
    /* Diagnostic-only synchronization. Compare against the owned immutable
     * scenario and current resident cursor before copying any imported state. */
    error = cudaMemcpyAsync(&expected,
        h->scenarios + (size_t)index * h->episodes_per_lane + in->scenario_index,
        sizeof(expected), cudaMemcpyDeviceToHost, s);
    if (error != cudaSuccess) return ai2b2_cuda_fail(h, error);
    error = cudaMemcpyAsync(&current_index, &h->state[index].scenario_index,
        sizeof(current_index), cudaMemcpyDeviceToHost, s);
    if (error != cudaSuccess) return ai2b2_cuda_fail(h, error);
    if ((rc = ai2b2_cuda_end(h, s, 1))) return rc;
    if (!ai2b2_lane_scenario_matches(in, &expected, current_index)) return AI2N_ERR_CONFIG;
    error = cudaMemcpyAsync(h->state + index, in, sizeof(*in), cudaMemcpyHostToDevice, s);
    return error == cudaSuccess ? ai2b2_cuda_end(h, s, 1) : ai2b2_cuda_fail(h, error);
}
