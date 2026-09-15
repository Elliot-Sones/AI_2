/* Device-resident whole games, one thread per game. No per-step host reads.
 * Build: nvcc -std=c++17 -O2 --fmad=false -shared -Xcompiler=-fPIC ...
 * Each context owns every allocation, tape and dependency event. Host calls
 * into one context must be serialized. No process-global game state exists.
 */
#include <cuda_runtime.h>
#include "game_lanes.h"
#include <stdlib.h>
#include <string.h>

struct AI2GLanesCuda {
    AI2NConfig cfg;
    double scale;
    int games, capacity, device, last_cuda_error;
    AI2GLane *lanes;
    AI2GPlatformTable *table;
    AI2GEpisodeRecord *records;
    int *counters; /* stored records, lost records */
    int *replay_status; /* mode, cursor, count, error per lane for the last diagnostic replay step. */
    int32_t *errors;
    uint64_t *valid_steps, *seeds;
    uint8_t *mask;
    AI2NDraw **tapes; /* Host array containing separately-owned device pointers. */
    int *tape_counts;
    int *tape_capacities;
    cudaEvent_t ready;
};

#define AI2G_LANES_THREADS 64

__global__ void ai2g_lanes_reset_kernel(AI2GLane *lanes, AI2NConfig cfg, double scale,
                                        const uint64_t *seeds, const uint8_t *mask, float *obs, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (mask[i]) {
        ai2n_rng_seed(&lanes[i].rng, seeds[i], (unsigned long long)i);
        ai2g_lane_begin(&lanes[i], &cfg, scale, obs + (size_t)i * 128, 1);
    } else ai2g_lane_obs(&lanes[i], obs + (size_t)i * 128);
}

__global__ void ai2g_lanes_step_kernel(AI2GLane *lanes, AI2NConfig cfg, const AI2GPlatformTable *table,
                                       double scale, const float *actions, float *obs, float *terminal,
                                       float *rewards, uint8_t *terminated, uint8_t *truncated,
                                       AI2GEpisodeRecord *records, int capacity, int *counters, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ai2g_lane_step(&lanes[i], i, &cfg, table, scale, actions + (size_t)i * 20,
        obs + (size_t)i * 128, terminal + (size_t)i * 128, rewards + (size_t)i * 2,
        terminated + i, truncated + i, records, capacity, counters, counters + 1);
}

__global__ void ai2g_lanes_status_kernel(const AI2GLane *lanes, int32_t *errors, uint64_t *steps, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    errors[i] = lanes[i].error; steps[i] = lanes[i].valid_steps;
}

__global__ void ai2g_lanes_replay_kernel(AI2GLane *lane, AI2NConfig cfg, double scale,
                                         const AI2NDraw *draws, int count, float *obs) {
    ai2n_rng_replay(&lane->rng, draws, count);
    ai2g_lane_begin(lane, &cfg, scale, obs, 1);
}

__global__ void ai2g_lanes_configure_episode_kernel(AI2GLane *lanes, int index,
                                                    int player0_stocks, int player1_stocks,
                                                    int max_timesteps) {
    if (player0_stocks >= 0) lanes[index].game.scene.players[0].ms.stocks = player0_stocks;
    if (player1_stocks >= 0) lanes[index].game.scene.players[1].ms.stocks = player1_stocks;
    if (max_timesteps > 0) lanes[index].game.max_timesteps = max_timesteps;
}

__global__ void ai2g_lanes_replay_step_kernel(AI2GLane *lanes, int index, AI2NConfig cfg,
                                              const AI2GPlatformTable *table, double scale,
                                              const float *actions, const AI2NDraw *draws, int draw_count,
                                              float *obs, float *terminal, float *rewards,
                                              uint8_t *terminated, uint8_t *truncated,
                                              AI2GEpisodeRecord *records, int capacity, int *counters,
                                              int *status) {
    AI2NRng saved = lanes[index].rng;
    int rc;
    ai2n_rng_replay(&lanes[index].rng, draws, draw_count);
    rc = ai2g_lane_step(&lanes[index], index, &cfg, table, scale, actions, obs, terminal, rewards,
        terminated, truncated, records, capacity, counters, counters + 1);
    status[0] = lanes[index].rng.mode;
    status[1] = lanes[index].rng.cursor;
    status[2] = lanes[index].rng.count;
    status[3] = lanes[index].rng.error ? lanes[index].rng.error : rc;
    lanes[index].rng = saved;
}

static int ai2g_cuda_fail(AI2GLanesCuda *h, cudaError_t error) {
    if (h && !h->last_cuda_error) h->last_cuda_error = (int)error;
    return AI2G_LANE_ERR_CUDA;
}

static int ai2g_cuda_owner(AI2GLanesCuda *h) {
    int device;
    cudaError_t err;
    if (!h) return AI2N_ERR_NULL;
    err = cudaGetDevice(&device);
    if (err != cudaSuccess) return ai2g_cuda_fail(h, err);
    return device == h->device ? 0 : AI2G_LANE_ERR_DEVICE;
}

/* Event ordering supports a different collector stream without a host sync. */
static int ai2g_cuda_begin(AI2GLanesCuda *h, cudaStream_t stream) {
    const int rc = ai2g_cuda_owner(h);
    cudaError_t err;
    if (rc) return rc;
    if (h->last_cuda_error) return AI2G_LANE_ERR_CUDA;
    err = cudaStreamWaitEvent(stream, h->ready, 0);
    return err == cudaSuccess ? 0 : ai2g_cuda_fail(h, err);
}

static int ai2g_cuda_end(AI2GLanesCuda *h, cudaStream_t stream, int sync) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return ai2g_cuda_fail(h, err);
    err = cudaEventRecord(h->ready, stream);
    if (err != cudaSuccess) return ai2g_cuda_fail(h, err);
    if (sync && (err = cudaStreamSynchronize(stream)) != cudaSuccess) return ai2g_cuda_fail(h, err);
    return 0;
}

/* Best-effort unwind checks every release and returns the first CUDA failure.
 * Called only on the owning device, including during partially-built create. */
static cudaError_t ai2g_cuda_release(AI2GLanesCuda *h) {
    cudaError_t first = cudaSuccess, err;
    int i;
#define AI2G_FREE(p) do { if ((p)) { err = cudaFree(p); if (first == cudaSuccess && err != cudaSuccess) first = err; } } while (0)
    if (h->tapes) for (i = 0; i < h->games; ++i) AI2G_FREE(h->tapes[i]);
    AI2G_FREE(h->lanes); AI2G_FREE(h->table); AI2G_FREE(h->records);
    AI2G_FREE(h->counters); AI2G_FREE(h->replay_status); AI2G_FREE(h->errors); AI2G_FREE(h->valid_steps);
    AI2G_FREE(h->seeds); AI2G_FREE(h->mask);
#undef AI2G_FREE
    if (h->ready) {
        err = cudaEventDestroy(h->ready);
        if (first == cudaSuccess && err != cudaSuccess) first = err;
    }
    free(h->tapes); free(h->tape_counts);
    free(h->tape_capacities);
    return first;
}

extern "C" size_t ai2g_lanes_cuda_lane_size(void) { return sizeof(AI2GLane); }
extern "C" size_t ai2g_lanes_cuda_record_size(void) { return sizeof(AI2GEpisodeRecord); }
extern "C" int ai2g_lanes_cuda_device(void *handle) {
    return handle ? ((AI2GLanesCuda*)handle)->device : -1;
}
extern "C" int ai2g_lanes_cuda_last_cuda_error(void *handle) {
    return handle ? ((AI2GLanesCuda*)handle)->last_cuda_error : -1;
}

extern "C" void *ai2g_lanes_cuda_create(const AI2NConfig *cfg, const AI2GPlatformTable *table,
                                        int games, int capacity, double scale, int *rc) {
    AI2GLanesCuda *h;
    cudaError_t err = cudaSuccess;
    size_t stack = 0;
    if (rc) *rc = AI2N_ERR_CONFIG;
    if (!ai2g_lanes_config_valid(cfg, table, games, capacity, scale)) return 0;
    if (rc) *rc = AI2N_ERR_CAPACITY;
    h = (AI2GLanesCuda*)calloc(1, sizeof(*h));
    if (!h) return 0;
    h->cfg = *cfg; h->games = games; h->capacity = capacity; h->scale = scale;
    h->tapes = (AI2NDraw**)calloc((size_t)games, sizeof(AI2NDraw*));
    h->tape_counts = (int*)calloc((size_t)games, sizeof(int));
    h->tape_capacities = (int*)calloc((size_t)games, sizeof(int));
    if (!h->tapes || !h->tape_counts || !h->tape_capacities) {
        free(h->tapes); free(h->tape_counts); free(h->tape_capacities); free(h); return 0;
    }
#define AI2G_CREATE_CHECK(expr) do { err = (expr); if (err != cudaSuccess) goto fail; } while (0)
    AI2G_CREATE_CHECK(cudaGetDevice(&h->device));
    /* Recursive contact helpers need stack space; only increase the owning
     * device's CUDA runtime limit at setup, never during a collector step. */
    AI2G_CREATE_CHECK(cudaDeviceGetLimit(&stack, cudaLimitStackSize));
    if (stack < 32768) AI2G_CREATE_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 32768));
    AI2G_CREATE_CHECK(cudaMalloc((void**)&h->lanes, (size_t)games * sizeof(AI2GLane)));
    AI2G_CREATE_CHECK(cudaMalloc((void**)&h->table, sizeof(AI2GPlatformTable)));
    AI2G_CREATE_CHECK(cudaMalloc((void**)&h->records, (size_t)capacity * sizeof(AI2GEpisodeRecord)));
    AI2G_CREATE_CHECK(cudaMalloc((void**)&h->counters, 2 * sizeof(int)));
    AI2G_CREATE_CHECK(cudaMalloc((void**)&h->replay_status, (size_t)games * 4 * sizeof(int)));
    AI2G_CREATE_CHECK(cudaMalloc((void**)&h->errors, (size_t)games * sizeof(int32_t)));
    AI2G_CREATE_CHECK(cudaMalloc((void**)&h->valid_steps, (size_t)games * sizeof(uint64_t)));
    AI2G_CREATE_CHECK(cudaMalloc((void**)&h->seeds, (size_t)games * sizeof(uint64_t)));
    AI2G_CREATE_CHECK(cudaMalloc((void**)&h->mask, (size_t)games));
    AI2G_CREATE_CHECK(cudaEventCreateWithFlags(&h->ready, cudaEventDisableTiming));
    AI2G_CREATE_CHECK(cudaMemcpy(h->table, table, sizeof(*table), cudaMemcpyHostToDevice));
    AI2G_CREATE_CHECK(cudaMemset(h->lanes, 0, (size_t)games * sizeof(AI2GLane)));
    AI2G_CREATE_CHECK(cudaMemset(h->counters, 0, 2 * sizeof(int)));
    AI2G_CREATE_CHECK(cudaMemset(h->replay_status, 0, (size_t)games * 4 * sizeof(int)));
    AI2G_CREATE_CHECK(cudaEventRecord(h->ready, 0));
    AI2G_CREATE_CHECK(cudaEventSynchronize(h->ready));
    if (rc) *rc = 0;
    return h;
fail:
    h->last_cuda_error = (int)err;
    /* Primary failure is already explicit; also check unwind failure. */
    if (ai2g_cuda_release(h) != cudaSuccess && rc) *rc = AI2G_LANE_ERR_CUDA;
    if (rc) *rc = AI2G_LANE_ERR_CUDA;
    free(h); return 0;
#undef AI2G_CREATE_CHECK
}

extern "C" int ai2g_lanes_cuda_destroy(void *handle) {
    AI2GLanesCuda *h = (AI2GLanesCuda*)handle;
    int rc = ai2g_cuda_owner(h);
    cudaError_t first, release;
    if (rc) return rc; /* Do not free the wrong device's resources. */
    first = cudaEventSynchronize(h->ready);
    release = ai2g_cuda_release(h);
    free(h);
    return first != cudaSuccess || release != cudaSuccess ? AI2G_LANE_ERR_CUDA : 0;
}

extern "C" int ai2g_lanes_cuda_reset(void *handle, const uint64_t *seeds, const uint8_t *mask,
                                     float *obs, void *stream) {
    AI2GLanesCuda *h = (AI2GLanesCuda*)handle;
    cudaStream_t s = (cudaStream_t)stream;
    cudaError_t err;
    int rc;
    if (!h || !seeds || !obs) return AI2N_ERR_NULL;
    if ((rc = ai2g_cuda_begin(h, s))) return rc;
    err = cudaMemcpyAsync(h->seeds, seeds, (size_t)h->games * sizeof(uint64_t), cudaMemcpyHostToDevice, s);
    if (err != cudaSuccess) return ai2g_cuda_fail(h, err);
    err = mask ? cudaMemcpyAsync(h->mask, mask, (size_t)h->games, cudaMemcpyHostToDevice, s) :
                 cudaMemsetAsync(h->mask, 1, (size_t)h->games, s);
    if (err != cudaSuccess) return ai2g_cuda_fail(h, err);
    const unsigned blocks = ((unsigned)h->games + AI2G_LANES_THREADS - 1) / AI2G_LANES_THREADS;
    ai2g_lanes_reset_kernel<<<blocks, AI2G_LANES_THREADS, 0, s>>>(h->lanes, h->cfg, h->scale, h->seeds, h->mask, obs, h->games);
    return ai2g_cuda_end(h, s, 1); /* Host seed/mask lifetime ends on return. */
}

extern "C" int ai2g_lanes_cuda_step(void *handle, const float *actions, float *obs, float *terminal,
                                    float *rewards, uint8_t *terminated, uint8_t *truncated, void *stream) {
    AI2GLanesCuda *h = (AI2GLanesCuda*)handle;
    cudaStream_t s = (cudaStream_t)stream;
    int rc;
    if (!h || !actions || !obs || !terminal || !rewards || !terminated || !truncated) return AI2N_ERR_NULL;
    if ((rc = ai2g_cuda_begin(h, s))) return rc;
    const unsigned blocks = ((unsigned)h->games + AI2G_LANES_THREADS - 1) / AI2G_LANES_THREADS;
    ai2g_lanes_step_kernel<<<blocks, AI2G_LANES_THREADS, 0, s>>>(h->lanes, h->cfg, h->table, h->scale,
        actions, obs, terminal, rewards, terminated, truncated, h->records, h->capacity, h->counters, h->games);
    return ai2g_cuda_end(h, s, 0);
}

extern "C" int ai2g_lanes_cuda_synchronize(void *handle, void *stream) {
    AI2GLanesCuda *h = (AI2GLanesCuda*)handle;
    cudaStream_t s = (cudaStream_t)stream;
    const int rc = ai2g_cuda_begin(h, s);
    return rc ? rc : ai2g_cuda_end(h, s, 1);
}

extern "C" int ai2g_lanes_cuda_drain(void *handle, AI2GEpisodeRecord *out, int capacity, int *count,
                                     int *overflow, int32_t *errors, uint64_t *valid_steps, void *stream) {
    AI2GLanesCuda *h = (AI2GLanesCuda*)handle;
    cudaStream_t s = (cudaStream_t)stream;
    cudaError_t err;
    int counters[2], rc;
    if (!h || !count || !overflow || !errors || !valid_steps) return AI2N_ERR_NULL;
    if ((rc = ai2g_cuda_begin(h, s))) return rc;
    if ((err = cudaMemcpyAsync(counters, h->counters, sizeof(counters), cudaMemcpyDeviceToHost, s)) != cudaSuccess)
        return ai2g_cuda_fail(h, err);
    if ((err = cudaStreamSynchronize(s)) != cudaSuccess) return ai2g_cuda_fail(h, err);
    if (capacity < counters[0]) return AI2N_ERR_CAPACITY;
    if (counters[0] && !out) return AI2N_ERR_NULL;
    const unsigned blocks = ((unsigned)h->games + AI2G_LANES_THREADS - 1) / AI2G_LANES_THREADS;
    ai2g_lanes_status_kernel<<<blocks, AI2G_LANES_THREADS, 0, s>>>(h->lanes, h->errors, h->valid_steps, h->games);
    if ((err = cudaGetLastError()) != cudaSuccess) return ai2g_cuda_fail(h, err);
    if (counters[0] && (err = cudaMemcpyAsync(out, h->records, (size_t)counters[0] * sizeof(*out), cudaMemcpyDeviceToHost, s)) != cudaSuccess)
        return ai2g_cuda_fail(h, err);
    if ((err = cudaMemcpyAsync(errors, h->errors, (size_t)h->games * sizeof(int32_t), cudaMemcpyDeviceToHost, s)) != cudaSuccess ||
        (err = cudaMemcpyAsync(valid_steps, h->valid_steps, (size_t)h->games * sizeof(uint64_t), cudaMemcpyDeviceToHost, s)) != cudaSuccess)
        return ai2g_cuda_fail(h, err);
    /* Clear records only after all host copies completed successfully. */
    if ((err = cudaStreamSynchronize(s)) != cudaSuccess) return ai2g_cuda_fail(h, err);
    if ((err = cudaMemsetAsync(h->counters, 0, 2 * sizeof(int), s)) != cudaSuccess) return ai2g_cuda_fail(h, err);
    if ((rc = ai2g_cuda_end(h, s, 1))) return rc;
    *count = counters[0]; *overflow = counters[1]; return 0;
}

extern "C" int ai2g_lanes_cuda_lane_get(void *handle, int index, AI2GLane *out, void *stream) {
    AI2GLanesCuda *h = (AI2GLanesCuda*)handle;
    cudaStream_t s = (cudaStream_t)stream;
    cudaError_t err;
    int rc;
    if (!h || !out) return AI2N_ERR_NULL;
    if (index < 0 || index >= h->games) return AI2N_ERR_CONFIG;
    if ((rc = ai2g_cuda_begin(h, s))) return rc;
    err = cudaMemcpyAsync(out, h->lanes + index, sizeof(*out), cudaMemcpyDeviceToHost, s);
    return err == cudaSuccess ? ai2g_cuda_end(h, s, 1) : ai2g_cuda_fail(h, err);
}

extern "C" int ai2g_lanes_cuda_lane_set(void *handle, int index, const AI2GLane *in, void *stream) {
    AI2GLanesCuda *h = (AI2GLanesCuda*)handle;
    cudaStream_t s = (cudaStream_t)stream;
    AI2GLane *copy;
    cudaError_t err;
    int rc;
    if (!h || !in) return AI2N_ERR_NULL;
    if (index < 0 || index >= h->games || (in->rng.mode != 0 && in->rng.mode != 1)) return AI2N_ERR_CONFIG;
    if (!in->rng.mode && (in->rng.count != h->tape_counts[index] || in->rng.cursor < 0 || in->rng.cursor > in->rng.count)) return AI2N_ERR_RNG;
    if ((rc = ai2g_cuda_begin(h, s))) return rc;
    copy = (AI2GLane*)malloc(sizeof(*copy));
    if (!copy) return AI2N_ERR_CAPACITY;
    *copy = *in; copy->rng.draws = in->rng.mode ? 0 : h->tapes[index];
    err = cudaMemcpyAsync(h->lanes + index, copy, sizeof(*copy), cudaMemcpyHostToDevice, s);
    rc = err == cudaSuccess ? ai2g_cuda_end(h, s, 1) : ai2g_cuda_fail(h, err);
    free(copy); return rc;
}

extern "C" int ai2g_lanes_cuda_replay_reset(void *handle, int index, const AI2NDraw *draws,
                                            int count, float *obs, void *stream) {
    AI2GLanesCuda *h = (AI2GLanesCuda*)handle;
    cudaStream_t s = (cudaStream_t)stream;
    AI2NDraw *tape = 0;
    cudaError_t err;
    int rc;
    if (!h || !obs) return AI2N_ERR_NULL;
    if (index < 0 || index >= h->games) return AI2N_ERR_CONFIG;
    if (!ai2g_lane_draws_valid(draws, count)) return AI2N_ERR_RNG;
    if ((size_t)count > SIZE_MAX / sizeof(AI2NDraw)) return AI2N_ERR_CAPACITY;
    if ((rc = ai2g_cuda_begin(h, s))) return rc;
    /* Setup-only synchronization permits replacing just this lane's tape. */
    if ((err = cudaStreamSynchronize(s)) != cudaSuccess) return ai2g_cuda_fail(h, err);
    if (count) {
        if ((err = cudaMalloc((void**)&tape, (size_t)count * sizeof(*tape))) != cudaSuccess) return ai2g_cuda_fail(h, err);
        err = cudaMemcpy(tape, draws, (size_t)count * sizeof(*tape), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            const cudaError_t released = cudaFree(tape);
            if (released != cudaSuccess) ai2g_cuda_fail(h, released);
            return ai2g_cuda_fail(h, err);
        }
    }
    if (h->tapes[index] && (err = cudaFree(h->tapes[index])) != cudaSuccess) {
        if (tape) { const cudaError_t released = cudaFree(tape); if (released != cudaSuccess) ai2g_cuda_fail(h, released); }
        return ai2g_cuda_fail(h, err);
    }
    h->tapes[index] = tape; h->tape_counts[index] = count;
    h->tape_capacities[index] = count;
    ai2g_lanes_replay_kernel<<<1, 1, 0, s>>>(h->lanes + index, h->cfg, h->scale, tape, count, obs);
    return ai2g_cuda_end(h, s, 1);
}

extern "C" int ai2g_lanes_cuda_configure_episode(void *handle, int index, int player0_stocks,
                                                  int player1_stocks, int max_timesteps, void *stream) {
    AI2GLanesCuda *h = (AI2GLanesCuda*)handle;
    cudaStream_t s = (cudaStream_t)stream;
    int rc;
    if (!h) return AI2N_ERR_NULL;
    if (index < 0 || index >= h->games) return AI2N_ERR_CONFIG;
    if ((rc = ai2g_cuda_begin(h, s))) return rc;
    ai2g_lanes_configure_episode_kernel<<<1, 1, 0, s>>>(h->lanes, index, player0_stocks, player1_stocks, max_timesteps);
    return ai2g_cuda_end(h, s, 1);
}

extern "C" int ai2g_lanes_cuda_rng_status(void *handle, int index, int *mode, int *cursor,
                                           int *count, int *error, void *stream) {
    AI2GLanesCuda *h = (AI2GLanesCuda*)handle;
    cudaStream_t s = (cudaStream_t)stream;
    int status[4], rc;
    cudaError_t err;
    if (!h || !mode || !cursor || !count || !error) return AI2N_ERR_NULL;
    if (index < 0 || index >= h->games) return AI2N_ERR_CONFIG;
    if ((rc = ai2g_cuda_begin(h, s))) return rc;
    err = cudaMemcpyAsync(status, h->replay_status + (size_t)index * 4, sizeof(status), cudaMemcpyDeviceToHost, s);
    if (err != cudaSuccess) return ai2g_cuda_fail(h, err);
    if ((err = cudaStreamSynchronize(s)) != cudaSuccess) return ai2g_cuda_fail(h, err);
    if ((rc = ai2g_cuda_end(h, s, 1))) return rc;
    *mode = status[0]; *cursor = status[1]; *count = status[2]; *error = status[3];
    return 0;
}

extern "C" int ai2g_lanes_cuda_replay_step(void *handle, int index, const float *actions,
                                            const AI2NDraw *draws, int count, float *obs,
                                            float *terminal, float *rewards, uint8_t *terminated,
                                            uint8_t *truncated, void *stream) {
    AI2GLanesCuda *h = (AI2GLanesCuda*)handle;
    cudaStream_t s = (cudaStream_t)stream;
    int rc, status[4];
    cudaError_t err;
    if (!h || !actions || !obs || !terminal || !rewards || !terminated || !truncated) return AI2N_ERR_NULL;
    if (index < 0 || index >= h->games) return AI2N_ERR_CONFIG;
    if (!ai2g_lane_draws_valid(draws, count)) return AI2N_ERR_RNG;
    if ((size_t)count > SIZE_MAX / sizeof(AI2NDraw)) return AI2N_ERR_CAPACITY;
    if ((rc = ai2g_cuda_begin(h, s))) return rc;
    if (count > h->tape_capacities[index]) {
        AI2NDraw *replacement = 0;
        if ((err = cudaStreamSynchronize(s)) != cudaSuccess) return ai2g_cuda_fail(h, err);
        if ((err = cudaMalloc((void**)&replacement, (size_t)count * sizeof(*replacement))) != cudaSuccess)
            return ai2g_cuda_fail(h, err);
        if (h->tapes[index] && (err = cudaFree(h->tapes[index])) != cudaSuccess) {
            cudaFree(replacement);
            return ai2g_cuda_fail(h, err);
        }
        h->tapes[index] = replacement;
        h->tape_capacities[index] = count;
    }
    h->tape_counts[index] = count;
    if (count) {
        if ((err = cudaMemcpyAsync(h->tapes[index], draws, (size_t)count * sizeof(*draws), cudaMemcpyHostToDevice, s)) != cudaSuccess)
            return ai2g_cuda_fail(h, err);
    }
    ai2g_lanes_replay_step_kernel<<<1, 1, 0, s>>>(h->lanes, index, h->cfg, h->table, h->scale,
        actions, h->tapes[index], count, obs, terminal, rewards, terminated, truncated, h->records, h->capacity,
        h->counters, h->replay_status + (size_t)index * 4);
    err = cudaGetLastError();
    if (err != cudaSuccess) return ai2g_cuda_fail(h, err);
    if ((err = cudaMemcpyAsync(status, h->replay_status + (size_t)index * 4, sizeof(status), cudaMemcpyDeviceToHost, s)) != cudaSuccess) {
        return ai2g_cuda_fail(h, err);
    }
    if ((err = cudaStreamSynchronize(s)) != cudaSuccess) {
        return ai2g_cuda_fail(h, err);
    }
    if ((rc = ai2g_cuda_end(h, s, 1))) return rc;
    if (status[3]) return status[3] == AI2N_ERR_RNG ? AI2N_ERR_RNG : status[3];
    return status[1] == status[2] ? 0 : AI2N_ERR_RNG;
}
