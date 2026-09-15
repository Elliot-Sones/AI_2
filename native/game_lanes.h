#ifndef AI2_GAME_LANES_H
#define AI2_GAME_LANES_H

/* One lane is one complete, unmasked two-player game. Only the transport
 * differs between .c and .cu. Frame stacking/policy inference are external.
 * Buffers: action[n][2][10], obs/terminal[n][2][64], reward[n][2], flags[n].
 * Errors freeze a lane: cached last-valid observation, zero reward/no done,
 * and no increment of valid_steps. A partially-mutated failed game is never
 * resumed except by an explicit reset. Record overflow also freezes its lane.
 * reset preserves undrained records and lifetime valid_steps, including when
 * selective; autoreset preserves cumulative completed-episode indexing.
 */
#include "game_episode.h"
#include <limits.h>
#include <stddef.h>

#define AI2G_LANE_ERR_RECORD_CAPACITY 301
#define AI2G_LANE_ERR_COUNTER 302
#define AI2G_LANE_ERR_CUDA 303
#define AI2G_LANE_ERR_DEVICE 304
#define AI2G_LANE_OBS 128
#define AI2G_LANE_ACTIONS 20

typedef struct AI2GLane {
    AI2GState game;
    AI2NRng rng;
    double returns[2];
    float last_obs[AI2G_LANE_OBS];
    uint64_t valid_steps; /* Successful game ticks over this context lifetime. */
    int32_t length, episodes, error, active;
} AI2GLane;

typedef struct AI2GEpisodeRecord {
    int32_t lane, length, terminated, truncated, win_mask, episode_index;
    int32_t stocks[2];
    double returns[2], damage_done[2], damage_taken[2];
} AI2GEpisodeRecord;

AI2N_HD static inline void ai2g_lane_obs(const AI2GLane *lane, float *obs) {
    int i;
    for (i = 0; i < AI2G_LANE_OBS; ++i) obs[i] = lane->last_obs[i];
}

AI2N_HD static inline void ai2g_lane_hold(const AI2GLane *lane, float *obs, float *terminal,
                                         float *reward, uint8_t *terminated, uint8_t *truncated) {
    ai2g_lane_obs(lane, obs); ai2g_lane_obs(lane, terminal);
    reward[0] = reward[1] = 0.0f; *terminated = *truncated = 0;
}

AI2N_HD static inline int ai2g_lane_begin(AI2GLane *lane, const AI2NConfig *cfg,
                                         double pixels_per_world, float *obs, int seeded) {
    AI2GStepResult out;
    int i, j, rc;
    lane->returns[0] = lane->returns[1] = 0.0;
    lane->length = 0; lane->error = 0; lane->active = 0;
    if (seeded) lane->episodes = 0;
    rc = ai2g_reset(&lane->game, cfg, &lane->rng, pixels_per_world, &out);
    if (!rc && lane->rng.error) rc = AI2N_ERR_RNG;
    if (rc) {
        lane->error = rc;
        ai2g_lane_obs(lane, obs);
        return rc;
    }
    for (i = 0; i < 2; ++i) for (j = 0; j < 64; ++j)
        lane->last_obs[i * 64 + j] = (float)out.obs[i][j];
    ai2g_lane_obs(lane, obs); lane->active = 1;
    return 0;
}

/* Bounded reservation prevents counter wrap even after many explicit resets.
 * GPU ordering of records between lanes is unspecified; lane/index identify
 * records. Ordering within each lane is stable. */
AI2N_HD static inline int ai2g_lane_reserve(int *count, int capacity, int *overflow) {
#if defined(__CUDA_ARCH__)
    int old = atomicAdd(count, 0);
    while (old < capacity) {
        const int seen = atomicCAS(count, old, old + 1);
        if (seen == old) return old;
        old = seen;
    }
    old = atomicAdd(overflow, 0);
    while (old < INT_MAX) {
        const int seen = atomicCAS(overflow, old, old + 1);
        if (seen == old) break;
        old = seen;
    }
#else
    if (*count < capacity) return (*count)++;
    if (*overflow < INT_MAX) ++*overflow;
#endif
    return -1;
}

AI2N_HD static inline int ai2g_lane_step(AI2GLane *lane, int index, const AI2NConfig *cfg,
                                        const AI2GPlatformTable *table, double pixels_per_world,
                                        const float *actions, float *obs, float *terminal, float *rewards,
                                        uint8_t *terminated, uint8_t *truncated, AI2GEpisodeRecord *records,
                                        int capacity, int *count, int *overflow) {
    double action[2][10];
    AI2GStepResult out;
    int i, j, rc, slot;
    if (!lane->error && !lane->active) lane->error = AI2G_ERR_UNINITIALIZED;
    if (!lane->error && (lane->length == INT_MAX || lane->episodes == INT_MAX || lane->valid_steps == UINT64_MAX))
        lane->error = AI2G_LANE_ERR_COUNTER;
    if (lane->error) {
        ai2g_lane_hold(lane, obs, terminal, rewards, terminated, truncated);
        return lane->error;
    }
    for (i = 0; i < 2; ++i) for (j = 0; j < 10; ++j) action[i][j] = actions[i * 10 + j];
    rc = ai2g_step(&lane->game, (const double (*)[10])action, &lane->rng, table, &out);
    if (!rc && lane->rng.error) rc = AI2N_ERR_RNG;
    if (rc) {
        lane->error = rc;
        ai2g_lane_hold(lane, obs, terminal, rewards, terminated, truncated);
        return rc;
    }
    ++lane->length; ++lane->valid_steps;
    for (i = 0; i < 2; ++i) {
        lane->returns[i] += out.reward[i]; rewards[i] = (float)out.reward[i];
        for (j = 0; j < 64; ++j) lane->last_obs[i * 64 + j] = (float)out.obs[i][j];
    }
    ai2g_lane_obs(lane, terminal); ai2g_lane_obs(lane, obs);
    *terminated = out.terminated != 0;
    *truncated = out.truncated != 0; /* Raw truth; SB3 derives truncated&&!terminated externally. */
    if (!out.terminated && !out.truncated) return 0;
    slot = ai2g_lane_reserve(count, capacity, overflow);
    if (slot < 0) { lane->error = AI2G_LANE_ERR_RECORD_CAPACITY; return lane->error; }
    {
        AI2GEpisodeRecord *r = &records[slot];
        r->lane = index; r->length = lane->length;
        r->terminated = out.terminated; r->truncated = out.truncated;
        r->win_mask = out.win_mask; r->episode_index = lane->episodes;
        for (i = 0; i < 2; ++i) {
            r->stocks[i] = lane->game.scene.players[i].ms.stocks;
            r->returns[i] = lane->returns[i];
            r->damage_done[i] = lane->game.combat[i].damage_done;
            r->damage_taken[i] = lane->game.combat[i].damage_taken_total;
        }
    }
    ++lane->episodes;
    /* On reset failure terminal/reward/flags still describe the valid finished
     * tick. begin retains last_obs, so no bogus reset observation is exposed. */
    return ai2g_lane_begin(lane, cfg, pixels_per_world, obs, 0);
}

/* Validate a complete host-side tape BEFORE any int conversion in raw reset.
 * Request-order/argument agreement is checked again by AI2NRng on consumption. */
static inline int ai2g_lane_draws_valid(const AI2NDraw *draws, int count) {
    int i;
    if (count < 0 || (count && !draws)) return 0;
    for (i = 0; i < count; ++i) {
        const AI2NDraw *d = &draws[i];
        if (!isfinite(d->a) || !isfinite(d->b) || !isfinite(d->value)) return 0;
        if (d->kind == AI2N_DRAW_GETRANDBITS1) {
            if (d->a != 1 || d->b != 0 || (d->value != 0 && d->value != 1)) return 0;
        } else if (d->kind == AI2N_DRAW_UNIFORM) {
            if (d->a > d->b || d->value < d->a || d->value > d->b) return 0;
        } else if (d->kind == AI2N_DRAW_RANDINT) {
            if (d->a < INT_MIN || d->b > INT_MAX || d->a > d->b ||
                floor(d->a) != d->a || floor(d->b) != d->b || floor(d->value) != d->value ||
                d->value < d->a || d->value > d->b) return 0;
        } else if (d->kind == AI2N_DRAW_NP_RANDINT) {
            if (d->a < 1 || d->a > INT_MAX || floor(d->a) != d->a || d->b != 0 ||
                d->value < 0 || d->value >= d->a || floor(d->value) != d->value) return 0;
        } else return 0;
    }
    return 1;
}

static inline int ai2g_lanes_config_valid(const AI2NConfig *cfg, const AI2GPlatformTable *table,
                                         int games, int capacity, double scale) {
    return cfg && table && ai2n_config_valid(cfg) && games > 0 && capacity > 0 &&
           (size_t)games <= SIZE_MAX / sizeof(AI2GLane) &&
           (size_t)capacity <= SIZE_MAX / sizeof(AI2GEpisodeRecord) &&
           table->count >= AI2N_RAW_MAX_TIMESTEPS && table->count <= AI2G_PLATFORM_TABLE_SIZE &&
           isfinite(scale) && scale > 0 && isfinite(40.0 / scale);
}

#ifdef __cplusplus
extern "C" {
#endif
/* CUDA reset/replay input arrays are HOST pointers and copied at setup only.
 * CUDA step buffers are DEVICE pointers, no host read/synchronization.
 * CUDA get/set/drain pointers are HOST, explicitly synchronized diagnostics.
 * All CUDA calls with work accept the collector stream; cross-stream calls
 * on one context are event-ordered. Caller serializes host calls per context.
 * Creation binds the current CUDA device; wrong-device calls fail explicitly.
 * Diagnostic lane_set never imports a foreign replay pointer. */
size_t ai2g_lanes_cpu_lane_size(void);
size_t ai2g_lanes_cpu_record_size(void);
void *ai2g_lanes_cpu_create(const AI2NConfig*, const AI2GPlatformTable*, int, int, double, int*);
int ai2g_lanes_cpu_destroy(void*);
int ai2g_lanes_cpu_reset(void*, const uint64_t*, const uint8_t*, float*);
int ai2g_lanes_cpu_step(void*, const float*, float*, float*, float*, uint8_t*, uint8_t*);
int ai2g_lanes_cpu_drain(void*, AI2GEpisodeRecord*, int, int*, int*, int32_t*, uint64_t*);
int ai2g_lanes_cpu_lane_get(void*, int, AI2GLane*);
int ai2g_lanes_cpu_lane_set(void*, int, const AI2GLane*);
int ai2g_lanes_cpu_replay_reset(void*, int, const AI2NDraw*, int, float*);
int ai2g_lanes_cpu_configure_episode(void*, int, int, int, int);
int ai2g_lanes_cpu_replay_step(void*, int, const float*, const AI2NDraw*, int,
                               float*, float*, float*, uint8_t*, uint8_t*);
int ai2g_lanes_cpu_rng_status(void*, int, int*, int*, int*, int*);
size_t ai2g_lanes_cuda_lane_size(void);
size_t ai2g_lanes_cuda_record_size(void);
void *ai2g_lanes_cuda_create(const AI2NConfig*, const AI2GPlatformTable*, int, int, double, int*);
int ai2g_lanes_cuda_destroy(void*);
int ai2g_lanes_cuda_reset(void*, const uint64_t*, const uint8_t*, float*, void*);
int ai2g_lanes_cuda_step(void*, const float*, float*, float*, float*, uint8_t*, uint8_t*, void*);
int ai2g_lanes_cuda_drain(void*, AI2GEpisodeRecord*, int, int*, int*, int32_t*, uint64_t*, void*);
int ai2g_lanes_cuda_lane_get(void*, int, AI2GLane*, void*);
int ai2g_lanes_cuda_lane_set(void*, int, const AI2GLane*, void*);
int ai2g_lanes_cuda_replay_reset(void*, int, const AI2NDraw*, int, float*, void*);
int ai2g_lanes_cuda_configure_episode(void*, int, int, int, int, void*);
int ai2g_lanes_cuda_replay_step(void*, int, const float*, const AI2NDraw*, int,
                                float*, float*, float*, uint8_t*, uint8_t*, void*);
int ai2g_lanes_cuda_rng_status(void*, int, int*, int*, int*, int*, void*);
int ai2g_lanes_cuda_synchronize(void*, void*);
int ai2g_lanes_cuda_last_cuda_error(void*);
int ai2g_lanes_cuda_device(void*);
#ifdef __cplusplus
}
#endif
#endif
