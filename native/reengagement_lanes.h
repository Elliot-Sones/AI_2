#ifndef AI2_REENGAGEMENT_LANES_H
#define AI2_REENGAGEMENT_LANES_H

/* One learner per raw game. Both transports call the identical B2 task core.
 * Actions[n][10], observations/terminal[n][64], rewards/flags[n]. The finite
 * lane-major scenario schedule is copied at creation; every explicit reset
 * and autoreset consumes the next entry. Exhaustion never rewinds the tape.
 * Failed lanes freeze on the last valid observation with zero reward/no done.
 * Valid terminal transitions survive record/reset failures, but drain reports
 * the error and the learner must reject the whole rollout before optimization.
 */
#include "reengagement_episode.h"
#include <limits.h>
#include <stddef.h>

#define AI2B2_LANE_ERR_RECORD_CAPACITY 501
#define AI2B2_LANE_ERR_COUNTER 502
#define AI2B2_LANE_ERR_CUDA 503
#define AI2B2_LANE_ERR_DEVICE 504
#define AI2B2_LANE_ERR_SCENARIO_EXHAUSTED 505
#define AI2B2_LANE_ERR_SCENARIO_CURSOR 506

typedef struct {
    AI2B2State episode;
    AI2NRng rng;
    double ret;
    float last_obs[64];
    uint64_t valid_steps;
    int32_t length, episodes, error, active, scenario_index;
} AI2B2Lane;

typedef struct {
    int32_t lane, length, terminated, truncated, episode_index, scenario_index;
    double ret;
    AI2B2Diagnostics diagnostics;
} AI2B2EpisodeRecord;

AI2N_HD static inline void ai2b2_lane_obs(const AI2B2Lane *lane, float *obs) {
    int i;
    for (i = 0; i < 64; ++i) obs[i] = lane->last_obs[i];
}

AI2N_HD static inline void ai2b2_lane_hold(const AI2B2Lane *lane, float *obs,
    float *terminal, float *reward, uint8_t *terminated, uint8_t *truncated) {
    ai2b2_lane_obs(lane, obs); ai2b2_lane_obs(lane, terminal);
    *reward = 0.0f; *terminated = *truncated = 0;
}

AI2N_HD static inline int ai2b2_lane_begin(AI2B2Lane *lane, int index,
    const AI2NConfig *cfg, double scale, const AI2B2Scenario *scenarios,
    int episodes_per_lane, float *obs) {
    AI2B2StepResult out;
    const AI2B2Scenario *scenario;
    int i, rc;
    if (lane->scenario_index < -1 || lane->scenario_index >= episodes_per_lane) {
        lane->error = AI2B2_LANE_ERR_SCENARIO_CURSOR;
        ai2b2_lane_obs(lane, obs); return lane->error;
    }
    if (lane->scenario_index == episodes_per_lane - 1) {
        lane->error = AI2B2_LANE_ERR_SCENARIO_EXHAUSTED;
        ai2b2_lane_obs(lane, obs); return lane->error;
    }
    ++lane->scenario_index;
    scenario = &scenarios[(size_t)index * episodes_per_lane + lane->scenario_index];
    lane->ret = 0.0; lane->length = 0; lane->error = 0; lane->active = 0;
    ai2n_rng_seed(&lane->rng, scenario->seed, (uint64_t)index);
    rc = ai2b2_reset(&lane->episode, cfg, &lane->rng, scale, scenario, &out);
    if (!rc && lane->rng.error) rc = AI2N_ERR_RNG;
    if (rc) { lane->error = rc; ai2b2_lane_obs(lane, obs); return rc; }
    for (i = 0; i < 64; ++i) lane->last_obs[i] = (float)out.obs[i];
    ai2b2_lane_obs(lane, obs); lane->active = 1;
    return 0;
}

/* Saturating, bounded reservation, shared by host and device. Record order
 * across lanes is unspecified; (lane, episode_index) identifies each row. */
AI2N_HD static inline int ai2b2_lane_reserve(int *count, int capacity, int *overflow) {
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

AI2N_HD static inline int ai2b2_lane_step(AI2B2Lane *lane, int index,
    const AI2NConfig *cfg, const AI2GPlatformTable *table, double scale,
    const AI2B2Scenario *scenarios, int episodes_per_lane, const float *actions,
    float *obs, float *terminal, float *reward, uint8_t *terminated, uint8_t *truncated,
    AI2B2EpisodeRecord *records, int capacity, int *count, int *overflow) {
    AI2B2StepResult out;
    double action[10];
    int i, rc, slot;
    if (!lane->error && !lane->active) lane->error = AI2G_ERR_UNINITIALIZED;
    if (!lane->error && (lane->length == INT_MAX || lane->episodes == INT_MAX || lane->valid_steps == UINT64_MAX))
        lane->error = AI2B2_LANE_ERR_COUNTER;
    if (lane->error) {
        ai2b2_lane_hold(lane, obs, terminal, reward, terminated, truncated);
        return lane->error;
    }
    for (i = 0; i < 10; ++i) action[i] = actions[i];
    rc = ai2b2_step(&lane->episode, action, &lane->rng, table, &out);
    if (!rc && lane->rng.error) rc = AI2N_ERR_RNG;
    if (rc) {
        lane->error = rc;
        ai2b2_lane_hold(lane, obs, terminal, reward, terminated, truncated);
        return rc;
    }
    ++lane->length; ++lane->valid_steps; lane->ret += out.reward;
    *reward = (float)out.reward;
    for (i = 0; i < 64; ++i) lane->last_obs[i] = (float)out.obs[i];
    ai2b2_lane_obs(lane, obs); ai2b2_lane_obs(lane, terminal);
    *terminated = out.terminated != 0; *truncated = out.truncated != 0;
    if (!out.terminated && !out.truncated) return 0;
    slot = ai2b2_lane_reserve(count, capacity, overflow);
    if (slot < 0) { lane->error = AI2B2_LANE_ERR_RECORD_CAPACITY; return lane->error; }
    {
        AI2B2EpisodeRecord *record = &records[slot];
        ai2n_zero(record, sizeof(*record));
        record->lane = index; record->length = lane->length;
        record->terminated = out.terminated; record->truncated = out.truncated;
        record->episode_index = lane->episodes; record->scenario_index = lane->scenario_index;
        record->ret = lane->ret; record->diagnostics = out.diagnostics;
    }
    ++lane->episodes;
    return ai2b2_lane_begin(lane, index, cfg, scale, scenarios, episodes_per_lane, obs);
}

static inline int ai2b2_lanes_config_valid(const AI2NConfig *cfg,
    const AI2GPlatformTable *table, int lanes, int capacity, double scale,
    const AI2B2Scenario *scenarios, int episodes_per_lane) {
    size_t i, count;
    if (!cfg || !table || !scenarios || !ai2n_config_valid(cfg) ||
        cfg->time_limit_steps != AI2B2_TIME_LIMIT || lanes <= 0 || capacity <= 0 || episodes_per_lane <= 0 ||
        (size_t)lanes > SIZE_MAX / sizeof(AI2B2Lane) ||
        (size_t)capacity > SIZE_MAX / sizeof(AI2B2EpisodeRecord) ||
        (size_t)lanes > SIZE_MAX / (size_t)episodes_per_lane / sizeof(AI2B2Scenario) ||
        table->count < AI2N_RAW_MAX_TIMESTEPS || table->count > AI2G_PLATFORM_TABLE_SIZE ||
        !isfinite(scale) || scale <= 0.0 || !isfinite(40.0 / scale)) return 0;
    count = (size_t)lanes * episodes_per_lane;
    for (i = 0; i < count; ++i) if (!ai2b2_scenario_valid(scenarios + i)) return 0;
    return 1;
}

static inline int ai2b2_lane_import_valid(const AI2B2Lane *lane, int episodes_per_lane) {
    int i;
    /* Only initialized diagnostic snapshots, never replay pointers or an
     * arbitrary serialized physics world. The task guard runs before any
     * state copy; malformed task indices must never reach the step kernel. */
    if (!lane || !ai2b2_state_valid(&lane->episode) ||
        lane->rng.mode != 1 || lane->rng.draws != NULL ||
        lane->rng.cursor != 0 || lane->rng.count != 0 || lane->rng.error != 0 ||
        lane->rng.error_kind != 0 || lane->rng.error_cursor != 0 ||
        lane->rng.error_a != 0.0 || lane->rng.error_b != 0.0 ||
        !(lane->rng.s[0] | lane->rng.s[1] | lane->rng.s[2] | lane->rng.s[3]) ||
        lane->length != lane->episode.diagnostics.tick || lane->episodes < 0 ||
        lane->scenario_index < 0 || lane->scenario_index >= episodes_per_lane ||
        lane->episodes > lane->scenario_index + (lane->episode.diagnostics.outcome != AI2B2_ONGOING) ||
        lane->valid_steps < (uint64_t)lane->length || lane->valid_steps < (uint64_t)lane->episodes ||
        lane->error < 0 || lane->active != 1 || !isfinite(lane->ret) ||
        (!lane->error && (lane->episode.error || lane->episode.game.error))) return 0;
    for (i = 0; i < 64; ++i) if (!isfinite(lane->last_obs[i])) return 0;
    return 1;
}

static inline int ai2b2_lane_scenario_matches(const AI2B2Lane *lane,
    const AI2B2Scenario *expected, int current_index) {
    const AI2B2Scenario *scenario = &lane->episode.scenario;
    return lane->scenario_index == current_index && scenario->seed == expected->seed &&
        scenario->player_slot == expected->player_slot && scenario->direction == expected->direction &&
        scenario->gap == expected->gap && scenario->midpoint == expected->midpoint;
}

#ifdef __cplusplus
extern "C" {
#endif
/* Creation owns a copy of the HOST scenario schedule. reset's optional HOST
 * mask is copied at setup; step buffers are DEVICE pointers for CUDA, HOST
 * pointers for CPU. CUDA diagnostics are explicitly synchronized HOST reads.
 * Native CUDA state is event-ordered across caller streams, device-owned,
 * and has no per-step host synchronization. Serialize host calls per context.
 */
size_t ai2b2_lanes_cpu_lane_size(void);
size_t ai2b2_lanes_cpu_record_size(void);
void *ai2b2_lanes_cpu_create(const AI2NConfig*, const AI2GPlatformTable*, int, int, double, const AI2B2Scenario*, int, int*);
int ai2b2_lanes_cpu_destroy(void*);
int ai2b2_lanes_cpu_reset(void*, const uint8_t*, float*);
int ai2b2_lanes_cpu_step(void*, const float*, float*, float*, float*, uint8_t*, uint8_t*);
int ai2b2_lanes_cpu_drain(void*, AI2B2EpisodeRecord*, int, int*, int*, int32_t*, uint64_t*);
int ai2b2_lanes_cpu_lane_get(void*, int, AI2B2Lane*);
int ai2b2_lanes_cpu_lane_set(void*, int, const AI2B2Lane*);
size_t ai2b2_lanes_cuda_lane_size(void);
size_t ai2b2_lanes_cuda_record_size(void);
void *ai2b2_lanes_cuda_create(const AI2NConfig*, const AI2GPlatformTable*, int, int, double, const AI2B2Scenario*, int, int*);
int ai2b2_lanes_cuda_destroy(void*);
int ai2b2_lanes_cuda_reset(void*, const uint8_t*, float*, void*);
int ai2b2_lanes_cuda_step(void*, const float*, float*, float*, float*, uint8_t*, uint8_t*, void*);
int ai2b2_lanes_cuda_drain(void*, AI2B2EpisodeRecord*, int, int*, int*, int32_t*, uint64_t*, void*);
int ai2b2_lanes_cuda_lane_get(void*, int, AI2B2Lane*, void*);
int ai2b2_lanes_cuda_lane_set(void*, int, const AI2B2Lane*, void*);
int ai2b2_lanes_cuda_synchronize(void*, void*);
int ai2b2_lanes_cuda_device(void*);
int ai2b2_lanes_cuda_last_cuda_error(void*);
#ifdef __cplusplus
}
#endif
#endif
