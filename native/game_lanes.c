#include "game_lanes.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    AI2NConfig cfg;
    AI2GPlatformTable table;
    double scale;
    int games, capacity, count, overflow;
    AI2GLane *lanes;
    AI2GEpisodeRecord *records;
    AI2NDraw **tapes;
    int *tape_counts;
} AI2GLanesCpu;

size_t ai2g_lanes_cpu_lane_size(void) { return sizeof(AI2GLane); }
size_t ai2g_lanes_cpu_record_size(void) { return sizeof(AI2GEpisodeRecord); }

int ai2g_lanes_cpu_destroy(void *handle) {
    AI2GLanesCpu *h = (AI2GLanesCpu*)handle;
    int i;
    if (!h) return AI2N_ERR_NULL;
    if (h->tapes) for (i = 0; i < h->games; ++i) free(h->tapes[i]);
    free(h->lanes); free(h->records); free(h->tapes); free(h->tape_counts); free(h);
    return 0;
}

void *ai2g_lanes_cpu_create(const AI2NConfig *cfg, const AI2GPlatformTable *table,
                           int games, int capacity, double scale, int *rc) {
    AI2GLanesCpu *h;
    if (rc) *rc = AI2N_ERR_CONFIG;
    if (!ai2g_lanes_config_valid(cfg, table, games, capacity, scale)) return 0;
    if (rc) *rc = AI2N_ERR_CAPACITY;
    h = (AI2GLanesCpu*)calloc(1, sizeof(*h));
    if (!h) return 0;
    h->games = games; h->capacity = capacity; h->scale = scale;
    h->cfg = *cfg; h->table = *table;
    h->lanes = (AI2GLane*)calloc((size_t)games, sizeof(AI2GLane));
    h->records = (AI2GEpisodeRecord*)calloc((size_t)capacity, sizeof(AI2GEpisodeRecord));
    h->tapes = (AI2NDraw**)calloc((size_t)games, sizeof(AI2NDraw*));
    h->tape_counts = (int*)calloc((size_t)games, sizeof(int));
    if (!h->lanes || !h->records || !h->tapes || !h->tape_counts) {
        ai2g_lanes_cpu_destroy(h); return 0;
    }
    if (rc) *rc = 0;
    return h;
}

int ai2g_lanes_cpu_reset(void *handle, const uint64_t *seeds, const uint8_t *mask, float *obs) {
    AI2GLanesCpu *h = (AI2GLanesCpu*)handle;
    int i, first = 0;
    if (!h || !seeds || !obs) return AI2N_ERR_NULL;
    for (i = 0; i < h->games; ++i) {
        int rc = 0;
        if (!mask || mask[i]) {
            ai2n_rng_seed(&h->lanes[i].rng, seeds[i], (unsigned long long)i);
            rc = ai2g_lane_begin(&h->lanes[i], &h->cfg, h->scale, obs + (size_t)i * 128, 1);
        } else ai2g_lane_obs(&h->lanes[i], obs + (size_t)i * 128);
        if (rc && !first) first = rc;
    }
    return first;
}

int ai2g_lanes_cpu_step(void *handle, const float *actions, float *obs, float *terminal,
                        float *rewards, uint8_t *terminated, uint8_t *truncated) {
    AI2GLanesCpu *h = (AI2GLanesCpu*)handle;
    int i, first = 0;
    if (!h || !actions || !obs || !terminal || !rewards || !terminated || !truncated) return AI2N_ERR_NULL;
    for (i = 0; i < h->games; ++i) {
        const int rc = ai2g_lane_step(&h->lanes[i], i, &h->cfg, &h->table, h->scale,
            actions + (size_t)i * 20, obs + (size_t)i * 128, terminal + (size_t)i * 128,
            rewards + (size_t)i * 2, terminated + i, truncated + i, h->records, h->capacity, &h->count, &h->overflow);
        if (rc && !first) first = rc;
    }
    return first;
}

int ai2g_lanes_cpu_drain(void *handle, AI2GEpisodeRecord *out, int capacity, int *count,
                         int *overflow, int32_t *errors, uint64_t *valid_steps) {
    AI2GLanesCpu *h = (AI2GLanesCpu*)handle;
    int i;
    if (!h || !count || !overflow || !errors || !valid_steps) return AI2N_ERR_NULL;
    if (capacity < h->count) return AI2N_ERR_CAPACITY;
    if (h->count && !out) return AI2N_ERR_NULL;
    if (h->count) memcpy(out, h->records, (size_t)h->count * sizeof(*out));
    *count = h->count; *overflow = h->overflow;
    for (i = 0; i < h->games; ++i) { errors[i] = h->lanes[i].error; valid_steps[i] = h->lanes[i].valid_steps; }
    h->count = h->overflow = 0;
    return 0;
}

int ai2g_lanes_cpu_lane_get(void *handle, int index, AI2GLane *out) {
    AI2GLanesCpu *h = (AI2GLanesCpu*)handle;
    if (!h || !out) return AI2N_ERR_NULL;
    if (index < 0 || index >= h->games) return AI2N_ERR_CONFIG;
    *out = h->lanes[index]; return 0;
}

int ai2g_lanes_cpu_lane_set(void *handle, int index, const AI2GLane *in) {
    AI2GLanesCpu *h = (AI2GLanesCpu*)handle;
    if (!h || !in) return AI2N_ERR_NULL;
    if (index < 0 || index >= h->games || (in->rng.mode != 0 && in->rng.mode != 1)) return AI2N_ERR_CONFIG;
    if (!in->rng.mode && (in->rng.count != h->tape_counts[index] || in->rng.cursor < 0 || in->rng.cursor > in->rng.count))
        return AI2N_ERR_RNG;
    h->lanes[index] = *in;
    h->lanes[index].rng.draws = in->rng.mode ? 0 : h->tapes[index];
    return 0;
}

int ai2g_lanes_cpu_replay_reset(void *handle, int index, const AI2NDraw *draws, int count, float *obs) {
    AI2GLanesCpu *h = (AI2GLanesCpu*)handle;
    AI2NDraw *tape = 0;
    if (!h || !obs) return AI2N_ERR_NULL;
    if (index < 0 || index >= h->games) return AI2N_ERR_CONFIG;
    if (!ai2g_lane_draws_valid(draws, count)) return AI2N_ERR_RNG;
    if (count) {
        if ((size_t)count > SIZE_MAX / sizeof(*tape)) return AI2N_ERR_CAPACITY;
        tape = (AI2NDraw*)malloc((size_t)count * sizeof(*tape));
        if (!tape) return AI2N_ERR_CAPACITY;
        memcpy(tape, draws, (size_t)count * sizeof(*tape));
    }
    free(h->tapes[index]); h->tapes[index] = tape; h->tape_counts[index] = count;
    ai2n_rng_replay(&h->lanes[index].rng, tape, count);
    return ai2g_lane_begin(&h->lanes[index], &h->cfg, h->scale, obs, 1);
}
