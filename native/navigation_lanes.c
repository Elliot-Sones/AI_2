/*
 * Host runtime for the navigation training lanes: the reference the CUDA
 * lanes are compared against, and the backend of the local tests.  Compiled
 * together with navigation_episode.c so one library carries the reference
 * exports (config finalize, platform table) and the lane exports.
 */
#include "navigation_lanes.h"

#include <stdlib.h>
#include <string.h>

typedef struct AI2NLanesCpu {
    AI2NConfig cfg;
    AI2NPlatformTable table;
    int lanes;
    int record_capacity;
    int record_count;
    int record_overflow;
    AI2NLane *lane;
    AI2NEpisodeRecord *records;
    AI2NDraw *replay;
    int replay_capacity;
} AI2NLanesCpu;

int ai2n_lanes_cpu_lane_size(void) { return (int)sizeof(AI2NLane); }
int ai2n_lanes_cpu_record_size(void) { return (int)sizeof(AI2NEpisodeRecord); }

void *ai2n_lanes_cpu_create(const AI2NConfig *cfg, const AI2NPlatformTable *table, int lanes, int record_capacity) {
    AI2NLanesCpu *h;
    if (cfg == 0 || table == 0 || lanes <= 0 || record_capacity <= 0) return 0;
    h = (AI2NLanesCpu *)calloc(1, sizeof(*h));
    if (h == 0) return 0;
    h->lane = (AI2NLane *)calloc((size_t)lanes, sizeof(AI2NLane));
    h->records = (AI2NEpisodeRecord *)calloc((size_t)record_capacity, sizeof(AI2NEpisodeRecord));
    if (h->lane == 0 || h->records == 0) { free(h->lane); free(h->records); free(h); return 0; }
    memcpy(&h->cfg, cfg, sizeof(*cfg));
    memcpy(&h->table, table, sizeof(*table));
    h->lanes = lanes;
    h->record_capacity = record_capacity;
    return h;
}

void ai2n_lanes_cpu_destroy(void *handle) {
    AI2NLanesCpu *h = (AI2NLanesCpu *)handle;
    if (h == 0) return;
    free(h->lane);
    free(h->records);
    free(h->replay);
    free(h);
}

int ai2n_lanes_cpu_reset(void *handle, const unsigned long long *seeds, float *obs) {
    AI2NLanesCpu *h = (AI2NLanesCpu *)handle;
    int i, first = AI2N_ERR_NONE;
    if (h == 0 || seeds == 0 || obs == 0) return AI2N_ERR_NULL;
    h->record_count = 0;
    h->record_overflow = 0;
    for (i = 0; i < h->lanes; ++i) {
        const int rc = ai2n_lane_reset(&h->lane[i], &h->cfg, seeds[i], (unsigned long long)i, obs + (size_t)i * AI2N_OBS_WIDTH);
        if (rc && first == AI2N_ERR_NONE) first = rc;
    }
    return first;
}

int ai2n_lanes_cpu_step(void *handle, const float *actions, float *obs, float *terminal, float *rewards, uint8_t *terminated, uint8_t *truncated) {
    AI2NLanesCpu *h = (AI2NLanesCpu *)handle;
    int i, first = AI2N_ERR_NONE;
    if (h == 0 || actions == 0 || obs == 0 || terminal == 0 || rewards == 0 || terminated == 0 || truncated == 0) return AI2N_ERR_NULL;
    for (i = 0; i < h->lanes; ++i) {
        const int rc = ai2n_lane_step(&h->lane[i], i, &h->cfg, &h->table, actions + (size_t)i * AI2N_ACTION_WIDTH,
                                      obs + (size_t)i * AI2N_OBS_WIDTH, terminal + (size_t)i * AI2N_OBS_WIDTH, rewards + i,
                                      terminated + i, truncated + i, h->records, h->record_capacity, &h->record_count, &h->record_overflow);
        if (rc && first == AI2N_ERR_NONE) first = rc;
    }
    return first;
}

/* Copy the completed-episode records out and clear them; `capacity` must hold every stored record. */
int ai2n_lanes_cpu_drain(void *handle, AI2NEpisodeRecord *out, int capacity, int *count, int *overflow) {
    AI2NLanesCpu *h = (AI2NLanesCpu *)handle;
    int stored;
    if (h == 0 || count == 0 || overflow == 0) return AI2N_ERR_NULL;
    stored = h->record_count < h->record_capacity ? h->record_count : h->record_capacity;
    if (stored > capacity) return AI2N_ERR_CAPACITY;
    if (stored > 0 && out == 0) return AI2N_ERR_NULL;
    if (stored > 0) memcpy(out, h->records, sizeof(AI2NEpisodeRecord) * (size_t)stored);
    *count = stored;
    *overflow = h->record_overflow;
    h->record_count = 0;
    h->record_overflow = 0;
    return AI2N_ERR_NONE;
}

/* First lane error, with its lane index; zero when every lane is healthy. */
int ai2n_lanes_cpu_error(void *handle, int *lane_index) {
    AI2NLanesCpu *h = (AI2NLanesCpu *)handle;
    int i;
    if (h == 0) return AI2N_ERR_NULL;
    if (lane_index) *lane_index = -1;
    for (i = 0; i < h->lanes; ++i) {
        if (h->lane[i].error) { if (lane_index) *lane_index = i; return h->lane[i].error; }
    }
    return AI2N_ERR_NONE;
}

int ai2n_lanes_cpu_lane_get(void *handle, int lane, AI2NLane *out) {
    AI2NLanesCpu *h = (AI2NLanesCpu *)handle;
    if (h == 0 || out == 0) return AI2N_ERR_NULL;
    if (lane < 0 || lane >= h->lanes) return AI2N_ERR_CONFIG;
    memcpy(out, &h->lane[lane], sizeof(AI2NLane));
    return AI2N_ERR_NONE;
}

int ai2n_lanes_cpu_lane_set(void *handle, int lane, const AI2NLane *in) {
    AI2NLanesCpu *h = (AI2NLanesCpu *)handle;
    if (h == 0 || in == 0) return AI2N_ERR_NULL;
    if (lane < 0 || lane >= h->lanes) return AI2N_ERR_CONFIG;
    memcpy(&h->lane[lane], in, sizeof(AI2NLane));
    return AI2N_ERR_NONE;
}

/* --- replay mode: drive one lane with the real game's recorded draws (exactness tests) --- */

static int ai2n_lanes_cpu_replay_load(AI2NLanesCpu *h, const AI2NDraw *draws, int count) {
    if (count < 0 || (count > 0 && draws == 0)) return AI2N_ERR_NULL;
    if (count > h->replay_capacity) {
        AI2NDraw *grown = (AI2NDraw *)realloc(h->replay, sizeof(AI2NDraw) * (size_t)(count < 64 ? 64 : count));
        if (grown == 0) return AI2N_ERR_CAPACITY;
        h->replay = grown;
        h->replay_capacity = count < 64 ? 64 : count;
    }
    if (count > 0) memcpy(h->replay, draws, sizeof(AI2NDraw) * (size_t)count);
    return AI2N_ERR_NONE;
}

int ai2n_lanes_cpu_replay_reset(void *handle, int lane, const AI2NDraw *draws, int count, float *obs) {
    AI2NLanesCpu *h = (AI2NLanesCpu *)handle;
    int rc;
    if (h == 0 || obs == 0) return AI2N_ERR_NULL;
    if (lane < 0 || lane >= h->lanes) return AI2N_ERR_CONFIG;
    rc = ai2n_lanes_cpu_replay_load(h, draws, count);
    if (rc) return rc;
    ai2n_rng_replay(&h->lane[lane].rng, h->replay, count);
    return ai2n_lane_begin(&h->lane[lane], &h->cfg, obs);
}

int ai2n_lanes_cpu_replay_step(void *handle, int lane, const AI2NDraw *draws, int count, const float *action, float *obs, float *terminal,
                               float *reward, uint8_t *terminated, uint8_t *truncated) {
    AI2NLanesCpu *h = (AI2NLanesCpu *)handle;
    int rc;
    if (h == 0 || action == 0 || obs == 0 || terminal == 0 || reward == 0 || terminated == 0 || truncated == 0) return AI2N_ERR_NULL;
    if (lane < 0 || lane >= h->lanes) return AI2N_ERR_CONFIG;
    rc = ai2n_lanes_cpu_replay_load(h, draws, count);
    if (rc) return rc;
    ai2n_rng_replay(&h->lane[lane].rng, h->replay, count);
    return ai2n_lane_step(&h->lane[lane], lane, &h->cfg, &h->table, action, obs, terminal, reward, terminated, truncated,
                          h->records, h->record_capacity, &h->record_count, &h->record_overflow);
}

int ai2n_lanes_cpu_rng_status(void *handle, int lane, int *error, int *cursor, int *count) {
    AI2NLanesCpu *h = (AI2NLanesCpu *)handle;
    if (h == 0) return AI2N_ERR_NULL;
    if (lane < 0 || lane >= h->lanes) return AI2N_ERR_CONFIG;
    if (error) *error = h->lane[lane].rng.error;
    if (cursor) *cursor = h->lane[lane].rng.cursor;
    if (count) *count = h->lane[lane].rng.count;
    return AI2N_ERR_NONE;
}
