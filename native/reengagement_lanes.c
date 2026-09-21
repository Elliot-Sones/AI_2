#include "reengagement_lanes.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    AI2NConfig cfg;
    AI2GPlatformTable table;
    double scale;
    int lanes, capacity, count, overflow, episodes_per_lane;
    AI2B2Lane *state;
    AI2B2EpisodeRecord *records;
    AI2B2Scenario *scenarios;
} AI2B2LanesCpu;

size_t ai2b2_lanes_cpu_lane_size(void) { return sizeof(AI2B2Lane); }
size_t ai2b2_lanes_cpu_record_size(void) { return sizeof(AI2B2EpisodeRecord); }

#define AI2B2_FIELD_TABLE(prefix, type) \
    size_t ai2b2_lanes_cpu_##prefix##_field_count(void) { \
        return sizeof(ai2b2_lanes_##prefix##_offsets) / sizeof(ai2b2_lanes_##prefix##_offsets[0]); \
    } \
    size_t ai2b2_lanes_cpu_##prefix##_field_offset(size_t index) { \
        return index < ai2b2_lanes_cpu_##prefix##_field_count() ? ai2b2_lanes_##prefix##_offsets[index] : (size_t)-1; \
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
AI2B2_FIELD_TABLE(lane, AI2B2Lane)

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
AI2B2_FIELD_TABLE(record, AI2B2EpisodeRecord)

int ai2b2_lanes_cpu_destroy(void *handle) {
    AI2B2LanesCpu *h = (AI2B2LanesCpu*)handle;
    if (!h) return AI2N_ERR_NULL;
    free(h->state); free(h->records); free(h->scenarios); free(h);
    return 0;
}

void *ai2b2_lanes_cpu_create(const AI2NConfig *cfg, const AI2GPlatformTable *table,
    int lanes, int capacity, double scale, const AI2B2Scenario *scenarios,
    int episodes_per_lane, int *rc) {
    AI2B2LanesCpu *h;
    int i;
    size_t scenario_bytes;
    if (rc) *rc = AI2N_ERR_CONFIG;
    if (!ai2b2_lanes_config_valid(cfg, table, lanes, capacity, scale, scenarios, episodes_per_lane)) return NULL;
    if (rc) *rc = AI2N_ERR_CAPACITY;
    h = (AI2B2LanesCpu*)calloc(1, sizeof(*h));
    if (!h) return NULL;
    h->cfg = *cfg; h->table = *table; h->lanes = lanes; h->capacity = capacity;
    h->scale = scale; h->episodes_per_lane = episodes_per_lane;
    scenario_bytes = (size_t)lanes * episodes_per_lane * sizeof(AI2B2Scenario);
    h->state = (AI2B2Lane*)calloc((size_t)lanes, sizeof(AI2B2Lane));
    h->records = (AI2B2EpisodeRecord*)calloc((size_t)capacity, sizeof(AI2B2EpisodeRecord));
    h->scenarios = (AI2B2Scenario*)malloc(scenario_bytes);
    if (!h->state || !h->records || !h->scenarios) { ai2b2_lanes_cpu_destroy(h); return NULL; }
    memcpy(h->scenarios, scenarios, scenario_bytes);
    for (i = 0; i < lanes; ++i) h->state[i].scenario_index = -1;
    if (rc) *rc = 0;
    return h;
}

int ai2b2_lanes_cpu_reset(void *handle, const uint8_t *mask, float *obs) {
    AI2B2LanesCpu *h = (AI2B2LanesCpu*)handle;
    int i, first = 0;
    if (!h || !obs) return AI2N_ERR_NULL;
    for (i = 0; i < h->lanes; ++i) {
        int rc = 0;
        if (!mask || mask[i]) rc = ai2b2_lane_begin(h->state + i, i, &h->cfg, h->scale,
            h->scenarios, h->episodes_per_lane, obs + (size_t)i * 64);
        else ai2b2_lane_obs(h->state + i, obs + (size_t)i * 64);
        if (rc && !first) first = rc;
    }
    return first;
}

int ai2b2_lanes_cpu_step(void *handle, const float *actions, float *obs, float *terminal,
    float *rewards, uint8_t *terminated, uint8_t *truncated) {
    AI2B2LanesCpu *h = (AI2B2LanesCpu*)handle;
    int i, first = 0;
    if (!h || !actions || !obs || !terminal || !rewards || !terminated || !truncated) return AI2N_ERR_NULL;
    for (i = 0; i < h->lanes; ++i) {
        const int rc = ai2b2_lane_step(h->state + i, i, &h->cfg, &h->table, h->scale,
            h->scenarios, h->episodes_per_lane, actions + (size_t)i * 10,
            obs + (size_t)i * 64, terminal + (size_t)i * 64, rewards + i,
            terminated + i, truncated + i, h->records, h->capacity, &h->count, &h->overflow);
        if (rc && !first) first = rc;
    }
    return first;
}

int ai2b2_lanes_cpu_drain(void *handle, AI2B2EpisodeRecord *out, int capacity,
    int *count, int *overflow, int32_t *errors, uint64_t *valid_steps) {
    AI2B2LanesCpu *h = (AI2B2LanesCpu*)handle;
    int i;
    if (!h || !count || !overflow || !errors || !valid_steps) return AI2N_ERR_NULL;
    if (capacity < h->count) return AI2N_ERR_CAPACITY;
    if (h->count && !out) return AI2N_ERR_NULL;
    if (h->count) memcpy(out, h->records, (size_t)h->count * sizeof(*out));
    *count = h->count; *overflow = h->overflow;
    for (i = 0; i < h->lanes; ++i) { errors[i] = h->state[i].error; valid_steps[i] = h->state[i].valid_steps; }
    h->count = h->overflow = 0;
    return 0;
}

int ai2b2_lanes_cpu_lane_get(void *handle, int index, AI2B2Lane *out) {
    AI2B2LanesCpu *h = (AI2B2LanesCpu*)handle;
    if (!h || !out) return AI2N_ERR_NULL;
    if (index < 0 || index >= h->lanes) return AI2N_ERR_CONFIG;
    *out = h->state[index]; return 0;
}

int ai2b2_lanes_cpu_lane_set(void *handle, int index, const AI2B2Lane *in) {
    AI2B2LanesCpu *h = (AI2B2LanesCpu*)handle;
    if (!h || !in) return AI2N_ERR_NULL;
    if (index < 0 || index >= h->lanes || !ai2b2_lane_import_valid(in, h->episodes_per_lane)) return AI2N_ERR_CONFIG;
    if (!ai2b2_lane_scenario_matches(in,
        h->scenarios + (size_t)index * h->episodes_per_lane + in->scenario_index,
        h->state[index].scenario_index)) return AI2N_ERR_CONFIG;
    h->state[index] = *in; return 0;
}
