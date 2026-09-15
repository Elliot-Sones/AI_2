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
    int *replay_status;
    int *replay_status_valid;
} AI2GLanesCpu;

size_t ai2g_lanes_cpu_lane_size(void) { return sizeof(AI2GLane); }
size_t ai2g_lanes_cpu_record_size(void) { return sizeof(AI2GEpisodeRecord); }

int ai2g_lanes_cpu_destroy(void *handle) {
    AI2GLanesCpu *h = (AI2GLanesCpu*)handle;
    int i;
    if (!h) return AI2N_ERR_NULL;
    if (h->tapes) for (i = 0; i < h->games; ++i) free(h->tapes[i]);
    free(h->lanes); free(h->records); free(h->tapes); free(h->tape_counts);
    free(h->replay_status); free(h->replay_status_valid); free(h);
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
    h->replay_status = (int*)calloc((size_t)games * 4, sizeof(int));
    h->replay_status_valid = (int*)calloc((size_t)games, sizeof(int));
    if (!h->lanes || !h->records || !h->tapes || !h->tape_counts || !h->replay_status || !h->replay_status_valid) {
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

int ai2g_lanes_cpu_configure_episode(void *handle, int index, int player0_stocks, int player1_stocks, int max_timesteps) {
    AI2GLanesCpu *h = (AI2GLanesCpu*)handle;
    AI2GLane *lane;
    if (!h) return AI2N_ERR_NULL;
    if (index < 0 || index >= h->games) return AI2N_ERR_CONFIG;
    lane = h->lanes + index;
    if (player0_stocks >= 0) lane->game.scene.players[0].ms.stocks = player0_stocks;
    if (player1_stocks >= 0) lane->game.scene.players[1].ms.stocks = player1_stocks;
    if (max_timesteps > 0) lane->game.max_timesteps = max_timesteps;
    return 0;
}

int ai2g_lanes_cpu_configure_reward(void *handle, int learner_slot, uint32_t enabled_mask,
                                    double fade, const double *terms) {
    AI2GLanesCpu *h = (AI2GLanesCpu*)handle;
    AI2CRCoefficients cfg;
    int i, rc;
    if (!h) return AI2N_ERR_NULL;
    rc = ai2g_lanes_normalized_reward_config(terms, enabled_mask, fade, &cfg);
    if (rc) return rc;
    for (i = 0; i < h->games; ++i)
        if (!ai2g_lane_can_apply_reward_config(&h->lanes[i], learner_slot, &cfg)) return AI2N_ERR_CONFIG;
    for (i = 0; i < h->games; ++i) {
        h->lanes[i].reward_cfg = cfg;
        h->lanes[i].reward_learner_slot = learner_slot;
        ai2n_zero(&h->lanes[i].last_reward, sizeof(h->lanes[i].last_reward));
        if (h->lanes[i].active && cfg.enabled_mask) {
            rc = ai2cr_reset(&h->lanes[i].reward_state, &h->lanes[i].game, learner_slot);
            if (rc) return rc;
        } else ai2n_zero(&h->lanes[i].reward_state, sizeof(h->lanes[i].reward_state));
    }
    return 0;
}

int ai2g_lanes_cpu_rng_status(void *handle, int index, int *mode, int *cursor, int *count, int *error) {
    AI2GLanesCpu *h = (AI2GLanesCpu*)handle;
    int *s;
    if (!h || !mode || !cursor || !count || !error) return AI2N_ERR_NULL;
    if (index < 0 || index >= h->games) return AI2N_ERR_CONFIG;
    s = h->replay_status + (size_t)index * 4;
    if (h->replay_status_valid[index]) {
        *mode = s[0]; *cursor = s[1]; *count = s[2]; *error = s[3];
    } else {
        *mode = h->lanes[index].rng.mode;
        *cursor = h->lanes[index].rng.cursor;
        *count = h->lanes[index].rng.count;
        *error = h->lanes[index].rng.error;
    }
    return 0;
}

int ai2g_lanes_cpu_replay_step(void *handle, int index, const float *actions, const AI2NDraw *draws,
                               int count, float *obs, float *terminal, float *rewards,
                               uint8_t *terminated, uint8_t *truncated) {
    AI2GLanesCpu *h = (AI2GLanesCpu*)handle;
    AI2NDraw *tape = 0;
    AI2NRng saved;
    int rc, *s;
    if (!h || !actions || !obs || !terminal || !rewards || !terminated || !truncated) return AI2N_ERR_NULL;
    if (index < 0 || index >= h->games) return AI2N_ERR_CONFIG;
    if (!ai2g_lane_draws_valid(draws, count)) return AI2N_ERR_RNG;
    if (count) {
        if ((size_t)count > SIZE_MAX / sizeof(*tape)) return AI2N_ERR_CAPACITY;
        tape = (AI2NDraw*)malloc((size_t)count * sizeof(*tape));
        if (!tape) return AI2N_ERR_CAPACITY;
        memcpy(tape, draws, (size_t)count * sizeof(*tape));
    }
    saved = h->lanes[index].rng;
    ai2n_rng_replay(&h->lanes[index].rng, tape, count);
    rc = ai2g_lane_step(&h->lanes[index], index, &h->cfg, &h->table, h->scale, actions,
        obs, terminal, rewards, terminated, truncated, h->records, h->capacity, &h->count, &h->overflow);
    s = h->replay_status + (size_t)index * 4;
    s[0] = h->lanes[index].rng.mode; s[1] = h->lanes[index].rng.cursor;
    s[2] = h->lanes[index].rng.count; s[3] = h->lanes[index].rng.error;
    h->replay_status_valid[index] = 1;
    h->lanes[index].rng = saved;
    if (saved.mode == 0) h->lanes[index].rng.draws = h->tapes[index];
    free(tape);
    if (!rc && (s[3] || s[1] != s[2])) return AI2N_ERR_RNG;
    return rc;
}
