#ifndef AI2_REENGAGEMENT_EPISODE_H
#define AI2_REENGAGEMENT_EPISODE_H

/* bridge_b2_reengagement-v1. Two distinct learner light hits, with a clean
 * relocation boundary and renewed pursuit. Target action is always zero. */
#include "game_episode.h"

#define AI2B2_ERR_SCENARIO 401
#define AI2B2_ERR_ACTION 402
#define AI2B2_ERR_TERMINAL 403
#define AI2B2_ERR_SUPPORT 404
#define AI2B2_ERR_STATE 405
#define AI2B2_ERR_EVIDENCE 406
#define AI2B2_REWARD_TERMS 8
#define AI2B2_TIME_LIMIT 300
#define AI2B2_ATTRIBUTED_EVENT_CAPACITY 2
#define AI2B2_OBS_RESIDUAL_VELOCITY_EPSILON 1e-15

enum {
    AI2B2_ONGOING = 0, AI2B2_SUCCESS = 1, AI2B2_OWN_STOCK_LOSS = 2,
    AI2B2_OPPONENT_STOCK_LOSS = 3, AI2B2_UNSAFE_SUPPORT = 4,
    AI2B2_RESPONSE_DEADLINE = 5, AI2B2_SECOND_HIT_DEADLINE = 6,
    AI2B2_TOTAL_DEADLINE = 7
};

enum {
    AI2B2_FAILURE_NONE = 0, AI2B2_FAILURE_SUCCESS = 1,
    AI2B2_FAILURE_OWN_STOCK_LOSS = 2, AI2B2_FAILURE_OPPONENT_STOCK_LOSS = 3,
    AI2B2_FAILURE_UNSAFE_SUPPORT = 4, AI2B2_FAILURE_RESPONSE_DEADLINE = 5,
    AI2B2_FAILURE_SECOND_HIT_DEADLINE = 6, AI2B2_FAILURE_TOTAL_DEADLINE = 7,
    AI2B2_FAILURE_FIRST_HIT_ONLY = 8, AI2B2_FAILURE_FAILED_RENEWED_PURSUIT = 9,
    AI2B2_FAILURE_PURSUED_MISSED_SECOND = 10, AI2B2_FAILURE_STARTED_MISSED = 11,
    AI2B2_FAILURE_REQUESTED_NO_START = 12, AI2B2_FAILURE_NO_APPROACH = 13,
    AI2B2_FAILURE_NO_ATTACK_REQUEST = 14, AI2B2_FAILURE_DEADLINE = 15
};

typedef struct {
    uint64_t seed;
    int32_t player_slot, direction;
    double gap, midpoint;
} AI2B2Scenario;

typedef struct {
    double x, y, vx, vy, damage, damage_done, damage_taken, clearance;
    int32_t player_slot, facing, stocks, state_id, grounded, fully_supported, stun_time_stored;
    uint64_t attack_id;
} AI2B2PlayerSnapshot;

/* Fixed diagnostic ABI. attributed_damage_* is a bounded two-entry record:
 * entry 0 is the credited first hit and entry 1 is the credited second hit. */
typedef struct {
    uint64_t seed, scenario_id_hash, first_hit_attack_id, second_hit_attack_id, hit_attack_id;
    int32_t player_slot, direction, tick, outcome;
    int32_t first_hit_tick, second_hit_tick, relocation_tick, pursuit_tick, pursuit_direction;
    int32_t terminal_reason, success, sequence_completed, failure_category;
    int32_t achieved_first_hit, achieved_relocation, achieved_pursuit;
    int32_t achieved_second_hit, achieved_safe_resolution;
    int32_t attributed_damage_event_count;
    uint64_t attributed_damage_attack_ids[AI2B2_ATTRIBUTED_EVENT_CAPACITY];
    int32_t attributed_damage_ticks[AI2B2_ATTRIBUTED_EVENT_CAPACITY];
    int32_t attributed_damage_sources[AI2B2_ATTRIBUTED_EVENT_CAPACITY];
    int32_t attributed_damage_targets[AI2B2_ATTRIBUTED_EVENT_CAPACITY];
    int32_t attributed_damage_moves[AI2B2_ATTRIBUTED_EVENT_CAPACITY];
    double attributed_damage_amounts[AI2B2_ATTRIBUTED_EVENT_CAPACITY];
    int32_t initial_stocks[2];
    int32_t light_request_ticks, light_press_count, max_light_hold, attack_ticks;
    int32_t movement_request_ticks, renewed_movement_request_ticks, attack_start_count;
    uint64_t attack_starts, attack_contacts, damage_events;
    double initial_learner_x, initial_target_x, spawn_gap, spawn_midpoint, initial_gap;
    double max_initial_approach_progress, relocation_gap, relocation_learner_x;
    double relocation_target_x, gap_closed, learner_progress;
    double reward_totals[AI2B2_REWARD_TERMS], min_clearance[2];
    int32_t unsupported_ticks[2], falls[2], stock_changes[2];
    AI2B2PlayerSnapshot relocation_pre_target, relocation_post_target;
    AI2B2PlayerSnapshot players[2];
} AI2B2Diagnostics;

typedef struct {
    AI2GState game;
    AI2B2Scenario scenario;
    AI2B2Diagnostics diagnostics;
    int32_t initial_stocks[2];
    int32_t initialized, error, previous_light, current_light_hold, second_stage, relocated;
    uint64_t first_attack_id, second_attack_id, last_second_duplicate_id;
} AI2B2State;

typedef struct {
    double obs[64], reward, reward_terms[AI2B2_REWARD_TERMS], raw_action[10], executed_action[10];
    int32_t terminated, truncated, error;
    AI2B2PlayerSnapshot pre_players[2];
    AI2B2Diagnostics diagnostics;
} AI2B2StepResult;

AI2N_HD static inline uint64_t ai2b2_scenario_hash(const AI2B2Scenario *s) {
    uint64_t h = 1469598103934665603ULL;
    int i;
    for (i = 0; i < 8; ++i) { h ^= (s->seed >> (8 * i)) & 0xffu; h *= 1099511628211ULL; }
    h ^= (uint64_t)(uint32_t)s->player_slot; h *= 1099511628211ULL;
    h ^= (uint64_t)(uint32_t)s->direction; h *= 1099511628211ULL;
    return h;
}

AI2N_HD static inline int ai2b2_scenario_valid(const AI2B2Scenario *s) {
    return s && (s->player_slot == 0 || s->player_slot == 1) &&
        (s->direction == -1 || s->direction == 1) && isfinite(s->gap) &&
        s->gap >= 3.1 && s->gap <= 3.4 && isfinite(s->midpoint) &&
        s->midpoint >= 4.35 && s->midpoint <= 4.65;
}

AI2N_HD static inline int ai2b2_failure_category_value(const AI2B2Diagnostics *d) {
    if (!d->outcome) return AI2B2_FAILURE_NONE;
    if (d->outcome == AI2B2_SUCCESS) return AI2B2_FAILURE_SUCCESS;
    if (d->outcome == AI2B2_OWN_STOCK_LOSS) return AI2B2_FAILURE_OWN_STOCK_LOSS;
    if (d->outcome == AI2B2_OPPONENT_STOCK_LOSS) return AI2B2_FAILURE_OPPONENT_STOCK_LOSS;
    if (d->outcome == AI2B2_UNSAFE_SUPPORT) return AI2B2_FAILURE_UNSAFE_SUPPORT;
    if (d->outcome == AI2B2_RESPONSE_DEADLINE ||
        d->outcome == AI2B2_SECOND_HIT_DEADLINE ||
        d->outcome == AI2B2_TOTAL_DEADLINE) {
        if (d->relocation_tick >= 0) {
            if (d->pursuit_tick < 0) {
                if (d->first_hit_tick >= 0 && d->renewed_movement_request_ticks == 0)
                    return AI2B2_FAILURE_FIRST_HIT_ONLY;
                return AI2B2_FAILURE_FAILED_RENEWED_PURSUIT;
            }
            if (d->second_hit_tick < 0) return AI2B2_FAILURE_PURSUED_MISSED_SECOND;
        }
        if (d->first_hit_tick < 0) {
            if (d->attack_start_count > 0) return AI2B2_FAILURE_STARTED_MISSED;
            if (d->light_request_ticks > 0) return AI2B2_FAILURE_REQUESTED_NO_START;
            if (d->movement_request_ticks > 0 && d->max_initial_approach_progress < 1.0)
                return AI2B2_FAILURE_NO_APPROACH;
            return AI2B2_FAILURE_NO_ATTACK_REQUEST;
        }
        if (d->relocation_tick < 0) return AI2B2_FAILURE_FIRST_HIT_ONLY;
    }
    return AI2B2_FAILURE_DEADLINE;
}

AI2N_HD static inline int ai2b2_light_move(int move) {
    return move == 0 || move == 1 || move == 2 || move == 6 || move == 7 || move == 8;
}

AI2N_HD static inline void ai2b2_sync_diagnostic_flags(AI2B2State *s) {
    AI2B2Diagnostics *d = &s->diagnostics;
    d->terminal_reason = d->outcome;
    d->success = d->outcome == AI2B2_SUCCESS;
    d->sequence_completed = d->success && d->first_hit_attack_id && d->second_hit_attack_id;
    d->achieved_first_hit = d->first_hit_attack_id != 0;
    d->achieved_relocation = d->relocation_tick >= 0;
    d->achieved_pursuit = d->pursuit_tick >= 0;
    d->achieved_second_hit = d->second_hit_attack_id != 0;
    d->achieved_safe_resolution = d->success;
    d->failure_category = ai2b2_failure_category_value(d);
}

AI2N_HD static inline int ai2b2_state_valid(const AI2B2State *s) {
    const AI2B2Diagnostics *d;
    int i;
    if (!s || !ai2b2_scenario_valid(&s->scenario) || s->initialized != 1 ||
        s->game.initialized != 1) return 0;
    d = &s->diagnostics;
    if (d->seed != s->scenario.seed || d->player_slot != s->scenario.player_slot ||
        d->direction != s->scenario.direction || d->scenario_id_hash != ai2b2_scenario_hash(&s->scenario) ||
        d->spawn_gap != s->scenario.gap ||
        d->spawn_midpoint != s->scenario.midpoint ||
        d->tick < 0 || d->tick > AI2B2_TIME_LIMIT || d->tick != s->game.scene.steps ||
        d->outcome < AI2B2_ONGOING || d->outcome > AI2B2_TOTAL_DEADLINE ||
        (d->outcome == AI2B2_ONGOING && d->tick == AI2B2_TIME_LIMIT) ||
        s->first_attack_id != d->first_hit_attack_id ||
        s->second_attack_id != d->second_hit_attack_id ||
        (s->last_second_duplicate_id && s->last_second_duplicate_id != s->second_attack_id) ||
        d->attributed_damage_event_count < 0 ||
        d->attributed_damage_event_count > AI2B2_ATTRIBUTED_EVENT_CAPACITY ||
        d->terminal_reason != d->outcome || d->success != (d->outcome == AI2B2_SUCCESS) ||
        d->sequence_completed != (d->success && d->first_hit_attack_id && d->second_hit_attack_id) ||
        d->failure_category != ai2b2_failure_category_value(d)) return 0;
    if (d->first_hit_attack_id) {
        if (d->first_hit_tick < 1 || d->first_hit_tick > d->tick ||
            d->attributed_damage_event_count < 1 ||
            d->attributed_damage_attack_ids[0] != d->first_hit_attack_id ||
            d->attributed_damage_ticks[0] != d->first_hit_tick ||
            d->achieved_first_hit != 1) return 0;
    } else if (d->first_hit_tick != -1 || d->attributed_damage_event_count != 0 ||
        d->hit_attack_id != 0 || d->achieved_first_hit != 0) return 0;
    if (d->second_hit_attack_id) {
        if (d->second_hit_tick < 1 || d->second_hit_tick > d->tick ||
            !d->first_hit_attack_id || d->second_hit_attack_id == d->first_hit_attack_id ||
            d->attributed_damage_event_count != 2 ||
            d->attributed_damage_attack_ids[1] != d->second_hit_attack_id ||
            d->attributed_damage_ticks[1] != d->second_hit_tick ||
            d->achieved_second_hit != 1) return 0;
    } else if (d->second_hit_tick != -1 || d->attributed_damage_event_count > 1 ||
        d->achieved_second_hit != 0) return 0;
    if (s->second_stage != 0 && s->second_stage != 1) return 0;
    if (s->relocated != 0 && s->relocated != 1) return 0;
    if (s->second_stage != s->relocated) return 0;
    if (s->second_stage && (!s->first_attack_id || d->relocation_tick < 1 ||
        d->relocation_tick > d->tick || d->relocation_gap < 2.0 ||
        d->achieved_relocation != 1 ||
        (d->pursuit_direction != -1 && d->pursuit_direction != 1) ||
        (d->pursuit_tick != -1 && (d->pursuit_tick <= d->relocation_tick ||
         d->pursuit_tick > d->tick || d->achieved_pursuit != 1)))) return 0;
    if (!s->second_stage && (d->relocation_tick != -1 || d->pursuit_tick != -1 ||
        d->pursuit_direction != 0 || d->achieved_relocation != 0 ||
        d->achieved_pursuit != 0)) return 0;
    if (d->pursuit_tick < 0 && d->achieved_pursuit != 0) return 0;
    if (d->achieved_safe_resolution != d->success) return 0;
    if (d->outcome == AI2B2_SUCCESS && (!s->second_attack_id || d->pursuit_tick < 0)) return 0;
    if (d->outcome == AI2B2_RESPONSE_DEADLINE &&
        (!s->second_stage || d->pursuit_tick >= 0 || d->tick - d->relocation_tick < 60)) return 0;
    if (d->outcome == AI2B2_SECOND_HIT_DEADLINE &&
        (!s->second_stage || d->pursuit_tick < 0 || s->second_attack_id ||
         d->tick - d->relocation_tick < 90)) return 0;
    if (d->outcome == AI2B2_TOTAL_DEADLINE && d->tick != AI2B2_TIME_LIMIT) return 0;
    if (d->light_request_ticks < 0 || d->light_request_ticks > d->tick ||
        d->light_press_count < 0 || d->light_press_count > d->light_request_ticks ||
        d->max_light_hold < 0 || d->max_light_hold > d->light_request_ticks ||
        d->attack_ticks < 0 || d->attack_ticks > d->tick ||
        d->movement_request_ticks < 0 || d->movement_request_ticks > d->tick ||
        d->renewed_movement_request_ticks < 0 || d->renewed_movement_request_ticks > d->movement_request_ticks ||
        (s->previous_light != 0 && s->previous_light != 1) ||
        s->current_light_hold < 0 || s->current_light_hold > d->max_light_hold ||
        (s->previous_light == 0 && s->current_light_hold != 0)) return 0;
    for (i = 0; i < 2; ++i) {
        if (d->unsupported_ticks[i] < 0 || d->unsupported_ticks[i] > d->tick ||
            d->falls[i] < 0 || d->stock_changes[i] < 0 ||
            s->initial_stocks[i] != d->initial_stocks[i]) return 0;
    }
    return 1;
}

AI2N_HD static inline int ai2b2_event_validity(const AI2B2State *s, const AI2GPlayerEvent *e) {
    const int learner = s->scenario.player_slot, target = 1 - learner;
    if (!e || e->kind != AI2G_EVENT_DAMAGE || e->amount <= 0.0) return 0;
    if (e->source != learner || e->target != target) return AI2B2_ERR_EVIDENCE;
    if (e->attack_id == 0 || !ai2b2_light_move(e->move)) return AI2B2_ERR_EVIDENCE;
    if (s->second_stage && s->first_attack_id && e->attack_id == s->first_attack_id)
        return AI2B2_ERR_EVIDENCE;
    return 1;
}

AI2N_HD static inline void ai2b2_record_attributed_damage(AI2B2Diagnostics *d,
                                                          const AI2GPlayerEvent *e) {
    const int idx = d->attributed_damage_event_count;
    if (idx < AI2B2_ATTRIBUTED_EVENT_CAPACITY) {
        d->attributed_damage_attack_ids[idx] = e->attack_id;
        d->attributed_damage_ticks[idx] = d->tick;
        d->attributed_damage_sources[idx] = e->source;
        d->attributed_damage_targets[idx] = e->target;
        d->attributed_damage_moves[idx] = e->move;
        d->attributed_damage_amounts[idx] = e->amount;
    }
    if (d->attributed_damage_event_count < AI2B2_ATTRIBUTED_EVENT_CAPACITY)
        ++d->attributed_damage_event_count;
}

AI2N_HD static inline void ai2b2_snapshot(AI2GState *g, int slot, AI2B2PlayerSnapshot *out) {
    AI2NEpisode *scene = &g->scene;
    AI2NPlayer *p = &scene->players[slot];
    AI2GCombatPlayer *c = &g->combat[slot];
    AI2CWorld *w = &scene->world;
    AI2CBody *b = &w->bodies[p->body];
    const AI2CVec pos = ai2n_body_position(b);
    const AI2CBB floor = ai2c_shape_cache(&w->shapes[0], w->bodies[w->shapes[0].body].tf);
    const AI2CBB body = ai2c_shape_cache(&w->shapes[p->shape], b->tf);
    ai2n_zero(out, sizeof(*out));
    out->x = pos.x; out->y = pos.y; out->vx = b->v.x; out->vy = b->v.y;
    out->damage = p->damage; out->damage_done = c->damage_done; out->damage_taken = c->damage_taken_total;
    out->clearance = ai2n_pymin(body.l - floor.l, floor.r - body.r);
    out->player_slot = slot; out->facing = p->ms.facing; out->stocks = p->ms.stocks;
    out->state_id = p->ms.state_id; out->grounded = ai2n_is_on_floor(scene, slot);
    out->fully_supported = out->grounded && out->clearance >= 0.0;
    out->stun_time_stored = ai2_mov_slot_get(&p->ms, p->ms.state_id, AI2_MOV_SLOT_STUN_TIME_STORED, 0);
    out->attack_id = p->ms.state_id == AI2G_STATE_ATTACK ? c->attack.attack_id : 0;
}

AI2N_HD static inline int ai2b2_fresh_player(AI2GState *g, int slot, double x, double y, int facing) {
    AI2NEpisode *scene = &g->scene;
    AI2CWorld *w = &scene->world;
    const int body = scene->players[slot].body, shape = scene->players[slot].shape;
    const int stocks = scene->players[slot].ms.stocks;
    const double damage = scene->players[slot].damage;
    const double damage_done = g->combat[slot].damage_done, damage_taken = g->combat[slot].damage_taken_total;
    AI2CVec verts[4];
    int rc, leaf;
    rc = ai2c_world_remove_shape(w, shape);
    if (rc) return AI2N_ERR_CONTACT;
    rc = ai2c_world_remove_body(w, body);
    if (rc) return AI2N_ERR_CONTACT;
    ai2n_body_init(&w->bodies[body], AI2C_BODY_DYNAMIC, 1.0, 1e9, ai2c_v(x, y));
    ai2n_box_verts(verts, 0.87, 1.0);
    ai2n_shape_init(&w->shapes[shape], body, verts, 0.0, 0.7, AI2N_TYPE_PLAYER + slot,
                    AI2N_CAT_ALL, AI2N_CAT_ALL, (unsigned long long)scene->shape_id_counter++);
    ai2n_player_init(&scene->players[slot], body, shape, x, y);
    ai2g_player_init(scene, g->combat, slot);
    scene->players[slot].ms.stocks = stocks;
    scene->players[slot].damage = damage;
    scene->players[slot].ms.facing = facing;
    scene->players[slot].ms.state_id = AI2_STATE_STANDING;
    ai2_mov_slot_set(&scene->players[slot].ms, AI2_STATE_STANDING, AI2_MOV_SLOT_DASH_TIMER, 0);
    g->combat[slot].damage_done = damage_done;
    g->combat[slot].damage_taken_total = damage_taken;
    g->combat[slot].hurtbox_x = x; g->combat[slot].hurtbox_y = y;
    rc = ai2c_world_add_shape(w, shape);
    if (rc) return AI2N_ERR_CONTACT;
    w->shapes[shape].categories = AI2N_CAT_PLAYER;
    w->shapes[shape].mask = AI2N_CAT_ALL & ~AI2N_CAT_WEAPON;
    (void)ai2c_shape_cache(&w->shapes[shape], w->bodies[body].tf);
    leaf = w->leaf_of_shape[shape];
    if (leaf >= 0) {
        if (ai2c_leaf_update(w, &w->dyn_tree, leaf)) {
            AI2CMarkContext ctx = {1, w->static_tree.root, 0};
            ai2c_mark_leaf(w, leaf, &ctx);
        }
        w->dyn_tree.stamp++;
    }
    return 0;
}

AI2N_HD static inline int ai2b2_fail(AI2B2State *s, AI2B2StepResult *out, int rc) {
    if (s) s->error = rc;
    if (out) out->error = rc;
    return rc;
}

AI2N_HD static inline int ai2b2_fail_restore(AI2B2State *s, AI2NRng *rng,
                                             const AI2B2State *state_before,
                                             const AI2NRng *rng_before,
                                             AI2B2StepResult *out, int rc) {
    if (s && state_before) *s = *state_before;
    if (rng && rng_before) *rng = *rng_before;
    if (out) out->error = rc;
    return rc;
}

AI2N_HD static inline void ai2b2_fill_obs(AI2B2State *s, int learner, AI2B2StepResult *out) {
    double obs[2][64];
    int i;
    if (!ai2g_observe(&s->game, obs)) {
        for (i = 0; i < 64; ++i) {
            const int velocity_slot = i == 2 || i == 3 || i == 34 || i == 35;
            float value = velocity_slot && fabs(obs[learner][i]) < AI2B2_OBS_RESIDUAL_VELOCITY_EPSILON
                ? 0.0f : (float)obs[learner][i];
            out->obs[i] = (double)value;
        }
    }
}

AI2N_HD static inline int ai2b2_reset(AI2B2State *s, const AI2NConfig *cfg, AI2NRng *rng,
                                     double scale, const AI2B2Scenario *scenario, AI2B2StepResult *out) {
    AI2GStepResult raw;
    AI2B2Diagnostics *d;
    AI2CBB floor;
    double y;
    int rc, i, learner;
    if (!s || !cfg || !rng || !scenario || !out) return AI2N_ERR_NULL;
    ai2n_zero(out, sizeof(*out));
    if (!ai2b2_scenario_valid(scenario)) { out->error = AI2B2_ERR_SCENARIO; return out->error; }
    if (cfg->time_limit_steps != AI2B2_TIME_LIMIT || !isfinite(scale) || scale <= 0.0) {
        out->error = AI2N_ERR_CONFIG; return out->error;
    }
    ai2n_zero(s, sizeof(*s));
    rc = ai2g_reset(&s->game, cfg, rng, scale, &raw);
    if (rc) return ai2b2_fail(s, out, rc);
    s->scenario = *scenario;
    d = &s->diagnostics; learner = scenario->player_slot;
    d->seed = scenario->seed; d->scenario_id_hash = ai2b2_scenario_hash(scenario);
    d->player_slot = learner; d->direction = scenario->direction;
    d->first_hit_tick = -1; d->second_hit_tick = -1; d->relocation_tick = -1; d->pursuit_tick = -1;
    d->spawn_gap = scenario->gap; d->spawn_midpoint = scenario->midpoint; d->initial_gap = scenario->gap;
    d->initial_learner_x = scenario->midpoint - scenario->direction * scenario->gap / 2.0;
    d->initial_target_x = scenario->midpoint + scenario->direction * scenario->gap / 2.0;
    floor = ai2c_shape_cache(&s->game.scene.world.shapes[0], s->game.scene.world.bodies[0].tf);
    y = floor.b - 0.5;
    rc = ai2b2_fresh_player(&s->game, learner, d->initial_learner_x, y, scenario->direction);
    if (!rc) rc = ai2b2_fresh_player(&s->game, 1-learner, d->initial_target_x, y, -scenario->direction);
    if (rc) return ai2b2_fail(s, out, rc);
    for (i = 0; i < 2; ++i) {
        ai2b2_snapshot(&s->game, i, &d->players[i]);
        if (!d->players[i].fully_supported || d->players[i].clearance < 0.2)
            return ai2b2_fail(s, out, AI2B2_ERR_SUPPORT);
        d->initial_stocks[i] = d->players[i].stocks;
        s->initial_stocks[i] = d->players[i].stocks;
        d->min_clearance[i] = d->players[i].clearance;
        out->pre_players[i] = d->players[i];
    }
    (void)raw;
    ai2b2_fill_obs(s, learner, out);
    s->initialized = 1; ai2b2_sync_diagnostic_flags(s); out->diagnostics = *d;
    return 0;
}

AI2N_HD static inline int ai2b2_choose_anchor(AI2GState *g, int target_slot, double learner_x,
                                             int direction, double *anchor_out) {
    AI2CWorld *w = &g->scene.world;
    AI2NPlayer *target = &g->scene.players[target_slot];
    AI2CBody *body = &w->bodies[target->body];
    AI2CBB floor = ai2c_shape_cache(&w->shapes[0], w->bodies[w->shapes[0].body].tf);
    AI2CBB bb = ai2c_shape_cache(&w->shapes[target->shape], body->tf);
    double half = (bb.r - bb.l) / 2.0;
    double left = floor.l + half + 0.2, right = floor.r - half - 0.2;
    double gl = fabs(left - learner_x), gr = fabs(right - learner_x);
    int vl = left <= right && gl >= 2.0, vr = left <= right && gr >= 2.0;
    if (!vl && !vr) return 0;
    if (vl && !vr) *anchor_out = left;
    else if (!vl && vr) *anchor_out = right;
    else if (gl == gr) *anchor_out = direction == 1 ? right : left;
    else *anchor_out = gl > gr ? left : right;
    return 1;
}

AI2N_HD static inline int ai2b2_relocate(AI2B2State *s) {
    AI2B2Diagnostics *d = &s->diagnostics;
    int learner = s->scenario.player_slot, target = 1 - learner;
    double anchor;
    if (!ai2b2_choose_anchor(&s->game, target, d->players[learner].x, s->scenario.direction, &anchor))
        return 0;
    ai2b2_snapshot(&s->game, target, &d->relocation_pre_target);
    d->relocation_tick = d->tick;
    d->relocation_learner_x = d->players[learner].x;
    d->relocation_target_x = anchor;
    d->relocation_gap = fabs(anchor - d->relocation_learner_x);
    d->pursuit_direction = anchor >= d->relocation_learner_x ? 1 : -1;
    if (ai2b2_fresh_player(&s->game, target, anchor,
        ai2c_shape_cache(&s->game.scene.world.shapes[0],
        s->game.scene.world.bodies[s->game.scene.world.shapes[0].body].tf).b - 0.5,
        -d->pursuit_direction)) return 0;
    ai2b2_snapshot(&s->game, target, &d->relocation_post_target);
    ai2b2_snapshot(&s->game, learner, &d->players[learner]);
    d->players[target] = d->relocation_post_target;
    s->second_stage = 1; s->relocated = 1; d->hit_attack_id = 0;
    return 1;
}

AI2N_HD static inline int ai2b2_debug_prepare_boundary(AI2B2State *s,
                                                       const AI2GPlatformTable *table,
                                                       int mode,
                                                       int elapsed_after_step,
                                                       int tick_after_step) {
    AI2B2Diagnostics *d;
    int learner, target, dir, before_tick;
    double current_gap;
    if (!s || !ai2b2_state_valid(s)) return AI2B2_ERR_STATE;
    if (mode < 1 || mode > 2 || elapsed_after_step < 1 ||
        tick_after_step <= elapsed_after_step || tick_after_step > AI2B2_TIME_LIMIT)
        return AI2B2_ERR_STATE;
    d = &s->diagnostics;
    learner = s->scenario.player_slot;
    target = 1 - learner;
    before_tick = tick_after_step - 1;
    ai2b2_snapshot(&s->game, learner, &d->players[learner]);
    ai2b2_snapshot(&s->game, target, &d->players[target]);
    dir = d->players[target].x >= d->players[learner].x ? 1 : -1;
    current_gap = fabs(d->players[target].x - d->players[learner].x);
    s->game.scene.steps = before_tick;
    if (table && table->count > before_tick) {
        AI2CBody *platform = &s->game.scene.world.bodies[s->game.scene.platform_body];
        platform->p = ai2c_v(table->px[before_tick], table->py[before_tick]);
        platform->v = ai2c_v(table->vx[before_tick], table->vy[before_tick]);
        ai2c_body_set_transform(platform, platform->p, platform->a);
        s->game.scene.moving_to_w2 = table->dir_after[before_tick] ? 1 : 0;
    }
    d->tick = before_tick;
    s->first_attack_id = 111;
    s->second_attack_id = 0;
    s->last_second_duplicate_id = 0;
    s->second_stage = 1;
    s->relocated = 1;
    d->outcome = AI2B2_ONGOING;
    d->first_hit_attack_id = 111;
    d->second_hit_attack_id = 0;
    d->hit_attack_id = 0;
    d->first_hit_tick = tick_after_step - elapsed_after_step - 1;
    if (d->first_hit_tick < 1) d->first_hit_tick = 1;
    d->second_hit_tick = -1;
    d->relocation_tick = tick_after_step - elapsed_after_step;
    d->pursuit_direction = dir;
    d->relocation_learner_x = d->players[learner].x - dir * 1.2;
    d->relocation_target_x = d->players[target].x;
    d->relocation_gap = current_gap + 1.2;
    d->gap_closed = mode == 2 ? 1.2 : 0.0;
    d->learner_progress = mode == 2 ? 1.2 : 0.0;
    d->pursuit_tick = mode == 2 ? d->relocation_tick + 1 : -1;
    d->attributed_damage_event_count = 1;
    d->attributed_damage_attack_ids[0] = 111;
    d->attributed_damage_ticks[0] = d->first_hit_tick;
    d->attributed_damage_sources[0] = learner;
    d->attributed_damage_targets[0] = target;
    d->attributed_damage_moves[0] = 0;
    d->attributed_damage_amounts[0] = 1.0;
    d->attributed_damage_attack_ids[1] = 0;
    d->attributed_damage_ticks[1] = 0;
    d->attributed_damage_sources[1] = 0;
    d->attributed_damage_targets[1] = 0;
    d->attributed_damage_moves[1] = 0;
    d->attributed_damage_amounts[1] = 0.0;
    d->relocation_pre_target = d->players[target];
    d->relocation_post_target = d->players[target];
    ai2b2_sync_diagnostic_flags(s);
    return ai2b2_state_valid(s) ? 0 : AI2B2_ERR_STATE;
}

AI2N_HD static inline int ai2b2_step_core(AI2B2State *s, const double action[10], AI2NRng *rng,
                                          const AI2GPlatformTable *table, AI2B2StepResult *out,
                                          int inject_stream, const AI2GPlayerEvent *inject_event) {
    double executed[2][10];
    AI2GStepResult raw;
    AI2B2State state_before;
    AI2NRng rng_before;
    AI2B2Diagnostics *d;
    AI2GCombatPlayer *combat, *stream_combat;
    int i, j, stream, rc, learner, target, requested, resolved = 1;
    if (!s || !action || !rng || !out) return AI2N_ERR_NULL;
    ai2n_zero(out, sizeof(*out));
    if (!s->initialized) { out->error = AI2G_ERR_UNINITIALIZED; return out->error; }
    if (s->error) { out->error = s->error; return out->error; }
    if (!ai2b2_scenario_valid(&s->scenario)) { out->error = AI2B2_ERR_SCENARIO; return out->error; }
    if (!ai2b2_state_valid(s)) { out->error = AI2B2_ERR_STATE; return out->error; }
    if (s->diagnostics.outcome) { out->error = AI2B2_ERR_TERMINAL; return out->error; }
    if ((inject_stream < -1 || inject_stream > 1) || (inject_stream >= 0 && !inject_event)) {
        out->error = AI2B2_ERR_EVIDENCE; return out->error;
    }
    ai2n_zero(executed, sizeof(executed));
    learner = s->scenario.player_slot; target = 1 - learner; d = &s->diagnostics;
    for (i = 0; i < 10; ++i) {
        const float value = (float)action[i];
        if (!isfinite(value)) { out->error = AI2B2_ERR_ACTION; return out->error; }
        out->raw_action[i] = (double)value;
        executed[learner][i] = i == 5 || i == 8 || i == 9 ? 0.0 : ai2g_clip(value, 0.0, 1.0);
        out->executed_action[i] = executed[learner][i];
    }
    for (i = 0; i < 2; ++i) ai2b2_snapshot(&s->game, i, &out->pre_players[i]);
    requested = executed[learner][7] > 0.5;
    state_before = *s;
    rng_before = *rng;
    rc = ai2g_step(&s->game, (const double (*)[10])executed, rng, table, &raw);
    if (rc) return ai2b2_fail(s, out, rc);
    if (inject_stream >= 0) {
        stream_combat = &s->game.combat[inject_stream];
        if (stream_combat->event_count < 0 ||
            stream_combat->event_count >= AI2G_PLAYER_EVENT_CAPACITY)
            return ai2b2_fail_restore(s, rng, &state_before, &rng_before, out, AI2B2_ERR_EVIDENCE);
        stream_combat->events[stream_combat->event_count++] = *inject_event;
        if (s->game.combat[0].total_tick_events < INT32_MAX)
            s->game.combat[0].total_tick_events++;
    }
    ++d->tick;
    d->light_request_ticks += requested;
    d->light_press_count += requested && !s->previous_light;
    s->current_light_hold = requested ? s->current_light_hold + 1 : 0;
    if (s->current_light_hold > d->max_light_hold) d->max_light_hold = s->current_light_hold;
    s->previous_light = requested;
    d->movement_request_ticks += executed[learner][1] > 0.5 || executed[learner][3] > 0.5;
    d->renewed_movement_request_ticks += s->second_stage &&
        (executed[learner][1] > 0.5 || executed[learner][3] > 0.5);
    combat = &s->game.combat[learner];
    d->attack_starts = combat->attack_starts; d->attack_contacts = combat->attack_contacts;
    d->damage_events = combat->damage_events; d->attack_start_count = (int32_t)combat->attack_starts;
    for (i = 0; i < 2; ++i) {
        int before_stocks = out->pre_players[i].stocks;
        ai2b2_snapshot(&s->game, i, &d->players[i]);
        if (d->players[i].clearance < d->min_clearance[i]) d->min_clearance[i] = d->players[i].clearance;
        d->unsupported_ticks[i] += d->players[i].grounded && !d->players[i].fully_supported;
        if (d->players[i].stocks < before_stocks) d->falls[i] += before_stocks - d->players[i].stocks;
        d->stock_changes[i] = s->initial_stocks[i] - d->players[i].stocks;
        if (d->players[i].state_id == AI2G_STATE_ATTACK || d->players[i].state_id == AI2G_STATE_STUN ||
            d->players[i].stun_time_stored != 0) resolved = 0;
    }
    d->attack_ticks += d->players[learner].state_id == AI2G_STATE_ATTACK;
    if (!s->second_stage) {
        const double gap = fabs(d->players[target].x - d->players[learner].x);
        const double progress = d->direction * (d->players[learner].x - d->initial_learner_x);
        const double closed = d->initial_gap - gap;
        const double bounded = closed < progress ? closed : progress;
        if (bounded > d->max_initial_approach_progress) d->max_initial_approach_progress = bounded;
    }
    out->reward_terms[7] = -0.001;
    for (stream = 0; stream < 2; ++stream) {
        stream_combat = &s->game.combat[stream];
        for (i = 0; i < stream_combat->event_count; ++i) {
            const AI2GPlayerEvent *e = &stream_combat->events[i];
            if (e->kind == AI2G_EVENT_DAMAGE && e->amount > 0.0 && e->source != stream)
                return ai2b2_fail_restore(s, rng, &state_before, &rng_before, out,
                                          AI2B2_ERR_EVIDENCE);
            rc = ai2b2_event_validity(s, e);
            if (rc == AI2B2_ERR_EVIDENCE)
                return ai2b2_fail_restore(s, rng, &state_before, &rng_before, out, rc);
            if (rc != 1) continue;
            if (!s->second_stage && !s->first_attack_id) {
                s->first_attack_id = e->attack_id; d->first_hit_attack_id = e->attack_id;
                d->hit_attack_id = e->attack_id; d->first_hit_tick = d->tick;
                ai2b2_record_attributed_damage(d, e);
                out->reward_terms[0] = 0.5;
            } else if (s->second_stage) {
                if (s->second_attack_id && e->attack_id == s->second_attack_id) {
                    s->last_second_duplicate_id = e->attack_id;
                } else if (!s->second_attack_id) {
                    s->second_attack_id = e->attack_id; d->second_hit_attack_id = e->attack_id;
                    d->hit_attack_id = e->attack_id; d->second_hit_tick = d->tick;
                    ai2b2_record_attributed_damage(d, e);
                    out->reward_terms[2] = 1.0;
                }
            }
        }
    }
    if (s->second_stage && d->pursuit_tick < 0) {
        int elapsed = d->tick - d->relocation_tick;
        if (elapsed >= 1 && elapsed <= 60) {
            d->gap_closed = d->relocation_gap - fabs(d->players[target].x - d->players[learner].x);
            d->learner_progress = d->pursuit_direction * (d->players[learner].x - d->relocation_learner_x);
            if (d->gap_closed >= 1.0 && d->learner_progress >= 1.0) {
                d->pursuit_tick = d->tick; out->reward_terms[1] = 0.5;
            }
        }
    }
    if (d->players[learner].stocks < s->initial_stocks[learner]) {
        d->outcome = AI2B2_OWN_STOCK_LOSS; out->reward_terms[4] = -5.0;
    } else if (d->players[target].stocks < s->initial_stocks[target]) {
        d->outcome = AI2B2_OPPONENT_STOCK_LOSS; out->reward_terms[4] = -1.0;
    }
    if (!d->outcome && d->hit_attack_id && resolved && d->players[0].grounded && d->players[1].grounded) {
        if (!d->players[0].fully_supported || !d->players[1].fully_supported || d->players[learner].clearance < 0.2) {
            d->outcome = AI2B2_UNSAFE_SUPPORT; out->reward_terms[5] = -2.0;
        } else if (!s->second_stage) {
            (void)ai2b2_relocate(s);
        } else if (s->second_attack_id && d->pursuit_tick >= 0) {
            d->outcome = AI2B2_SUCCESS; out->reward_terms[3] = 2.0;
        }
    }
    if (!d->outcome && s->second_stage) {
        int elapsed = d->tick - d->relocation_tick;
        if (elapsed >= 60 && d->pursuit_tick < 0) {
            d->outcome = AI2B2_RESPONSE_DEADLINE; out->reward_terms[6] = -1.0;
        } else if (elapsed >= 90 && !s->second_attack_id) {
            d->outcome = AI2B2_SECOND_HIT_DEADLINE; out->reward_terms[6] = -1.0;
        }
    }
    if (!d->outcome && d->tick >= AI2B2_TIME_LIMIT) {
        d->outcome = AI2B2_TOTAL_DEADLINE; out->reward_terms[6] = -1.0;
    }
    for (j = 0; j < AI2B2_REWARD_TERMS; ++j) {
        d->reward_totals[j] += out->reward_terms[j];
        out->reward += out->reward_terms[j];
    }
    (void)raw;
    ai2b2_fill_obs(s, learner, out);
    out->terminated = d->outcome != AI2B2_ONGOING;
    ai2b2_sync_diagnostic_flags(s);
    out->diagnostics = *d;
    return 0;
}

AI2N_HD static inline int ai2b2_step(AI2B2State *s, const double action[10], AI2NRng *rng,
                                    const AI2GPlatformTable *table, AI2B2StepResult *out) {
    return ai2b2_step_core(s, action, rng, table, out, -1, (const AI2GPlayerEvent *)0);
}

#endif
