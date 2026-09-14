#ifndef AI2_CONTACT_EPISODE_H
#define AI2_CONTACT_EPISODE_H

/* bridge_b1_contact-v1. Raw gameplay is unchanged; one learner per game.
 * Scenario geometry is prepared with NumPy's seed-derived generator at setup.
 * Only raw-game draws are consumed from rng. All task endings are true terminals.
 * Reward terms are contact, completion, own-stock-loss, timeout, tick, in order.
 */
#include "game_episode.h"

#define AI2B1_ERR_SCENARIO 401
#define AI2B1_ERR_ACTION 402
#define AI2B1_ERR_TERMINAL 403
#define AI2B1_ERR_SUPPORT 404
#define AI2B1_ERR_STATE 405
#define AI2B1_REWARD_TERMS 5
#define AI2B1_TIME_LIMIT 300

enum {
    AI2B1_ONGOING = 0, AI2B1_SUCCESS = 1, AI2B1_OWN_STOCK_LOSS = 2,
    AI2B1_OPPONENT_STOCK_LOSS = 3, AI2B1_UNSAFE_SUPPORT = 4,
    AI2B1_INSUFFICIENT_APPROACH = 5, AI2B1_TIMEOUT = 6
};

typedef struct {
    uint64_t seed;
    int32_t player_slot, direction;
    double gap, midpoint;
} AI2B1Scenario;

typedef struct {
    double x, y, vx, vy, damage, damage_done, damage_taken, clearance;
    int32_t player_slot, facing, stocks, state_id, grounded, fully_supported, stun_time_stored;
    uint64_t attack_id;
} AI2B1PlayerSnapshot;

typedef struct {
    uint64_t seed, hit_attack_id;
    int32_t player_slot, direction, tick, outcome, first_hit_tick, initial_stocks[2];
    int32_t light_request_ticks, light_press_count, max_light_hold, attack_ticks;
    uint64_t attack_starts, attack_contacts, damage_events;
    double initial_learner_x, initial_target_x, spawn_gap, spawn_midpoint;
    double approach_progress, approach_progress_at_first_hit, reward_totals[AI2B1_REWARD_TERMS];
    double min_clearance[2];
    int32_t unsupported_ticks[2];
    AI2B1PlayerSnapshot players[2];
} AI2B1Diagnostics;

typedef struct {
    AI2GState game;
    AI2B1Scenario scenario;
    AI2B1Diagnostics diagnostics;
    int32_t initialized, error, previous_light, current_light_hold;
} AI2B1State;

typedef struct {
    double obs[64], reward, reward_terms[AI2B1_REWARD_TERMS], raw_action[10], executed_action[10];
    int32_t terminated, truncated, error;
    AI2B1PlayerSnapshot pre_players[2];
    AI2B1Diagnostics diagnostics;
} AI2B1StepResult;

AI2N_HD static inline int ai2b1_scenario_valid(const AI2B1Scenario *s) {
    return s && (s->player_slot == 0 || s->player_slot == 1) &&
        (s->direction == -1 || s->direction == 1) && isfinite(s->gap) &&
        s->gap >= 3.1 && s->gap <= 3.4 && isfinite(s->midpoint) &&
        s->midpoint >= 4.35 && s->midpoint <= 4.65;
}

/* Task bookkeeping guard for stepping and diagnostic lane imports. This does
 * not deserialize or validate arbitrary raw-physics world memory. A normal
 * terminal snapshot is valid, but an ongoing episode cannot reach tick 300.
 */
AI2N_HD static inline int ai2b1_state_valid(const AI2B1State *s) {
    const AI2B1Diagnostics *d;
    int i;
    if (!s || !ai2b1_scenario_valid(&s->scenario) || s->initialized != 1 ||
        s->game.initialized != 1) return 0;
    d = &s->diagnostics;
    if (d->seed != s->scenario.seed || d->player_slot != s->scenario.player_slot ||
        d->direction != s->scenario.direction || d->spawn_gap != s->scenario.gap ||
        d->spawn_midpoint != s->scenario.midpoint ||
        d->tick < 0 || d->tick > AI2B1_TIME_LIMIT || d->tick != s->game.scene.steps ||
        d->outcome < AI2B1_ONGOING || d->outcome > AI2B1_TIMEOUT ||
        (d->outcome == AI2B1_ONGOING && d->tick == AI2B1_TIME_LIMIT)) return 0;
    if (d->hit_attack_id ? d->first_hit_tick < 1 || d->first_hit_tick > d->tick :
        d->first_hit_tick != -1) return 0;
    if (d->light_request_ticks < 0 || d->light_request_ticks > d->tick ||
        d->light_press_count < 0 || d->light_press_count > d->light_request_ticks ||
        d->max_light_hold < 0 || d->max_light_hold > d->light_request_ticks ||
        d->attack_ticks < 0 || d->attack_ticks > d->tick ||
        (s->previous_light != 0 && s->previous_light != 1) ||
        s->current_light_hold < 0 || s->current_light_hold > d->max_light_hold ||
        (s->previous_light == 0 && s->current_light_hold != 0)) return 0;
    for (i = 0; i < 2; ++i)
        if (d->unsupported_ticks[i] < 0 || d->unsupported_ticks[i] > d->tick) return 0;
    return 1;
}

AI2N_HD static inline void ai2b1_snapshot(AI2GState *g, int slot, AI2B1PlayerSnapshot *out) {
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

/* Equivalent of constructing a new Player after space.remove(shape, body),
 * then setting StandingState and calling space.reindex_shapes_for_body.
 * Reusing storage slots does not reuse body state or collision shape identity.
 */
AI2N_HD static inline int ai2b1_fresh_player(AI2GState *g, int slot, double x, double y, int facing) {
    AI2NEpisode *scene = &g->scene;
    AI2CWorld *w = &scene->world;
    const int body = scene->players[slot].body, shape = scene->players[slot].shape;
    const int stocks = scene->players[slot].ms.stocks;
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

AI2N_HD static inline int ai2b1_fail(AI2B1State *s, AI2B1StepResult *out, int rc) {
    s->error = rc; out->error = rc;
    return rc;
}

AI2N_HD static inline int ai2b1_reset(AI2B1State *s, const AI2NConfig *cfg, AI2NRng *rng,
                                     double scale, const AI2B1Scenario *scenario, AI2B1StepResult *out) {
    AI2GStepResult raw;
    AI2B1Diagnostics *d;
    AI2CBB floor;
    double y;
    int rc, i, learner;
    if (!s || !cfg || !rng || !scenario || !out) return AI2N_ERR_NULL;
    ai2n_zero(out, sizeof(*out));
    if (!ai2b1_scenario_valid(scenario)) { out->error = AI2B1_ERR_SCENARIO; return out->error; }
    if (cfg->time_limit_steps != AI2B1_TIME_LIMIT || !isfinite(scale) || scale <= 0.0) {
        out->error = AI2N_ERR_CONFIG; return out->error;
    }
    s->initialized = 0; s->error = 0; s->previous_light = 0; s->current_light_hold = 0;
    rc = ai2g_reset(&s->game, cfg, rng, scale, &raw);
    if (rc) return ai2b1_fail(s, out, rc);
    s->scenario = *scenario;
    ai2n_zero(&s->diagnostics, sizeof(s->diagnostics));
    d = &s->diagnostics; learner = scenario->player_slot;
    d->seed = scenario->seed; d->player_slot = learner; d->direction = scenario->direction;
    d->first_hit_tick = -1;
    d->spawn_gap = scenario->gap; d->spawn_midpoint = scenario->midpoint;
    d->initial_learner_x = scenario->midpoint - scenario->direction * scenario->gap / 2.0;
    d->initial_target_x = scenario->midpoint + scenario->direction * scenario->gap / 2.0;
    floor = ai2c_shape_cache(&s->game.scene.world.shapes[0], s->game.scene.world.bodies[0].tf);
    y = floor.b - 0.5;
    rc = ai2b1_fresh_player(&s->game, learner, d->initial_learner_x, y, scenario->direction);
    if (!rc) rc = ai2b1_fresh_player(&s->game, 1-learner, d->initial_target_x, y, -scenario->direction);
    if (rc) return ai2b1_fail(s, out, rc);
    for (i = 0; i < 2; ++i) {
        ai2b1_snapshot(&s->game, i, &d->players[i]);
        if (!d->players[i].fully_supported || d->players[i].clearance < 0.2)
            return ai2b1_fail(s, out, AI2B1_ERR_SUPPORT);
        d->initial_stocks[i] = d->players[i].stocks;
        d->min_clearance[i] = d->players[i].clearance;
        out->pre_players[i] = d->players[i];
    }
    rc = ai2g_observe(&s->game, raw.obs);
    if (rc) return ai2b1_fail(s, out, rc);
    for (i = 0; i < 64; ++i) out->obs[i] = (double)(float)raw.obs[learner][i];
    s->initialized = 1; out->diagnostics = *d;
    return 0;
}

AI2N_HD static inline int ai2b1_light_move(int move) {
    /* Native move_idx is weapon*11 + MoveType-2. */
    return move == 0 || move == 1 || move == 2 || move == 6 || move == 7 || move == 8;
}

AI2N_HD static inline int ai2b1_step(AI2B1State *s, const double action[10], AI2NRng *rng,
                                    const AI2GPlatformTable *table, AI2B1StepResult *out) {
    double executed[2][10];
    AI2GStepResult raw;
    AI2B1Diagnostics *d;
    AI2GCombatPlayer *combat;
    int i, j, rc, learner, requested, resolved = 1;
    if (!s || !action || !rng || !out) return AI2N_ERR_NULL;
    ai2n_zero(out, sizeof(*out));
    if (!s->initialized) { out->error = AI2G_ERR_UNINITIALIZED; return out->error; }
    if (s->error) { out->error = s->error; return out->error; }
    if (!ai2b1_scenario_valid(&s->scenario)) { out->error = AI2B1_ERR_SCENARIO; return out->error; }
    if (!ai2b1_state_valid(s)) { out->error = AI2B1_ERR_STATE; return out->error; }
    if (s->diagnostics.outcome) { out->error = AI2B1_ERR_TERMINAL; return out->error; }
    ai2n_zero(executed, sizeof(executed));
    learner = s->scenario.player_slot; d = &s->diagnostics;
    for (i = 0; i < 10; ++i) {
        const float value = (float)action[i];
        if (!isfinite(value)) { out->error = AI2B1_ERR_ACTION; return out->error; }
        out->raw_action[i] = (double)value;
        executed[learner][i] = i == 5 || i == 8 || i == 9 ? 0.0 : ai2g_clip(value, 0.0, 1.0);
        out->executed_action[i] = executed[learner][i];
    }
    for (i = 0; i < 2; ++i) ai2b1_snapshot(&s->game, i, &out->pre_players[i]);
    requested = executed[learner][7] > 0.5;
    rc = ai2g_step(&s->game, (const double (*)[10])executed, rng, table, &raw);
    if (rc) return ai2b1_fail(s, out, rc);
    ++d->tick;
    d->light_request_ticks += requested;
    d->light_press_count += requested && !s->previous_light;
    s->current_light_hold = requested ? s->current_light_hold + 1 : 0;
    if (s->current_light_hold > d->max_light_hold) d->max_light_hold = s->current_light_hold;
    s->previous_light = requested;
    combat = &s->game.combat[learner];
    d->attack_starts = combat->attack_starts; d->attack_contacts = combat->attack_contacts;
    d->damage_events = combat->damage_events;
    for (i = 0; i < 2; ++i) {
        ai2b1_snapshot(&s->game, i, &d->players[i]);
        if (d->players[i].clearance < d->min_clearance[i]) d->min_clearance[i] = d->players[i].clearance;
        d->unsupported_ticks[i] += !d->players[i].fully_supported;
        if (d->players[i].state_id == AI2G_STATE_ATTACK || d->players[i].state_id == AI2G_STATE_STUN ||
            d->players[i].stun_time_stored != 0) resolved = 0;
    }
    d->attack_ticks += d->players[learner].state_id == AI2G_STATE_ATTACK;
    d->approach_progress = d->direction * (d->players[learner].x - d->initial_learner_x);
    out->reward_terms[4] = -0.001;
    if (d->players[learner].stocks < d->initial_stocks[learner]) {
        d->outcome = AI2B1_OWN_STOCK_LOSS; out->reward_terms[2] = -5.0;
    } else if (d->players[1-learner].stocks < d->initial_stocks[1-learner]) {
        d->outcome = AI2B1_OPPONENT_STOCK_LOSS;
    } else {
        if (!d->hit_attack_id) for (i = 0; i < combat->event_count; ++i) {
            const AI2GPlayerEvent *e = &combat->events[i];
            if (e->kind == AI2G_EVENT_DAMAGE && e->source == learner && e->target == 1-learner &&
                e->amount > 0.0 && e->attack_id != 0 && ai2b1_light_move(e->move)) {
                d->hit_attack_id = e->attack_id; d->first_hit_tick = d->tick;
                d->approach_progress_at_first_hit = d->approach_progress;
                out->reward_terms[0] = 1.0;
                break;
            }
        }
        if (d->hit_attack_id && resolved) {
            if (d->approach_progress_at_first_hit < 0.25) d->outcome = AI2B1_INSUFFICIENT_APPROACH;
            else if (d->players[0].grounded && d->players[1].grounded) {
                if (d->players[0].stocks > 0 && d->players[1].stocks > 0 &&
                    d->players[0].fully_supported && d->players[1].fully_supported && d->players[learner].clearance >= 0.2) {
                    d->outcome = AI2B1_SUCCESS; out->reward_terms[1] = 2.0;
                } else d->outcome = AI2B1_UNSAFE_SUPPORT;
            }
        }
        if (!d->outcome && d->tick >= AI2B1_TIME_LIMIT) {
            d->outcome = AI2B1_TIMEOUT; out->reward_terms[3] = -1.0;
        }
    }
    for (j = 0; j < AI2B1_REWARD_TERMS; ++j) {
        d->reward_totals[j] += out->reward_terms[j];
        out->reward += out->reward_terms[j];
    }
    for (i = 0; i < 64; ++i) out->obs[i] = (double)(float)raw.obs[learner][i];
    out->terminated = d->outcome != AI2B1_ONGOING;
    out->diagnostics = *d;
    return 0;
}

#endif
