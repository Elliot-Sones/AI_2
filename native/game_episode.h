#ifndef AI2_GAME_EPISODE_H
#define AI2_GAME_EPISODE_H

/* Raw two-player gameplay. Curriculum rewards/autoreset live outside this
 * module; neither player's inputs are masked or replaced. */
#include "game_player.h"
#include "game_weapons.h"

#define AI2G_PLATFORM_TABLE_SIZE (AI2N_RAW_MAX_TIMESTEPS + 1)
#define AI2G_ERR_UNINITIALIZED 201
#define AI2G_ERR_ACTION 202

typedef struct {
    int32_t count, pad;
    double px[AI2G_PLATFORM_TABLE_SIZE], py[AI2G_PLATFORM_TABLE_SIZE];
    double vx[AI2G_PLATFORM_TABLE_SIZE], vy[AI2G_PLATFORM_TABLE_SIZE];
    int32_t dir_after[AI2G_PLATFORM_TABLE_SIZE];
} AI2GPlatformTable;

typedef struct {
    AI2NEpisode scene;
    AI2GCombatPlayer combat[2];
    AI2GWeapons weapons;
    AI2GWeaponPlayerAux weapon_aux[2];
    int32_t initialized, error, win_mask, max_timesteps;
} AI2GState;

typedef struct {
    double obs[2][64], reward[2];
    int32_t terminated, truncated, error, win_mask;
} AI2GStepResult;

AI2N_HD static inline int ai2g_fail(AI2GState *g, AI2GStepResult *out, int error) {
    if (g && !g->error) g->error = error;
    if (out) out->error = error;
    return error;
}

AI2N_HD static inline double ai2g_clip(double x, double low, double high) {
    return x < low ? low : (x > high ? high : x);
}

AI2N_HD static inline int ai2g_observe(AI2GState *g, double out[2][64]) {
    double rows[2][32];
    int i, j;
    if (!g || !out) return AI2N_ERR_NULL;
    if (!g->initialized) return AI2G_ERR_UNINITIALIZED;
    for (i = 0; i < 2; ++i) {
        AI2NPlayer *p = &g->scene.players[i];
        AI2MovementState *s = &p->ms;
        const AI2GCombatPlayer *e = &g->combat[i];
        const AI2CBody *b = &g->scene.world.bodies[p->body];
        const AI2CBody *platform = &g->scene.world.bodies[g->scene.platform_body];
        const AI2CVec pos = ai2n_body_position(b), plat = ai2n_body_position(platform);
        double *r = rows[i];
        r[0] = ai2g_clip(pos.x, -18.0, 18.0); r[1] = ai2g_clip(pos.y, -7.0, 7.0);
        r[2] = ai2g_clip(b->v.x, -10.0, 10.0); r[3] = ai2g_clip(b->v.y, -10.0, 10.0);
        r[4] = s->facing == 1 ? 1.0 : 0.0;
        r[5] = ai2n_is_on_floor(&g->scene, i) ? 1.0 : 0.0;
        r[6] = r[5] == 1.0 ? 0.0 : 1.0;
        r[7] = (double)ai2_mov_slot_get(s, s->state_id, AI2_MOV_SLOT_JUMPS_LEFT, 0);
        r[8] = (double)s->state_id;
        r[9] = (double)ai2_mov_slot_get(s, s->state_id, AI2_MOV_SLOT_RECOVERIES_LEFT, 0);
        r[10] = (double)ai2_mov_slot_get(s, s->state_id, AI2_MOV_SLOT_DODGE_TIMER, 0);
        r[11] = s->state_id == AI2G_STATE_STUN ? (double)e->stun_frames : 0.0;
        r[12] = p->damage / 700.0; r[13] = (double)s->stocks;
        r[14] = s->state_id == AI2G_STATE_ATTACK && e->attack.move_idx >= 0 ? (double)(e->attack.move_idx % 11 + 2) : 0.0;
        r[15] = (double)p->weapon;
        for (j = 0; j < 4; ++j) {
            const AI2GSpawner *sp = j < g->weapons.spawner_count ? &g->weapons.spawners[j] : 0;
            const int at = 16 + j * 3;
            if (sp && sp->weapon_slot >= 0) {
                r[at] = ai2g_clip(sp->world_pos[0], -18.0, 18.0);
                r[at+1] = ai2g_clip(sp->world_pos[1], -7.0, 7.0);
                r[at+2] = sp->kind == AI2G_WEAPON_DROPPED ? (double)(sp->weapon + 1) : 1.0;
            } else r[at] = r[at+1] = r[at+2] = 0.0;
        }
        r[28] = ai2g_clip(plat.x, -18.0, 18.0); r[29] = ai2g_clip(plat.y, -7.0, 7.0);
        r[30] = ai2g_clip(platform->v.x, -10.0, 10.0); r[31] = ai2g_clip(platform->v.y, -10.0, 10.0);
    }
    for (i = 0; i < 2; ++i) for (j = 0; j < 32; ++j) {
        out[i][j] = rows[i][j]; out[i][32+j] = rows[1-i][j];
    }
    return 0;
}

/* State must initially be zero initialized; subsequent resets retain only the
 * original environment's monotonically increasing dropped-spawner ID. */
AI2N_HD static inline int ai2g_reset(AI2GState *g, const AI2NConfig *cfg, AI2NRng *rng,
                                    double pixels_per_world, AI2GStepResult *out) {
    int i, rc;
    if (!g || !cfg || !rng || !out) return AI2N_ERR_NULL;
    /* Raw-game horizon is bounded by the full-match platform table. Validate
     * these public configuration values before touching state or RNG. */
    if (!isfinite(cfg->dt) || !(cfg->dt > 0.0) ||
        cfg->time_limit_steps < 1 || cfg->time_limit_steps > AI2N_RAW_MAX_TIMESTEPS) {
        ai2n_zero(out, sizeof(*out)); out->error = AI2N_ERR_CONFIG;
        return AI2N_ERR_CONFIG;
    }
    ai2n_zero(out, sizeof(*out));
    g->initialized = 0; g->error = 0; g->win_mask = 0;
    g->max_timesteps = cfg->time_limit_steps;
    rc = ai2n_reset_scene(&g->scene, cfg, rng);
    if (rc) return ai2g_fail(g, out, rc);
    for (i = 0; i < 2; ++i) ai2g_player_init(&g->scene, g->combat, i);
    rc = ai2g_weapons_init_from_scene(&g->scene, &g->weapons, g->weapon_aux, pixels_per_world);
    if (rc) return ai2g_fail(g, out, rc);
    g->initialized = 1;
    return ai2g_observe(g, out->obs);
}

AI2N_HD static inline int ai2g_step(AI2GState *g, const double actions[2][10], AI2NRng *rng,
                                   const AI2GPlatformTable *table, AI2GStepResult *out) {
    AI2NEpisode *s;
    AI2CBody *platform;
    double vx = 0.0, vy = 0.0;
    uint8_t next = 0;
    int i, j, rc;
    if (!g || !actions || !rng || !out) return AI2N_ERR_NULL;
    ai2n_zero(out, sizeof(*out));
    if (!g->initialized) return ai2g_fail(g, out, AI2G_ERR_UNINITIALIZED);
    if (g->error) return ai2g_fail(g, out, g->error);
    for (i = 0; i < 2; ++i) for (j = 0; j < 10; ++j)
        if (!isfinite(actions[i][j])) return ai2g_fail(g, out, AI2G_ERR_ACTION);
    s = &g->scene;
    s->raw_terminated = 0; g->win_mask = 0;
    for (i = 0; i < 2; ++i) ai2g_player_pre_process(s, g->combat, i);
    for (i = 0; i < 2; ++i) {
        const double x = ai2n_body_position(&s->world.bodies[s->players[i].body]).x;
        rc = ai2_input_step_core(actions[i], s->players[i].input_state, s->players[i].input_state);
        if (rc) return ai2g_fail(g, out, AI2N_ERR_INPUT);
        /* Python resets each accumulator to integer zero, then adds movement.
         * Assignment would retain -0.0 where its addition produces +0.0. */
        out->reward[i] += (x < 0.0 ? 1.0 : -1.0) * (x - s->players[i].ms.prev_x);
        if (s->players[i].ms.stocks <= 0) { s->raw_terminated = 1; g->win_mask |= 1 << (1-i); }
    }
    platform = &s->world.bodies[s->platform_body];
    if (table) {
        const int k = s->steps;
        if (k < 0 || k >= table->count || k >= AI2G_PLATFORM_TABLE_SIZE ||
            platform->p.x != table->px[k] || platform->p.y != table->py[k])
            return ai2g_fail(g, out, AI2N_ERR_PLATFORM);
        vx = table->vx[k]; vy = table->vy[k]; next = (uint8_t)table->dir_after[k];
    } else {
        const AI2CVec pos = ai2n_body_position(platform);
        ai2_navigation_platform_step_math(pos.x, pos.y, s->waypoint1[0], s->waypoint1[1],
            s->waypoint2[0], s->waypoint2[1], (uint8_t)s->moving_to_w2, &vx, &vy, &next);
    }
    platform->v = ai2c_v(vx, vy); s->moving_to_w2 = next ? 1 : 0;
    for (i = 0; i < 2; ++i) {
        rc = ai2g_player_physics(s, g->combat, i, rng);
        if (rc) return ai2g_fail(g, out, rc);
        g->weapon_aux[i].hurtbox_x = g->combat[i].hurtbox_x;
        g->weapon_aux[i].hurtbox_y = g->combat[i].hurtbox_y;
        g->weapon_aux[i].hurtbox_width = g->combat[i].hurtbox_width;
        g->weapon_aux[i].hurtbox_height = g->combat[i].hurtbox_height;
    }
    rc = ai2g_weapons_mirror_before_physics(s, &g->weapons);
    if (rc) return ai2g_fail(g, out, rc);
    rc = ai2c_world_step(&s->world, s->cfg.dt, s->cfg.bias_coef, s->cfg.damping_dt);
    if (rc != AI2C_OK) return ai2g_fail(g, out, AI2N_ERR_CONTACT);
    s->steps += 1;
    rc = ai2g_weapons_drop_all(s, &g->weapons, g->weapon_aux, rng);
    if (rc) return ai2g_fail(g, out, rc);
    rc = ai2g_weapons_pickup_all(s, &g->weapons, g->weapon_aux, rng);
    if (rc) return ai2g_fail(g, out, rc);
    rc = ai2g_weapons_update_and_prune(s, &g->weapons, rng);
    if (rc) return ai2g_fail(g, out, rc);
    s->raw_truncated = s->steps >= g->max_timesteps;
    out->terminated = s->raw_terminated; out->truncated = s->raw_truncated; out->win_mask = g->win_mask;
    return ai2g_observe(g, out->obs);
}

#endif
