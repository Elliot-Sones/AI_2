#include "game_episode.h"
#include <stddef.h>

size_t ai2g_state_size(void) { return sizeof(AI2GState); }
size_t ai2g_result_size(void) { return sizeof(AI2GStepResult); }
size_t ai2g_table_size(void) { return sizeof(AI2GPlatformTable); }
size_t ai2g_combat_size(void) { return sizeof(AI2GCombatPlayer); }
size_t ai2g_weapons_size(void) { return sizeof(AI2GWeapons); }
size_t ai2g_state_offset(int field) {
    const size_t fields[] = {offsetof(AI2GState,scene), offsetof(AI2GState,combat), offsetof(AI2GState,weapons),
        offsetof(AI2GState,weapon_aux), offsetof(AI2GState,initialized), offsetof(AI2GState,error), offsetof(AI2GState,win_mask)};
    return field >= 0 && field < 7 ? fields[field] : (size_t)-1;
}
int ai2g_reset_export(AI2GState *g, const AI2NConfig *cfg, AI2NRng *rng, double pixels_per_world, AI2GStepResult *out) {
    return ai2g_reset(g, cfg, rng, pixels_per_world, out);
}
int ai2g_step_export(AI2GState *g, const double *actions, AI2NRng *rng, const AI2GPlatformTable *table, AI2GStepResult *out) {
    return ai2g_step(g, (const double (*)[10])actions, rng, table, out);
}
int ai2g_observe_export(AI2GState *g, double *out) { return ai2g_observe(g, (double (*)[64])out); }
void ai2g_table_build(AI2GPlatformTable *table, double dt) {
    AI2CBody body;
    int dir = 1, k;
    ai2n_body_init(&body, AI2C_BODY_KINEMATIC, 0.0, 0.0, ai2c_v(0.0,1.0));
    for (k = 0; k < AI2G_PLATFORM_TABLE_SIZE; ++k) {
        double vx = 0.0, vy = 0.0;
        uint8_t next = 0;
        const AI2CVec pos = ai2n_body_position(&body);
        ai2_navigation_platform_step_math(pos.x,pos.y,1.0,0.0,-1.0,2.0,(uint8_t)dir,&vx,&vy,&next);
        table->px[k] = body.p.x; table->py[k] = body.p.y;
        table->vx[k] = vx; table->vy[k] = vy; table->dir_after[k] = next ? 1 : 0;
        body.v = ai2c_v(vx,vy); ai2c_body_update_position(&body,dt); dir = next ? 1 : 0;
    }
    table->count = AI2G_PLATFORM_TABLE_SIZE; table->pad = 0;
}
int ai2g_place_player(AI2GState *g, int player, double x, double y, double vx, double vy) {
    AI2NPlayer *p;
    AI2CBody *b;
    if (!g || player < 0 || player > 1 || !isfinite(x) || !isfinite(y) || !isfinite(vx) || !isfinite(vy)) return AI2G_ERR_ACTION;
    p = &g->scene.players[player]; b = &g->scene.world.bodies[p->body];
    ai2n_body_teleport(b, x, y);
    b->v = ai2c_v(vx,vy); p->ms.x=x; p->ms.y=y; p->ms.vx=vx; p->ms.vy=vy;
    return 0;
}

/* Reset-only test fixture equivalent of space.reindex_shapes_for_body.
 * Chipmunk 7.0.3-0593976 cpBBTreeReindexObject updates pairs only after a
 * changed leaf, but increments its tree stamp for every existing leaf. */
int ai2g_fixture_player_position(AI2GState *g, int player, double x, double y) {
    AI2CWorld *w;
    int shape, body, leaf;
    if (!g || !g->initialized || g->scene.steps != 0 || player < 0 || player > 1 || !isfinite(x) || !isfinite(y))
        return AI2G_ERR_ACTION;
    w = &g->scene.world; body = g->scene.players[player].body;
    ai2n_body_teleport(&w->bodies[body], x, y);
    g->combat[player].hurtbox_x = x; g->combat[player].hurtbox_y = y;
    for (shape = 0; shape < w->shape_count; ++shape) {
        if (!w->shape_active[shape] || w->shapes[shape].body != body) continue;
        (void)ai2c_shape_cache(&w->shapes[shape], w->bodies[body].tf);
        leaf = w->leaf_of_shape[shape];
        if (leaf >= 0) {
            if (ai2c_leaf_update(w, &w->dyn_tree, leaf)) {
                AI2CMarkContext ctx = {1, w->static_tree.root, 0};
                ai2c_mark_leaf(w, leaf, &ctx);
            }
            w->dyn_tree.stamp++;
        }
    }
    return 0;
}
