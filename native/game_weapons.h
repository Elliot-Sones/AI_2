#ifndef AI2_GAME_WEAPONS_H
#define AI2_GAME_WEAPONS_H

/* Frozen environment.py c0116fc7... WeaponGO/Pool/Spawner/Controller/Dropped.
 * All simulation is host/device code; no navigation wrapper is invoked.
 * A zero-initialized AI2GWeapons is required before first init. Subsequent
 * init follows a fresh raw scene reset and preserves lazy next-spawner ID.
 * Phase order: mirror AFTER both player physics and BEFORE world integration;
 * then world.step, scene.steps++, drop_all, pickup_all, update_and_prune.
 * Mirror clears this tick's ordered drop/equip events. The caller owns player
 * hurtbox snapshots (including cast-frame size changes) and animations.
 * Compile with -ffp-contract=off / --fmad=false.
 */
#include "navigation_episode.h"
#include "game_capsule.h"
#include <limits.h>

#define AI2G_WEAPON_CAPACITY 8
#define AI2G_SPAWNER_CAPACITY 8
#define AI2G_WEAPON_EVENT_CAPACITY 8
#define AI2G_WEAPON_REGULAR 0
#define AI2G_WEAPON_DROPPED 1
#define AI2G_WEAPON_EVENT_DROP 1
#define AI2G_WEAPON_EVENT_EQUIP 2
#define AI2G_WEAPON_ERR_NULL 101
#define AI2G_WEAPON_ERR_STATE 102
#define AI2G_WEAPON_ERR_GEOMETRY 103
#define AI2G_WEAPON_ERR_CAPACITY 104
#define AI2G_WEAPON_ERR_RNG 105
#define AI2G_WEAPON_ERR_PHYSICS 106
/* spawnervfx/spawn.gif: 18 frames x 60ms; max(1,round(.060*30)) each.
 * Asset SHA256 b76d02d162556865d56366d37a9648591a461deab40f989e20671c3263f77de2. */
#define AI2G_REGULAR_SPAWN_STEPS 36

typedef struct AI2GWeaponPlayerAux {
    double hurtbox_x, hurtbox_y, hurtbox_width, hurtbox_height;
    int32_t pickup_lock_until, pad;
} AI2GWeaponPlayerAux;

typedef struct AI2GWeapon {
    double world_pos[2]; /* Mirror, NOT necessarily the current body position. */
    int32_t weapon, active, body, shape, spawn_frame, physics_on;
} AI2GWeapon;

typedef struct AI2GSpawner {
    double world_pos[2]; /* Anchor used by Player.get_obs, never body-following. */
    int32_t id, kind, og_cooldown_frames, cooldown_frames, last_spawn_frame;
    int32_t despawn_frames, weapon_slot, flag, spawned_once, done, flipped;
    int32_t weapon; /* Fixed dropped kind: 0 Punch, 1 Spear, 2 Hammer. */
} AI2GSpawner;

typedef struct AI2GWeaponEvent {
    int32_t kind, agent, weapon, spawner_id;
} AI2GWeaponEvent;

typedef struct AI2GWeapons {
    double px_per_world; /* Frozen LOW camera=48; also accepts explicit other resolutions. */
    int32_t initialized, next_spawner_id, spawner_count, weapon_count, pool_count;
    int32_t event_count, last_error, pad;
    AI2GSpawner spawners[AI2G_SPAWNER_CAPACITY]; /* Controller insertion order. */
    AI2GWeapon weapons[AI2G_WEAPON_CAPACITY]; /* Stable object handles. */
    int32_t pool[AI2G_WEAPON_CAPACITY]; /* Inactive objects, return order. */
    AI2GWeaponEvent events[AI2G_WEAPON_EVENT_CAPACITY];
} AI2GWeapons;

AI2N_HD static inline int ai2gw_fail(AI2GWeapons *s, int error) {
    if (s && !s->last_error) s->last_error = error;
    return error;
}

AI2N_HD static inline int ai2gw_world_counts_valid(const AI2CWorld *w) {
    return w->body_count >= 0 && w->body_count <= AI2C_MAX_BODIES &&
        w->shape_count >= 0 && w->shape_count <= AI2C_MAX_SHAPES &&
        w->arbiter_count >= 0 && w->arbiter_count <= AI2C_MAX_ARBITERS &&
        w->solve_count >= 0 && w->solve_count <= AI2C_MAX_ARBITERS &&
        w->handler_count >= 0 && w->handler_count <= AI2C_MAX_HANDLERS;
}

AI2N_HD static inline int ai2gw_check(const AI2NEpisode *scene, AI2GWeapons *s) {
    if (!scene || !s) return ai2gw_fail(s, AI2G_WEAPON_ERR_NULL);
    if (s->last_error) return s->last_error;
    if (!ai2gw_world_counts_valid(&scene->world)) return ai2gw_fail(s, AI2G_WEAPON_ERR_STATE);
    if (!s->initialized || s->spawner_count < 2 || s->spawner_count > AI2G_SPAWNER_CAPACITY ||
        s->weapon_count < 0 || s->weapon_count > AI2G_WEAPON_CAPACITY ||
        s->pool_count < 0 || s->pool_count > AI2G_WEAPON_CAPACITY ||
        s->event_count < 0 || s->event_count > AI2G_WEAPON_EVENT_CAPACITY || scene->steps < 0)
        return ai2gw_fail(s, AI2G_WEAPON_ERR_STATE);
    if (!isfinite(s->px_per_world) || s->px_per_world <= 0.0)
        return ai2gw_fail(s, AI2G_WEAPON_ERR_GEOMETRY);
    return 0;
}

AI2N_HD static inline int ai2gw_capsule_valid(const AI2GCapsule *capsule) {
    AI2GCParts parts;
    return capsule->width > 0.0 && capsule->height > 0.0 && ai2gc_parts(capsule, &parts);
}

AI2N_HD static inline int ai2gw_draw(AI2GWeapons *s, AI2NRng *rng, int kind,
                                    double a, double b, double *out) {
    double value;
    if (!rng) return ai2gw_fail(s, AI2G_WEAPON_ERR_NULL);
    value = ai2n_rng_request(rng, kind, a, b);
    if (rng->error || !isfinite(value) || value < a || value > b ||
        (kind == AI2N_DRAW_RANDINT && value != floor(value)))
        return ai2gw_fail(s, AI2G_WEAPON_ERR_RNG);
    *out = value;
    return 0;
}

/* No RNG draws: import the two already-created regular spawners after reset.
 * Active imported spawners are rejected: init is not a mid-episode migration. */
AI2N_HD static inline int ai2g_weapons_init_from_scene(AI2NEpisode *scene, AI2GWeapons *s,
                                                      AI2GWeaponPlayerAux aux[2], double px_per_world) {
    int i, next_id;
    if (!scene || !s || !aux) return ai2gw_fail(s, AI2G_WEAPON_ERR_NULL);
    if (!ai2gw_world_counts_valid(&scene->world)) return ai2gw_fail(s, AI2G_WEAPON_ERR_STATE);
    if (!isfinite(px_per_world) || px_per_world <= 0.0 ||
        !isfinite(40.0 / px_per_world) || !isfinite(16.0 / px_per_world))
        return ai2gw_fail(s, AI2G_WEAPON_ERR_GEOMETRY);
    for (i = 0; i < 2; ++i) {
        if (scene->spawners[i].active) return ai2gw_fail(s, AI2G_WEAPON_ERR_STATE);
        if (!isfinite(scene->spawners[i].world_pos[0]) || !isfinite(scene->spawners[i].world_pos[1]))
            return ai2gw_fail(s, AI2G_WEAPON_ERR_GEOMETRY);
    }
    next_id = s->initialized ? s->next_spawner_id : 0;
    ai2n_zero(s, sizeof(*s));
    ai2n_zero(aux, 2 * sizeof(*aux));
    s->initialized = 1;
    s->next_spawner_id = next_id;
    s->px_per_world = px_per_world;
    s->spawner_count = 2;
    for (i = 0; i < 2; ++i) {
        const AI2NSpawner *source = &scene->spawners[i];
        AI2GSpawner *sp = &s->spawners[i];
        sp->id = i;
        sp->world_pos[0] = source->world_pos[0];
        sp->world_pos[1] = source->world_pos[1];
        sp->og_cooldown_frames = source->og_cooldown_frames;
        sp->cooldown_frames = source->cooldown_frames;
        sp->last_spawn_frame = source->last_spawn_frame;
        sp->despawn_frames = 350;
        sp->weapon_slot = -1;
        /* Player constructor starts hurtbox at (0,0), not at its spawn body. */
        aux[i].hurtbox_width = (290 / 2) * 2 * (1.024 / 320);
        aux[i].hurtbox_height = (320 / 2) * 2 * (1.024 / 320);
        aux[i].pickup_lock_until = -1;
    }
    return 0;
}

AI2N_HD static inline int ai2g_weapons_mirror_before_physics(AI2NEpisode *scene, AI2GWeapons *s) {
    int i, rc = ai2gw_check(scene, s);
    if (rc) return rc;
    s->event_count = 0;
    for (i = 0; i < s->weapon_count; ++i) {
        AI2GWeapon *weapon = &s->weapons[i];
        if (weapon->active && weapon->physics_on) {
            AI2CVec p;
            if (weapon->body < 0 || weapon->body >= scene->world.body_count ||
                !scene->world.body_active[weapon->body]) return ai2gw_fail(s, AI2G_WEAPON_ERR_STATE);
            p = ai2n_body_position(&scene->world.bodies[weapon->body]);
            if (!isfinite(p.x) || !isfinite(p.y)) return ai2gw_fail(s, AI2G_WEAPON_ERR_GEOMETRY);
            weapon->world_pos[0] = p.x;
            weapon->world_pos[1] = p.y;
        }
    }
    return 0;
}

AI2N_HD static inline int ai2gw_event(AI2GWeapons *s, int kind, int agent, int weapon, int id) {
    AI2GWeaponEvent *event;
    if (s->event_count >= AI2G_WEAPON_EVENT_CAPACITY) return ai2gw_fail(s, AI2G_WEAPON_ERR_CAPACITY);
    event = &s->events[s->event_count++];
    event->kind = kind; event->agent = agent; event->weapon = weapon; event->spawner_id = id;
    return 0;
}

/* WeaponPool.get_weapon: first matching inactive object in return order. */
AI2N_HD static inline int ai2gw_activate(AI2NEpisode *scene, AI2GWeapons *s,
                                       AI2GSpawner *sp, int kind) {
    AI2CWorld *world = &scene->world;
    AI2CVec verts[4];
    AI2GWeapon *weapon;
    int i, j, handle = -1, pool_index = -1, body, shape, rc;
    double width = 40 / s->px_per_world, height = 16 / s->px_per_world + 0.5;
    if (kind < 1 || kind > 2) return ai2gw_fail(s, AI2G_WEAPON_ERR_STATE);
    if (!isfinite(sp->world_pos[0]) || !isfinite(sp->world_pos[1]) ||
        !isfinite(width) || !isfinite(height) || width <= 0 || height <= 0)
        return ai2gw_fail(s, AI2G_WEAPON_ERR_GEOMETRY);
    for (i = 0; i < s->pool_count; ++i) {
        int k = s->pool[i];
        if (k < 0 || k >= s->weapon_count || s->weapons[k].active)
            return ai2gw_fail(s, AI2G_WEAPON_ERR_STATE);
        if (s->weapons[k].weapon == kind) { handle = k; pool_index = i; break; }
    }
    if (handle < 0 && s->weapon_count >= AI2G_WEAPON_CAPACITY)
        return ai2gw_fail(s, AI2G_WEAPON_ERR_CAPACITY);
    body = ai2n_free_slot(world->body_active, world->body_count, AI2C_MAX_BODIES);
    shape = ai2n_free_slot(world->shape_active, world->shape_count, AI2C_MAX_SHAPES);
    if (body < 0 || shape < 0 || scene->shape_id_counter == INT_MAX)
        return ai2gw_fail(s, AI2G_WEAPON_ERR_CAPACITY);
    if (handle < 0) {
        handle = s->weapon_count++;
        ai2n_zero(&s->weapons[handle], sizeof(s->weapons[handle]));
    } else {
        for (j = pool_index; j + 1 < s->pool_count; ++j) s->pool[j] = s->pool[j + 1];
        s->pool_count--;
    }
    weapon = &s->weapons[handle];
    weapon->weapon = kind;
    weapon->active = weapon->physics_on = 1;
    weapon->world_pos[0] = sp->world_pos[0]; weapon->world_pos[1] = sp->world_pos[1];
    weapon->spawn_frame = scene->steps;
    weapon->body = body; weapon->shape = shape;
    ai2n_body_init(&world->bodies[body], AI2C_BODY_DYNAMIC, 100.0, 1e9,
                   ai2c_v(sp->world_pos[0], sp->world_pos[1]));
    ai2n_box_verts(verts, width, height);
    ai2n_shape_init(&world->shapes[shape], body, verts, 1.0, 100000000000.0,
                    AI2N_TYPE_WEAPON, AI2N_CAT_ALL, AI2N_CAT_ALL,
                    (unsigned long long)scene->shape_id_counter++);
    if (body >= world->body_count) world->body_count = body + 1;
    if (shape >= world->shape_count) world->shape_count = shape + 1;
    rc = ai2c_world_add_shape(world, shape);
    if (rc) return ai2gw_fail(s, AI2G_WEAPON_ERR_PHYSICS);
    /* Original assigns filter only after space.add(body, shape). */
    world->shapes[shape].categories = AI2N_CAT_WEAPON;
    world->shapes[shape].mask = AI2N_CAT_GROUND;
    sp->weapon_slot = handle;
    sp->last_spawn_frame = scene->steps;
    return 0;
}

AI2N_HD static inline int ai2gw_despawn(AI2NEpisode *scene, AI2GWeapons *s,
                                      AI2GSpawner *sp, AI2NRng *rng) {
    AI2GWeapon *weapon;
    double cooldown;
    int rc, slot = sp->weapon_slot;
    if (slot < 0) return 0;
    if (slot >= s->weapon_count) return ai2gw_fail(s, AI2G_WEAPON_ERR_STATE);
    if (s->pool_count >= AI2G_WEAPON_CAPACITY) return ai2gw_fail(s, AI2G_WEAPON_ERR_CAPACITY);
    if (sp->kind == AI2G_WEAPON_REGULAR) {
        rc = ai2gw_draw(s, rng, AI2N_DRAW_RANDINT, (int)(sp->og_cooldown_frames * 0.5),
                        (int)(sp->og_cooldown_frames * 1.5), &cooldown);
        if (rc) return rc;
        sp->cooldown_frames = (int)cooldown;
    }
    weapon = &s->weapons[slot];
    if (!weapon->active) return ai2gw_fail(s, AI2G_WEAPON_ERR_STATE);
    weapon->active = 0;
    /* Pymunk remove(body,shape) argument order is observable to contact cache. */
    if (ai2c_world_remove_body(&scene->world, weapon->body) ||
        ai2c_world_remove_shape(&scene->world, weapon->shape))
        return ai2gw_fail(s, AI2G_WEAPON_ERR_PHYSICS);
    weapon->body = weapon->shape = -1;
    s->pool[s->pool_count++] = slot;
    sp->weapon_slot = -1;
    return 0;
}

AI2N_HD static inline int ai2g_weapons_drop_all(AI2NEpisode *scene, AI2GWeapons *s,
                                               AI2GWeaponPlayerAux aux[2], AI2NRng *rng) {
    int i, rc = ai2gw_check(scene, s);
    if (rc) return rc;
    if (!aux || !rng) return ai2gw_fail(s, AI2G_WEAPON_ERR_NULL);
    if (!s->next_spawner_id) s->next_spawner_id = 2; /* try_drop's lazy initialization. */
    for (i = 0; i < 2; ++i) {
        AI2NPlayer *player = &scene->players[i];
        AI2GSpawner *sp;
        AI2CVec pos;
        double unused_phase;
        int flipped, a;
        if (player->ms.state_id == 8 || !player->input_state[15] || player->weapon == 0) continue;
        if (player->weapon < 1 || player->weapon > 2 || player->body < 0 ||
            player->body >= scene->world.body_count) return ai2gw_fail(s, AI2G_WEAPON_ERR_STATE);
        if (s->spawner_count >= AI2G_SPAWNER_CAPACITY || s->event_count >= AI2G_WEAPON_EVENT_CAPACITY ||
            s->next_spawner_id == INT_MAX || scene->steps > INT_MAX - 15)
            return ai2gw_fail(s, AI2G_WEAPON_ERR_CAPACITY);
        pos = ai2n_body_position(&scene->world.bodies[player->body]);
        if (!isfinite(pos.x) || !isfinite(pos.y)) return ai2gw_fail(s, AI2G_WEAPON_ERR_GEOMETRY);
        rc = ai2gw_draw(s, rng, AI2N_DRAW_RANDINT, 0, 350, &unused_phase);
        if (rc) return rc;
        sp = &s->spawners[s->spawner_count++];
        ai2n_zero(sp, sizeof(*sp));
        sp->id = s->next_spawner_id++;
        sp->kind = AI2G_WEAPON_DROPPED;
        sp->weapon = player->weapon;
        flipped = player->ms.facing == -1;
        a = flipped ? 1 : -1;
        sp->world_pos[0] = pos.x - a * 0.1;
        sp->world_pos[1] = pos.y + 0.27;
        sp->og_cooldown_frames = sp->cooldown_frames = 1000000000;
        sp->last_spawn_frame = -(1000000000 - (int)unused_phase);
        sp->despawn_frames = 250;
        sp->weapon_slot = -1;
        sp->flipped = flipped;
        aux[i].pickup_lock_until = scene->steps + 15;
        player->weapon = 0;
        rc = ai2gw_event(s, AI2G_WEAPON_EVENT_DROP, i, sp->weapon, sp->id);
        if (rc) return rc;
    }
    return 0;
}

AI2N_HD static inline int ai2g_weapons_pickup_all(AI2NEpisode *scene, AI2GWeapons *s,
                                                 AI2GWeaponPlayerAux aux[2], AI2NRng *rng) {
    int i, j, rc = ai2gw_check(scene, s);
    if (rc) return rc;
    if (!aux || !rng) return ai2gw_fail(s, AI2G_WEAPON_ERR_NULL);
    for (i = 0; i < s->spawner_count; ++i) {
        AI2GSpawner *sp = &s->spawners[i];
        if (sp->weapon_slot < 0 || (sp->kind == AI2G_WEAPON_REGULAR && !sp->flag)) continue;
        if (sp->weapon_slot >= s->weapon_count) return ai2gw_fail(s, AI2G_WEAPON_ERR_STATE);
        for (j = 0; j < 2 && sp->weapon_slot >= 0; ++j) {
            AI2NPlayer *player = &scene->players[j];
            AI2GWeapon *weapon = &s->weapons[sp->weapon_slot];
            AI2GCapsule hurtbox, pickup;
            int pressed;
            if (player->ms.state_id == 8 || player->weapon != 0) continue;
            if (sp->kind == AI2G_WEAPON_DROPPED && scene->steps < aux[j].pickup_lock_until) continue;
            pressed = sp->kind == AI2G_WEAPON_DROPPED ? player->input_state[15] :
                      (player->input_state[5] || player->input_state[15]);
            hurtbox.x = aux[j].hurtbox_x; hurtbox.y = aux[j].hurtbox_y;
            hurtbox.width = aux[j].hurtbox_width; hurtbox.height = aux[j].hurtbox_height;
            pickup.x = weapon->world_pos[0]; pickup.y = weapon->world_pos[1];
            pickup.width = 1.5;
            pickup.height = sp->kind == AI2G_WEAPON_REGULAR ? 0.83 : 16 / s->px_per_world;
            if (!ai2gw_capsule_valid(&hurtbox) || !ai2gw_capsule_valid(&pickup))
                return ai2gw_fail(s, AI2G_WEAPON_ERR_GEOMETRY);
            if (!pressed || !ai2g_capsule_intersects(&hurtbox, &pickup)) continue;
            if (s->event_count >= AI2G_WEAPON_EVENT_CAPACITY) return ai2gw_fail(s, AI2G_WEAPON_ERR_CAPACITY);
            player->weapon = weapon->weapon;
            rc = ai2gw_event(s, AI2G_WEAPON_EVENT_EQUIP, j, weapon->weapon, sp->id);
            if (rc) return rc;
            if (sp->kind == AI2G_WEAPON_REGULAR) sp->last_spawn_frame = scene->steps;
            rc = ai2gw_despawn(scene, s, sp, rng);
            if (rc) return rc;
        }
    }
    return 0;
}

AI2N_HD static inline int ai2g_weapons_update_and_prune(AI2NEpisode *scene, AI2GWeapons *s,
                                                      AI2NRng *rng) {
    int i, kept, rc = ai2gw_check(scene, s);
    if (rc) return rc;
    if (!rng) return ai2gw_fail(s, AI2G_WEAPON_ERR_NULL);
    /* Controller passes len(spawners), including inactive and done entries,
     * to every update; it prunes only after ALL updates have run. */
    for (i = 0; i < s->spawner_count; ++i) {
        AI2GSpawner *sp = &s->spawners[i];
        if (sp->weapon_slot >= s->weapon_count || (sp->kind != 0 && sp->kind != 1))
            return ai2gw_fail(s, AI2G_WEAPON_ERR_STATE);
        if (sp->kind == AI2G_WEAPON_DROPPED && !sp->spawned_once) {
            rc = ai2gw_activate(scene, s, sp, sp->weapon);
            if (rc) return rc;
            sp->spawned_once = 1;
            continue;
        }
        if (sp->weapon_slot >= 0 && s->weapons[sp->weapon_slot].active) {
            AI2GWeapon *weapon = &s->weapons[sp->weapon_slot];
            int64_t age = (int64_t)scene->steps - weapon->spawn_frame;
            if (sp->kind == AI2G_WEAPON_REGULAR &&
                (int64_t)scene->steps - sp->last_spawn_frame >= AI2G_REGULAR_SPAWN_STEPS) sp->flag = 1;
            if ((sp->kind == AI2G_WEAPON_REGULAR && age > sp->despawn_frames) ||
                (sp->kind == AI2G_WEAPON_DROPPED && age >= sp->despawn_frames)) {
                rc = ai2gw_despawn(scene, s, sp, rng);
                if (rc) return rc;
                if (sp->kind == AI2G_WEAPON_DROPPED) sp->done = 1;
            }
            continue;
        }
        if (sp->kind == AI2G_WEAPON_DROPPED) {
            sp->done = 1;
        } else if ((int64_t)scene->steps - sp->last_spawn_frame >= sp->cooldown_frames && s->spawner_count <= 2) {
            double x, name;
            rc = ai2gw_draw(s, rng, AI2N_DRAW_UNIFORM, 2.6, 6.5, &x);
            if (rc) return rc;
            sp->world_pos[0] = sp->world_pos[1] == 0 + 0.7 ? x : -x;
            rc = ai2gw_draw(s, rng, AI2N_DRAW_RANDINT, 0, 1, &name);
            if (rc) return rc;
            rc = ai2gw_activate(scene, s, sp, (int)name + 1);
            if (rc) return rc;
            sp->flag = 0;
        }
    }
    kept = 0;
    for (i = 0; i < s->spawner_count; ++i) {
        if (s->spawners[i].kind == AI2G_WEAPON_DROPPED && s->spawners[i].done) continue;
        if (kept != i) s->spawners[kept] = s->spawners[i];
        ++kept;
    }
    s->spawner_count = kept;
    return 0;
}

#endif
