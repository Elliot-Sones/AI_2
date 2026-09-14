#ifndef AI2_NAVIGATION_EPISODE_H
#define AI2_NAVIGATION_EPISODE_H

/*
 * Closed-loop native tick of the frozen navigation environment (phase 0b).
 *
 * One AI2NEpisode is one lane: the raw WarehouseBrawl scene (Chipmunk contact
 * world, both player state machines, the moving platform, the two weapon
 * spawners) plus the FrozenOpponentWrapper of the frozen train.py.
 * ai2n_reset() rebuilds the scene the way WarehouseBrawl.reset() and
 * FrozenOpponentWrapper.reset() do; ai2n_step() runs one frame in the frozen
 * order: input decode, platform velocity, both player state machines, the
 * contact step, weapon spawn/despawn, then the wrapper (opponent freeze,
 * reward, settle/hold bookkeeping, target handoff, termination, observation).
 *
 * Randomness is explicit.  Every random draw the Python game makes
 * (random.getrandbits, random.uniform, random.randint, np.random.randint) is
 * requested from an AI2NRng.  In replay mode each request must match the next
 * recorded draw (kind and arguments), so a verifier can drive the lane with the
 * real game's outcomes; in generator mode a lane-local xoshiro256** stream
 * supplies them.  The Python game's opponent-selection draw (random.choices in
 * OpponentsCfg.on_env_reset) has no effect on the game and is not modelled.
 *
 * Source anchors (frozen environment.py sha256 c0116fc7..., train.py sha256
 * b93585d9..., control.yaml sha256 33b0103e...): WarehouseBrawl.step/_setup,
 * Ground/Stage/Player constructors, Player.physics_process/get_obs/is_on_floor,
 * Stage.physics_process, WeaponGO._ensure_body/activate/deactivate,
 * WeaponSpawner.update/spawn_weapon/despawn_weapon, FrozenOpponentWrapper.
 *
 * Not modelled (no effect on the training environment): pickup and drop
 * (index 5 is masked in movement control mode and the opponent's action is all
 * zeros), the spawner VFX pickup flag, attack/stun/taunt states, and the
 * wrapper's cumulative logging counters.
 */

#include <math.h>
#include <stdint.h>

/* The contact step must be compiled with the same floating-point contraction
 * rule as the Pymunk wheel that produced the reference (on for the macOS
 * wheel, off for manylinux), so it is included before the pragma below.
 * Everything after the pragma reproduces arithmetic that CPython performs one
 * operation at a time, so fused multiply-add is disabled for it regardless of
 * the command line. */
#include "navigation_contact.h"

#if defined(__clang__) && !defined(__CUDACC__)
#pragma clang fp contract(off)
#elif defined(__GNUC__) && !defined(__CUDACC__)
#pragma GCC optimize("fp-contract=off")
#endif

#include "navigation_input.h"
#include "navigation_movement.h"
#include "navigation_platform.h"
#include "navigation_observation.h"
#include "navigation_positions.h"

#ifdef __CUDACC__
#define AI2N_HD __host__ __device__
#else
#define AI2N_HD
#endif

#define AI2N_OBS_WIDTH 64
#define AI2N_ACTION_WIDTH 10
#define AI2N_RAW_MAX_TIMESTEPS 9000 /* WarehouseBrawl.max_timesteps = fps * 300 */
#define AI2N_DESPAWN_FRAMES 350
#define AI2N_SPAWNER_COUNT 2
#define AI2N_INITIAL_SHAPES 5

#define AI2N_TYPE_GROUND 1ull
#define AI2N_TYPE_PLAYER 3ull   /* PLAYER + agent_id */
#define AI2N_TYPE_WEAPON 5ull
#define AI2N_TYPE_PLATFORM 6ull /* PLATFORM + platform_id 1 */
#define AI2N_CAT_PLAYER 1u
#define AI2N_CAT_GROUND 2u
#define AI2N_CAT_WEAPON 4u
#define AI2N_CAT_ALL 0xFFFFFFFFu

#define AI2N_ERR_NONE 0
#define AI2N_ERR_NULL 1
#define AI2N_ERR_CONFIG 2
#define AI2N_ERR_RNG 3
#define AI2N_ERR_MOVEMENT 4
#define AI2N_ERR_CONTACT 5
#define AI2N_ERR_CAPACITY 6
#define AI2N_ERR_OBSERVATION 7
#define AI2N_ERR_TARGET 8
#define AI2N_ERR_INPUT 9
#define AI2N_ERR_PLATFORM 10
#define AI2N_PLATFORM_TABLE_SIZE 1024

/* ------------------------------------------------------------------ random */

enum { AI2N_DRAW_GETRANDBITS1 = 1, AI2N_DRAW_UNIFORM = 2, AI2N_DRAW_RANDINT = 3, AI2N_DRAW_NP_RANDINT = 4 };

typedef struct AI2NDraw {
    int32_t kind;
    int32_t pad;
    double a;
    double b;
    double value;
} AI2NDraw;

typedef struct AI2NRng {
    int32_t mode;   /* 0 = replay recorded draws, 1 = lane generator */
    int32_t error;  /* first replay failure: 1 exhausted, 2 kind, 3 arguments */
    int32_t cursor;
    int32_t count;
    const AI2NDraw *draws;
    unsigned long long s[4];
    int32_t error_kind;
    int32_t error_cursor;
    double error_a;
    double error_b;
} AI2NRng;

AI2N_HD static inline unsigned long long ai2n_rotl(unsigned long long x, int k) { return (x << k) | (x >> (64 - k)); }

AI2N_HD static inline unsigned long long ai2n_splitmix(unsigned long long *x) {
    unsigned long long z = (*x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

AI2N_HD static inline void ai2n_rng_seed(AI2NRng *r, unsigned long long seed, unsigned long long lane) {
    unsigned long long x = seed ^ (lane * 0x9E3779B97F4A7C15ull) ^ 0xD1B54A32D192ED03ull;
    int i;
    r->mode = 1;
    r->error = 0;
    r->cursor = 0;
    r->count = 0;
    r->draws = 0;
    for (i = 0; i < 4; ++i) r->s[i] = ai2n_splitmix(&x);
    r->error_kind = 0;
    r->error_cursor = 0;
    r->error_a = 0.0;
    r->error_b = 0.0;
}

AI2N_HD static inline void ai2n_rng_replay(AI2NRng *r, const AI2NDraw *draws, int count) {
    r->mode = 0;
    r->error = 0;
    r->cursor = 0;
    r->count = count;
    r->draws = draws;
    r->s[0] = r->s[1] = r->s[2] = r->s[3] = 0ull;
    r->error_kind = 0;
    r->error_cursor = 0;
    r->error_a = 0.0;
    r->error_b = 0.0;
}

AI2N_HD static inline unsigned long long ai2n_rng_next(AI2NRng *r) {
    unsigned long long *s = r->s;
    const unsigned long long result = ai2n_rotl(s[1] * 5ull, 7) * 9ull;
    const unsigned long long t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = ai2n_rotl(s[3], 45);
    return result;
}

AI2N_HD static inline double ai2n_rng_unit(AI2NRng *r) {
    return (double)(ai2n_rng_next(r) >> 11) * (1.0 / 9007199254740992.0);
}

AI2N_HD static inline double ai2n_rng_request(AI2NRng *r, int kind, double a, double b) {
    if (r->mode == 0) {
        const AI2NDraw *d;
        if (r->error) return 0.0;
        if (r->cursor >= r->count) {
            r->error = 1; r->error_kind = kind; r->error_cursor = r->cursor; r->error_a = a; r->error_b = b;
            return 0.0;
        }
        d = &r->draws[r->cursor];
        if (d->kind != kind) {
            r->error = 2; r->error_kind = kind; r->error_cursor = r->cursor; r->error_a = a; r->error_b = b;
            return 0.0;
        }
        if (d->a != a || d->b != b) {
            r->error = 3; r->error_kind = kind; r->error_cursor = r->cursor; r->error_a = a; r->error_b = b;
            return 0.0;
        }
        r->cursor += 1;
        return d->value;
    }
    if (kind == AI2N_DRAW_GETRANDBITS1) return (double)(ai2n_rng_next(r) >> 63);
    if (kind == AI2N_DRAW_UNIFORM) return a + (b - a) * ai2n_rng_unit(r);
    if (kind == AI2N_DRAW_RANDINT) {
        double span = b - a + 1.0;
        double v = a + floor(ai2n_rng_unit(r) * span);
        return v > b ? b : v;
    }
    if (kind == AI2N_DRAW_NP_RANDINT) {
        double v = floor(ai2n_rng_unit(r) * a);
        return v >= a ? a - 1.0 : v;
    }
    r->error = 4;
    return 0.0;
}

AI2N_HD static inline int ai2n_rng_getrandbits1(AI2NRng *r) { return (int)ai2n_rng_request(r, AI2N_DRAW_GETRANDBITS1, 1.0, 0.0); }
AI2N_HD static inline double ai2n_rng_uniform(AI2NRng *r, double a, double b) { return ai2n_rng_request(r, AI2N_DRAW_UNIFORM, a, b); }
AI2N_HD static inline int ai2n_rng_randint(AI2NRng *r, int a, int b) { return (int)ai2n_rng_request(r, AI2N_DRAW_RANDINT, (double)a, (double)b); }
AI2N_HD static inline int ai2n_rng_np_randint(AI2NRng *r, int n) { return (int)ai2n_rng_request(r, AI2N_DRAW_NP_RANDINT, (double)n, 0.0); }

/* ------------------------------------------------------------------ types */

typedef struct AI2NConfig {
    int32_t completion_mode;         /* 0 = horizon, 1 = sequence */
    int32_t control_mode;            /* 0 = full, 1 = movement (mask indices 5, 7, 8, 9) */
    int32_t max_targets;
    int32_t time_limit_steps;
    int32_t settle_steps;
    int32_t post_goal_required_steps;
    int32_t pad[2];
    double settle_speed;
    double fall_penalty;
    double min_dist;
    double max_dist;
    double dt;               /* 1 / 30 */
    double collision_bias;   /* Chipmunk space->collision_bias as stored by the live space */
    double bias_coef;        /* 1 - pow(collision_bias, dt), host computed */
    double damping_dt;       /* pow(damping, dt), host computed */
} AI2NConfig;

typedef struct AI2NPlayer {
    AI2MovementState ms;
    int32_t input_state[AI2_INPUT_WIDTH];
    int32_t body;
    int32_t shape;
    int32_t weapon;   /* Player.weapon_mapping[weapon]; always 0 (Punch) in navigation */
    int32_t pad;
    double damage;    /* always 0 in navigation */
} AI2NPlayer;

typedef struct AI2NSpawner {
    double world_pos[2];
    int32_t og_cooldown_frames;
    int32_t cooldown_frames;
    int32_t last_spawn_frame;
    int32_t active;        /* active_weapon is not None */
    int32_t weapon_name;   /* 0 Spear, 1 Hammer */
    int32_t spawn_frame;
    int32_t body;
    int32_t shape;
    int32_t flag;
    int32_t pad;
} AI2NSpawner;

typedef struct AI2NWrapper {
    double target_x;
    double target_y;
    double last_distance;
    int32_t target_valid;
    int32_t last_distance_valid;
    int32_t episode_steps;
    int32_t targets_this_episode;
    int32_t hold_steps;
    int32_t post_goal_steps;
    int32_t goal_reached_step;
    int32_t goal_reached_valid;
    int32_t last_stocks;
    int32_t fell_this_episode;
    int32_t target_touched;
    int32_t targets_reached;
    int32_t falls;
    int32_t pad;
} AI2NWrapper;

typedef struct AI2NStepResult {
    float obs[AI2N_OBS_WIDTH];
    double reward;
    double nav_reward;
    int32_t terminated;
    int32_t truncated;
    int32_t underlying_done;
    int32_t touched;
    int32_t reached;
    int32_t fall;
    int32_t handoff;
    int32_t success;
    int32_t horizon;
    int32_t error;
} AI2NStepResult;

typedef struct AI2NEpisode {
    AI2CWorld world;
    AI2MovementConfig mcfg;
    AI2NConfig cfg;
    AI2NPlayer players[2];
    AI2NSpawner spawners[AI2N_SPAWNER_COUNT];
    AI2NWrapper wrap;
    double waypoint1[2];
    double waypoint2[2];
    int32_t platform_body;
    int32_t platform_shape;
    int32_t moving_to_w2;
    int32_t steps;            /* WarehouseBrawl.steps */
    int32_t raw_terminated;
    int32_t raw_truncated;
    int32_t p1_right;
    int32_t shape_id_counter; /* cpSpace shapeIDCounter */
    int32_t last_error;
    int32_t pad[3];
    float obs[AI2N_OBS_WIDTH];
} AI2NEpisode;

/* The moving platform never depends on the players (kinematic body, velocity
 * set from its own position every frame), so its per-frame velocity sequence
 * from reset is a constant of the scene.  The host tabulates it with libm sin
 * (the same call CPython makes) and both the CPU reference and the device read
 * the table, which keeps the device free of a transcendental whose last bit is
 * not the host's.  Each entry also records the platform position it expects, so
 * a lane whose physics drifted fails closed instead of reading a wrong row. */
typedef struct AI2NPlatformTable {
    int32_t count;
    int32_t pad;
    double px[AI2N_PLATFORM_TABLE_SIZE];
    double py[AI2N_PLATFORM_TABLE_SIZE];
    double vx[AI2N_PLATFORM_TABLE_SIZE];
    double vy[AI2N_PLATFORM_TABLE_SIZE];
    int32_t dir_after[AI2N_PLATFORM_TABLE_SIZE];
} AI2NPlatformTable;

/* ---------------------------------------------------------------- helpers */

/* CPython evaluates x ** 2.0 with the C library's pow.  Apple's libm pow is
 * not correctly rounded for y == 2, so it can differ from x * x in the last
 * bit; clang would otherwise rewrite pow(x, 2.0) as x * x.  Calling through a
 * volatile pointer keeps the libm call, so the CPU reference matches the
 * Python game on the same host.  CUDA has no libm; the device pow is used and
 * its last-bit deviation is measured by the lane parity gate. */
#ifdef __CUDACC__
#define AI2N_POW(x, y) pow((x), (y))
#else
static double (*volatile ai2n_libm_pow)(double, double) = pow;
#define AI2N_POW(x, y) ai2n_libm_pow((x), (y))
#endif
/* x ** 2 for the wrapper distances.  On the device the correctly rounded square
 * is used: it equals the host libm pow(x, 2.0) for 99.9% of inputs (macOS libm
 * and glibc 2.35 both differ from x * x in about 0.1% of inputs, in the last
 * bit), while the device pow differs far more often.  Only rewards and the
 * wrapper's last_distance can be affected; the lane parity gate counts it. */
#if defined(__CUDA_ARCH__)
#define AI2N_SQ(x) ((x) * (x))
#else
#define AI2N_SQ(x) AI2N_POW((x), 2.0)
#endif

/* cpBodyGetPosition returns cpTransformPoint(transform, cpvzero), which adds a
 * signed zero to the stored position; cpBodySetPosition adds the transformed
 * centre of gravity (zero) to the requested value.  Both turn -0.0 into +0.0. */
AI2N_HD static inline AI2CVec ai2n_body_position(const AI2CBody *b) { return ai2c_v(0.0 + b->p.x, 0.0 + b->p.y); }

/* a * b + c evaluated under the tick's contraction rule (must equal the unfused result). */
AI2N_HD static inline double ai2n_fma_probe(double a, double b, double c) {
    volatile double x = a, y = b, z = c;
    return x * y + z;
}

AI2N_HD static inline void ai2n_zero(void *ptr, unsigned long bytes) {
    unsigned char *p = (unsigned char *)ptr;
    unsigned long i;
    for (i = 0; i < bytes; ++i) p[i] = 0;
}

/* Python's min(a, b): b only when b < a. */
AI2N_HD static inline double ai2n_pymin(double a, double b) { return (b < a) ? b : a; }
AI2N_HD static inline int ai2n_imin(int a, int b) { return (b < a) ? b : a; }
AI2N_HD static inline int ai2n_imax(int a, int b) { return (b > a) ? b : a; }

/* Ground(space, x, y, width_ground=10): environment.py Ground.__init__. */
AI2N_HD static inline void ai2n_ground_verts(AI2CVec *v) {
    const double width = 10.0;
    const double height_up = 0.0;
    const double height_down = -4.0;
    v[0] = ai2c_v(-width / 4, height_up - 0.15);
    v[1] = ai2c_v(width / 4, height_up - 0.15);
    v[2] = ai2c_v(width / 4, -height_down);
    v[3] = ai2c_v(-width / 4, -height_down);
}

/* cpBoxShapeNew(body, width, height, 0): (r,b), (r,t), (l,t), (l,b). */
AI2N_HD static inline void ai2n_box_verts(AI2CVec *v, double width, double height) {
    const double hw = width / 2.0;
    const double hh = height / 2.0;
    v[0] = ai2c_v(hw, -hh);
    v[1] = ai2c_v(hw, hh);
    v[2] = ai2c_v(-hw, hh);
    v[3] = ai2c_v(-hw, -hh);
}

/* cpBodyNew / cpBodyNewStatic / cpBodyNewKinematic followed by cpBodySetPosition. */
AI2N_HD static inline void ai2n_body_init(AI2CBody *b, int type, double m, double i, AI2CVec p) {
    b->type = type;
    b->velocity_mode = AI2C_VELOCITY_DEFAULT;
    if (type == AI2C_BODY_DYNAMIC) {
        b->m = m; b->m_inv = 1.0 / m; b->i = i; b->i_inv = 1.0 / i;
    } else {
        b->m = (double)INFINITY; b->m_inv = 0.0; b->i = (double)INFINITY; b->i_inv = 0.0;
    }
    b->cog = ai2c_vzero();
    b->p = p;
    b->v = ai2c_vzero();
    b->f = ai2c_vzero();
    b->a = 0.0;
    b->w = 0.0;
    b->t = 0.0;
    b->v_bias = ai2c_vzero();
    b->w_bias = 0.0;
    ai2c_body_set_transform(b, p, 0.0);
}

AI2N_HD static inline void ai2n_shape_init(AI2CShape *s, int body, const AI2CVec *verts, double e, double u,
                                          unsigned long long type, unsigned int categories, unsigned int mask,
                                          unsigned long long hashid) {
    int k;
    ai2c_shape_set_verts(s, 4, verts);
    s->body = body;
    for (k = 0; k < AI2C_MAX_VERTS; ++k) { s->tv[k] = ai2c_vzero(); s->tn[k] = ai2c_vzero(); }
    s->bb = ai2c_bb(0.0, 0.0, 0.0, 0.0);
    s->r = 0.0;
    s->e = e;
    s->u = u;
    s->surface_v = ai2c_vzero();
    s->hashid = hashid;
    s->collision_type = type;
    s->group = 0ull;
    s->categories = categories;
    s->mask = mask;
    s->sensor = 0;
}

/* cpBodySetPosition keeps the angle; cpBodySetVelocity leaves v_bias alone. */
AI2N_HD static inline void ai2n_body_teleport(AI2CBody *b, double x, double y) {
    b->p = ai2c_v(0.0 + x, 0.0 + y);
    ai2c_body_set_transform(b, b->p, b->a);
    b->v = ai2c_vzero();
}

/* Tabulate Stage.physics_process for every frame after reset: velocity for frame k
 * and the platform position expected at the start of frame k.  The position is
 * integrated with the contact step's own body integrator so the table matches
 * the world bit for bit. */
AI2N_HD static inline void ai2n_platform_table_build(AI2NPlatformTable *table, double dt) {
    AI2CBody body;
    int dir = 1;
    int k;
    ai2n_body_init(&body, AI2C_BODY_KINEMATIC, 0.0, 0.0, ai2c_v(0.0, 1.0));
    for (k = 0; k < AI2N_PLATFORM_TABLE_SIZE; ++k) {
        double vx = 0.0, vy = 0.0;
        uint8_t next = 0;
        const AI2CVec pos = ai2n_body_position(&body);
        ai2_navigation_platform_step_math(pos.x, pos.y, 1.0, 0.0, -1.0, 2.0, (uint8_t)dir, &vx, &vy, &next);
        table->px[k] = body.p.x;
        table->py[k] = body.p.y;
        table->vx[k] = vx;
        table->vy[k] = vy;
        table->dir_after[k] = next ? 1 : 0;
        body.v = ai2c_v(vx, vy);
        ai2c_body_update_position(&body, dt);
        dir = next ? 1 : 0;
    }
    table->count = AI2N_PLATFORM_TABLE_SIZE;
    table->pad = 0;
}

/* Player.is_on_floor(): cache_bb() recomputes each bound from the live transform. */
AI2N_HD static inline int ai2n_is_on_floor(AI2NEpisode *ep, int player) {
    AI2CWorld *w = &ep->world;
    const AI2NPlayer *pl = &ep->players[player];
    AI2CBody *pb = &w->bodies[pl->body];
    AI2CBB pbb = ai2c_shape_cache(&w->shapes[pl->shape], pb->tf);
    AI2CBB other;
    const AI2CBody *ob;
    ob = &w->bodies[w->shapes[0].body];
    other = ai2c_shape_cache(&w->shapes[0], ob->tf);
    if (ai2c_bb_intersects(pbb, other) && pb->p.y <= ob->p.y) return 1;
    pbb = ai2c_shape_cache(&w->shapes[pl->shape], pb->tf);
    ob = &w->bodies[w->shapes[1].body];
    other = ai2c_shape_cache(&w->shapes[1], ob->tf);
    if (ai2c_bb_intersects(pbb, other) && pb->p.y <= ob->p.y) return 1;
    pbb = ai2c_shape_cache(&w->shapes[pl->shape], pb->tf);
    ob = &w->bodies[ep->platform_body];
    other = ai2c_shape_cache(&w->shapes[ep->platform_shape], ob->tf);
    if (ai2c_bb_intersects(pbb, other) && pb->p.y <= ob->p.y - 0.01 && (ob->p.x - 1.25) <= pb->p.x && pb->p.x <= (ob->p.x + 1.25)) return 1;
    return 0;
}

/* shape.cache_bb().intersects(platform1.shape.cache_bb()) in Walking/StandingState. */
AI2N_HD static inline int ai2n_platform_overlap(AI2NEpisode *ep, int player) {
    AI2CWorld *w = &ep->world;
    const AI2NPlayer *pl = &ep->players[player];
    AI2CBB pbb = ai2c_shape_cache(&w->shapes[pl->shape], w->bodies[pl->body].tf);
    AI2CBB other = ai2c_shape_cache(&w->shapes[ep->platform_shape], w->bodies[ep->platform_body].tf);
    return ai2c_bb_intersects(pbb, other);
}

/* Player.__init__ movement constants. */
AI2N_HD static inline void ai2n_movement_config(AI2MovementConfig *c) {
    c->stage_width_tiles = 29.8;
    c->stage_height_tiles = 16.8;
    c->move_speed = 6.75;
    c->jump_speed = 8.9;
    c->max_fall_speed = 12.0;
    c->fast_fall_ease = 14.25 / 30.0;
    c->in_air_ease = 10.75 / 30.0;
    c->run_speed = 8.0;
    c->dash_speed = 10.0;
    c->backdash_speed = 4.0;
    c->turnaround_time = 1;
    c->backdash_time = 7;
    c->dodge_time = 10;
    c->grounded_dodge_cooldown = 30;
    c->air_dodge_cooldown = 82;
    c->invincible_time = 90;
    c->jump_cooldown = 6;
    c->dash_time = 9;
}

/* Player.__init__: every state object carries the three PlayerObjectState timers;
 * the initial InAirState has jumps_left/jump_timer/recoveries_left = 0 and is_base. */
AI2N_HD static inline void ai2n_player_init(AI2NPlayer *pl, int body, int shape, double start_x, double start_y) {
    AI2MovementState *s = &pl->ms;
    int i;
    ai2n_zero(pl, sizeof(*pl));
    s->x = start_x;
    s->y = start_y;
    s->start_x = start_x;
    s->start_y = start_y;
    s->prev_x = start_x;
    s->prev_y = start_y;
    s->facing = start_x < 0.0 ? 1 : -1;
    s->state_id = AI2_STATE_IN_AIR;
    s->stocks = 3;
    s->velocity_func = AI2_VELOCITY_DEFAULT;
    s->body_type = 0;
    for (i = 0; i < AI2_MOVEMENT_STATE_COUNT; ++i) ai2_mov_ensure_slot(s, i);
    ai2_mov_slot_set(s, AI2_STATE_IN_AIR, AI2_MOV_SLOT_JUMP_TIMER, 0);
    ai2_mov_slot_set(s, AI2_STATE_IN_AIR, AI2_MOV_SLOT_JUMPS_LEFT, 0);
    ai2_mov_slot_set(s, AI2_STATE_IN_AIR, AI2_MOV_SLOT_RECOVERIES_LEFT, 0);
    ai2_mov_slot_set(s, AI2_STATE_IN_AIR, AI2_MOV_SLOT_IS_BASE, 1);
    ai2_input_init_core(pl->input_state);
    pl->body = body;
    pl->shape = shape;
    pl->weapon = 0;
    pl->damage = 0.0;
}

/* FrozenOpponentWrapper._get_distance_to_target: np.sqrt((px - tx)**2 + (py - ty)**2). */
AI2N_HD static inline double ai2n_distance(const AI2NWrapper *wr, double px, double py) {
    if (!wr->target_valid) return 0.0;
    return sqrt(AI2N_SQ(px - wr->target_x) + AI2N_SQ(py - wr->target_y));
}

/* FrozenOpponentWrapper._spawn_target with positions.json grounded positions. */
AI2N_HD static inline int ai2n_spawn_target(AI2NEpisode *ep, AI2NRng *rng, double px, double py) {
    const AI2NConfig *c = &ep->cfg;
    AI2NWrapper *wr = &ep->wrap;
    const double min_base = c->completion_mode == 1 ? (c->min_dist > 1.5 ? c->min_dist : 1.5) : c->min_dist;
    const double min_dist_sq = AI2N_SQ(min_base);
    const double max_dist_sq = AI2N_SQ(c->max_dist);
    int count = 0, pick, seen = 0, i;
    for (i = 0; i < AI2N_POSITIONS_COUNT; ++i) {
        const double d_sq = AI2N_SQ(ai2n_position(i, 0) - px) + AI2N_SQ(ai2n_position(i, 1) - py);
        if (min_dist_sq <= d_sq && d_sq <= max_dist_sq) count += 1;
    }
    if (count > 0) {
        pick = ai2n_rng_np_randint(rng, count);
        if (rng->error) return AI2N_ERR_RNG;
        for (i = 0; i < AI2N_POSITIONS_COUNT; ++i) {
            const double d_sq = AI2N_SQ(ai2n_position(i, 0) - px) + AI2N_SQ(ai2n_position(i, 1) - py);
            if (min_dist_sq <= d_sq && d_sq <= max_dist_sq) {
                if (seen == pick) { wr->target_x = ai2n_position(i, 0); wr->target_y = ai2n_position(i, 1); wr->target_valid = 1; return AI2N_ERR_NONE; }
                seen += 1;
            }
        }
        return AI2N_ERR_TARGET;
    }
    if (c->completion_mode == 1) {
        for (i = 0; i < AI2N_POSITIONS_COUNT; ++i) {
            const double d_sq = AI2N_SQ(ai2n_position(i, 0) - px) + AI2N_SQ(ai2n_position(i, 1) - py);
            if (d_sq >= min_dist_sq) count += 1;
        }
        if (count == 0) return AI2N_ERR_TARGET;
        pick = ai2n_rng_np_randint(rng, count);
        if (rng->error) return AI2N_ERR_RNG;
        for (i = 0; i < AI2N_POSITIONS_COUNT; ++i) {
            const double d_sq = AI2N_SQ(ai2n_position(i, 0) - px) + AI2N_SQ(ai2n_position(i, 1) - py);
            if (d_sq >= min_dist_sq) {
                if (seen == pick) { wr->target_x = ai2n_position(i, 0); wr->target_y = ai2n_position(i, 1); wr->target_valid = 1; return AI2N_ERR_NONE; }
                seen += 1;
            }
        }
        return AI2N_ERR_TARGET;
    }
    pick = ai2n_rng_np_randint(rng, AI2N_POSITIONS_COUNT);
    if (rng->error) return AI2N_ERR_RNG;
    wr->target_x = ai2n_position(pick, 0);
    wr->target_y = ai2n_position(pick, 1);
    wr->target_valid = 1;
    return AI2N_ERR_NONE;
}

AI2N_HD static inline void ai2n_freeze_opponent(AI2NEpisode *ep) {
    if (ep->wrap.target_valid) ai2n_body_teleport(&ep->world.bodies[ep->players[1].body], ep->wrap.target_x, ep->wrap.target_y);
}

/* WarehouseBrawl.observe(0): Player.get_obs for the player then the opponent. */
AI2N_HD static inline int ai2n_observe(AI2NEpisode *ep, float *out) {
    double p[32];
    double s[12];
    double plat[4];
    int i;
    for (i = 0; i < 2; ++i) {
        const AI2NPlayer *pl = &ep->players[i];
        const AI2CBody *body = &ep->world.bodies[pl->body];
        const AI2MovementState *ms = &pl->ms;
        const AI2MovementSlot *slot = &ms->bank[ms->state_id];
        double *row = p + 16 * i;
        row[0] = ai2n_body_position(body).x;
        row[1] = ai2n_body_position(body).y;
        row[2] = body->v.x;
        row[3] = body->v.y;
        row[4] = (double)ms->facing;
        row[5] = ai2n_is_on_floor(ep, i) ? 1.0 : 0.0;
        row[6] = 0.0;
        row[7] = slot->present[AI2_MOV_SLOT_JUMPS_LEFT] ? (double)slot->value[AI2_MOV_SLOT_JUMPS_LEFT] : 0.0;
        row[8] = (double)ms->state_id;
        row[9] = slot->present[AI2_MOV_SLOT_RECOVERIES_LEFT] ? (double)slot->value[AI2_MOV_SLOT_RECOVERIES_LEFT] : 0.0;
        row[10] = slot->present[AI2_MOV_SLOT_DODGE_TIMER] ? (double)slot->value[AI2_MOV_SLOT_DODGE_TIMER] : 0.0;
        row[11] = 0.0;
        row[12] = pl->damage;
        row[13] = (double)ms->stocks;
        row[14] = 0.0;
        row[15] = (double)pl->weapon;
    }
    for (i = 0; i < 4; ++i) {
        if (i < AI2N_SPAWNER_COUNT && ep->spawners[i].active) {
            s[3 * i] = ep->spawners[i].world_pos[0];
            s[3 * i + 1] = ep->spawners[i].world_pos[1];
            s[3 * i + 2] = 1.0;
        } else {
            s[3 * i] = 0.0;
            s[3 * i + 1] = 0.0;
            s[3 * i + 2] = 0.0;
        }
    }
    plat[0] = ai2n_body_position(&ep->world.bodies[ep->platform_body]).x;
    plat[1] = ai2n_body_position(&ep->world.bodies[ep->platform_body]).y;
    plat[2] = ep->world.bodies[ep->platform_body].v.x;
    plat[3] = ep->world.bodies[ep->platform_body].v.y;
    return ai2_obs_pack(p, s, plat, out) ? AI2N_ERR_OBSERVATION : AI2N_ERR_NONE;
}

AI2N_HD static inline int ai2n_config_valid(const AI2NConfig *c) {
    if (!(c->completion_mode == 0 || c->completion_mode == 1)) return 0;
    if (!(c->control_mode == 0 || c->control_mode == 1)) return 0;
    if (c->max_targets < 1 || c->time_limit_steps <= 0 || c->settle_steps <= 0 || c->post_goal_required_steps <= 0) return 0;
    if (c->completion_mode == 1 && c->max_targets < 2) return 0;
    if (!(c->settle_speed > 0.0) || !isfinite(c->settle_speed) || !isfinite(c->fall_penalty)) return 0;
    if (!isfinite(c->min_dist) || !isfinite(c->max_dist) || !(c->dt > 0.0)) return 0;
    if (!isfinite(c->collision_bias) || !isfinite(c->bias_coef) || !isfinite(c->damping_dt)) return 0;
    return 1;
}

/* ------------------------------------------------------------------ reset */

/* Raw scene reset is shared with full combat. It deliberately consumes no
 * curriculum target draw and never relocates or freezes either player. */
AI2N_HD static inline int ai2n_reset_scene(AI2NEpisode *ep, const AI2NConfig *cfg, AI2NRng *rng) {
    AI2CWorld *w;
    AI2CVec verts[4];
    AI2CBB add_bb[AI2N_INITIAL_SHAPES];
    AI2CVec add_v[AI2N_INITIAL_SHAPES];
    int add_order[AI2N_INITIAL_SHAPES];
    double p1_x, p2_x;
    int i, rc;
    if (ep == 0 || cfg == 0 || rng == 0) return AI2N_ERR_NULL;
    if (!ai2n_config_valid(cfg)) return AI2N_ERR_CONFIG;
    ai2n_zero(ep, sizeof(*ep));
    ep->cfg = *cfg;
    ai2n_movement_config(&ep->mcfg);
    w = &ep->world;
    /* pymunk.Space() defaults plus WarehouseBrawl.reset gravity. */
    w->stamp = 0;
    w->curr_dt = 0.0;
    w->iterations = 10;
    w->gravity = ai2c_v(0.0, 17.808);
    w->damping = 1.0;
    w->slop = (double)0.1f;
    w->bias = cfg->collision_bias;
    w->persistence = 3u;
    /* _setup: collision handlers. */
    w->handler_count = 3;
    w->handler_type_a[0] = AI2N_TYPE_PLAYER; w->handler_type_b[0] = AI2N_TYPE_PLAYER + 1; w->handler_begin_false[0] = 1;
    w->handler_type_a[1] = AI2N_TYPE_PLAYER; w->handler_type_b[1] = AI2N_TYPE_PLATFORM; w->handler_begin_false[1] = 0;
    w->handler_type_a[2] = AI2N_TYPE_PLAYER + 1; w->handler_type_b[2] = AI2N_TYPE_PLATFORM; w->handler_begin_false[2] = 0;
    /* Players spawn on random sides. */
    ep->p1_right = ai2n_rng_getrandbits1(rng);
    if (rng->error) return AI2N_ERR_RNG;
    p1_x = ep->p1_right ? 5.0 : -5.0;
    p2_x = ep->p1_right ? -5.0 : 5.0;
    /* Bodies in creation order: ground1, ground2, platform1, player, opponent. */
    ai2n_body_init(&w->bodies[0], AI2C_BODY_STATIC, 0.0, 0.0, ai2c_v(4.5, 1.0));
    ai2n_body_init(&w->bodies[1], AI2C_BODY_STATIC, 0.0, 0.0, ai2c_v(-4.5, 3.0));
    ai2n_body_init(&w->bodies[2], AI2C_BODY_KINEMATIC, 0.0, 0.0, ai2c_v(0.0, 1.0));
    ai2n_body_init(&w->bodies[3], AI2C_BODY_DYNAMIC, 1.0, 1e9, ai2c_v(p1_x, 0.0));
    ai2n_body_init(&w->bodies[4], AI2C_BODY_DYNAMIC, 1.0, 1e9, ai2c_v(p2_x, 0.0));
    w->body_count = 5;
    ai2n_ground_verts(verts);
    ai2n_shape_init(&w->shapes[0], 0, verts, 0.0, 0.7, AI2N_TYPE_GROUND, AI2N_CAT_GROUND, AI2N_CAT_ALL, 0ull);
    ai2n_shape_init(&w->shapes[1], 1, verts, 0.0, 0.7, AI2N_TYPE_GROUND, AI2N_CAT_GROUND, AI2N_CAT_ALL, 1ull);
    ai2n_box_verts(verts, 2.0, 1 * 0.1 + 0.1);
    ai2n_shape_init(&w->shapes[2], 2, verts, 0.0, 0.9, AI2N_TYPE_PLATFORM, AI2N_CAT_ALL, AI2N_CAT_ALL, 2ull);
    ai2n_box_verts(verts, 0.87, 1.0);
    ai2n_shape_init(&w->shapes[3], 3, verts, 0.0, 0.7, AI2N_TYPE_PLAYER, AI2N_CAT_PLAYER, AI2N_CAT_ALL & ~AI2N_CAT_WEAPON, 3ull);
    ai2n_shape_init(&w->shapes[4], 4, verts, 0.0, 0.7, AI2N_TYPE_PLAYER + 1, AI2N_CAT_PLAYER, AI2N_CAT_ALL & ~AI2N_CAT_WEAPON, 4ull);
    w->shape_count = AI2N_INITIAL_SHAPES;
    ep->shape_id_counter = AI2N_INITIAL_SHAPES;
    for (i = 0; i < AI2N_INITIAL_SHAPES; ++i) {
        add_order[i] = i;
        add_bb[i] = ai2c_shape_cache(&w->shapes[i], w->bodies[w->shapes[i].body].tf);
        add_v[i] = ai2c_vzero();
    }
    rc = ai2c_world_build_trees(w, add_order, add_bb, add_v, AI2N_INITIAL_SHAPES);
    if (rc != AI2C_OK) return AI2N_ERR_CONTACT;
    ep->platform_body = 2;
    ep->platform_shape = 2;
    ep->waypoint1[0] = 1.0; ep->waypoint1[1] = 0.0;
    ep->waypoint2[0] = -1.0; ep->waypoint2[1] = 2.0;
    ep->moving_to_w2 = 1;
    ai2n_player_init(&ep->players[0], 3, 3, p1_x, 0.0);
    ai2n_player_init(&ep->players[1], 4, 4, p2_x, 0.0);
    /* Weapon spawners: constructor arguments draw left to right, then __init__ draws the phase. */
    for (i = 0; i < AI2N_SPAWNER_COUNT; ++i) {
        AI2NSpawner *sp = &ep->spawners[i];
        const double u = ai2n_rng_uniform(rng, 2.6, 6.5);
        const int cooldown = ai2n_rng_randint(rng, 500, 700);
        const int phase = ai2n_rng_randint(rng, 0, 350);
        if (rng->error) return AI2N_ERR_RNG;
        sp->world_pos[0] = i == 0 ? u : -u;
        sp->world_pos[1] = i == 0 ? 0 + 0.7 : 2 + 0.7;
        sp->og_cooldown_frames = cooldown;
        sp->cooldown_frames = cooldown;
        sp->last_spawn_frame = -(cooldown - phase);
        sp->active = 0;
        sp->weapon_name = 0;
        sp->spawn_frame = 0;
        sp->body = -1;
        sp->shape = -1;
        sp->flag = 0;
    }
    ep->steps = 0;
    ep->raw_terminated = 0;
    ep->raw_truncated = 0;
    return rng->error ? AI2N_ERR_RNG : AI2N_ERR_NONE;
}

AI2N_HD static inline int ai2n_reset(AI2NEpisode *ep, const AI2NConfig *cfg, AI2NRng *rng) {
    int rc = ai2n_reset_scene(ep, cfg, rng);
    AI2CWorld *w;
    if (rc) return rc;
    w = &ep->world;
    /* FrozenOpponentWrapper.reset */
    ai2n_zero(&ep->wrap, sizeof(ep->wrap));
    rc = ai2n_spawn_target(ep, rng, ai2n_body_position(&w->bodies[3]).x, ai2n_body_position(&w->bodies[3]).y);
    if (rc) return rc;
    ai2n_freeze_opponent(ep);
    ep->wrap.last_distance = ai2n_distance(&ep->wrap, ai2n_body_position(&w->bodies[3]).x, ai2n_body_position(&w->bodies[3]).y);
    ep->wrap.last_distance_valid = 1;
    ep->wrap.last_stocks = ep->players[0].ms.stocks;
    rc = ai2n_observe(ep, ep->obs);
    if (rc) return rc;
    return rng->error ? AI2N_ERR_RNG : AI2N_ERR_NONE;
}

/* ------------------------------------------------------------------- step */

AI2N_HD static inline int ai2n_player_physics(AI2NEpisode *ep, int i) {
    AI2NPlayer *pl = &ep->players[i];
    AI2CBody *body = &ep->world.bodies[pl->body];
    const AI2CBody *platform = &ep->world.bodies[ep->platform_body];
    AI2MovementInput in;
    AI2MovementResult res;
    int k, rc;
    for (k = 0; k < AI2_INPUT_WIDTH; ++k) in.decoded[k] = pl->input_state[k];
    in.grounded = (uint8_t)ai2n_is_on_floor(ep, i);
    in.platform_overlap = (uint8_t)ai2n_platform_overlap(ep, i);
    in.platform_vx = platform->v.x;
    in.platform_vy = platform->v.y;
    pl->ms.x = ai2n_body_position(body).x;
    pl->ms.y = ai2n_body_position(body).y;
    pl->ms.vx = body->v.x;
    pl->ms.vy = body->v.y;
    pl->ms.velocity_func = body->velocity_mode;
    rc = ai2_movement_step_core(&ep->mcfg, &pl->ms, &in, &res);
    if (rc != 0) return AI2N_ERR_MOVEMENT;
    pl->ms = res.state;
    if (res.event_mask & AI2_MOV_EVENT_RESPAWN) {
        body->p = ai2c_v(0.0 + pl->ms.x, 0.0 + pl->ms.y);
        ai2c_body_set_transform(body, body->p, body->a);
    }
    body->v = ai2c_v(pl->ms.vx, pl->ms.vy);
    body->velocity_mode = pl->ms.velocity_func;
    return AI2N_ERR_NONE;
}

AI2N_HD static inline int ai2n_free_slot(const int *active, int count, int max) {
    int k;
    for (k = AI2N_INITIAL_SHAPES; k < max; ++k) {
        if (k >= count || !active[k]) return k;
    }
    return -1;
}

/* WeaponSpawner.spawn_weapon + WeaponGO.activate/_ensure_body. */
AI2N_HD static inline int ai2n_spawn_weapon(AI2NEpisode *ep, AI2NRng *rng, int index) {
    AI2NSpawner *sp = &ep->spawners[index];
    AI2CWorld *w = &ep->world;
    AI2CVec verts[4];
    /* Camera.scale_gtp() at LOW resolution: (720 // 29.8) * zoom(2.0). */
    const double px_per_world = floor(720.0 / 29.8) * 2.0;
    const double w_world = 40 / px_per_world;
    const double h_world = 16 / px_per_world;
    const double y_pos = sp->world_pos[1];
    const double u = ai2n_rng_uniform(rng, 2.6, 6.5);
    double x_pos;
    int name, body, shape, rc;
    if (rng->error) return AI2N_ERR_RNG;
    x_pos = (y_pos == 0 + 0.7) ? u : -u;
    sp->world_pos[0] = x_pos;
    sp->world_pos[1] = y_pos;
    name = ai2n_rng_randint(rng, 0, 1) == 0 ? 0 : 1;
    if (rng->error) return AI2N_ERR_RNG;
    body = ai2n_free_slot(w->body_active, w->body_count, AI2C_MAX_BODIES);
    shape = ai2n_free_slot(w->shape_active, w->shape_count, AI2C_MAX_SHAPES);
    if (body < 0 || shape < 0) return AI2N_ERR_CAPACITY;
    ai2n_body_init(&w->bodies[body], AI2C_BODY_DYNAMIC, 100.0, 1e9, ai2c_v(x_pos, y_pos));
    ai2n_box_verts(verts, w_world, h_world + 0.5);
    ai2n_shape_init(&w->shapes[shape], body, verts, 1.0, 100000000000.0, AI2N_TYPE_WEAPON, AI2N_CAT_ALL, AI2N_CAT_ALL,
                    (unsigned long long)ep->shape_id_counter);
    ep->shape_id_counter += 1;
    if (body >= w->body_count) w->body_count = body + 1;
    if (shape >= w->shape_count) w->shape_count = shape + 1;
    rc = ai2c_world_add_shape(w, shape);
    if (rc != AI2C_OK) return AI2N_ERR_CONTACT;
    /* shape.filter is assigned after space.add. */
    w->shapes[shape].categories = AI2N_CAT_WEAPON;
    w->shapes[shape].mask = AI2N_CAT_GROUND;
    sp->last_spawn_frame = ep->steps;
    sp->spawn_frame = ep->steps;
    sp->active = 1;
    sp->weapon_name = name;
    sp->body = body;
    sp->shape = shape;
    return AI2N_ERR_NONE;
}

/* WeaponSpawnController.update after the space step. */
AI2N_HD static inline int ai2n_weapons_update(AI2NEpisode *ep, AI2NRng *rng) {
    int i, rc;
    for (i = 0; i < AI2N_SPAWNER_COUNT; ++i) {
        AI2NSpawner *sp = &ep->spawners[i];
        if (sp->active) {
            /* The VFX pickup flag only gates try_pick_up, which is unreachable here. */
            if (ep->steps - sp->spawn_frame > AI2N_DESPAWN_FRAMES) {
                const int lo = (int)(sp->og_cooldown_frames * 0.5);
                const int hi = (int)(sp->og_cooldown_frames * 1.5);
                sp->cooldown_frames = ai2n_rng_randint(rng, lo, hi);
                if (rng->error) return AI2N_ERR_RNG;
                /* space.remove(body, shape): Pymunk removes in argument order. */
                ai2c_world_remove_body(&ep->world, sp->body);
                rc = ai2c_world_remove_shape(&ep->world, sp->shape);
                if (rc != AI2C_OK) return AI2N_ERR_CONTACT;
                sp->active = 0;
                sp->body = -1;
                sp->shape = -1;
            }
            continue;
        }
        if (ep->steps - sp->last_spawn_frame >= sp->cooldown_frames) {
            rc = ai2n_spawn_weapon(ep, rng, i);
            if (rc) return rc;
            sp->flag = 0;
        }
    }
    return AI2N_ERR_NONE;
}

/* FrozenOpponentWrapper._is_settled_at_target; on_platform is never set by the frozen game. */
AI2N_HD static inline int ai2n_is_settled(AI2NEpisode *ep, double distance) {
    const AI2CBody *body = &ep->world.bodies[ep->players[0].body];
    double relative_speed;
    if (distance >= 1.0 || !ai2n_is_on_floor(ep, 0)) return 0;
    relative_speed = hypot(body->v.x - 0.0, body->v.y - 0.0);
    return distance < 1.0 && relative_speed <= ep->cfg.settle_speed;
}

AI2N_HD static inline int ai2n_step(AI2NEpisode *ep, const double *action, AI2NRng *rng, const AI2NPlatformTable *table, AI2NStepResult *out) {
    double masked[AI2N_ACTION_WIDTH];
    double zeros[AI2N_ACTION_WIDTH];
    const AI2NConfig *c;
    AI2NWrapper *wr;
    AI2CBody *platform;
    double vx = 0.0, vy = 0.0, px, py, current_dist, nav_reward = 0.0, x_pos;
    uint8_t next_dir = 0;
    int i, rc, underlying_done, terminated, truncated, active, current_stocks;
    int touched = 0, reached = 0, fall = 0, handoff = 0, horizon, completed, success;
    if (ep == 0 || action == 0 || rng == 0 || out == 0) return AI2N_ERR_NULL;
    c = &ep->cfg;
    wr = &ep->wrap;
    ai2n_zero(out, sizeof(*out));
    for (i = 0; i < AI2N_ACTION_WIDTH; ++i) { masked[i] = action[i]; zeros[i] = 0.0; }
    if (c->control_mode == 1) { masked[5] = 0.0; masked[7] = 0.0; masked[8] = 0.0; masked[9] = 0.0; }
    /* --- WarehouseBrawl.step --- */
    ep->raw_terminated = 0;
    if (ai2_input_step_core(masked, ep->players[0].input_state, ep->players[0].input_state) != 0) return AI2N_ERR_INPUT;
    if (ai2_input_step_core(zeros, ep->players[1].input_state, ep->players[1].input_state) != 0) return AI2N_ERR_INPUT;
    for (i = 0; i < 2; ++i) {
        if (ep->players[i].ms.stocks <= 0) ep->raw_terminated = 1;
        /* on_platform is never assigned by the frozen handlers, so no velocity override. */
    }
    /* physics_process in objects order: grounds (no-op), platform1, player, opponent, VFX and weapons (no-op). */
    platform = &ep->world.bodies[ep->platform_body];
    if (table != 0 && table->count > 0) {
        const int k = ep->steps;
        if (k < 0 || k >= table->count) { ep->last_error = AI2N_ERR_PLATFORM; out->error = AI2N_ERR_PLATFORM; return AI2N_ERR_PLATFORM; }
        if (platform->p.x != table->px[k] || platform->p.y != table->py[k]) { ep->last_error = AI2N_ERR_PLATFORM; out->error = AI2N_ERR_PLATFORM; return AI2N_ERR_PLATFORM; }
        vx = table->vx[k];
        vy = table->vy[k];
        next_dir = (uint8_t)table->dir_after[k];
    } else {
        ai2_navigation_platform_step_math(ai2n_body_position(platform).x, ai2n_body_position(platform).y, ep->waypoint1[0], ep->waypoint1[1], ep->waypoint2[0], ep->waypoint2[1],
                                          (uint8_t)ep->moving_to_w2, &vx, &vy, &next_dir);
    }
    platform->v = ai2c_v(vx, vy);
    ep->moving_to_w2 = next_dir ? 1 : 0;
    for (i = 0; i < 2; ++i) {
        rc = ai2n_player_physics(ep, i);
        if (rc) { ep->last_error = rc; out->error = rc; return rc; }
    }
    rc = ai2c_world_step(&ep->world, c->dt, c->bias_coef, c->damping_dt);
    if (rc != AI2C_OK) { ep->last_error = AI2N_ERR_CONTACT; out->error = AI2N_ERR_CONTACT; return AI2N_ERR_CONTACT; }
    ep->steps += 1;
    /* DroppedWeaponSpawner.try_drop and try_pick_up_all cannot act: pickup is masked and the opponent is idle. */
    rc = ai2n_weapons_update(ep, rng);
    if (rc) { ep->last_error = rc; out->error = rc; return rc; }
    ep->raw_truncated = ep->steps >= AI2N_RAW_MAX_TIMESTEPS ? 1 : 0;
    /* --- FrozenOpponentWrapper.step --- */
    underlying_done = (ep->raw_terminated || ep->raw_truncated) ? 1 : 0;
    terminated = ep->raw_terminated;
    truncated = ep->raw_truncated;
    wr->episode_steps += 1;
    px = ai2n_body_position(&ep->world.bodies[ep->players[0].body]).x;
    py = ai2n_body_position(&ep->world.bodies[ep->players[0].body]).y;
    ai2n_freeze_opponent(ep);
    current_dist = ai2n_distance(wr, px, py);
    if (wr->last_distance_valid) nav_reward += (wr->last_distance - current_dist) * 2.0;
    current_stocks = ep->players[0].ms.stocks;
    if (current_stocks < wr->last_stocks) {
        nav_reward += c->fall_penalty;
        wr->falls += 1;
        wr->fell_this_episode = 1;
        terminated = 1;
        fall = 1;
    }
    wr->last_stocks = current_stocks;
    active = !(underlying_done || wr->fell_this_episode);
    if (active && current_dist < 1.0 && !wr->target_touched) {
        wr->target_touched = 1;
        touched = 1;
    }
    if (active && ai2n_is_settled(ep, current_dist)) {
        int hold_limit = c->settle_steps;
        if (c->completion_mode == 1 && wr->targets_this_episode >= c->max_targets) hold_limit += c->post_goal_required_steps;
        wr->hold_steps = ai2n_imin(wr->hold_steps + 1, hold_limit);
    } else {
        wr->hold_steps = 0;
    }
    if (active && wr->hold_steps >= c->settle_steps && wr->targets_this_episode < c->max_targets) {
        nav_reward += 5.0;
        wr->targets_reached += 1;
        reached = 1;
        wr->targets_this_episode += 1;
        if (wr->targets_this_episode < c->max_targets) {
            rc = ai2n_spawn_target(ep, rng, px, py);
            if (rc) { ep->last_error = rc; out->error = rc; return rc; }
            ai2n_freeze_opponent(ep);
            wr->hold_steps = 0;
            wr->target_touched = 0;
            handoff = 1;
        } else {
            wr->goal_reached_step = wr->episode_steps;
            wr->goal_reached_valid = 1;
        }
    }
    if (wr->goal_reached_valid) {
        if (c->completion_mode == 1) wr->post_goal_steps = ai2n_imax(0, wr->hold_steps - c->settle_steps);
        else wr->post_goal_steps = wr->episode_steps - wr->goal_reached_step;
    }
    wr->last_distance = ai2n_distance(wr, px, py);
    wr->last_distance_valid = 1;
    x_pos = fabs(px);
    if (x_pos > 8.0) {
        const double edge_factor = (x_pos - 8.0) / 4.0;
        nav_reward -= 0.1 * ai2n_pymin(edge_factor, 1.0);
    }
    if (py > 3.0) {
        const double y_edge_factor = (py - 3.0) / 3.0;
        nav_reward -= 0.1 * ai2n_pymin(y_edge_factor, 1.0);
    }
    horizon = (wr->episode_steps >= c->time_limit_steps && !underlying_done) ? 1 : 0;
    completed = (active && wr->targets_this_episode >= c->max_targets && wr->hold_steps >= c->settle_steps &&
                 wr->post_goal_steps >= c->post_goal_required_steps) ? 1 : 0;
    success = (completed && (c->completion_mode == 1 ? !horizon : horizon)) ? 1 : 0;
    if (success && c->completion_mode == 1) terminated = 1;
    if (horizon) truncated = 1;
    rc = ai2n_observe(ep, ep->obs);
    if (rc) { ep->last_error = rc; out->error = rc; return rc; }
    for (i = 0; i < AI2N_OBS_WIDTH; ++i) out->obs[i] = ep->obs[i];
    out->reward = 0.0 + nav_reward;
    out->nav_reward = nav_reward;
    out->terminated = terminated;
    out->truncated = truncated;
    out->underlying_done = underlying_done;
    out->touched = touched;
    out->reached = reached;
    out->fall = fall;
    out->handoff = handoff;
    out->success = success;
    out->horizon = horizon;
    if (rng->error) { ep->last_error = AI2N_ERR_RNG; out->error = AI2N_ERR_RNG; return AI2N_ERR_RNG; }
    out->error = 0;
    return AI2N_ERR_NONE;
}

#endif /* AI2_NAVIGATION_EPISODE_H */
