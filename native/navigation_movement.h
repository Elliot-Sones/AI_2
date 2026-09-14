#ifndef AI2_NAVIGATION_MOVEMENT_H
#define AI2_NAVIGATION_MOVEMENT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __CUDACC__
#include <math.h>
#define AI2_MOV_HD __host__ __device__
#else
#include <math.h>
#include <string.h>
#define AI2_MOV_HD
#endif

#define AI2_INPUT_WIDTH 35
#define AI2_MOVEMENT_STATE_COUNT 13
#define AI2_MOVEMENT_SLOT_FIELD_COUNT 14

#define AI2_STATE_WALKING 0
#define AI2_STATE_STANDING 1
#define AI2_STATE_TURNAROUND 2
#define AI2_STATE_AIR_TURNAROUND 3
#define AI2_STATE_SPRINTING 4
#define AI2_STATE_IN_AIR 6
#define AI2_STATE_DODGE 7
#define AI2_STATE_DASH 9
#define AI2_STATE_BACKDASH 10
#define AI2_STATE_KO 11

#define AI2_VELOCITY_DEFAULT 0
#define AI2_VELOCITY_NO_GRAVITY 1

#define AI2_MOVEMENT_ERR_INVALID 1
#define AI2_MOVEMENT_ERR_UNSUPPORTED_STATE 2
#define AI2_MOVEMENT_ERR_COMBAT_INPUT 3
#define AI2_MOVEMENT_ERR_NONFINITE 4

typedef struct {
    double stage_width_tiles;
    double stage_height_tiles;
    double move_speed;
    double jump_speed;
    double max_fall_speed;
    double fast_fall_ease;
    double in_air_ease;
    double run_speed;
    double dash_speed;
    double backdash_speed;
    int32_t turnaround_time;
    int32_t backdash_time;
    int32_t dodge_time;
    int32_t grounded_dodge_cooldown;
    int32_t air_dodge_cooldown;
    int32_t invincible_time;
    int32_t jump_cooldown;
    int32_t dash_time;
} AI2MovementConfig;

typedef struct {
    int32_t value[AI2_MOVEMENT_SLOT_FIELD_COUNT];
    uint8_t present[AI2_MOVEMENT_SLOT_FIELD_COUNT];
} AI2MovementSlot;

typedef struct {
    double x;
    double y;
    double vx;
    double vy;
    double start_x;
    double start_y;
    double damage_vx;
    double damage_vy;
    double target_vx;
    double target_vy;
    double prev_x;
    double prev_y;
    int32_t facing;
    int32_t state_id;
    int32_t stocks;
    int32_t velocity_func;
    int32_t body_type;
    AI2MovementSlot bank[AI2_MOVEMENT_STATE_COUNT];
} AI2MovementState;

typedef struct {
    int32_t decoded[AI2_INPUT_WIDTH];
    uint8_t grounded;
    uint8_t platform_overlap;
    double platform_vx;
    double platform_vy;
} AI2MovementInput;

typedef struct {
    AI2MovementState state;
    uint8_t transitioned;
    int32_t from_state;
    int32_t to_state;
    uint32_t event_mask;
} AI2MovementResult;

int ai2_movement_default_config(AI2MovementConfig *config);
int ai2_movement_default_state(AI2MovementState *state, int32_t state_id);
int ai2_movement_step(const AI2MovementConfig *config,
                      const AI2MovementState *before,
                      const AI2MovementInput *input,
                      AI2MovementResult *result);

enum {
    AI2_MOV_SLOT_VALID = 0,
    AI2_MOV_SLOT_INVINCIBLE_TIMER = 1,
    AI2_MOV_SLOT_DODGE_COOLDOWN = 2,
    AI2_MOV_SLOT_STUN_TIME_STORED = 3,
    AI2_MOV_SLOT_JUMP_TIMER = 4,
    AI2_MOV_SLOT_JUMPS_LEFT = 5,
    AI2_MOV_SLOT_RECOVERIES_LEFT = 6,
    AI2_MOV_SLOT_IS_BASE = 7,
    AI2_MOV_SLOT_TURNAROUND_TIMER = 8,
    AI2_MOV_SLOT_DASH_TIMER = 9,
    AI2_MOV_SLOT_BACKDASH_TIMER = 10,
    AI2_MOV_SLOT_DODGE_TIMER = 11,
    AI2_MOV_SLOT_DODGE_IS_GROUNDED = 12,
    AI2_MOV_SLOT_KO_TIMER = 13,
};

enum {
    AI2_MOV_KEY_W = 0,
    AI2_MOV_KEY_A = 1,
    AI2_MOV_KEY_S = 2,
    AI2_MOV_KEY_D = 3,
    AI2_MOV_KEY_SPACE = 4,
    AI2_MOV_KEY_H = 5,
    AI2_MOV_KEY_L = 6,
    AI2_MOV_KEY_J = 7,
    AI2_MOV_KEY_K = 8,
    AI2_MOV_KEY_G = 9,
    AI2_MOV_OFF_HELD = 0,
    AI2_MOV_OFF_PRESSED = 10,
    AI2_MOV_OFF_RAW_VERTICAL = 30,
    AI2_MOV_OFF_RAW_HORIZONTAL = 31,
    AI2_MOV_OFF_NO_HORIZONTAL = 32,
    AI2_MOV_OFF_HORIZONTAL_STATE = 33,
};

enum {
    AI2_MOV_EVENT_TRANSITION = 1u << 0,
    AI2_MOV_EVENT_ENTER_TURNAROUND = 1u << 1,
    AI2_MOV_EVENT_ENTER_AIR_TURNAROUND = 1u << 2,
    AI2_MOV_EVENT_ENTER_DASH = 1u << 3,
    AI2_MOV_EVENT_ENTER_BACKDASH = 1u << 4,
    AI2_MOV_EVENT_ENTER_DODGE = 1u << 5,
    AI2_MOV_EVENT_EXIT_DODGE = 1u << 6,
    AI2_MOV_EVENT_ENTER_KO = 1u << 7,
    AI2_MOV_EVENT_EXIT_KO = 1u << 8,
    AI2_MOV_EVENT_RESPAWN = 1u << 9,
};

AI2_MOV_HD static inline int ai2_mov_supported_state(int32_t s) {
    return s == AI2_STATE_WALKING || s == AI2_STATE_STANDING ||
           s == AI2_STATE_TURNAROUND || s == AI2_STATE_AIR_TURNAROUND ||
           s == AI2_STATE_SPRINTING || s == AI2_STATE_IN_AIR ||
           s == AI2_STATE_DODGE || s == AI2_STATE_DASH ||
           s == AI2_STATE_BACKDASH || s == AI2_STATE_KO;
}

AI2_MOV_HD static inline int ai2_mov_abs_int(int v) { return v < 0 ? -v : v; }

AI2_MOV_HD static inline int ai2_mov_slot_get(const AI2MovementState *s, int state, int field, int fallback) {
    return s->bank[state].present[field] ? s->bank[state].value[field] : fallback;
}

AI2_MOV_HD static inline void ai2_mov_slot_set(AI2MovementState *s, int state, int field, int value) {
    s->bank[state].present[field] = 1;
    s->bank[state].value[field] = value;
}

AI2_MOV_HD static inline void ai2_mov_ensure_slot(AI2MovementState *s, int state) {
    ai2_mov_slot_set(s, state, AI2_MOV_SLOT_VALID, 1);
    if (!s->bank[state].present[AI2_MOV_SLOT_INVINCIBLE_TIMER]) ai2_mov_slot_set(s, state, AI2_MOV_SLOT_INVINCIBLE_TIMER, 0);
    if (!s->bank[state].present[AI2_MOV_SLOT_DODGE_COOLDOWN]) ai2_mov_slot_set(s, state, AI2_MOV_SLOT_DODGE_COOLDOWN, 0);
    if (!s->bank[state].present[AI2_MOV_SLOT_STUN_TIME_STORED]) ai2_mov_slot_set(s, state, AI2_MOV_SLOT_STUN_TIME_STORED, 0);
}

AI2_MOV_HD static inline int ai2_mov_sign_from_direction(int d) { return d > 0 ? 1 : -1; }
AI2_MOV_HD static inline int ai2_mov_turn_check(int facing, int direction) { return (facing == 1 && direction < 0) || (facing == -1 && direction > 0); }
AI2_MOV_HD static inline int ai2_mov_hstate_last_int(int state) {
    if (state == 1 || state == 4) return -1;
    if (state == 2 || state == 3) return 1;
    return 0;
}

AI2_MOV_HD static inline double ai2_mov_move_toward(double current, double target, double delta) {
    if (current == target) return target;
    const double diff = target - current;
    const double direction = diff / fabs(diff);
    if (delta >= 0.0) {
        const double candidate = current + delta * direction;
        return direction > 0.0 ? fmin(candidate, target) : fmax(candidate, target);
    }
    return current - fabs(delta) * direction;
}

AI2_MOV_HD static inline int ai2_mov_ground_state_for_input(const AI2MovementInput *in) {
    return ai2_mov_abs_int(in->decoded[AI2_MOV_OFF_RAW_HORIZONTAL]) > 0 ? AI2_STATE_WALKING : AI2_STATE_STANDING;
}

AI2_MOV_HD static inline void ai2_mov_refresh_in_air(AI2MovementState *s) {
    ai2_mov_ensure_slot(s, AI2_STATE_IN_AIR);
    ai2_mov_slot_set(s, AI2_STATE_IN_AIR, AI2_MOV_SLOT_JUMP_TIMER, 0);
    ai2_mov_slot_set(s, AI2_STATE_IN_AIR, AI2_MOV_SLOT_JUMPS_LEFT, 2);
    ai2_mov_slot_set(s, AI2_STATE_IN_AIR, AI2_MOV_SLOT_RECOVERIES_LEFT, 1);
}

AI2_MOV_HD static inline int ai2_mov_finite_state(const AI2MovementConfig *c, const AI2MovementState *s, const AI2MovementInput *in) {
    return isfinite(c->stage_width_tiles) && isfinite(c->stage_height_tiles) &&
           isfinite(c->move_speed) && isfinite(c->jump_speed) &&
           isfinite(c->max_fall_speed) && isfinite(c->fast_fall_ease) &&
           isfinite(c->in_air_ease) && isfinite(c->run_speed) &&
           isfinite(c->dash_speed) && isfinite(c->backdash_speed) &&
           isfinite(s->x) && isfinite(s->y) && isfinite(s->vx) && isfinite(s->vy) &&
           isfinite(s->start_x) && isfinite(s->start_y) &&
           isfinite(s->damage_vx) && isfinite(s->damage_vy) &&
           isfinite(s->target_vx) && isfinite(s->target_vy) &&
           isfinite(s->prev_x) && isfinite(s->prev_y) &&
           isfinite(in->platform_vx) && isfinite(in->platform_vy);
}

AI2_MOV_HD static inline void ai2_mov_common_reset(AI2MovementState *s, int target, int old) {
    ai2_mov_ensure_slot(s, target);
    ai2_mov_slot_set(s, target, AI2_MOV_SLOT_STUN_TIME_STORED, 0);
    ai2_mov_slot_set(s, target, AI2_MOV_SLOT_INVINCIBLE_TIMER, ai2_mov_slot_get(s, old, AI2_MOV_SLOT_INVINCIBLE_TIMER, 0));
    ai2_mov_slot_set(s, target, AI2_MOV_SLOT_DODGE_COOLDOWN, ai2_mov_slot_get(s, old, AI2_MOV_SLOT_DODGE_COOLDOWN, 0));
    if (target == AI2_STATE_WALKING || target == AI2_STATE_STANDING ||
        target == AI2_STATE_SPRINTING || target == AI2_STATE_TURNAROUND ||
        target == AI2_STATE_DASH || target == AI2_STATE_BACKDASH ||
        target == AI2_STATE_KO) {
        ai2_mov_slot_set(s, target, AI2_MOV_SLOT_DASH_TIMER, ai2_mov_slot_get(s, old, AI2_MOV_SLOT_DASH_TIMER, 0));
    }
}

AI2_MOV_HD static inline void ai2_mov_exit(AI2MovementState *s, int old, uint32_t *events) {
    if (old == AI2_STATE_DODGE) {
        s->body_type = 0;
        s->velocity_func = AI2_VELOCITY_DEFAULT;
        s->vx = 0.0;
        s->vy = 0.0;
        *events |= AI2_MOV_EVENT_EXIT_DODGE;
    } else if (old == AI2_STATE_KO) {
        ai2_mov_slot_set(s, AI2_STATE_KO, AI2_MOV_SLOT_INVINCIBLE_TIMER, 90);
        s->body_type = 0;
        s->velocity_func = AI2_VELOCITY_DEFAULT;
        s->vx = 0.0;
        s->vy = 0.0;
        *events |= AI2_MOV_EVENT_EXIT_KO;
    }
}

AI2_MOV_HD static inline void ai2_mov_enter(AI2MovementState *s, const AI2MovementConfig *c, int target, uint32_t *events) {
    ai2_mov_ensure_slot(s, target);
    if (target == AI2_STATE_IN_AIR) {
        ai2_mov_slot_set(s, AI2_STATE_IN_AIR, AI2_MOV_SLOT_IS_BASE, 1);
    } else if (target == AI2_STATE_TURNAROUND) {
        ai2_mov_slot_set(s, target, AI2_MOV_SLOT_TURNAROUND_TIMER, c->turnaround_time);
        *events |= AI2_MOV_EVENT_ENTER_TURNAROUND;
    } else if (target == AI2_STATE_AIR_TURNAROUND) {
        ai2_mov_slot_set(s, target, AI2_MOV_SLOT_TURNAROUND_TIMER, c->turnaround_time);
        s->vx /= 3.0;
        ai2_mov_slot_set(s, target, AI2_MOV_SLOT_IS_BASE, 0);
        *events |= AI2_MOV_EVENT_ENTER_AIR_TURNAROUND;
    } else if (target == AI2_STATE_DASH) {
        ai2_mov_slot_set(s, target, AI2_MOV_SLOT_DASH_TIMER, c->dash_time);
        *events |= AI2_MOV_EVENT_ENTER_DASH;
    } else if (target == AI2_STATE_BACKDASH) {
        ai2_mov_slot_set(s, target, AI2_MOV_SLOT_BACKDASH_TIMER, c->backdash_time);
        *events |= AI2_MOV_EVENT_ENTER_BACKDASH;
    } else if (target == AI2_STATE_DODGE) {
        ai2_mov_slot_set(s, target, AI2_MOV_SLOT_DODGE_TIMER, c->dodge_time);
        s->velocity_func = AI2_VELOCITY_NO_GRAVITY;
        s->vx = 0.0;
        s->vy = 0.0;
        *events |= AI2_MOV_EVENT_ENTER_DODGE;
    } else if (target == AI2_STATE_KO) {
        ai2_mov_slot_set(s, target, AI2_MOV_SLOT_KO_TIMER, 90);
        if (s->stocks > 0) s->stocks -= 1;
        s->velocity_func = AI2_VELOCITY_NO_GRAVITY;
        s->vx = 0.0;
        s->vy = 0.0;
        *events |= AI2_MOV_EVENT_ENTER_KO;
    }
}

AI2_MOV_HD static inline int ai2_mov_transition_to(AI2MovementState *s, const AI2MovementConfig *c, int old, int target, uint32_t *events) {
    if (!ai2_mov_supported_state(target)) return AI2_MOVEMENT_ERR_UNSUPPORTED_STATE;
    ai2_mov_common_reset(s, target, old);
    ai2_mov_exit(s, old, events);
    s->state_id = target;
    ai2_mov_enter(s, c, target, events);
    *events |= AI2_MOV_EVENT_TRANSITION;
    return 0;
}

AI2_MOV_HD static inline int ai2_mov_common_physics(const AI2MovementConfig *c, AI2MovementState *s, int current, int *target) {
    const int sides = fabs(s->x) > floor(c->stage_width_tiles / 2.0);
    const int tops = fabs(s->y) > floor(c->stage_height_tiles / 2.0);
    if (sides || tops) {
        *target = AI2_STATE_KO;
        return 1;
    }
    if (ai2_mov_slot_get(s, current, AI2_MOV_SLOT_STUN_TIME_STORED, 0) > 0) {
        return -AI2_MOVEMENT_ERR_UNSUPPORTED_STATE;
    }
    const int inv = ai2_mov_slot_get(s, current, AI2_MOV_SLOT_INVINCIBLE_TIMER, 0);
    const int dodge = ai2_mov_slot_get(s, current, AI2_MOV_SLOT_DODGE_COOLDOWN, 0);
    ai2_mov_slot_set(s, current, AI2_MOV_SLOT_INVINCIBLE_TIMER, inv > 0 ? inv - 1 : 0);
    ai2_mov_slot_set(s, current, AI2_MOV_SLOT_DODGE_COOLDOWN, dodge > 0 ? dodge - 1 : 0);
    *target = -1;
    return 0;
}

AI2_MOV_HD static inline int ai2_mov_ground_common(const AI2MovementConfig *c, AI2MovementState *s, const AI2MovementInput *in, int current, int *target) {
    int r = ai2_mov_common_physics(c, s, current, target);
    if (r != 0) return r;
    const int direction = in->decoded[AI2_MOV_OFF_RAW_HORIZONTAL];
    const int near_still = ai2_mov_abs_int(direction) == 0;
    if (in->decoded[AI2_MOV_OFF_PRESSED + AI2_MOV_KEY_SPACE] && in->grounded) {
        s->vy = -c->jump_speed;
        if (!near_still) s->facing = ai2_mov_sign_from_direction(direction);
        ai2_mov_refresh_in_air(s);
        *target = AI2_STATE_IN_AIR;
        return 1;
    }
    if (!in->grounded) {
        ai2_mov_refresh_in_air(s);
        *target = AI2_STATE_IN_AIR;
        return 1;
    }
    if (near_still && in->decoded[AI2_MOV_OFF_PRESSED + AI2_MOV_KEY_L] &&
        ai2_mov_slot_get(s, current, AI2_MOV_SLOT_DODGE_COOLDOWN, 0) <= 0) {
        ai2_mov_slot_set(s, current, AI2_MOV_SLOT_DODGE_COOLDOWN, c->grounded_dodge_cooldown);
        ai2_mov_slot_set(s, AI2_STATE_DODGE, AI2_MOV_SLOT_DODGE_IS_GROUNDED, 1);
        *target = AI2_STATE_DODGE;
        return 1;
    }
    *target = -1;
    return 0;
}

AI2_MOV_HD static inline int ai2_mov_state_physics(const AI2MovementConfig *c, AI2MovementState *s, const AI2MovementInput *in, int current, int *target, uint32_t *events) {
    int r = 0;
    if (current == AI2_STATE_STANDING) {
        r = ai2_mov_ground_common(c, s, in, current, target);
        if (r != 0) return r;
        const int direction = in->decoded[AI2_MOV_OFF_RAW_HORIZONTAL];
        if (ai2_mov_turn_check(s->facing, direction)) {
            *target = in->decoded[AI2_MOV_OFF_PRESSED + AI2_MOV_KEY_L] ? AI2_STATE_BACKDASH : AI2_STATE_TURNAROUND;
            return 1;
        }
        if (ai2_mov_abs_int(direction) > 0) {
            s->facing = ai2_mov_sign_from_direction(direction);
            *target = AI2_STATE_WALKING;
            return 1;
        }
        if (in->platform_overlap && !in->decoded[AI2_MOV_OFF_HELD + AI2_MOV_KEY_S] && in->grounded) {
            s->vx = in->platform_vx;
            s->vy = in->platform_vy;
        } else {
            s->vx = ai2_mov_move_toward(s->vx, 0.0, c->move_speed);
        }
        return 0;
    }
    if (current == AI2_STATE_WALKING || current == AI2_STATE_SPRINTING) {
        r = ai2_mov_ground_common(c, s, in, current, target);
        if (r != 0) return r;
        const int original = s->facing == 1 ? AI2_MOV_KEY_D : AI2_MOV_KEY_A;
        const int opposite = s->facing == 1 ? AI2_MOV_KEY_A : AI2_MOV_KEY_D;
        if (in->decoded[AI2_MOV_OFF_PRESSED + opposite] ||
            (!in->decoded[AI2_MOV_OFF_HELD + original] && in->decoded[AI2_MOV_OFF_HELD + opposite])) {
            *target = in->decoded[AI2_MOV_OFF_PRESSED + AI2_MOV_KEY_L] ? AI2_STATE_BACKDASH : AI2_STATE_TURNAROUND;
            return 1;
        }
        if (in->decoded[AI2_MOV_OFF_NO_HORIZONTAL]) {
            *target = AI2_STATE_STANDING;
            return 1;
        }
        if (in->decoded[AI2_MOV_OFF_PRESSED + AI2_MOV_KEY_L]) {
            *target = AI2_STATE_DASH;
            return 1;
        }
        const double speed = current == AI2_STATE_WALKING ? c->move_speed : c->run_speed;
        if (current == AI2_STATE_WALKING && in->platform_overlap && !in->decoded[AI2_MOV_OFF_HELD + AI2_MOV_KEY_S] && in->grounded) {
            s->vx = s->facing * speed + in->platform_vx;
            s->vy += in->platform_vy;
        } else {
            s->vx = s->facing * speed;
        }
        return 0;
    }
    if (current == AI2_STATE_TURNAROUND) {
        r = ai2_mov_ground_common(c, s, in, current, target);
        if (r != 0) return r;
        int timer = ai2_mov_slot_get(s, current, AI2_MOV_SLOT_TURNAROUND_TIMER, 0);
        if (timer <= 0) {
            const int last = ai2_mov_hstate_last_int(in->decoded[AI2_MOV_OFF_HORIZONTAL_STATE]);
            const int old_facing = s->facing;
            s->facing = -s->facing;
            if (-old_facing == last) *target = AI2_STATE_WALKING;
            else if (old_facing == last) *target = AI2_STATE_TURNAROUND;
            else *target = AI2_STATE_STANDING;
            return 1;
        }
        if (in->decoded[AI2_MOV_OFF_PRESSED + AI2_MOV_KEY_SPACE] && in->grounded) {
            s->facing = -s->facing;
            s->vy = -c->jump_speed;
            *target = AI2_STATE_IN_AIR;
            return 1;
        }
        if (in->decoded[AI2_MOV_OFF_PRESSED + AI2_MOV_KEY_L]) {
            *target = AI2_STATE_BACKDASH;
            return 1;
        }
        ai2_mov_slot_set(s, current, AI2_MOV_SLOT_TURNAROUND_TIMER, timer > 0 ? timer - 1 : 0);
        return 0;
    }
    if (current == AI2_STATE_IN_AIR) {
        r = ai2_mov_common_physics(c, s, current, target);
        if (r != 0) return r;
        const int direction = in->decoded[AI2_MOV_OFF_RAW_HORIZONTAL];
        if (ai2_mov_slot_get(s, current, AI2_MOV_SLOT_IS_BASE, 1) && ai2_mov_turn_check(s->facing, direction)) {
            ai2_mov_slot_set(s, AI2_STATE_AIR_TURNAROUND, AI2_MOV_SLOT_JUMP_TIMER, ai2_mov_slot_get(s, current, AI2_MOV_SLOT_JUMP_TIMER, 0));
            ai2_mov_slot_set(s, AI2_STATE_AIR_TURNAROUND, AI2_MOV_SLOT_JUMPS_LEFT, ai2_mov_slot_get(s, current, AI2_MOV_SLOT_JUMPS_LEFT, 0));
            ai2_mov_slot_set(s, AI2_STATE_AIR_TURNAROUND, AI2_MOV_SLOT_RECOVERIES_LEFT, ai2_mov_slot_get(s, current, AI2_MOV_SLOT_RECOVERIES_LEFT, 0));
            *target = AI2_STATE_AIR_TURNAROUND;
            return 1;
        }
        s->vy = (in->decoded[AI2_MOV_OFF_RAW_VERTICAL] < 0 && s->vy > 0.0) ? ai2_mov_move_toward(s->vy, c->max_fall_speed, c->fast_fall_ease) : fmin(s->vy, c->max_fall_speed);
        s->vx = ai2_mov_move_toward(s->vx, direction * c->move_speed, c->in_air_ease);
        if (in->grounded) {
            *target = ai2_mov_ground_state_for_input(in);
            return 1;
        }
        int jump_timer = ai2_mov_slot_get(s, current, AI2_MOV_SLOT_JUMP_TIMER, 0);
        int jumps_left = ai2_mov_slot_get(s, current, AI2_MOV_SLOT_JUMPS_LEFT, 0);
        if (in->decoded[AI2_MOV_OFF_PRESSED + AI2_MOV_KEY_SPACE] && jump_timer <= 0 && jumps_left > 0) {
            s->vy = -c->jump_speed;
            s->facing = ai2_mov_sign_from_direction(direction);
            jump_timer = c->jump_cooldown;
            jumps_left -= 1;
        }
        jump_timer = jump_timer > 0 ? jump_timer - 1 : 0;
        ai2_mov_slot_set(s, current, AI2_MOV_SLOT_JUMP_TIMER, jump_timer);
        ai2_mov_slot_set(s, current, AI2_MOV_SLOT_JUMPS_LEFT, jumps_left);
        if (in->decoded[AI2_MOV_OFF_PRESSED + AI2_MOV_KEY_L] && ai2_mov_slot_get(s, current, AI2_MOV_SLOT_DODGE_COOLDOWN, 0) <= 0) {
            ai2_mov_slot_set(s, current, AI2_MOV_SLOT_DODGE_COOLDOWN, c->air_dodge_cooldown);
            ai2_mov_slot_set(s, AI2_STATE_DODGE, AI2_MOV_SLOT_JUMP_TIMER, jump_timer);
            ai2_mov_slot_set(s, AI2_STATE_DODGE, AI2_MOV_SLOT_JUMPS_LEFT, jumps_left);
            ai2_mov_slot_set(s, AI2_STATE_DODGE, AI2_MOV_SLOT_RECOVERIES_LEFT, ai2_mov_slot_get(s, current, AI2_MOV_SLOT_RECOVERIES_LEFT, 0));
            ai2_mov_slot_set(s, AI2_STATE_DODGE, AI2_MOV_SLOT_DODGE_IS_GROUNDED, 0);
            *target = AI2_STATE_DODGE;
            return 1;
        }
        return 0;
    }
    if (current == AI2_STATE_AIR_TURNAROUND) {
        r = ai2_mov_common_physics(c, s, current, target);
        if (r != 0) return r;
        const int direction = in->decoded[AI2_MOV_OFF_RAW_HORIZONTAL];
        s->vy = (in->decoded[AI2_MOV_OFF_RAW_VERTICAL] < 0 && s->vy > 0.0) ? ai2_mov_move_toward(s->vy, c->max_fall_speed, c->fast_fall_ease) : fmin(s->vy, c->max_fall_speed);
        s->vx = ai2_mov_move_toward(s->vx, direction * c->move_speed, c->in_air_ease);
        if (in->grounded) {
            *target = ai2_mov_ground_state_for_input(in);
            return 1;
        }
        int jump_timer = ai2_mov_slot_get(s, current, AI2_MOV_SLOT_JUMP_TIMER, 0);
        int jumps_left = ai2_mov_slot_get(s, current, AI2_MOV_SLOT_JUMPS_LEFT, 0);
        if (in->decoded[AI2_MOV_OFF_PRESSED + AI2_MOV_KEY_SPACE] && jump_timer <= 0 && jumps_left > 0) {
            s->vy = -c->jump_speed;
            s->facing = ai2_mov_sign_from_direction(direction);
            jump_timer = c->jump_cooldown;
            jumps_left -= 1;
        }
        jump_timer = jump_timer > 0 ? jump_timer - 1 : 0;
        ai2_mov_slot_set(s, current, AI2_MOV_SLOT_JUMP_TIMER, jump_timer);
        ai2_mov_slot_set(s, current, AI2_MOV_SLOT_JUMPS_LEFT, jumps_left);
        if (in->decoded[AI2_MOV_OFF_PRESSED + AI2_MOV_KEY_L] && ai2_mov_slot_get(s, current, AI2_MOV_SLOT_DODGE_COOLDOWN, 0) <= 0) {
            ai2_mov_slot_set(s, current, AI2_MOV_SLOT_DODGE_COOLDOWN, c->air_dodge_cooldown);
            ai2_mov_slot_set(s, AI2_STATE_DODGE, AI2_MOV_SLOT_JUMP_TIMER, jump_timer);
            ai2_mov_slot_set(s, AI2_STATE_DODGE, AI2_MOV_SLOT_JUMPS_LEFT, jumps_left);
            ai2_mov_slot_set(s, AI2_STATE_DODGE, AI2_MOV_SLOT_RECOVERIES_LEFT, ai2_mov_slot_get(s, current, AI2_MOV_SLOT_RECOVERIES_LEFT, 0));
            ai2_mov_slot_set(s, AI2_STATE_DODGE, AI2_MOV_SLOT_DODGE_IS_GROUNDED, 0);
            *target = AI2_STATE_DODGE;
            return 1;
        }
        int timer = ai2_mov_slot_get(s, current, AI2_MOV_SLOT_TURNAROUND_TIMER, 0);
        if (timer <= 0) {
            const int last = ai2_mov_hstate_last_int(in->decoded[AI2_MOV_OFF_HORIZONTAL_STATE]);
            const int old_facing = s->facing;
            s->facing = -s->facing;
            const int dest = (-old_facing == last) ? AI2_STATE_IN_AIR : (old_facing == last ? AI2_STATE_AIR_TURNAROUND : AI2_STATE_IN_AIR);
            ai2_mov_slot_set(s, dest, AI2_MOV_SLOT_JUMP_TIMER, jump_timer);
            ai2_mov_slot_set(s, dest, AI2_MOV_SLOT_JUMPS_LEFT, jumps_left);
            ai2_mov_slot_set(s, dest, AI2_MOV_SLOT_RECOVERIES_LEFT, ai2_mov_slot_get(s, current, AI2_MOV_SLOT_RECOVERIES_LEFT, 0));
            *target = dest;
            return 1;
        }
        ai2_mov_slot_set(s, current, AI2_MOV_SLOT_TURNAROUND_TIMER, timer > 0 ? timer - 1 : 0);
        return 0;
    }
    if (current == AI2_STATE_DASH) {
        r = ai2_mov_ground_common(c, s, in, current, target);
        if (r != 0) return r;
        s->vx = s->facing * c->dash_speed;
        int timer = ai2_mov_slot_get(s, current, AI2_MOV_SLOT_DASH_TIMER, 0);
        timer = timer > 0 ? timer - 1 : 0;
        ai2_mov_slot_set(s, current, AI2_MOV_SLOT_DASH_TIMER, timer);
        if (timer <= 0) {
            *target = AI2_STATE_SPRINTING;
            return 1;
        }
        return 0;
    }
    if (current == AI2_STATE_BACKDASH) {
        r = ai2_mov_common_physics(c, s, current, target);
        if (r != 0) return r;
        s->vx = -s->facing * c->backdash_speed;
        int timer = ai2_mov_slot_get(s, current, AI2_MOV_SLOT_BACKDASH_TIMER, 0);
        timer = timer > 0 ? timer - 1 : 0;
        ai2_mov_slot_set(s, current, AI2_MOV_SLOT_BACKDASH_TIMER, timer);
        if (timer <= 0) {
            *target = ai2_mov_ground_state_for_input(in);
            return 1;
        }
        return 0;
    }
    if (current == AI2_STATE_DODGE) {
        r = ai2_mov_common_physics(c, s, current, target);
        if (r != 0) return r;
        int timer = ai2_mov_slot_get(s, current, AI2_MOV_SLOT_DODGE_TIMER, 0);
        timer = timer > 0 ? timer - 1 : 0;
        ai2_mov_slot_set(s, current, AI2_MOV_SLOT_DODGE_TIMER, timer);
        if (timer <= 0) {
            if (ai2_mov_slot_get(s, current, AI2_MOV_SLOT_DODGE_IS_GROUNDED, 0)) {
                *target = ai2_mov_ground_state_for_input(in);
            } else {
                ai2_mov_slot_set(s, AI2_STATE_IN_AIR, AI2_MOV_SLOT_JUMPS_LEFT, ai2_mov_slot_get(s, current, AI2_MOV_SLOT_JUMPS_LEFT, 0));
                ai2_mov_slot_set(s, AI2_STATE_IN_AIR, AI2_MOV_SLOT_JUMP_TIMER, 0);
                ai2_mov_slot_set(s, AI2_STATE_IN_AIR, AI2_MOV_SLOT_RECOVERIES_LEFT, ai2_mov_slot_get(s, current, AI2_MOV_SLOT_RECOVERIES_LEFT, 0));
                *target = AI2_STATE_IN_AIR;
            }
            return 1;
        }
        return 0;
    }
    if (current == AI2_STATE_KO) {
        (void)ai2_mov_common_physics(c, s, current, target);
        int timer = ai2_mov_slot_get(s, current, AI2_MOV_SLOT_KO_TIMER, 0) - 1;
        ai2_mov_slot_set(s, current, AI2_MOV_SLOT_KO_TIMER, timer);
        if (timer <= 0) {
            s->x = s->start_x;
            s->y = s->start_y;
            s->vx = 0.0;
            s->vy = 0.0;
            s->damage_vx = 0.0;
            s->damage_vy = 0.0;
            s->target_vx = 0.0;
            s->target_vy = 0.0;
            ai2_mov_slot_set(s, AI2_STATE_IN_AIR, AI2_MOV_SLOT_JUMPS_LEFT, 0);
            ai2_mov_slot_set(s, AI2_STATE_IN_AIR, AI2_MOV_SLOT_RECOVERIES_LEFT, 0);
            *events |= AI2_MOV_EVENT_RESPAWN;
            *target = AI2_STATE_IN_AIR;
            return 1;
        }
        *target = -1;
        return 0;
    }
    return -AI2_MOVEMENT_ERR_UNSUPPORTED_STATE;
}

AI2_MOV_HD static inline int ai2_movement_step_core(
    const AI2MovementConfig *config,
    const AI2MovementState *before,
    const AI2MovementInput *input,
    AI2MovementResult *result) {
    if (!config || !before || !input || !result) return AI2_MOVEMENT_ERR_INVALID;
    if (!ai2_mov_supported_state(before->state_id)) return AI2_MOVEMENT_ERR_UNSUPPORTED_STATE;
    if (!ai2_mov_finite_state(config, before, input)) return AI2_MOVEMENT_ERR_NONFINITE;
    for (int i = 0; i < AI2_INPUT_WIDTH; ++i) {
        if (input->decoded[i] < -1 || input->decoded[i] > 4) return AI2_MOVEMENT_ERR_INVALID;
    }
    if (input->decoded[AI2_MOV_OFF_HORIZONTAL_STATE] < 0 || input->decoded[AI2_MOV_OFF_HORIZONTAL_STATE] > 4) return AI2_MOVEMENT_ERR_INVALID;
    if (input->decoded[AI2_MOV_OFF_HELD + AI2_MOV_KEY_H] ||
        input->decoded[AI2_MOV_OFF_HELD + AI2_MOV_KEY_J] ||
        input->decoded[AI2_MOV_OFF_HELD + AI2_MOV_KEY_K] ||
        input->decoded[AI2_MOV_OFF_HELD + AI2_MOV_KEY_G]) {
        return AI2_MOVEMENT_ERR_COMBAT_INPUT;
    }
    AI2MovementState s = *before;
    ai2_mov_ensure_slot(&s, s.state_id);
    uint32_t events = 0;
    const int old = s.state_id;
    int target = -1;
    const int status = ai2_mov_state_physics(config, &s, input, old, &target, &events);
    if (status < 0) return -status;
    s.vx += s.damage_vx + s.target_vx;
    s.vy += s.damage_vy + s.target_vy;
    if (target >= 0) {
        const int e = ai2_mov_transition_to(&s, config, old, target, &events);
        if (e) return e;
    }
    s.prev_x = s.x;
    s.prev_y = s.y;
    s.damage_vx = 0.0;
    s.damage_vy = 0.0;
    s.target_vx = 0.0;
    s.target_vy = 0.0;
    result->state = s;
    result->transitioned = (uint8_t)(target >= 0);
    result->from_state = old;
    result->to_state = s.state_id;
    result->event_mask = events;
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif
