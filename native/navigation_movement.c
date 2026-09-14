#include "navigation_movement.h"

#include <string.h>

int ai2_movement_default_config(AI2MovementConfig *config) {
    if (!config) return AI2_MOVEMENT_ERR_INVALID;
    memset(config, 0, sizeof(*config));
    config->stage_width_tiles = 29.8;
    config->stage_height_tiles = 16.8;
    config->move_speed = 6.75;
    config->jump_speed = 8.9;
    config->max_fall_speed = 12.0;
    config->fast_fall_ease = 14.25 / 30.0;
    config->in_air_ease = 10.75 / 30.0;
    config->run_speed = 8.0;
    config->dash_speed = 10.0;
    config->backdash_speed = 4.0;
    config->turnaround_time = 1;
    config->backdash_time = 7;
    config->dodge_time = 10;
    config->grounded_dodge_cooldown = 30;
    config->air_dodge_cooldown = 82;
    config->invincible_time = 90;
    config->jump_cooldown = 6;
    config->dash_time = 9;
    return 0;
}

int ai2_movement_default_state(AI2MovementState *state, int32_t state_id) {
    if (!state || !ai2_mov_supported_state(state_id)) return AI2_MOVEMENT_ERR_UNSUPPORTED_STATE;
    memset(state, 0, sizeof(*state));
    state->facing = 1;
    state->state_id = state_id;
    state->stocks = 3;
    state->velocity_func = AI2_VELOCITY_DEFAULT;
    state->body_type = 0;
    for (int i = 0; i < AI2_MOVEMENT_STATE_COUNT; ++i) ai2_mov_ensure_slot(state, i);
    ai2_mov_slot_set(state, AI2_STATE_IN_AIR, AI2_MOV_SLOT_JUMP_TIMER, 0);
    ai2_mov_slot_set(state, AI2_STATE_IN_AIR, AI2_MOV_SLOT_JUMPS_LEFT, 0);
    ai2_mov_slot_set(state, AI2_STATE_IN_AIR, AI2_MOV_SLOT_RECOVERIES_LEFT, 0);
    ai2_mov_slot_set(state, AI2_STATE_IN_AIR, AI2_MOV_SLOT_IS_BASE, 1);
    return 0;
}

int ai2_movement_step(const AI2MovementConfig *config,
                      const AI2MovementState *before,
                      const AI2MovementInput *input,
                      AI2MovementResult *result) {
    return ai2_movement_step_core(config, before, input, result);
}
