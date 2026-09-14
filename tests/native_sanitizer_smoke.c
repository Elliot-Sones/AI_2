#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "navigation_contract.h"
#include "navigation_input.h"
#include "navigation_movement.h"
#include "navigation_observation.h"
#include "navigation_physics.h"
#include "navigation_platform.h"

void ai2_input_init(int32_t *state);
int ai2_input_step(const double *action, const int32_t *before, int32_t *after);
void navigation_platform_step_one(double current_x, double current_y,
                                  double waypoint1_x, double waypoint1_y,
                                  double waypoint2_x, double waypoint2_y,
                                  uint8_t moving_to_w2, double *velocity_x,
                                  double *velocity_y, uint8_t *next_moving_to_w2);
void navigation_platform_step_kinematic_one(double current_x, double current_y,
                                            double waypoint1_x, double waypoint1_y,
                                            double waypoint2_x, double waypoint2_y,
                                            uint8_t moving_to_w2, double dt,
                                            double *next_x, double *next_y,
                                            double *velocity_x, double *velocity_y,
                                            uint8_t *next_moving_to_w2);
int navigation_physics_step_one_cpu(const double *state, double gravity_x,
                                    double gravity_y, double damping, double dt,
                                    double *out);
int ai2_observation_cpu(const double *p, const double *s, const double *platform,
                        float *out, int *errors, size_t lanes);

#define CHECK(expr)                                                                 \
    do {                                                                            \
        if (!(expr)) {                                                              \
            fprintf(stderr, "check failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
            return 1;                                                               \
        }                                                                           \
    } while (0)

static uint32_t rng_step(uint32_t *state) {
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static double rand_unit(uint32_t *state) {
    return ((double)(rng_step(state) & 0xffffu) / 32768.0) - 1.0;
}

static int test_input(void) {
    int32_t before[AI2_INPUT_WIDTH];
    int32_t after[AI2_INPUT_WIDTH];
    double action[AI2_INPUT_BUTTONS] = {0};
    ai2_input_init(before);
    action[AI2_INPUT_BUTTONS - 1] = 1.0;
    CHECK(ai2_input_step(action, before, after) == 0);
    CHECK(after[AI2_INPUT_HELD + AI2_INPUT_BUTTONS - 1] == 1);
    before[AI2_INPUT_HORIZONTAL_STATE] = 99;
    memcpy(after, before, sizeof(after));
    CHECK(ai2_input_step(action, before, after) != 0);
    CHECK(ai2_input_step(NULL, before, after) != 0);
    return 0;
}

static int test_platform(void) {
    double vx = -123.0;
    double vy = -456.0;
    double next_x = 777.0;
    double next_y = 888.0;
    uint8_t moving = 1u;
    navigation_platform_step_one(0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1u, &vx, &vy, &moving);
    CHECK(isfinite(vx));
    CHECK(isfinite(vy));
    CHECK(moving == 1u);
    navigation_platform_step_kinematic_one(0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1u, 1.0 / 30.0,
                                           &next_x, &next_y, &vx, &vy, &moving);
    CHECK(isfinite(next_x));
    CHECK(isfinite(next_y));
    return 0;
}

static void fill_physics_state(double *state) {
    for (size_t i = 0; i < AI2_PHYSICS_BODY_STATE_FIELDS; ++i) state[i] = 0.0;
    state[0] = AI2_PHYSICS_BODY_TYPE_DYNAMIC;
    state[1] = AI2_PHYSICS_VELOCITY_MODE_DEFAULT;
    state[2] = 1.0;
    state[3] = 2.0;
    state[4] = 3.0;
    state[5] = 4.0;
    state[11] = 2.0;
    state[12] = 3.0;
}

static int test_physics(void) {
    double guard[AI2_PHYSICS_BODY_STATE_FIELDS + 2];
    double out[AI2_PHYSICS_BODY_STATE_FIELDS + 2];
    fill_physics_state(guard + 1);
    guard[0] = 12345.0;
    guard[AI2_PHYSICS_BODY_STATE_FIELDS + 1] = 67890.0;
    out[0] = 111.0;
    out[AI2_PHYSICS_BODY_STATE_FIELDS + 1] = 222.0;
    CHECK(navigation_physics_step_one_cpu(guard + 1, 0.0, -9.8, 0.99, 1.0 / 30.0, out + 1) == AI2_PHYSICS_OK);
    CHECK(guard[0] == 12345.0);
    CHECK(guard[AI2_PHYSICS_BODY_STATE_FIELDS + 1] == 67890.0);
    CHECK(out[0] == 111.0);
    CHECK(out[AI2_PHYSICS_BODY_STATE_FIELDS + 1] == 222.0);
    CHECK(navigation_physics_step_one_cpu(guard + 1, 0.0, -9.8, 0.99, NAN, out + 1) == AI2_PHYSICS_ERR_INVALID_TIMESTEP);
    CHECK(navigation_physics_step_one_cpu(NULL, 0.0, -9.8, 0.99, 1.0 / 30.0, out + 1) == AI2_PHYSICS_ERR_NULL);
    guard[2] = NAN;
    CHECK(navigation_physics_step_one_cpu(guard + 1, 0.0, -9.8, 0.99, 1.0 / 30.0, out + 1) == AI2_PHYSICS_ERR_NONFINITE);
    return 0;
}

static int test_movement(void) {
    AI2MovementConfig config;
    AI2MovementState state;
    AI2MovementInput input;
    AI2MovementResult result;
    CHECK(ai2_movement_default_config(&config) == 0);
    CHECK(ai2_movement_default_state(&state, AI2_STATE_STANDING) == 0);
    memset(&input, 0, sizeof(input));
    input.grounded = 1u;
    input.decoded[AI2_MOV_OFF_NO_HORIZONTAL] = 1;
    CHECK(ai2_movement_step(&config, &state, &input, &result) == 0);
    input.decoded[AI2_MOV_OFF_HELD + AI2_MOV_KEY_H] = 1;
    CHECK(ai2_movement_step(&config, &state, &input, &result) == AI2_MOVEMENT_ERR_COMBAT_INPUT);
    state.x = NAN;
    input.decoded[AI2_MOV_OFF_HELD + AI2_MOV_KEY_H] = 0;
    CHECK(ai2_movement_step(&config, &state, &input, &result) == AI2_MOVEMENT_ERR_NONFINITE);
    return 0;
}

static int test_contract(void) {
    NavigationConfigC config = {0};
    NavigationStateC state = {0};
    NavigationPostPhysicsC post = {0};
    NavigationStepResultC out;
    config.completion_mode = 0;
    config.control_mode = 1;
    config.max_targets = 1;
    config.time_limit_steps = 1200;
    config.settle_steps = 3;
    config.post_goal_required_steps = 2;
    config.settle_speed = 0.5;
    config.fall_penalty = -5.0;
    state.target_valid = 1u;
    state.last_distance_valid = 1u;
    state.last_distance = 2.0;
    state.last_stocks = 3;
    post.player_present = 1u;
    post.grounded = 1u;
    post.stocks = 3;
    CHECK(navigation_contract_step_one(&config, &state, &post, &out) == 0);
    CHECK(out.state.episode_steps == 1);
    config.settle_speed = NAN;
    CHECK(navigation_contract_step_one(&config, &state, &post, &out) == 3);
    CHECK(navigation_contract_step_one(NULL, &state, &post, &out) == 1);
    return 0;
}

static int test_observation(void) {
    double players[32] = {0};
    double spawners[12] = {0};
    double platform[4] = {0};
    float out[64];
    int errors[1] = {999};
    players[4] = 1.0;
    players[5] = 1.0;
    players[20] = -1.0;
    players[21] = 0.0;
    CHECK(ai2_observation_cpu(players, spawners, platform, out, errors, 1) == 0);
    CHECK(errors[0] == 0);
    players[4] = 7.0;
    CHECK(ai2_observation_cpu(players, spawners, platform, out, errors, 1) == 0);
    CHECK(errors[0] != 0);
    CHECK(ai2_observation_cpu(NULL, spawners, platform, out, errors, 1) != 0);
    return 0;
}

static int test_fuzz(void) {
    uint32_t rng = 0xA1202609u;
    int32_t input_before[AI2_INPUT_WIDTH];
    int32_t input_after[AI2_INPUT_WIDTH];
    AI2MovementConfig movement_config;
    AI2MovementState movement_state;
    AI2MovementInput movement_input;
    AI2MovementResult movement_result;
    NavigationConfigC nav_config = {0};
    NavigationStateC nav_state = {0};
    NavigationPostPhysicsC nav_post = {0};
    NavigationStepResultC nav_out;
    double physics_state[AI2_PHYSICS_BODY_STATE_FIELDS];
    double physics_out[AI2_PHYSICS_BODY_STATE_FIELDS];

    CHECK(ai2_movement_default_config(&movement_config) == 0);
    nav_config.completion_mode = 0;
    nav_config.control_mode = 1;
    nav_config.max_targets = 1;
    nav_config.time_limit_steps = 1200;
    nav_config.settle_steps = 3;
    nav_config.post_goal_required_steps = 2;
    nav_config.settle_speed = 10.0;
    nav_config.fall_penalty = -1.0;

    for (int i = 0; i < 1000; ++i) {
        double action[AI2_INPUT_BUTTONS] = {0};
        ai2_input_init(input_before);
        for (int j = 0; j < AI2_INPUT_BUTTONS; ++j) action[j] = (rng_step(&rng) & 1u) ? 1.0 : 0.0;
        CHECK(ai2_input_step(action, input_before, input_after) == 0);

        CHECK(ai2_movement_default_state(&movement_state, (i & 1) ? AI2_STATE_STANDING : AI2_STATE_IN_AIR) == 0);
        memset(&movement_input, 0, sizeof(movement_input));
        memcpy(movement_input.decoded, input_after, sizeof(input_after));
        movement_input.decoded[AI2_MOV_OFF_HELD + AI2_MOV_KEY_H] = 0;
        movement_input.decoded[AI2_MOV_OFF_HELD + AI2_MOV_KEY_J] = 0;
        movement_input.decoded[AI2_MOV_OFF_HELD + AI2_MOV_KEY_K] = 0;
        movement_input.decoded[AI2_MOV_OFF_HELD + AI2_MOV_KEY_G] = 0;
        movement_input.grounded = (uint8_t)(i & 1);
        CHECK(ai2_movement_step(&movement_config, &movement_state, &movement_input, &movement_result) == 0);

        fill_physics_state(physics_state);
        physics_state[2] = rand_unit(&rng) * 10.0;
        physics_state[3] = rand_unit(&rng) * 5.0;
        physics_state[4] = rand_unit(&rng) * 3.0;
        physics_state[5] = rand_unit(&rng) * 3.0;
        CHECK(navigation_physics_step_one_cpu(physics_state, 0.0, -9.8, 0.99, 1.0 / 30.0, physics_out) == AI2_PHYSICS_OK);

        nav_state.target_x = rand_unit(&rng);
        nav_state.target_y = rand_unit(&rng);
        nav_state.target_valid = 1u;
        nav_state.last_distance = fabs(rand_unit(&rng));
        nav_state.last_distance_valid = 1u;
        nav_state.last_stocks = 3;
        nav_post.player_present = 1u;
        nav_post.player_x = rand_unit(&rng);
        nav_post.player_y = rand_unit(&rng);
        nav_post.velocity_x = rand_unit(&rng);
        nav_post.velocity_y = rand_unit(&rng);
        nav_post.grounded = 1u;
        nav_post.stocks = 3;
        CHECK(navigation_contract_step_one(&nav_config, &nav_state, &nav_post, &nav_out) == 0);
    }
    return 0;
}

int main(void) {
    CHECK(test_input() == 0);
    CHECK(test_platform() == 0);
    CHECK(test_physics() == 0);
    CHECK(test_movement() == 0);
    CHECK(test_contract() == 0);
    CHECK(test_observation() == 0);
    CHECK(test_fuzz() == 0);
    puts("native sanitizer smoke ok fuzz_cases=1000");
    return 0;
}
