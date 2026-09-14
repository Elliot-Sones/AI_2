#ifndef AI2_NAVIGATION_CONTRACT_H
#define AI2_NAVIGATION_CONTRACT_H

#include <stdint.h>

#ifdef __CUDACC__
#include <math.h>
#define AI2_NAV_HD __host__ __device__
#else
#include <math.h>
#define AI2_NAV_HD
#endif

#define AI2_NAV_INT32_MAX 2147483647

typedef struct {
    int32_t completion_mode; /* 0=horizon, 1=sequence */
    int32_t control_mode;    /* 0=full, 1=movement */
    int32_t max_targets;
    int32_t time_limit_steps;
    int32_t settle_steps;
    int32_t post_goal_required_steps;
    double settle_speed;
    double fall_penalty;
} NavigationConfigC;

typedef struct {
    double target_x;
    double target_y;
    uint8_t target_valid;
    double last_distance;
    uint8_t last_distance_valid;
    int32_t episode_steps;
    int32_t targets_this_episode;
    int32_t hold_steps;
    int32_t post_goal_steps;
    int32_t goal_reached_step;
    uint8_t goal_reached_step_valid;
    int32_t last_stocks;
    uint8_t fell_this_episode;
    uint8_t target_touched;
    int32_t targets_reached;
    int32_t total_targets;
    int32_t falls;
    int32_t steps_since_log;
    int32_t interval_targets_reached;
    int32_t interval_falls;
    int32_t interval_episodes;
} NavigationStateC;

typedef struct {
    uint8_t player_present;
    double player_x;
    double player_y;
    double velocity_x;
    double velocity_y;
    double support_velocity_x;
    double support_velocity_y;
    uint8_t support_velocity_valid;
    uint8_t grounded;
    int32_t stocks;
    double raw_reward;
    uint8_t raw_terminated;
    uint8_t raw_truncated;
    double next_target_x;
    double next_target_y;
    uint8_t next_target_valid;
} NavigationPostPhysicsC;

typedef struct {
    NavigationStateC state;
    double reward;
    double nav_reward;
    uint8_t terminated;
    uint8_t truncated;
    uint8_t underlying_done;
    uint8_t touched;
    uint8_t reached;
    uint8_t fall;
    uint8_t handoff;
    uint8_t success;
    uint8_t horizon;
} NavigationStepResultC;

AI2_NAV_HD static inline double navigation_contract_distance(
    double px,
    double py,
    double tx,
    double ty,
    uint8_t valid) {
    if (!valid) {
        return 0.0;
    }
    const double dx = px - tx;
    const double dy = py - ty;
    return sqrt(dx * dx + dy * dy);
}

AI2_NAV_HD static inline double navigation_contract_min_double(double a, double b) {
    return a < b ? a : b;
}

AI2_NAV_HD static inline int navigation_contract_is_settled(
    const NavigationConfigC *config,
    const NavigationPostPhysicsC *post,
    double distance) {
    if (distance >= 1.0 || !post->grounded) {
        return 0;
    }
    const double sx = post->support_velocity_valid ? post->support_velocity_x : 0.0;
    const double sy = post->support_velocity_valid ? post->support_velocity_y : 0.0;
    const double rvx = post->velocity_x - sx;
    const double rvy = post->velocity_y - sy;
    const double relative_speed = hypot(rvx, rvy);
    return distance < 1.0 && relative_speed <= config->settle_speed;
}

AI2_NAV_HD static inline void navigation_contract_zero_result(NavigationStepResultC *out) {
    out->state.target_x = 0.0;
    out->state.target_y = 0.0;
    out->state.target_valid = 0;
    out->state.last_distance = 0.0;
    out->state.last_distance_valid = 0;
    out->state.episode_steps = 0;
    out->state.targets_this_episode = 0;
    out->state.hold_steps = 0;
    out->state.post_goal_steps = 0;
    out->state.goal_reached_step = 0;
    out->state.goal_reached_step_valid = 0;
    out->state.last_stocks = 0;
    out->state.fell_this_episode = 0;
    out->state.target_touched = 0;
    out->state.targets_reached = 0;
    out->state.total_targets = 0;
    out->state.falls = 0;
    out->state.steps_since_log = 0;
    out->state.interval_targets_reached = 0;
    out->state.interval_falls = 0;
    out->state.interval_episodes = 0;
    out->reward = 0.0;
    out->nav_reward = 0.0;
    out->terminated = 0;
    out->truncated = 0;
    out->underlying_done = 0;
    out->touched = 0;
    out->reached = 0;
    out->fall = 0;
    out->handoff = 0;
    out->success = 0;
    out->horizon = 0;
}

AI2_NAV_HD static inline int navigation_contract_bool01(uint8_t value) {
    return value == 0 || value == 1;
}

AI2_NAV_HD static inline int navigation_contract_finite(double value) {
    return isfinite(value);
}

AI2_NAV_HD static inline int navigation_contract_valid_config(const NavigationConfigC *config) {
    if (!(config->completion_mode == 0 || config->completion_mode == 1)) {
        return 0;
    }
    if (!(config->control_mode == 0 || config->control_mode == 1)) {
        return 0;
    }
    if (config->max_targets < 1 || config->time_limit_steps <= 0 ||
        config->settle_steps <= 0 || config->post_goal_required_steps <= 0) {
        return 0;
    }
    if (config->settle_steps > AI2_NAV_INT32_MAX - config->post_goal_required_steps) {
        return 0;
    }
    if (config->completion_mode == 1 && config->max_targets < 2) {
        return 0;
    }
    if (!navigation_contract_finite(config->settle_speed) ||
        !navigation_contract_finite(config->fall_penalty) ||
        config->settle_speed <= 0.0) {
        return 0;
    }
    return 1;
}

AI2_NAV_HD static inline int navigation_contract_valid_state(const NavigationStateC *state) {
    if (!navigation_contract_finite(state->target_x) ||
        !navigation_contract_finite(state->target_y) ||
        !navigation_contract_finite(state->last_distance)) {
        return 0;
    }
    if (state->episode_steps < 0 || state->targets_this_episode < 0 ||
        state->hold_steps < 0 || state->post_goal_steps < 0 ||
        state->goal_reached_step < 0 || state->last_stocks < 0 ||
        state->targets_reached < 0 || state->total_targets < 0 ||
        state->falls < 0 || state->steps_since_log < 0 ||
        state->interval_targets_reached < 0 || state->interval_falls < 0 ||
        state->interval_episodes < 0) {
        return 0;
    }
    if (!navigation_contract_bool01(state->target_valid) ||
        !navigation_contract_bool01(state->last_distance_valid) ||
        !navigation_contract_bool01(state->goal_reached_step_valid) ||
        !navigation_contract_bool01(state->fell_this_episode) ||
        !navigation_contract_bool01(state->target_touched)) {
        return 0;
    }
    if (state->episode_steps >= AI2_NAV_INT32_MAX ||
        state->steps_since_log >= AI2_NAV_INT32_MAX ||
        state->hold_steps >= AI2_NAV_INT32_MAX ||
        state->falls >= AI2_NAV_INT32_MAX ||
        state->interval_falls >= AI2_NAV_INT32_MAX ||
        state->targets_reached >= AI2_NAV_INT32_MAX ||
        state->interval_targets_reached >= AI2_NAV_INT32_MAX ||
        state->targets_this_episode >= AI2_NAV_INT32_MAX ||
        state->total_targets >= AI2_NAV_INT32_MAX) {
        return 0;
    }
    return 1;
}

AI2_NAV_HD static inline int navigation_contract_valid_post(const NavigationPostPhysicsC *post) {
    if (!navigation_contract_finite(post->player_x) ||
        !navigation_contract_finite(post->player_y) ||
        !navigation_contract_finite(post->velocity_x) ||
        !navigation_contract_finite(post->velocity_y) ||
        !navigation_contract_finite(post->support_velocity_x) ||
        !navigation_contract_finite(post->support_velocity_y) ||
        !navigation_contract_finite(post->raw_reward) ||
        !navigation_contract_finite(post->next_target_x) ||
        !navigation_contract_finite(post->next_target_y)) {
        return 0;
    }
    if (post->stocks < 0) {
        return 0;
    }
    if (!navigation_contract_bool01(post->player_present) ||
        !navigation_contract_bool01(post->support_velocity_valid) ||
        !navigation_contract_bool01(post->grounded) ||
        !navigation_contract_bool01(post->raw_terminated) ||
        !navigation_contract_bool01(post->raw_truncated) ||
        !navigation_contract_bool01(post->next_target_valid)) {
        return 0;
    }
    return 1;
}

AI2_NAV_HD static inline int navigation_contract_step_core(
    const NavigationConfigC *config,
    const NavigationStateC *state,
    const NavigationPostPhysicsC *post,
    NavigationStepResultC *out) {
    if (config == 0 || state == 0 || post == 0 || out == 0) {
        return 1;
    }
    navigation_contract_zero_result(out);
    if (!navigation_contract_valid_config(config) ||
        !navigation_contract_valid_state(state) ||
        !navigation_contract_valid_post(post)) {
        return 3;
    }

    NavigationStateC next = *state;
    next.episode_steps += 1;
    next.steps_since_log += 1;

    if (!post->player_present) {
        out->state = next;
        out->reward = post->raw_reward;
        out->nav_reward = 0.0;
        out->terminated = post->raw_terminated != 0;
        out->truncated = post->raw_truncated != 0;
        out->underlying_done = (post->raw_terminated || post->raw_truncated) != 0;
        return 0;
    }

    double nav_reward = 0.0;
    uint8_t terminated = post->raw_terminated != 0;
    uint8_t truncated = post->raw_truncated != 0;
    const uint8_t underlying_done = (terminated || truncated) != 0;
    uint8_t touched = 0;
    uint8_t reached = 0;
    uint8_t fall = 0;
    uint8_t handoff = 0;

    const double current_dist = navigation_contract_distance(
        post->player_x, post->player_y, next.target_x, next.target_y, next.target_valid);
    if (next.last_distance_valid) {
        nav_reward += (next.last_distance - current_dist) * 2.0;
    }

    if (post->stocks < next.last_stocks) {
        nav_reward += config->fall_penalty;
        next.falls += 1;
        next.interval_falls += 1;
        next.fell_this_episode = 1;
        terminated = 1;
        fall = 1;
    }
    next.last_stocks = post->stocks;

    const uint8_t active = !(underlying_done || next.fell_this_episode);

    if (active && current_dist < 1.0 && !next.target_touched) {
        next.target_touched = 1;
        touched = 1;
    }

    if (active && navigation_contract_is_settled(config, post, current_dist)) {
        int32_t hold_limit = config->settle_steps;
        if (config->completion_mode == 1 && next.targets_this_episode >= config->max_targets) {
            hold_limit += config->post_goal_required_steps;
        }
        next.hold_steps = next.hold_steps + 1;
        if (next.hold_steps > hold_limit) {
            next.hold_steps = hold_limit;
        }
    } else {
        next.hold_steps = 0;
    }

    if (active && next.hold_steps >= config->settle_steps && next.targets_this_episode < config->max_targets) {
        nav_reward += 5.0;
        next.targets_reached += 1;
        next.interval_targets_reached += 1;
        reached = 1;
        next.targets_this_episode += 1;
        if (next.targets_this_episode < config->max_targets) {
            if (!post->next_target_valid) {
                return 2;
            }
            next.target_x = post->next_target_x;
            next.target_y = post->next_target_y;
            next.target_valid = 1;
            next.total_targets += 1;
            next.hold_steps = 0;
            next.target_touched = 0;
            handoff = 1;
        } else {
            next.goal_reached_step = next.episode_steps;
            next.goal_reached_step_valid = 1;
        }
    }

    if (next.goal_reached_step_valid) {
        if (config->completion_mode == 1) {
            next.post_goal_steps = next.hold_steps - config->settle_steps;
            if (next.post_goal_steps < 0) {
                next.post_goal_steps = 0;
            }
        } else {
            next.post_goal_steps = next.episode_steps - next.goal_reached_step;
        }
    }

    next.last_distance = navigation_contract_distance(
        post->player_x, post->player_y, next.target_x, next.target_y, next.target_valid);
    next.last_distance_valid = 1;

    const double x_pos = fabs(post->player_x);
    if (x_pos > 8.0) {
        const double edge_factor = (x_pos - 8.0) / 4.0;
        nav_reward -= 0.1 * navigation_contract_min_double(edge_factor, 1.0);
    }
    if (post->player_y > 3.0) {
        const double y_edge_factor = (post->player_y - 3.0) / 3.0;
        nav_reward -= 0.1 * navigation_contract_min_double(y_edge_factor, 1.0);
    }

    const uint8_t horizon = (next.episode_steps >= config->time_limit_steps && !underlying_done) != 0;
    const uint8_t completed = (
        active
        && next.targets_this_episode >= config->max_targets
        && next.hold_steps >= config->settle_steps
        && next.post_goal_steps >= config->post_goal_required_steps) != 0;
    const uint8_t success = (
        completed && (config->completion_mode == 1 ? !horizon : horizon)) != 0;
    if (success && config->completion_mode == 1) {
        terminated = 1;
    }
    if (horizon) {
        truncated = 1;
    }

    out->state = next;
    out->reward = post->raw_reward + nav_reward;
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
    return 0;
}

#ifdef __cplusplus
extern "C" {
#endif

int navigation_contract_step_one(
    const NavigationConfigC *config,
    const NavigationStateC *state,
    const NavigationPostPhysicsC *post,
    NavigationStepResultC *out);

#ifdef __cplusplus
}
#endif

#endif
