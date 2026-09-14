#include "navigation_contract.h"

#include <cuda_runtime.h>
#include <stdint.h>

static __device__ NavigationStateC navigation_contract_load_state(
    const double *state_xy,
    const double *last_distance,
    const int32_t *state_i32,
    const uint8_t *flags,
    int lane) {
    const int s2 = lane * 2;
    const int si = lane * 13;
    const int fi = lane * 11;
    NavigationStateC state;
    state.target_x = state_xy[s2 + 0];
    state.target_y = state_xy[s2 + 1];
    state.target_valid = flags[fi + 0];
    state.last_distance = last_distance[lane];
    state.last_distance_valid = flags[fi + 1];
    state.episode_steps = state_i32[si + 0];
    state.targets_this_episode = state_i32[si + 1];
    state.hold_steps = state_i32[si + 2];
    state.post_goal_steps = state_i32[si + 3];
    state.goal_reached_step = state_i32[si + 4];
    state.goal_reached_step_valid = flags[fi + 2];
    state.last_stocks = state_i32[si + 5];
    state.fell_this_episode = flags[fi + 3];
    state.target_touched = flags[fi + 4];
    state.targets_reached = state_i32[si + 6];
    state.total_targets = state_i32[si + 7];
    state.falls = state_i32[si + 8];
    state.steps_since_log = state_i32[si + 9];
    state.interval_targets_reached = state_i32[si + 10];
    state.interval_falls = state_i32[si + 11];
    state.interval_episodes = state_i32[si + 12];
    return state;
}

static __device__ NavigationPostPhysicsC navigation_contract_load_post(
    const uint8_t *flags,
    const double *player_xy,
    const double *velocity_xy,
    const double *support_velocity_xy,
    const double *raw_reward,
    const int32_t *post_i32,
    const double *next_target_xy,
    int lane) {
    const int s2 = lane * 2;
    const int fi = lane * 11;
    NavigationPostPhysicsC post;
    post.player_present = flags[fi + 5];
    post.player_x = player_xy[s2 + 0];
    post.player_y = player_xy[s2 + 1];
    post.velocity_x = velocity_xy[s2 + 0];
    post.velocity_y = velocity_xy[s2 + 1];
    post.support_velocity_x = support_velocity_xy[s2 + 0];
    post.support_velocity_y = support_velocity_xy[s2 + 1];
    post.support_velocity_valid = flags[fi + 6];
    post.grounded = flags[fi + 7];
    post.stocks = post_i32[lane];
    post.raw_reward = raw_reward[lane];
    post.raw_terminated = flags[fi + 8];
    post.raw_truncated = flags[fi + 9];
    post.next_target_x = next_target_xy[s2 + 0];
    post.next_target_y = next_target_xy[s2 + 1];
    post.next_target_valid = flags[fi + 10];
    return post;
}

static __device__ void navigation_contract_store_result(
    const NavigationStepResultC *result,
    double *out_state_xy,
    double *out_last_distance,
    int32_t *out_state_i32,
    uint8_t *out_flags,
    double *out_reward,
    double *out_nav_reward,
    int lane) {
    const int s2 = lane * 2;
    const int si = lane * 13;
    const int of = lane * 14;
    out_state_xy[s2 + 0] = result->state.target_x;
    out_state_xy[s2 + 1] = result->state.target_y;
    out_last_distance[lane] = result->state.last_distance;
    out_state_i32[si + 0] = result->state.episode_steps;
    out_state_i32[si + 1] = result->state.targets_this_episode;
    out_state_i32[si + 2] = result->state.hold_steps;
    out_state_i32[si + 3] = result->state.post_goal_steps;
    out_state_i32[si + 4] = result->state.goal_reached_step;
    out_state_i32[si + 5] = result->state.last_stocks;
    out_state_i32[si + 6] = result->state.targets_reached;
    out_state_i32[si + 7] = result->state.total_targets;
    out_state_i32[si + 8] = result->state.falls;
    out_state_i32[si + 9] = result->state.steps_since_log;
    out_state_i32[si + 10] = result->state.interval_targets_reached;
    out_state_i32[si + 11] = result->state.interval_falls;
    out_state_i32[si + 12] = result->state.interval_episodes;
    out_flags[of + 0] = result->state.target_valid;
    out_flags[of + 1] = result->state.last_distance_valid;
    out_flags[of + 2] = result->state.goal_reached_step_valid;
    out_flags[of + 3] = result->state.fell_this_episode;
    out_flags[of + 4] = result->state.target_touched;
    out_flags[of + 5] = result->terminated;
    out_flags[of + 6] = result->truncated;
    out_flags[of + 7] = result->underlying_done;
    out_flags[of + 8] = result->touched;
    out_flags[of + 9] = result->reached;
    out_flags[of + 10] = result->fall;
    out_flags[of + 11] = result->handoff;
    out_flags[of + 12] = result->success;
    out_flags[of + 13] = result->horizon;
    out_reward[lane] = result->reward;
    out_nav_reward[lane] = result->nav_reward;
}

__global__ void navigation_contract_kernel(
    NavigationConfigC config,
    const double *state_xy,
    const double *last_distance,
    const int32_t *state_i32,
    const uint8_t *flags,
    const double *player_xy,
    const double *velocity_xy,
    const double *support_velocity_xy,
    const double *raw_reward,
    const int32_t *post_i32,
    const double *next_target_xy,
    double *out_state_xy,
    double *out_last_distance,
    int32_t *out_state_i32,
    uint8_t *out_flags,
    double *out_reward,
    double *out_nav_reward,
    int32_t *out_errors,
    int lanes) {
    const int lane = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (lane >= lanes) {
        return;
    }
    out_errors[lane] = 0;

    const NavigationStateC state = navigation_contract_load_state(state_xy, last_distance, state_i32, flags, lane);
    const NavigationPostPhysicsC post = navigation_contract_load_post(
        flags, player_xy, velocity_xy, support_velocity_xy, raw_reward, post_i32, next_target_xy, lane);
    NavigationStepResultC result;
    const int status = navigation_contract_step_core(&config, &state, &post, &result);
    if (status != 0) {
        out_errors[lane] = status;
        return;
    }
    navigation_contract_store_result(
        &result,
        out_state_xy,
        out_last_distance,
        out_state_i32,
        out_flags,
        out_reward,
        out_nav_reward,
        lane);
}

extern "C" int navigation_contract_step_cuda(
    const NavigationConfigC *config,
    const double *state_xy,
    const double *last_distance,
    const int32_t *state_i32,
    const uint8_t *flags,
    const double *player_xy,
    const double *velocity_xy,
    const double *support_velocity_xy,
    const double *raw_reward,
    const int32_t *post_i32,
    const double *next_target_xy,
    double *out_state_xy,
    double *out_last_distance,
    int32_t *out_state_i32,
    uint8_t *out_flags,
    double *out_reward,
    double *out_nav_reward,
    int32_t *out_errors,
    int lanes,
    void *stream) {
    if (!config || !state_xy || !last_distance || !state_i32 || !flags || !player_xy ||
        !velocity_xy || !support_velocity_xy || !raw_reward || !post_i32 || !next_target_xy ||
        !out_state_xy || !out_last_distance || !out_state_i32 || !out_flags || !out_reward ||
        !out_nav_reward || !out_errors || lanes <= 0) {
        return 1;
    }
    const int threads = 256;
    const int blocks = (lanes + threads - 1) / threads;
    navigation_contract_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        *config,
        state_xy,
        last_distance,
        state_i32,
        flags,
        player_xy,
        velocity_xy,
        support_velocity_xy,
        raw_reward,
        post_i32,
        next_target_xy,
        out_state_xy,
        out_last_distance,
        out_state_i32,
        out_flags,
        out_reward,
        out_nav_reward,
        out_errors,
        lanes);
    return (int)cudaGetLastError();
}
