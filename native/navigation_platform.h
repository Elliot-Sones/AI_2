#ifndef AI2_NATIVE_NAVIGATION_PLATFORM_H
#define AI2_NATIVE_NAVIGATION_PLATFORM_H

#include <math.h>
#include <stdint.h>

#ifdef __CUDACC__
#define AI2_NAV_HD __host__ __device__
#else
#define AI2_NAV_HD
#endif

AI2_NAV_HD static inline void ai2_navigation_platform_step_math(
    double current_x,
    double current_y,
    double waypoint1_x,
    double waypoint1_y,
    double waypoint2_x,
    double waypoint2_y,
    uint8_t moving_to_w2,
    double *velocity_x,
    double *velocity_y,
    uint8_t *next_moving_to_w2
) {
    const double target_x = moving_to_w2 ? waypoint2_x : waypoint1_x;
    const double target_y = moving_to_w2 ? waypoint2_y : waypoint1_y;

    const double dx = target_x - current_x;
    const double dy = target_y - current_y;
    const double dist = sqrt(dx * dx + dy * dy);

    if (dist < 0.1) {
        *velocity_x = 0.0;
        *velocity_y = 0.0;
        *next_moving_to_w2 = moving_to_w2 ? 0u : 1u;
        return;
    }

    const double dir_x = dx / dist;
    const double dir_y = dy / dist;

    const double seg_x = waypoint2_x - waypoint1_x;
    const double seg_y = waypoint2_y - waypoint1_y;
    double seg_len = sqrt(seg_x * seg_x + seg_y * seg_y);
    if (seg_len < 0.05) {
        seg_len = 0.05;
    }

    const double rel_x = current_x - waypoint1_x;
    const double rel_y = current_y - waypoint1_y;
    double progress = (rel_x * seg_x + rel_y * seg_y) / (seg_len * seg_len);
    if (progress < 0.0) {
        progress = 0.0;
    } else if (progress > 1.0) {
        progress = 1.0;
    }

    const double speed = sin(progress * 3.14159265358979323846) + 0.03;
    *velocity_x = dir_x * speed;
    *velocity_y = dir_y * speed;
    *next_moving_to_w2 = moving_to_w2 ? 1u : 0u;
}

AI2_NAV_HD static inline void ai2_navigation_platform_step_kinematic_math(
    double current_x,
    double current_y,
    double waypoint1_x,
    double waypoint1_y,
    double waypoint2_x,
    double waypoint2_y,
    uint8_t moving_to_w2,
    double dt,
    double *next_x,
    double *next_y,
    double *velocity_x,
    double *velocity_y,
    uint8_t *next_moving_to_w2
) {
    ai2_navigation_platform_step_math(
        current_x,
        current_y,
        waypoint1_x,
        waypoint1_y,
        waypoint2_x,
        waypoint2_y,
        moving_to_w2,
        velocity_x,
        velocity_y,
        next_moving_to_w2
    );
    *next_x = current_x + (*velocity_x * dt);
    *next_y = current_y + (*velocity_y * dt);
}

#endif
