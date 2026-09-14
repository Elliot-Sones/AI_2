#include "navigation_platform.h"

#include <stddef.h>

void navigation_platform_step_one(
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
}

void navigation_platform_step_batch_cpu(
    const double *positions_xy,
    const double *waypoint1_xy,
    const double *waypoint2_xy,
    const uint8_t *moving_to_w2,
    double *velocities_xy,
    uint8_t *next_moving_to_w2,
    size_t lanes
) {
    for (size_t lane = 0; lane < lanes; ++lane) {
        const size_t offset = lane * 2u;
        ai2_navigation_platform_step_math(
            positions_xy[offset],
            positions_xy[offset + 1u],
            waypoint1_xy[offset],
            waypoint1_xy[offset + 1u],
            waypoint2_xy[offset],
            waypoint2_xy[offset + 1u],
            moving_to_w2[lane],
            &velocities_xy[offset],
            &velocities_xy[offset + 1u],
            &next_moving_to_w2[lane]
        );
    }
}

void navigation_platform_step_kinematic_one(
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
    ai2_navigation_platform_step_kinematic_math(
        current_x,
        current_y,
        waypoint1_x,
        waypoint1_y,
        waypoint2_x,
        waypoint2_y,
        moving_to_w2,
        dt,
        next_x,
        next_y,
        velocity_x,
        velocity_y,
        next_moving_to_w2
    );
}

void navigation_platform_step_kinematic_batch_cpu(
    const double *positions_xy,
    const double *waypoint1_xy,
    const double *waypoint2_xy,
    const uint8_t *moving_to_w2,
    double dt,
    double *next_positions_xy,
    double *velocities_xy,
    uint8_t *next_moving_to_w2,
    size_t lanes
) {
    for (size_t lane = 0; lane < lanes; ++lane) {
        const size_t offset = lane * 2u;
        ai2_navigation_platform_step_kinematic_math(
            positions_xy[offset],
            positions_xy[offset + 1u],
            waypoint1_xy[offset],
            waypoint1_xy[offset + 1u],
            waypoint2_xy[offset],
            waypoint2_xy[offset + 1u],
            moving_to_w2[lane],
            dt,
            &next_positions_xy[offset],
            &next_positions_xy[offset + 1u],
            &velocities_xy[offset],
            &velocities_xy[offset + 1u],
            &next_moving_to_w2[lane]
        );
    }
}
