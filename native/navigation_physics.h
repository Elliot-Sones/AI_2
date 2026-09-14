#ifndef AI2_NATIVE_NAVIGATION_PHYSICS_H
#define AI2_NATIVE_NAVIGATION_PHYSICS_H

/*
 * First AI_2 native physics foundation.
 *
 * The body integration formulas are source-faithful to Chipmunk 7.0.3 at
 * commit 0593976ef47fcb3957166bd342f6b2bafe4d0e44, as bundled by
 * Pymunk 6.2.1. They are derived from cpBodyUpdatePosition(),
 * cpBodyUpdateVelocity(), and the cpSpaceStep() ordering where positions are
 * integrated before velocities.
 *
 * Chipmunk is MIT licensed:
 * Copyright (c) 2013 Scott Lembcke and Howling Moon Software.
 * This file is a small, explicit reimplementation of the body integration
 * equations only; it is not a port of Chipmunk's contact solver.
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __CUDACC__
#define AI2_PHYSICS_HD __host__ __device__
#else
#define AI2_PHYSICS_HD
#endif

#define AI2_PHYSICS_BODY_STATE_FIELDS 18u

#define AI2_PHYSICS_BODY_TYPE_DYNAMIC 0
#define AI2_PHYSICS_BODY_TYPE_KINEMATIC 1
#define AI2_PHYSICS_BODY_TYPE_STATIC 2

#define AI2_PHYSICS_VELOCITY_MODE_DEFAULT 0
#define AI2_PHYSICS_VELOCITY_MODE_NO_GRAVITY 1

#define AI2_PHYSICS_OK 0
#define AI2_PHYSICS_ERR_NULL 1
#define AI2_PHYSICS_ERR_INVALID_LANES 2
#define AI2_PHYSICS_ERR_NONFINITE 3
#define AI2_PHYSICS_ERR_INVALID_BODY_TYPE 4
#define AI2_PHYSICS_ERR_INVALID_VELOCITY_MODE 5
#define AI2_PHYSICS_ERR_INVALID_DYNAMIC_MASS 6
#define AI2_PHYSICS_ERR_INVALID_TIMESTEP 7

AI2_PHYSICS_HD static inline int ai2_physics_is_finite_or_inf(double value) {
    return !isnan(value);
}

AI2_PHYSICS_HD static inline int ai2_physics_state_validate(const double *state) {
    if (state == 0) {
        return AI2_PHYSICS_ERR_NULL;
    }

    if (!isfinite(state[0]) || !isfinite(state[1])) {
        return AI2_PHYSICS_ERR_NONFINITE;
    }
    const int body_type = (int)state[0];
    const int velocity_mode = (int)state[1];
    if (state[0] != (double)body_type) {
        return AI2_PHYSICS_ERR_INVALID_BODY_TYPE;
    }
    if (state[1] != (double)velocity_mode) {
        return AI2_PHYSICS_ERR_INVALID_VELOCITY_MODE;
    }
    if (body_type != AI2_PHYSICS_BODY_TYPE_DYNAMIC &&
        body_type != AI2_PHYSICS_BODY_TYPE_KINEMATIC &&
        body_type != AI2_PHYSICS_BODY_TYPE_STATIC) {
        return AI2_PHYSICS_ERR_INVALID_BODY_TYPE;
    }
    if (velocity_mode != AI2_PHYSICS_VELOCITY_MODE_DEFAULT &&
        velocity_mode != AI2_PHYSICS_VELOCITY_MODE_NO_GRAVITY) {
        return AI2_PHYSICS_ERR_INVALID_VELOCITY_MODE;
    }

    for (size_t i = 0u; i < AI2_PHYSICS_BODY_STATE_FIELDS; ++i) {
        if (i == 11u || i == 12u) {
            if (!ai2_physics_is_finite_or_inf(state[i])) {
                return AI2_PHYSICS_ERR_NONFINITE;
            }
        } else if (!isfinite(state[i])) {
            return AI2_PHYSICS_ERR_NONFINITE;
        }
    }

    if (body_type == AI2_PHYSICS_BODY_TYPE_DYNAMIC &&
        (!(state[11] > 0.0) || !(state[12] > 0.0) || !isfinite(state[11]) || !isfinite(state[12]))) {
        return AI2_PHYSICS_ERR_INVALID_DYNAMIC_MASS;
    }
    if (state[13] != 0.0 || state[14] != 0.0) {
        return AI2_PHYSICS_ERR_INVALID_BODY_TYPE;
    }

    return AI2_PHYSICS_OK;
}

AI2_PHYSICS_HD static inline int ai2_physics_step_body_math(
    const double *state,
    double gravity_x,
    double gravity_y,
    double damping_factor,
    double dt,
    double *out
) {
    if (state == 0 || out == 0) {
        return AI2_PHYSICS_ERR_NULL;
    }
    if (!isfinite(gravity_x) || !isfinite(gravity_y) || !isfinite(damping_factor) || damping_factor < 0.0 ||
        !isfinite(dt) || dt < 0.0) {
        return AI2_PHYSICS_ERR_INVALID_TIMESTEP;
    }

    const int valid = ai2_physics_state_validate(state);
    if (valid != AI2_PHYSICS_OK) {
        return valid;
    }

    for (size_t i = 0u; i < AI2_PHYSICS_BODY_STATE_FIELDS; ++i) {
        out[i] = state[i];
    }

    const int body_type = (int)state[0];
    const int velocity_mode = (int)state[1];
    if (body_type == AI2_PHYSICS_BODY_TYPE_STATIC) {
        return AI2_PHYSICS_OK;
    }

    const double position_x = state[2];
    const double position_y = state[3];
    const double velocity_x = state[4];
    const double velocity_y = state[5];
    const double force_x = state[6];
    const double force_y = state[7];
    const double angle = state[8];
    const double angular_velocity = state[9];
    const double torque = state[10];
    const double mass = state[11];
    const double moment = state[12];
    const double velocity_bias_x = state[15];
    const double velocity_bias_y = state[16];
    const double angular_velocity_bias = state[17];

    out[2] = position_x + (velocity_x + velocity_bias_x) * dt;
    out[3] = position_y + (velocity_y + velocity_bias_y) * dt;
    out[8] = angle + (angular_velocity + angular_velocity_bias) * dt;
    out[15] = 0.0;
    out[16] = 0.0;
    out[17] = 0.0;

    if (body_type == AI2_PHYSICS_BODY_TYPE_KINEMATIC) {
        return AI2_PHYSICS_OK;
    }

    const double effective_gravity_x =
        velocity_mode == AI2_PHYSICS_VELOCITY_MODE_NO_GRAVITY ? 0.0 : gravity_x;
    const double effective_gravity_y =
        velocity_mode == AI2_PHYSICS_VELOCITY_MODE_NO_GRAVITY ? 0.0 : gravity_y;
    out[4] = velocity_x * damping_factor + (effective_gravity_x + force_x / mass) * dt;
    out[5] = velocity_y * damping_factor + (effective_gravity_y + force_y / mass) * dt;
    out[9] = angular_velocity * damping_factor + torque / moment * dt;
    out[6] = 0.0;
    out[7] = 0.0;
    out[10] = 0.0;

    return AI2_PHYSICS_OK;
}

#endif
