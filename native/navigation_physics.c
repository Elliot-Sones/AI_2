#include "navigation_physics.h"

#include <stddef.h>

int navigation_physics_step_one_cpu(
    const double *state,
    double *out,
    double gravity_x,
    double gravity_y,
    double damping,
    double dt
) {
    return ai2_physics_step_body_math(state, gravity_x, gravity_y, damping, dt, out);
}

int navigation_physics_step_batch_cpu(
    const double *states,
    double *out,
    size_t lanes,
    double gravity_x,
    double gravity_y,
    double damping,
    double dt
) {
    if (states == 0 || out == 0) {
        return AI2_PHYSICS_ERR_NULL;
    }
    if (!isfinite(gravity_x) || !isfinite(gravity_y) || !isfinite(damping) || !isfinite(dt) || dt < 0.0) {
        return AI2_PHYSICS_ERR_INVALID_TIMESTEP;
    }

    for (size_t lane = 0u; lane < lanes; ++lane) {
        const double *state = states + lane * AI2_PHYSICS_BODY_STATE_FIELDS;
        const int valid = ai2_physics_state_validate(state);
        if (valid != AI2_PHYSICS_OK) {
            return valid;
        }
    }

    for (size_t lane = 0u; lane < lanes; ++lane) {
        const double *state = states + lane * AI2_PHYSICS_BODY_STATE_FIELDS;
        double *lane_out = out + lane * AI2_PHYSICS_BODY_STATE_FIELDS;
        const int result = ai2_physics_step_body_math(state, gravity_x, gravity_y, damping, dt, lane_out);
        if (result != AI2_PHYSICS_OK) {
            return result;
        }
    }

    return AI2_PHYSICS_OK;
}
