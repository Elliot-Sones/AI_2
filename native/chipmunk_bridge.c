#include "chipmunk_bridge.h"

#include <math.h>

typedef void (*ai2_cp_space_step_fn)(void *, ai2_cp_float);
typedef ai2_cp_vect (*ai2_cp_body_get_vect_fn)(void *);
typedef void (*ai2_cp_body_set_vect_fn)(void *, ai2_cp_vect);
typedef ai2_cp_float (*ai2_cp_body_get_float_fn)(void *);
typedef void (*ai2_cp_body_set_float_fn)(void *, ai2_cp_float);

int chipmunk_bridge_step_spaces(
    uintptr_t cpSpaceStep_addr,
    const uintptr_t *space_ptrs,
    size_t count,
    double dt
) {
    if (cpSpaceStep_addr == 0u || space_ptrs == 0) {
        return AI2_CHIPMUNK_BRIDGE_ERR_NULL;
    }
    if (!isfinite(dt) || dt < 0.0) {
        return AI2_CHIPMUNK_BRIDGE_ERR_INVALID_TIMESTEP;
    }

    ai2_cp_space_step_fn cpSpaceStep = (ai2_cp_space_step_fn)cpSpaceStep_addr;
    for (size_t lane = 0u; lane < count; ++lane) {
        if (space_ptrs[lane] == 0u) {
            return AI2_CHIPMUNK_BRIDGE_ERR_NULL;
        }
    }
    for (size_t lane = 0u; lane < count; ++lane) {
        cpSpaceStep((void *)space_ptrs[lane], (ai2_cp_float)dt);
    }
    return AI2_CHIPMUNK_BRIDGE_OK;
}

int chipmunk_bridge_get_body_state(
    uintptr_t cpBodyGetPosition_addr,
    uintptr_t cpBodyGetVelocity_addr,
    uintptr_t cpBodyGetForce_addr,
    uintptr_t cpBodyGetAngle_addr,
    uintptr_t cpBodyGetAngularVelocity_addr,
    uintptr_t cpBodyGetTorque_addr,
    uintptr_t cpBodyGetMass_addr,
    uintptr_t cpBodyGetMoment_addr,
    uintptr_t body_ptr,
    double *out
) {
    if (cpBodyGetPosition_addr == 0u || cpBodyGetVelocity_addr == 0u ||
        cpBodyGetForce_addr == 0u || cpBodyGetAngle_addr == 0u ||
        cpBodyGetAngularVelocity_addr == 0u || cpBodyGetTorque_addr == 0u ||
        cpBodyGetMass_addr == 0u || cpBodyGetMoment_addr == 0u ||
        body_ptr == 0u || out == 0) {
        return AI2_CHIPMUNK_BRIDGE_ERR_NULL;
    }

    ai2_cp_body_get_vect_fn cpBodyGetPosition = (ai2_cp_body_get_vect_fn)cpBodyGetPosition_addr;
    ai2_cp_body_get_vect_fn cpBodyGetVelocity = (ai2_cp_body_get_vect_fn)cpBodyGetVelocity_addr;
    ai2_cp_body_get_vect_fn cpBodyGetForce = (ai2_cp_body_get_vect_fn)cpBodyGetForce_addr;
    ai2_cp_body_get_float_fn cpBodyGetAngle = (ai2_cp_body_get_float_fn)cpBodyGetAngle_addr;
    ai2_cp_body_get_float_fn cpBodyGetAngularVelocity =
        (ai2_cp_body_get_float_fn)cpBodyGetAngularVelocity_addr;
    ai2_cp_body_get_float_fn cpBodyGetTorque = (ai2_cp_body_get_float_fn)cpBodyGetTorque_addr;
    ai2_cp_body_get_float_fn cpBodyGetMass = (ai2_cp_body_get_float_fn)cpBodyGetMass_addr;
    ai2_cp_body_get_float_fn cpBodyGetMoment = (ai2_cp_body_get_float_fn)cpBodyGetMoment_addr;

    void *body = (void *)body_ptr;
    ai2_cp_vect position = cpBodyGetPosition(body);
    ai2_cp_vect velocity = cpBodyGetVelocity(body);
    ai2_cp_vect force = cpBodyGetForce(body);
    out[0] = position.x;
    out[1] = position.y;
    out[2] = velocity.x;
    out[3] = velocity.y;
    out[4] = force.x;
    out[5] = force.y;
    out[6] = cpBodyGetAngle(body);
    out[7] = cpBodyGetAngularVelocity(body);
    out[8] = cpBodyGetTorque(body);
    out[9] = cpBodyGetMass(body);
    out[10] = cpBodyGetMoment(body);
    return AI2_CHIPMUNK_BRIDGE_OK;
}

int chipmunk_bridge_set_body_state(
    uintptr_t cpBodySetPosition_addr,
    uintptr_t cpBodySetVelocity_addr,
    uintptr_t cpBodySetAngle_addr,
    uintptr_t cpBodySetAngularVelocity_addr,
    uintptr_t body_ptr,
    double position_x,
    double position_y,
    double velocity_x,
    double velocity_y,
    double angle,
    double angular_velocity
) {
    if (cpBodySetPosition_addr == 0u || cpBodySetVelocity_addr == 0u ||
        cpBodySetAngle_addr == 0u || cpBodySetAngularVelocity_addr == 0u ||
        body_ptr == 0u) {
        return AI2_CHIPMUNK_BRIDGE_ERR_NULL;
    }
    if (!isfinite(position_x) || !isfinite(position_y) ||
        !isfinite(velocity_x) || !isfinite(velocity_y) ||
        !isfinite(angle) || !isfinite(angular_velocity)) {
        return AI2_CHIPMUNK_BRIDGE_ERR_INVALID_TIMESTEP;
    }

    ai2_cp_body_set_vect_fn cpBodySetPosition = (ai2_cp_body_set_vect_fn)cpBodySetPosition_addr;
    ai2_cp_body_set_vect_fn cpBodySetVelocity = (ai2_cp_body_set_vect_fn)cpBodySetVelocity_addr;
    ai2_cp_body_set_float_fn cpBodySetAngle = (ai2_cp_body_set_float_fn)cpBodySetAngle_addr;
    ai2_cp_body_set_float_fn cpBodySetAngularVelocity =
        (ai2_cp_body_set_float_fn)cpBodySetAngularVelocity_addr;

    void *body = (void *)body_ptr;
    ai2_cp_vect position = {position_x, position_y};
    ai2_cp_vect velocity = {velocity_x, velocity_y};
    cpBodySetPosition(body, position);
    cpBodySetVelocity(body, velocity);
    cpBodySetAngle(body, (ai2_cp_float)angle);
    cpBodySetAngularVelocity(body, (ai2_cp_float)angular_velocity);
    return AI2_CHIPMUNK_BRIDGE_OK;
}
