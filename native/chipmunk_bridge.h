#ifndef AI2_NATIVE_CHIPMUNK_BRIDGE_H
#define AI2_NATIVE_CHIPMUNK_BRIDGE_H

/*
 * Exact-solver CPU bridge for the installed Pymunk 6.2.1 Chipmunk library.
 *
 * This bridge does not vendor or reimplement Chipmunk. It receives raw
 * cpSpace/cpBody pointers owned by live Pymunk objects and function
 * addresses resolved from Pymunk's already-installed _chipmunk.abi3.so.
 *
 * Safety contract:
 * - Python owns all allocation/lifetime/freeing.
 * - Call through ctypes.PyDLL from Python so the GIL is retained for scalar
 *   bridges and any CFFI/Python callbacks.
 * - Do not use OpenMP/off-GIL stepping until collision callbacks and Pymunk's
 *   deferred add/remove/post-step queues have native equivalents.
 */

#include <stddef.h>
#include <stdint.h>

typedef double ai2_cp_float;

typedef struct ai2_cp_vect {
    ai2_cp_float x;
    ai2_cp_float y;
} ai2_cp_vect;

#define AI2_CHIPMUNK_BRIDGE_OK 0
#define AI2_CHIPMUNK_BRIDGE_ERR_NULL 1
#define AI2_CHIPMUNK_BRIDGE_ERR_INVALID_COUNT 2
#define AI2_CHIPMUNK_BRIDGE_ERR_INVALID_TIMESTEP 3

int chipmunk_bridge_step_spaces(
    uintptr_t cpSpaceStep_addr,
    const uintptr_t *space_ptrs,
    size_t count,
    double dt
);

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
);

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
);

#endif
