#ifndef AI2_NAVIGATION_GEOMETRY_H
#define AI2_NAVIGATION_GEOMETRY_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double x;
    double y;
} AI2CpVect;

typedef struct {
    double l;
    double b;
    double r;
    double t;
} AI2CpBB;

typedef AI2CpBB (*AI2CpShapeCacheBBFn)(void *shape);
typedef AI2CpVect (*AI2CpBodyGetPositionFn)(void *body);

typedef struct {
    AI2CpShapeCacheBBFn cp_shape_cache_bb;
    AI2CpBodyGetPositionFn cp_body_get_position;
} AI2NavigationGeometryFns;

typedef struct {
    void *player_shape;
    void *player_body;
    void *ground1_shape;
    void *ground1_body;
    void *ground2_shape;
    void *ground2_body;
    void *platform_shape;
    void *platform_body;
} AI2NavigationFloorPointers;

enum {
    AI2_GEOMETRY_OK = 0,
    AI2_GEOMETRY_ERROR_NULL_ARGUMENT = 1,
    AI2_GEOMETRY_ERROR_NULL_CALLBACK = 2,
    AI2_GEOMETRY_ERROR_NULL_POINTER = 3
};

int ai2_navigation_is_on_floor(
    const AI2NavigationGeometryFns *fns,
    const AI2NavigationFloorPointers *ptrs,
    uint8_t *out_on_floor);

#ifdef __cplusplus
}
#endif

#endif
