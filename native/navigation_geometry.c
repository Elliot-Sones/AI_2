#include "navigation_geometry.h"

static int ai2_bb_intersects(AI2CpBB a, AI2CpBB b) {
    return (a.l <= b.r) && (b.l <= a.r) && (a.b <= b.t) && (b.b <= a.t);
}

int ai2_navigation_is_on_floor(
    const AI2NavigationGeometryFns *fns,
    const AI2NavigationFloorPointers *ptrs,
    uint8_t *out_on_floor) {
    AI2CpShapeCacheBBFn cache_bb;
    AI2CpBodyGetPositionFn body_pos;
    AI2CpBB player_bb;
    AI2CpBB other_bb;
    AI2CpVect player_pos;
    AI2CpVect other_pos;

    if (fns == 0 || ptrs == 0 || out_on_floor == 0) {
        return AI2_GEOMETRY_ERROR_NULL_ARGUMENT;
    }
    if (fns->cp_shape_cache_bb == 0 || fns->cp_body_get_position == 0) {
        return AI2_GEOMETRY_ERROR_NULL_CALLBACK;
    }
    if (ptrs->player_shape == 0 || ptrs->player_body == 0 ||
        ptrs->ground1_shape == 0 || ptrs->ground1_body == 0 ||
        ptrs->ground2_shape == 0 || ptrs->ground2_body == 0 ||
        ptrs->platform_shape == 0 || ptrs->platform_body == 0) {
        return AI2_GEOMETRY_ERROR_NULL_POINTER;
    }

    cache_bb = fns->cp_shape_cache_bb;
    body_pos = fns->cp_body_get_position;
    *out_on_floor = 0;

    /* Preserve frozen Python short-circuit order exactly:
       player cache + ground1 cache, then positions only on BB overlap. */
    player_bb = cache_bb(ptrs->player_shape);
    other_bb = cache_bb(ptrs->ground1_shape);
    if (ai2_bb_intersects(player_bb, other_bb)) {
        player_pos = body_pos(ptrs->player_body);
        other_pos = body_pos(ptrs->ground1_body);
        if (player_pos.y <= other_pos.y) {
            *out_on_floor = 1;
            return AI2_GEOMETRY_OK;
        }
    }

    player_bb = cache_bb(ptrs->player_shape);
    other_bb = cache_bb(ptrs->ground2_shape);
    if (ai2_bb_intersects(player_bb, other_bb)) {
        player_pos = body_pos(ptrs->player_body);
        other_pos = body_pos(ptrs->ground2_body);
        if (player_pos.y <= other_pos.y) {
            *out_on_floor = 1;
            return AI2_GEOMETRY_OK;
        }
    }

    player_bb = cache_bb(ptrs->player_shape);
    other_bb = cache_bb(ptrs->platform_shape);
    if (ai2_bb_intersects(player_bb, other_bb)) {
        player_pos = body_pos(ptrs->player_body);
        other_pos = body_pos(ptrs->platform_body);
        if (player_pos.y <= other_pos.y - 0.01 &&
            (other_pos.x - 1.25) <= player_pos.x &&
            player_pos.x <= (other_pos.x + 1.25)) {
            *out_on_floor = 1;
            return AI2_GEOMETRY_OK;
        }
    }

    return AI2_GEOMETRY_OK;
}
