#include "navigation_contact.h"

int ai2c_world_size(void) { return (int)sizeof(AI2CWorld); }
int ai2c_body_size(void) { return (int)sizeof(AI2CBody); }
int ai2c_shape_size(void) { return (int)sizeof(AI2CShape); }
int ai2c_arbiter_size(void) { return (int)sizeof(AI2CArbiter); }

unsigned long long ai2c_hash_pair_export(unsigned long long a, unsigned long long b) { return ai2c_hash_pair(a, b); }

void ai2c_coefficients_export(const AI2CWorld *w, double dt, double *bias_coef, double *damping_dt) {
    ai2c_step_coefficients(w, dt, bias_coef, damping_dt);
}

int ai2c_shape_set_verts_export(AI2CWorld *w, int shape, int count, const double *xy) {
    AI2CVec verts[AI2C_MAX_VERTS];
    int i;
    if (w == 0 || xy == 0) return AI2C_ERR_NULL;
    if (shape < 0 || shape >= AI2C_MAX_SHAPES || count < 3 || count > AI2C_MAX_VERTS) return AI2C_ERR_INVALID;
    for (i = 0; i < count; ++i) verts[i] = ai2c_v(xy[2 * i], xy[2 * i + 1]);
    ai2c_shape_set_verts(&w->shapes[shape], count, verts);
    return AI2C_OK;
}

int ai2c_shape_cache_export(AI2CWorld *w, int shape) {
    if (w == 0) return AI2C_ERR_NULL;
    if (shape < 0 || shape >= w->shape_count) return AI2C_ERR_INVALID;
    (void)ai2c_shape_cache(&w->shapes[shape], w->bodies[w->shapes[shape].body].tf);
    return AI2C_OK;
}

int ai2c_body_set_transform_export(AI2CWorld *w, int body) {
    if (w == 0) return AI2C_ERR_NULL;
    if (body < 0 || body >= w->body_count) return AI2C_ERR_INVALID;
    ai2c_body_set_transform(&w->bodies[body], w->bodies[body].p, w->bodies[body].a);
    return AI2C_OK;
}

int ai2c_world_build_trees_export(AI2CWorld *w, const int *add_order, const double *add_bb, const double *add_v, int count) {
    AI2CBB bbs[AI2C_MAX_SHAPES];
    AI2CVec vs[AI2C_MAX_SHAPES];
    int i;
    if (w == 0 || add_order == 0 || add_bb == 0 || add_v == 0) return AI2C_ERR_NULL;
    if (count < 0 || count > AI2C_MAX_SHAPES) return AI2C_ERR_INVALID;
    for (i = 0; i < count; ++i) {
        bbs[i] = ai2c_bb(add_bb[4 * i], add_bb[4 * i + 1], add_bb[4 * i + 2], add_bb[4 * i + 3]);
        vs[i] = ai2c_v(add_v[2 * i], add_v[2 * i + 1]);
    }
    return ai2c_world_build_trees(w, add_order, bbs, vs, count);
}

int ai2c_world_step_export(AI2CWorld *w, double dt, double bias_coef, double damping_dt) {
    return ai2c_world_step(w, dt, bias_coef, damping_dt);
}

int ai2c_world_add_shape_export(AI2CWorld *w, int shape) {
    if (w == 0) return AI2C_ERR_NULL;
    return ai2c_world_add_shape(w, shape);
}

int ai2c_world_remove_shape_export(AI2CWorld *w, int shape) {
    if (w == 0) return AI2C_ERR_NULL;
    return ai2c_world_remove_shape(w, shape);
}

int ai2c_world_remove_body_export(AI2CWorld *w, int body) {
    if (w == 0) return AI2C_ERR_NULL;
    return ai2c_world_remove_body(w, body);
}
