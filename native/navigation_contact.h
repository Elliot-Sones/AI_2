#ifndef AI2_NATIVE_NAVIGATION_CONTACT_H
#define AI2_NATIVE_NAVIGATION_CONTACT_H

/*
 * AI_2 navigation contact step: a fixed-capacity re-implementation of
 * cpSpaceStep() for the frozen navigation scene (convex polygon shapes only,
 * no joints, sleeping disabled, no sensors).
 *
 * Every formula, constant and evaluation order is transcribed from
 * Chipmunk2D 7.0.3 at commit 0593976ef47fcb3957166bd342f6b2bafe4d0e44 as
 * bundled by Pymunk 6.2.1: cpSpaceStep.c, cpArbiter.c, cpCollision.c,
 * cpBody.c, cpPolyShape.c, cpBBTree.c, cpRobust.c, cpVect.h, cpBB.h.
 * Chipmunk2D is MIT licensed, Copyright (c) 2013 Scott Lembcke and Howling
 * Moon Software. No Chipmunk source file is vendored here; this is a derived,
 * restricted re-implementation carrying that attribution.
 *
 * cpFloat is double. Exactness against the real game additionally requires
 * no fused multiply-add contraction (-ffp-contract=off / -fmad=false) and the
 * host-computed step coefficients passed to ai2c_world_step().
 */

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __CUDACC__
#define AI2C_HD __host__ __device__
#else
#define AI2C_HD
#endif

#define AI2C_MAX_BODIES 12
#define AI2C_MAX_SHAPES 12
#define AI2C_MAX_VERTS 4
#define AI2C_MAX_ARBITERS 24
#define AI2C_MAX_CONTACTS 2
#define AI2C_MAX_NODES 32
#define AI2C_MAX_PAIRS 48
#define AI2C_MAX_HANDLERS 4
#define AI2C_MAX_GJK_ITERATIONS 30
#define AI2C_MAX_EPA_ITERATIONS 30
#define AI2C_MAX_HULL (AI2C_MAX_EPA_ITERATIONS + 4)
#define AI2C_HASH_COEF 3344921057ull
#define AI2C_LEAF_HASH_INITIAL_SIZE 5u

#define AI2C_BODY_DYNAMIC 0
#define AI2C_BODY_KINEMATIC 1
#define AI2C_BODY_STATIC 2

#define AI2C_VELOCITY_DEFAULT 0
#define AI2C_VELOCITY_NO_GRAVITY 1

#define AI2C_STATE_FIRST_COLLISION 0
#define AI2C_STATE_NORMAL 1
#define AI2C_STATE_IGNORE 2
#define AI2C_STATE_CACHED 3
#define AI2C_STATE_INVALIDATED 4

#define AI2C_WILDCARD_TYPE (~(unsigned long long)0)

#define AI2C_OK 0
#define AI2C_ERR_NULL 1
#define AI2C_ERR_CAPACITY 2
#define AI2C_ERR_INVALID 3
#define AI2C_ERR_NONFINITE 4

typedef struct AI2CVec { double x, y; } AI2CVec;
typedef struct AI2CBB { double l, b, r, t; } AI2CBB;

typedef struct AI2CBody {
    int type;
    int velocity_mode;
    double m, m_inv, i, i_inv;
    AI2CVec cog, p, v, f;
    double a, w, t;
    double tf[6];      /* cpTransform: a b c d tx ty */
    AI2CVec v_bias;
    double w_bias;
} AI2CBody;

typedef struct AI2CShape {
    int body;
    int count;
    AI2CVec v0[AI2C_MAX_VERTS];   /* untransformed vertices (Chipmunk stored order) */
    AI2CVec n0[AI2C_MAX_VERTS];   /* untransformed plane normals */
    AI2CVec tv[AI2C_MAX_VERTS];   /* transformed vertices */
    AI2CVec tn[AI2C_MAX_VERTS];   /* transformed normals */
    AI2CBB bb;
    double r, e, u;
    AI2CVec surface_v;
    unsigned long long hashid;
    unsigned long long collision_type;
    unsigned long long group;
    unsigned int categories, mask;
    int sensor;
} AI2CShape;

typedef struct AI2CContact {
    AI2CVec r1, r2;
    double nMass, tMass, bounce, jnAcc, jtAcc, jBias, bias;
    unsigned long long hash;
} AI2CContact;

typedef struct AI2CArbiter {
    int active;
    int a, b;               /* shape indexes, current roles */
    int body_a, body_b;
    int state;
    unsigned int stamp;
    int swapped;
    double e, u;
    AI2CVec surface_vr, n;
    int count;
    AI2CContact contacts[AI2C_MAX_CONTACTS];
} AI2CArbiter;

typedef struct AI2CNode {
    int active;
    int obj;        /* shape index for leaves, -1 for internal nodes */
    AI2CBB bb;
    int parent, A, B;
    unsigned int stamp;
    int pairs;      /* head of the leaf's pair list */
} AI2CNode;

typedef struct AI2CPair {
    int active;
    int a_leaf, a_prev, a_next;
    int b_leaf, b_prev, b_next;
    unsigned int id;
} AI2CPair;

typedef struct AI2CTree {
    int root;
    unsigned int stamp;   /* only meaningful on the dynamic tree (master) */
    int is_static;
} AI2CTree;

typedef struct AI2CWorld {
    unsigned int stamp;
    double curr_dt;
    int iterations;
    AI2CVec gravity;
    double damping, slop, bias;
    unsigned int persistence;
    int body_count;
    AI2CBody bodies[AI2C_MAX_BODIES];
    int shape_count;
    AI2CShape shapes[AI2C_MAX_SHAPES];
    int arbiter_count;
    AI2CArbiter arbiters[AI2C_MAX_ARBITERS];
    int solve_count;
    int solve[AI2C_MAX_ARBITERS];
    AI2CTree dyn_tree, static_tree;
    AI2CNode nodes[AI2C_MAX_NODES];
    AI2CPair pairs[AI2C_MAX_PAIRS];
    int leaf_of_shape[AI2C_MAX_SHAPES];
    int shape_active[AI2C_MAX_SHAPES];
    int body_active[AI2C_MAX_BODIES];
    /* Model of the dynamic tree's leaves cpHashSet: bins by hashid % size, chains newest-first. */
    unsigned int leaf_set_size;
    unsigned int leaf_set_entries;
    int leaf_set_rank[AI2C_MAX_SHAPES];     /* chain rank within the bin (lower = nearer the head) */
    int leaf_set_member[AI2C_MAX_SHAPES];   /* 1 if the shape's leaf is in the dynamic set */
    int handler_count;
    unsigned long long handler_type_a[AI2C_MAX_HANDLERS];
    unsigned long long handler_type_b[AI2C_MAX_HANDLERS];
    int handler_begin_false[AI2C_MAX_HANDLERS];
    int last_error;
    /* scratch */
    AI2CContact scratch_contacts[AI2C_MAX_CONTACTS];
} AI2CWorld;

/* ---------------------------------------------------------------- vectors */

AI2C_HD static inline AI2CVec ai2c_v(double x, double y) { AI2CVec r; r.x = x; r.y = y; return r; }
AI2C_HD static inline AI2CVec ai2c_vzero(void) { return ai2c_v(0.0, 0.0); }
AI2C_HD static inline AI2CVec ai2c_vadd(AI2CVec a, AI2CVec b) { return ai2c_v(a.x + b.x, a.y + b.y); }
AI2C_HD static inline AI2CVec ai2c_vsub(AI2CVec a, AI2CVec b) { return ai2c_v(a.x - b.x, a.y - b.y); }
AI2C_HD static inline AI2CVec ai2c_vneg(AI2CVec a) { return ai2c_v(-a.x, -a.y); }
AI2C_HD static inline AI2CVec ai2c_vmult(AI2CVec a, double s) { return ai2c_v(a.x * s, a.y * s); }
AI2C_HD static inline double ai2c_vdot(AI2CVec a, AI2CVec b) { return a.x * b.x + a.y * b.y; }
AI2C_HD static inline double ai2c_vcross(AI2CVec a, AI2CVec b) { return a.x * b.y - a.y * b.x; }
AI2C_HD static inline AI2CVec ai2c_vperp(AI2CVec a) { return ai2c_v(-a.y, a.x); }
AI2C_HD static inline AI2CVec ai2c_vrperp(AI2CVec a) { return ai2c_v(a.y, -a.x); }
AI2C_HD static inline AI2CVec ai2c_vrotate(AI2CVec a, AI2CVec b) { return ai2c_v(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
AI2C_HD static inline double ai2c_vlengthsq(AI2CVec a) { return ai2c_vdot(a, a); }
AI2C_HD static inline double ai2c_vlength(AI2CVec a) { return sqrt(ai2c_vdot(a, a)); }
AI2C_HD static inline AI2CVec ai2c_vlerp(AI2CVec a, AI2CVec b, double t) { return ai2c_vadd(ai2c_vmult(a, 1.0 - t), ai2c_vmult(b, t)); }
AI2C_HD static inline AI2CVec ai2c_vnormalize(AI2CVec a) { return ai2c_vmult(a, 1.0 / (ai2c_vlength(a) + DBL_MIN)); }
AI2C_HD static inline AI2CVec ai2c_vforangle(double a) { return ai2c_v(cos(a), sin(a)); }
AI2C_HD static inline int ai2c_veql(AI2CVec a, AI2CVec b) { return a.x == b.x && a.y == b.y; }

AI2C_HD static inline double ai2c_fmax(double a, double b) { return (a > b) ? a : b; }
AI2C_HD static inline double ai2c_fmin(double a, double b) { return (a < b) ? a : b; }
AI2C_HD static inline double ai2c_fabs(double f) { return (f < 0) ? -f : f; }
AI2C_HD static inline double ai2c_fclamp(double f, double lo, double hi) { return ai2c_fmin(ai2c_fmax(f, lo), hi); }
AI2C_HD static inline double ai2c_fclamp01(double f) { return ai2c_fmax(0.0, ai2c_fmin(f, 1.0)); }

AI2C_HD static inline unsigned long long ai2c_hash_pair(unsigned long long a, unsigned long long b) {
    return (a * AI2C_HASH_COEF) ^ (b * AI2C_HASH_COEF);
}

AI2C_HD static inline AI2CBB ai2c_bb(double l, double b, double r, double t) { AI2CBB o; o.l = l; o.b = b; o.r = r; o.t = t; return o; }
AI2C_HD static inline int ai2c_bb_intersects(AI2CBB a, AI2CBB b) { return (a.l <= b.r && b.l <= a.r && a.b <= b.t && b.b <= a.t); }
AI2C_HD static inline int ai2c_bb_contains(AI2CBB bb, AI2CBB o) { return (bb.l <= o.l && bb.r >= o.r && bb.b <= o.b && bb.t >= o.t); }
AI2C_HD static inline AI2CBB ai2c_bb_merge(AI2CBB a, AI2CBB b) { return ai2c_bb(ai2c_fmin(a.l, b.l), ai2c_fmin(a.b, b.b), ai2c_fmax(a.r, b.r), ai2c_fmax(a.t, b.t)); }
AI2C_HD static inline double ai2c_bb_area(AI2CBB bb) { return (bb.r - bb.l) * (bb.t - bb.b); }
AI2C_HD static inline double ai2c_bb_merged_area(AI2CBB a, AI2CBB b) { return (ai2c_fmax(a.r, b.r) - ai2c_fmin(a.l, b.l)) * (ai2c_fmax(a.t, b.t) - ai2c_fmin(a.b, b.b)); }
AI2C_HD static inline double ai2c_bb_proximity(AI2CBB a, AI2CBB b) { return ai2c_fabs(a.l + a.r - b.l - b.r) + ai2c_fabs(a.b + a.t - b.b - b.t); }
AI2C_HD static inline AI2CVec ai2c_bb_center(AI2CBB bb) { return ai2c_vlerp(ai2c_v(bb.l, bb.b), ai2c_v(bb.r, bb.t), 0.5); }

AI2C_HD static inline int ai2c_check_point_greater(AI2CVec a, AI2CVec b, AI2CVec c) {
    return (b.y - a.y) * (a.x + b.x - 2 * c.x) > (b.x - a.x) * (a.y + b.y - 2 * c.y);
}
AI2C_HD static inline int ai2c_check_axis(AI2CVec v0, AI2CVec v1, AI2CVec p, AI2CVec n) {
    return ai2c_vdot(p, n) <= ai2c_fmax(ai2c_vdot(v0, n), ai2c_vdot(v1, n));
}

/* ------------------------------------------------------------------ bodies */

AI2C_HD static inline void ai2c_body_set_transform(AI2CBody *body, AI2CVec p, double a) {
    AI2CVec rot = ai2c_vforangle(a);
    AI2CVec c = body->cog;
    /* cpTransformNewTranspose(rot.x, -rot.y, tx, rot.y, rot.x, ty) -> {a, b, c, d, tx, ty} */
    body->tf[0] = rot.x;
    body->tf[1] = rot.y;
    body->tf[2] = -rot.y;
    body->tf[3] = rot.x;
    body->tf[4] = p.x - (c.x * rot.x - c.y * rot.y);
    body->tf[5] = p.y - (c.x * rot.y + c.y * rot.x);
}

AI2C_HD static inline AI2CVec ai2c_transform_point(const double *t, AI2CVec p) {
    return ai2c_v(t[0] * p.x + t[2] * p.y + t[4], t[1] * p.x + t[3] * p.y + t[5]);
}
AI2C_HD static inline AI2CVec ai2c_transform_vect(const double *t, AI2CVec v) {
    return ai2c_v(t[0] * v.x + t[2] * v.y, t[1] * v.x + t[3] * v.y);
}

AI2C_HD static inline void ai2c_body_update_position(AI2CBody *body, double dt) {
    AI2CVec p = body->p = ai2c_vadd(body->p, ai2c_vmult(ai2c_vadd(body->v, body->v_bias), dt));
    double a = body->a = body->a + (body->w + body->w_bias) * dt;
    ai2c_body_set_transform(body, p, a);
    body->v_bias = ai2c_vzero();
    body->w_bias = 0.0;
}

AI2C_HD static inline void ai2c_body_update_velocity(AI2CBody *body, AI2CVec gravity, double damping, double dt) {
    if (body->type == AI2C_BODY_KINEMATIC) return;
    if (body->velocity_mode == AI2C_VELOCITY_NO_GRAVITY) gravity = ai2c_vzero();
    body->v = ai2c_vadd(ai2c_vmult(body->v, damping), ai2c_vmult(ai2c_vadd(gravity, ai2c_vmult(body->f, body->m_inv)), dt));
    body->w = body->w * damping + body->t * body->i_inv * dt;
    body->f = ai2c_vzero();
    body->t = 0.0;
}

/* ------------------------------------------------------------------ shapes */

AI2C_HD static inline void ai2c_shape_set_verts(AI2CShape *shape, int count, const AI2CVec *verts) {
    int i;
    shape->count = count;
    for (i = 0; i < count; ++i) {
        AI2CVec a = verts[(i - 1 + count) % count];
        AI2CVec b = verts[i];
        AI2CVec n = ai2c_vnormalize(ai2c_vrperp(ai2c_vsub(b, a)));
        shape->v0[i] = b;
        shape->n0[i] = n;
    }
}

AI2C_HD static inline AI2CBB ai2c_shape_cache(AI2CShape *shape, const double *tf) {
    int i;
    double l = (double)INFINITY, r = -(double)INFINITY;
    double b = (double)INFINITY, t = -(double)INFINITY;
    double radius = shape->r;
    for (i = 0; i < shape->count; ++i) {
        AI2CVec v = ai2c_transform_point(tf, shape->v0[i]);
        AI2CVec n = ai2c_transform_vect(tf, shape->n0[i]);
        shape->tv[i] = v;
        shape->tn[i] = n;
        l = ai2c_fmin(l, v.x);
        r = ai2c_fmax(r, v.x);
        b = ai2c_fmin(b, v.y);
        t = ai2c_fmax(t, v.y);
    }
    shape->bb = ai2c_bb(l - radius, b - radius, r + radius, t + radius);
    return shape->bb;
}

AI2C_HD static inline int ai2c_filter_reject(const AI2CShape *a, const AI2CShape *b) {
    return ((a->group != 0 && a->group == b->group) || (a->categories & b->mask) == 0 || (b->categories & a->mask) == 0);
}

/* --------------------------------------------------------------- collision */

typedef struct AI2CSupportPoint { AI2CVec p; unsigned int index; } AI2CSupportPoint;
typedef struct AI2CMinkowskiPoint { AI2CVec a, b, ab; unsigned int id; } AI2CMinkowskiPoint;
typedef struct AI2CEdgePoint { AI2CVec p; unsigned long long hash; } AI2CEdgePoint;
typedef struct AI2CEdge { AI2CEdgePoint a, b; double r; AI2CVec n; } AI2CEdge;
typedef struct AI2CClosestPoints { AI2CVec a, b, n; double d; unsigned int id; } AI2CClosestPoints;
typedef struct AI2CCollisionInfo { int a, b; unsigned int id; AI2CVec n; int count; AI2CContact *arr; } AI2CCollisionInfo;

AI2C_HD static inline int ai2c_poly_support_index(const AI2CShape *poly, AI2CVec n) {
    double max = -(double)INFINITY;
    int index = 0;
    int i;
    for (i = 0; i < poly->count; ++i) {
        double d = ai2c_vdot(poly->tv[i], n);
        if (d > max) { max = d; index = i; }
    }
    return index;
}

AI2C_HD static inline AI2CSupportPoint ai2c_poly_support_point(const AI2CShape *poly, AI2CVec n) {
    AI2CSupportPoint sp;
    int i = ai2c_poly_support_index(poly, n);
    sp.p = poly->tv[i];
    sp.index = (unsigned int)i;
    return sp;
}

AI2C_HD static inline AI2CSupportPoint ai2c_shape_point(const AI2CShape *poly, unsigned int i) {
    AI2CSupportPoint sp;
    unsigned int index = ((int)i < poly->count) ? i : 0u;
    sp.p = poly->tv[index];
    sp.index = index;
    return sp;
}

AI2C_HD static inline AI2CMinkowskiPoint ai2c_minkowski(AI2CSupportPoint a, AI2CSupportPoint b) {
    AI2CMinkowskiPoint m;
    m.a = a.p;
    m.b = b.p;
    m.ab = ai2c_vsub(b.p, a.p);
    m.id = ((a.index & 0xFFu) << 8) | (b.index & 0xFFu);
    return m;
}

AI2C_HD static inline AI2CMinkowskiPoint ai2c_support(const AI2CShape *s1, const AI2CShape *s2, AI2CVec n) {
    AI2CSupportPoint a = ai2c_poly_support_point(s1, ai2c_vneg(n));
    AI2CSupportPoint b = ai2c_poly_support_point(s2, n);
    return ai2c_minkowski(a, b);
}

AI2C_HD static inline AI2CEdge ai2c_support_edge(const AI2CShape *poly, AI2CVec n) {
    AI2CEdge edge;
    int count = poly->count;
    int i1 = ai2c_poly_support_index(poly, n);
    int i0 = (i1 - 1 + count) % count;
    int i2 = (i1 + 1) % count;
    unsigned long long hashid = poly->hashid;
    if (ai2c_vdot(n, poly->tn[i1]) > ai2c_vdot(n, poly->tn[i2])) {
        edge.a.p = poly->tv[i0]; edge.a.hash = ai2c_hash_pair(hashid, (unsigned long long)i0);
        edge.b.p = poly->tv[i1]; edge.b.hash = ai2c_hash_pair(hashid, (unsigned long long)i1);
        edge.r = poly->r; edge.n = poly->tn[i1];
    } else {
        edge.a.p = poly->tv[i1]; edge.a.hash = ai2c_hash_pair(hashid, (unsigned long long)i1);
        edge.b.p = poly->tv[i2]; edge.b.hash = ai2c_hash_pair(hashid, (unsigned long long)i2);
        edge.r = poly->r; edge.n = poly->tn[i2];
    }
    return edge;
}

AI2C_HD static inline double ai2c_closest_t(AI2CVec a, AI2CVec b) {
    AI2CVec delta = ai2c_vsub(b, a);
    double d2 = ai2c_vlengthsq(delta);
    return d2 ? -ai2c_fclamp(ai2c_vdot(delta, ai2c_vadd(a, b)) / d2, -1.0, 1.0) : 0.0;
}

AI2C_HD static inline AI2CVec ai2c_lerp_t(AI2CVec a, AI2CVec b, double t) {
    double ht = 0.5 * t;
    return ai2c_vadd(ai2c_vmult(a, 0.5 - ht), ai2c_vmult(b, 0.5 + ht));
}

AI2C_HD static inline AI2CClosestPoints ai2c_closest_points_new(AI2CMinkowskiPoint v0, AI2CMinkowskiPoint v1) {
    AI2CClosestPoints points;
    double t = ai2c_closest_t(v0.ab, v1.ab);
    AI2CVec p = ai2c_lerp_t(v0.ab, v1.ab, t);
    AI2CVec pa = ai2c_lerp_t(v0.a, v1.a, t);
    AI2CVec pb = ai2c_lerp_t(v0.b, v1.b, t);
    unsigned int id = ((v0.id & 0xFFFFu) << 16) | (v1.id & 0xFFFFu);
    AI2CVec delta = ai2c_vsub(v1.ab, v0.ab);
    AI2CVec n = ai2c_vnormalize(ai2c_vrperp(delta));
    double d = ai2c_vdot(n, p);
    if (d <= 0.0 || (-1.0 < t && t < 1.0)) {
        points.a = pa; points.b = pb; points.n = n; points.d = d; points.id = id;
    } else {
        double d2 = ai2c_vlength(p);
        AI2CVec n2 = ai2c_vmult(p, 1.0 / (d2 + DBL_MIN));
        points.a = pa; points.b = pb; points.n = n2; points.d = d2; points.id = id;
    }
    return points;
}

AI2C_HD static inline double ai2c_closest_dist(AI2CVec v0, AI2CVec v1) {
    return ai2c_vlengthsq(ai2c_lerp_t(v0, v1, ai2c_closest_t(v0, v1)));
}

/* EPARecurse, iterative form with two hull buffers. */
AI2C_HD static inline AI2CClosestPoints ai2c_epa(const AI2CShape *s1, const AI2CShape *s2,
                                                 AI2CMinkowskiPoint a0, AI2CMinkowskiPoint a1, AI2CMinkowskiPoint a2) {
    AI2CMinkowskiPoint hull_a[AI2C_MAX_HULL];
    AI2CMinkowskiPoint hull_b[AI2C_MAX_HULL];
    AI2CMinkowskiPoint *hull = hull_a;
    AI2CMinkowskiPoint *hull2 = hull_b;
    int count = 3;
    int iteration = 1;
    hull[0] = a0; hull[1] = a1; hull[2] = a2;
    for (;;) {
        int mini = 0;
        double minDist = (double)INFINITY;
        int i, j;
        AI2CMinkowskiPoint v0, v1, p;
        int duplicate;
        for (j = 0, i = count - 1; j < count; i = j, j++) {
            double d = ai2c_closest_dist(hull[i].ab, hull[j].ab);
            if (d < minDist) { minDist = d; mini = i; }
        }
        v0 = hull[mini];
        v1 = hull[(mini + 1) % count];
        p = ai2c_support(s1, s2, ai2c_vperp(ai2c_vsub(v1.ab, v0.ab)));
        duplicate = (p.id == v0.id || p.id == v1.id);
        if (!duplicate && ai2c_check_point_greater(v0.ab, v1.ab, p.ab) && iteration < AI2C_MAX_EPA_ITERATIONS) {
            int count2 = 1;
            AI2CMinkowskiPoint *tmp;
            hull2[0] = p;
            for (i = 0; i < count; ++i) {
                int index = (mini + 1 + i) % count;
                AI2CVec h0 = hull2[count2 - 1].ab;
                AI2CVec h1 = hull[index].ab;
                AI2CVec h2 = (i + 1 < count ? hull[(index + 1) % count] : p).ab;
                if (ai2c_check_point_greater(h0, h2, h1)) {
                    hull2[count2] = hull[index];
                    count2++;
                }
            }
            tmp = hull; hull = hull2; hull2 = tmp;
            count = count2;
            iteration++;
        } else {
            return ai2c_closest_points_new(v0, v1);
        }
    }
}

/* GJKRecurse, iterative form. */
AI2C_HD static inline AI2CClosestPoints ai2c_gjk_recurse(const AI2CShape *s1, const AI2CShape *s2,
                                                         AI2CMinkowskiPoint v0, AI2CMinkowskiPoint v1) {
    int iteration = 1;
    for (;;) {
        double t;
        AI2CVec n;
        AI2CMinkowskiPoint p;
        if (iteration > AI2C_MAX_GJK_ITERATIONS) {
            return ai2c_closest_points_new(v0, v1);
        }
        if (ai2c_check_point_greater(v1.ab, v0.ab, ai2c_vzero())) {
            AI2CMinkowskiPoint tmp = v0; v0 = v1; v1 = tmp;
            continue;
        }
        t = ai2c_closest_t(v0.ab, v1.ab);
        n = (-1.0 < t && t < 1.0 ? ai2c_vperp(ai2c_vsub(v1.ab, v0.ab)) : ai2c_vneg(ai2c_lerp_t(v0.ab, v1.ab, t)));
        p = ai2c_support(s1, s2, n);
        if (ai2c_check_point_greater(p.ab, v0.ab, ai2c_vzero()) && ai2c_check_point_greater(v1.ab, p.ab, ai2c_vzero())) {
            return ai2c_epa(s1, s2, v0, p, v1);
        } else {
            if (ai2c_check_axis(v0.ab, v1.ab, p.ab, n)) {
                return ai2c_closest_points_new(v0, v1);
            } else {
                if (ai2c_closest_dist(v0.ab, p.ab) < ai2c_closest_dist(p.ab, v1.ab)) {
                    v1 = p;
                } else {
                    v0 = p;
                }
                iteration++;
            }
        }
    }
}

AI2C_HD static inline AI2CClosestPoints ai2c_gjk(const AI2CShape *s1, const AI2CShape *s2, unsigned int *id) {
    AI2CMinkowskiPoint v0, v1;
    AI2CClosestPoints points;
    if (*id) {
        v0 = ai2c_minkowski(ai2c_shape_point(s1, (*id >> 24) & 0xFFu), ai2c_shape_point(s2, (*id >> 16) & 0xFFu));
        v1 = ai2c_minkowski(ai2c_shape_point(s1, (*id >> 8) & 0xFFu), ai2c_shape_point(s2, (*id) & 0xFFu));
    } else {
        AI2CVec axis = ai2c_vperp(ai2c_vsub(ai2c_bb_center(s1->bb), ai2c_bb_center(s2->bb)));
        v0 = ai2c_support(s1, s2, axis);
        v1 = ai2c_support(s1, s2, ai2c_vneg(axis));
    }
    points = ai2c_gjk_recurse(s1, s2, v0, v1);
    *id = points.id;
    return points;
}

AI2C_HD static inline void ai2c_info_push_contact(AI2CCollisionInfo *info, AI2CVec p1, AI2CVec p2, unsigned long long hash) {
    AI2CContact *con = &info->arr[info->count];
    con->r1 = p1;
    con->r2 = p2;
    con->hash = hash;
    info->count++;
}

AI2C_HD static inline void ai2c_contact_points(AI2CEdge e1, AI2CEdge e2, AI2CClosestPoints points, AI2CCollisionInfo *info) {
    double mindist = e1.r + e2.r;
    if (points.d <= mindist) {
        AI2CVec n = info->n = points.n;
        double d_e1_a = ai2c_vcross(e1.a.p, n);
        double d_e1_b = ai2c_vcross(e1.b.p, n);
        double d_e2_a = ai2c_vcross(e2.a.p, n);
        double d_e2_b = ai2c_vcross(e2.b.p, n);
        double e1_denom = 1.0 / (d_e1_b - d_e1_a + DBL_MIN);
        double e2_denom = 1.0 / (d_e2_b - d_e2_a + DBL_MIN);
        {
            AI2CVec p1 = ai2c_vadd(ai2c_vmult(n, e1.r), ai2c_vlerp(e1.a.p, e1.b.p, ai2c_fclamp01((d_e2_b - d_e1_a) * e1_denom)));
            AI2CVec p2 = ai2c_vadd(ai2c_vmult(n, -e2.r), ai2c_vlerp(e2.a.p, e2.b.p, ai2c_fclamp01((d_e1_a - d_e2_a) * e2_denom)));
            double dist = ai2c_vdot(ai2c_vsub(p2, p1), n);
            if (dist <= 0.0) {
                ai2c_info_push_contact(info, p1, p2, ai2c_hash_pair(e1.a.hash, e2.b.hash));
            }
        }
        {
            AI2CVec p1 = ai2c_vadd(ai2c_vmult(n, e1.r), ai2c_vlerp(e1.a.p, e1.b.p, ai2c_fclamp01((d_e2_a - d_e1_a) * e1_denom)));
            AI2CVec p2 = ai2c_vadd(ai2c_vmult(n, -e2.r), ai2c_vlerp(e2.a.p, e2.b.p, ai2c_fclamp01((d_e1_b - d_e2_a) * e2_denom)));
            double dist = ai2c_vdot(ai2c_vsub(p2, p1), n);
            if (dist <= 0.0) {
                ai2c_info_push_contact(info, p1, p2, ai2c_hash_pair(e1.b.hash, e2.a.hash));
            }
        }
    }
}

AI2C_HD static inline AI2CCollisionInfo ai2c_collide(const AI2CWorld *w, int a, int b, unsigned int id, AI2CContact *contacts) {
    AI2CCollisionInfo info;
    const AI2CShape *poly1 = &w->shapes[a];
    const AI2CShape *poly2 = &w->shapes[b];
    AI2CClosestPoints points;
    info.a = a; info.b = b; info.id = id; info.n = ai2c_vzero(); info.count = 0; info.arr = contacts;
    /* Both shapes are polygons: no type-order swap. */
    points = ai2c_gjk(poly1, poly2, &info.id);
    if (points.d - poly1->r - poly2->r <= 0.0) {
        ai2c_contact_points(ai2c_support_edge(poly1, points.n), ai2c_support_edge(poly2, ai2c_vneg(points.n)), points, &info);
    }
    return info;
}

/* ---------------------------------------------------------------- arbiters */

AI2C_HD static inline AI2CVec ai2c_relative_velocity(const AI2CBody *a, const AI2CBody *b, AI2CVec r1, AI2CVec r2) {
    AI2CVec v1_sum = ai2c_vadd(a->v, ai2c_vmult(ai2c_vperp(r1), a->w));
    AI2CVec v2_sum = ai2c_vadd(b->v, ai2c_vmult(ai2c_vperp(r2), b->w));
    return ai2c_vsub(v2_sum, v1_sum);
}

AI2C_HD static inline double ai2c_normal_relative_velocity(const AI2CBody *a, const AI2CBody *b, AI2CVec r1, AI2CVec r2, AI2CVec n) {
    return ai2c_vdot(ai2c_relative_velocity(a, b, r1, r2), n);
}

AI2C_HD static inline void ai2c_apply_impulse(AI2CBody *body, AI2CVec j, AI2CVec r) {
    body->v = ai2c_vadd(body->v, ai2c_vmult(j, body->m_inv));
    body->w += body->i_inv * ai2c_vcross(r, j);
}

AI2C_HD static inline void ai2c_apply_impulses(AI2CBody *a, AI2CBody *b, AI2CVec r1, AI2CVec r2, AI2CVec j) {
    ai2c_apply_impulse(a, ai2c_vneg(j), r1);
    ai2c_apply_impulse(b, j, r2);
}

AI2C_HD static inline void ai2c_apply_bias_impulse(AI2CBody *body, AI2CVec j, AI2CVec r) {
    body->v_bias = ai2c_vadd(body->v_bias, ai2c_vmult(j, body->m_inv));
    body->w_bias += body->i_inv * ai2c_vcross(r, j);
}

AI2C_HD static inline void ai2c_apply_bias_impulses(AI2CBody *a, AI2CBody *b, AI2CVec r1, AI2CVec r2, AI2CVec j) {
    ai2c_apply_bias_impulse(a, ai2c_vneg(j), r1);
    ai2c_apply_bias_impulse(b, j, r2);
}

AI2C_HD static inline double ai2c_k_scalar_body(const AI2CBody *body, AI2CVec r, AI2CVec n) {
    double rcn = ai2c_vcross(r, n);
    return body->m_inv + body->i_inv * rcn * rcn;
}

AI2C_HD static inline double ai2c_k_scalar(const AI2CBody *a, const AI2CBody *b, AI2CVec r1, AI2CVec r2, AI2CVec n) {
    return ai2c_k_scalar_body(a, r1, n) + ai2c_k_scalar_body(b, r2, n);
}

AI2C_HD static inline int ai2c_find_handler(const AI2CWorld *w, unsigned long long ta, unsigned long long tb) {
    int h;
    for (h = 0; h < w->handler_count; ++h) {
        if ((w->handler_type_a[h] == ta && w->handler_type_b[h] == tb) || (w->handler_type_b[h] == ta && w->handler_type_a[h] == tb)) return h;
    }
    return -1;
}

AI2C_HD static inline void ai2c_arbiter_update(AI2CWorld *w, AI2CArbiter *arb, AI2CCollisionInfo *info) {
    const AI2CShape *a = &w->shapes[info->a];
    const AI2CShape *b = &w->shapes[info->b];
    AI2CContact old[AI2C_MAX_CONTACTS];
    int old_count = arb->count;
    int i, j, h;
    unsigned long long typeA, typeB;
    AI2CVec surface_vr;
    for (i = 0; i < old_count; ++i) old[i] = arb->contacts[i];
    arb->a = info->a; arb->body_a = a->body;
    arb->b = info->b; arb->body_b = b->body;
    for (i = 0; i < info->count; ++i) {
        AI2CContact *con = &info->arr[i];
        con->r1 = ai2c_vsub(con->r1, w->bodies[a->body].p);
        con->r2 = ai2c_vsub(con->r2, w->bodies[b->body].p);
        con->jnAcc = con->jtAcc = 0.0;
        for (j = 0; j < old_count; ++j) {
            if (con->hash == old[j].hash) {
                con->jnAcc = old[j].jnAcc;
                con->jtAcc = old[j].jtAcc;
            }
        }
    }
    for (i = 0; i < info->count; ++i) arb->contacts[i] = info->arr[i];
    arb->count = info->count;
    arb->n = info->n;
    arb->e = a->e * b->e;
    arb->u = a->u * b->u;
    surface_vr = ai2c_vsub(b->surface_v, a->surface_v);
    arb->surface_vr = ai2c_vsub(surface_vr, ai2c_vmult(info->n, ai2c_vdot(surface_vr, info->n)));
    typeA = a->collision_type; typeB = b->collision_type;
    h = ai2c_find_handler(w, typeA, typeB);
    arb->swapped = (h >= 0) ? (typeA != w->handler_type_a[h] && w->handler_type_a[h] != AI2C_WILDCARD_TYPE) : 0;
    if (arb->state == AI2C_STATE_CACHED) arb->state = AI2C_STATE_FIRST_COLLISION;
}

AI2C_HD static inline void ai2c_arbiter_prestep(AI2CWorld *w, AI2CArbiter *arb, double dt, double slop, double bias) {
    AI2CBody *a = &w->bodies[arb->body_a];
    AI2CBody *b = &w->bodies[arb->body_b];
    AI2CVec n = arb->n;
    AI2CVec body_delta = ai2c_vsub(b->p, a->p);
    int i;
    for (i = 0; i < arb->count; ++i) {
        AI2CContact *con = &arb->contacts[i];
        double dist;
        con->nMass = 1.0 / ai2c_k_scalar(a, b, con->r1, con->r2, n);
        con->tMass = 1.0 / ai2c_k_scalar(a, b, con->r1, con->r2, ai2c_vperp(n));
        dist = ai2c_vdot(ai2c_vadd(ai2c_vsub(con->r2, con->r1), body_delta), n);
        con->bias = -bias * ai2c_fmin(0.0, dist + slop) / dt;
        con->jBias = 0.0;
        con->bounce = ai2c_normal_relative_velocity(a, b, con->r1, con->r2, n) * arb->e;
    }
}

AI2C_HD static inline void ai2c_arbiter_apply_cached_impulse(AI2CWorld *w, AI2CArbiter *arb, double dt_coef) {
    AI2CBody *a;
    AI2CBody *b;
    AI2CVec n;
    int i;
    if (arb->state == AI2C_STATE_FIRST_COLLISION) return;
    a = &w->bodies[arb->body_a];
    b = &w->bodies[arb->body_b];
    n = arb->n;
    for (i = 0; i < arb->count; ++i) {
        AI2CContact *con = &arb->contacts[i];
        AI2CVec j = ai2c_vrotate(n, ai2c_v(con->jnAcc, con->jtAcc));
        ai2c_apply_impulses(a, b, con->r1, con->r2, ai2c_vmult(j, dt_coef));
    }
}

AI2C_HD static inline void ai2c_arbiter_apply_impulse(AI2CWorld *w, AI2CArbiter *arb) {
    AI2CBody *a = &w->bodies[arb->body_a];
    AI2CBody *b = &w->bodies[arb->body_b];
    AI2CVec n = arb->n;
    AI2CVec surface_vr = arb->surface_vr;
    double friction = arb->u;
    int i;
    for (i = 0; i < arb->count; ++i) {
        AI2CContact *con = &arb->contacts[i];
        double nMass = con->nMass;
        AI2CVec r1 = con->r1;
        AI2CVec r2 = con->r2;
        AI2CVec vb1 = ai2c_vadd(a->v_bias, ai2c_vmult(ai2c_vperp(r1), a->w_bias));
        AI2CVec vb2 = ai2c_vadd(b->v_bias, ai2c_vmult(ai2c_vperp(r2), b->w_bias));
        AI2CVec vr = ai2c_vadd(ai2c_relative_velocity(a, b, r1, r2), surface_vr);
        double vbn = ai2c_vdot(ai2c_vsub(vb2, vb1), n);
        double vrn = ai2c_vdot(vr, n);
        double vrt = ai2c_vdot(vr, ai2c_vperp(n));
        double jbn = (con->bias - vbn) * nMass;
        double jbnOld = con->jBias;
        double jn, jnOld, jtMax, jt, jtOld;
        con->jBias = ai2c_fmax(jbnOld + jbn, 0.0);
        jn = -(con->bounce + vrn) * nMass;
        jnOld = con->jnAcc;
        con->jnAcc = ai2c_fmax(jnOld + jn, 0.0);
        jtMax = friction * con->jnAcc;
        jt = -vrt * con->tMass;
        jtOld = con->jtAcc;
        con->jtAcc = ai2c_fclamp(jtOld + jt, -jtMax, jtMax);
        ai2c_apply_bias_impulses(a, b, r1, r2, ai2c_vmult(n, con->jBias - jbnOld));
        ai2c_apply_impulses(a, b, r1, r2, ai2c_vrotate(n, ai2c_v(con->jnAcc - jnOld, con->jtAcc - jtOld)));
    }
}

/* ------------------------------------------------------------ broadphase */

AI2C_HD static inline int ai2c_node_is_leaf(const AI2CWorld *w, int node) { return w->nodes[node].obj >= 0; }

AI2C_HD static inline int ai2c_node_alloc(AI2CWorld *w) {
    int i;
    for (i = 0; i < AI2C_MAX_NODES; ++i) {
        if (!w->nodes[i].active) {
            w->nodes[i].active = 1;
            w->nodes[i].obj = -1;
            w->nodes[i].parent = w->nodes[i].A = w->nodes[i].B = -1;
            w->nodes[i].stamp = 0;
            w->nodes[i].pairs = -1;
            return i;
        }
    }
    w->last_error = AI2C_ERR_CAPACITY;
    return -1;
}

AI2C_HD static inline void ai2c_node_recycle(AI2CWorld *w, int node) { w->nodes[node].active = 0; }

AI2C_HD static inline int ai2c_pair_alloc(AI2CWorld *w) {
    int i;
    for (i = 0; i < AI2C_MAX_PAIRS; ++i) {
        if (!w->pairs[i].active) { w->pairs[i].active = 1; return i; }
    }
    w->last_error = AI2C_ERR_CAPACITY;
    return -1;
}

AI2C_HD static inline AI2CBB ai2c_tree_get_bb(const AI2CWorld *w, const AI2CTree *tree, int shape) {
    AI2CBB bb = w->shapes[shape].bb;
    if (!tree->is_static) {
        /* Chipmunk writes these as the float literal 0.1f, which promotes to 0.10000000149011612. */
        double coef = (double)0.1f;
        double x = (bb.r - bb.l) * coef;
        double y = (bb.t - bb.b) * coef;
        AI2CVec v = ai2c_vmult(w->bodies[w->shapes[shape].body].v, (double)0.1f);
        return ai2c_bb(bb.l + ai2c_fmin(-x, v.x), bb.b + ai2c_fmin(-y, v.y), bb.r + ai2c_fmax(x, v.x), bb.t + ai2c_fmax(y, v.y));
    }
    return bb;
}

AI2C_HD static inline void ai2c_node_set_a(AI2CWorld *w, int node, int value) { w->nodes[node].A = value; w->nodes[value].parent = node; }
AI2C_HD static inline void ai2c_node_set_b(AI2CWorld *w, int node, int value) { w->nodes[node].B = value; w->nodes[value].parent = node; }

AI2C_HD static inline int ai2c_node_new(AI2CWorld *w, int a, int b) {
    int node = ai2c_node_alloc(w);
    if (node < 0) return -1;
    w->nodes[node].obj = -1;
    w->nodes[node].bb = ai2c_bb_merge(w->nodes[a].bb, w->nodes[b].bb);
    w->nodes[node].parent = -1;
    ai2c_node_set_a(w, node, a);
    ai2c_node_set_b(w, node, b);
    return node;
}

AI2C_HD static inline int ai2c_node_other(const AI2CWorld *w, int node, int child) {
    return (w->nodes[node].A == child ? w->nodes[node].B : w->nodes[node].A);
}

AI2C_HD static inline void ai2c_node_replace_child(AI2CWorld *w, int parent, int child, int value) {
    int node;
    if (w->nodes[parent].A == child) {
        ai2c_node_recycle(w, w->nodes[parent].A);
        ai2c_node_set_a(w, parent, value);
    } else {
        ai2c_node_recycle(w, w->nodes[parent].B);
        ai2c_node_set_b(w, parent, value);
    }
    for (node = parent; node >= 0; node = w->nodes[node].parent) {
        w->nodes[node].bb = ai2c_bb_merge(w->nodes[w->nodes[node].A].bb, w->nodes[w->nodes[node].B].bb);
    }
}

AI2C_HD static inline int ai2c_subtree_insert(AI2CWorld *w, int subtree, int leaf) {
    if (subtree < 0) {
        return leaf;
    } else if (ai2c_node_is_leaf(w, subtree)) {
        return ai2c_node_new(w, leaf, subtree);
    } else {
        int A = w->nodes[subtree].A, B = w->nodes[subtree].B;
        double cost_a = ai2c_bb_area(w->nodes[B].bb) + ai2c_bb_merged_area(w->nodes[A].bb, w->nodes[leaf].bb);
        double cost_b = ai2c_bb_area(w->nodes[A].bb) + ai2c_bb_merged_area(w->nodes[B].bb, w->nodes[leaf].bb);
        if (cost_a == cost_b) {
            cost_a = ai2c_bb_proximity(w->nodes[A].bb, w->nodes[leaf].bb);
            cost_b = ai2c_bb_proximity(w->nodes[B].bb, w->nodes[leaf].bb);
        }
        if (cost_b < cost_a) {
            ai2c_node_set_b(w, subtree, ai2c_subtree_insert(w, B, leaf));
        } else {
            ai2c_node_set_a(w, subtree, ai2c_subtree_insert(w, A, leaf));
        }
        w->nodes[subtree].bb = ai2c_bb_merge(w->nodes[subtree].bb, w->nodes[leaf].bb);
        return subtree;
    }
}

AI2C_HD static inline int ai2c_subtree_remove(AI2CWorld *w, int subtree, int leaf) {
    if (leaf == subtree) {
        return -1;
    } else {
        int parent = w->nodes[leaf].parent;
        if (parent == subtree) {
            int other = ai2c_node_other(w, subtree, leaf);
            w->nodes[other].parent = w->nodes[subtree].parent;
            ai2c_node_recycle(w, subtree);
            return other;
        } else {
            ai2c_node_replace_child(w, w->nodes[parent].parent, parent, ai2c_node_other(w, parent, leaf));
            return subtree;
        }
    }
}

AI2C_HD static inline void ai2c_thread_unlink(AI2CWorld *w, int pair, int side_b) {
    int leaf = side_b ? w->pairs[pair].b_leaf : w->pairs[pair].a_leaf;
    int next = side_b ? w->pairs[pair].b_next : w->pairs[pair].a_next;
    int prev = side_b ? w->pairs[pair].b_prev : w->pairs[pair].a_prev;
    if (next >= 0) {
        if (w->pairs[next].a_leaf == leaf) w->pairs[next].a_prev = prev; else w->pairs[next].b_prev = prev;
    }
    if (prev >= 0) {
        if (w->pairs[prev].a_leaf == leaf) w->pairs[prev].a_next = next; else w->pairs[prev].b_next = next;
    } else {
        w->nodes[leaf].pairs = next;
    }
}

AI2C_HD static inline void ai2c_pairs_clear(AI2CWorld *w, int leaf) {
    int pair = w->nodes[leaf].pairs;
    w->nodes[leaf].pairs = -1;
    while (pair >= 0) {
        if (w->pairs[pair].a_leaf == leaf) {
            int next = w->pairs[pair].a_next;
            ai2c_thread_unlink(w, pair, 1);
            w->pairs[pair].active = 0;
            pair = next;
        } else {
            int next = w->pairs[pair].b_next;
            ai2c_thread_unlink(w, pair, 0);
            w->pairs[pair].active = 0;
            pair = next;
        }
    }
}

AI2C_HD static inline void ai2c_pair_insert(AI2CWorld *w, int a, int b) {
    int nextA = w->nodes[a].pairs, nextB = w->nodes[b].pairs;
    int pair = ai2c_pair_alloc(w);
    if (pair < 0) return;
    w->pairs[pair].a_prev = -1; w->pairs[pair].a_leaf = a; w->pairs[pair].a_next = nextA;
    w->pairs[pair].b_prev = -1; w->pairs[pair].b_leaf = b; w->pairs[pair].b_next = nextB;
    w->pairs[pair].id = 0;
    w->nodes[a].pairs = w->nodes[b].pairs = pair;
    if (nextA >= 0) {
        if (w->pairs[nextA].a_leaf == a) w->pairs[nextA].a_prev = pair; else w->pairs[nextA].b_prev = pair;
    }
    if (nextB >= 0) {
        if (w->pairs[nextB].a_leaf == b) w->pairs[nextB].a_prev = pair; else w->pairs[nextB].b_prev = pair;
    }
}

/* Query callback (cpSpaceCollideShapes). Returns the new collision id. */
AI2C_HD static inline unsigned int ai2c_collide_shapes(AI2CWorld *w, int a, int b, unsigned int id);
AI2C_HD static inline void ai2c_leaf_set_insert(AI2CWorld *w, int shape);
AI2C_HD static inline void ai2c_leaf_set_remove(AI2CWorld *w, int shape);

typedef struct AI2CMarkContext { int tree_is_dynamic; int static_root; int use_func; } AI2CMarkContext;

AI2C_HD static inline void ai2c_mark_leaf_query(AI2CWorld *w, int subtree, int leaf, int left, const AI2CMarkContext *ctx) {
    if (ai2c_bb_intersects(w->nodes[leaf].bb, w->nodes[subtree].bb)) {
        if (ai2c_node_is_leaf(w, subtree)) {
            if (left) {
                ai2c_pair_insert(w, leaf, subtree);
            } else {
                if (w->nodes[subtree].stamp < w->nodes[leaf].stamp) ai2c_pair_insert(w, subtree, leaf);
                if (ctx->use_func) (void)ai2c_collide_shapes(w, w->nodes[leaf].obj, w->nodes[subtree].obj, 0u);
            }
        } else {
            ai2c_mark_leaf_query(w, w->nodes[subtree].A, leaf, left, ctx);
            ai2c_mark_leaf_query(w, w->nodes[subtree].B, leaf, left, ctx);
        }
    }
}

AI2C_HD static inline void ai2c_mark_leaf(AI2CWorld *w, int leaf, const AI2CMarkContext *ctx) {
    if (w->nodes[leaf].stamp == w->dyn_tree.stamp) {
        int node;
        if (ctx->static_root >= 0) ai2c_mark_leaf_query(w, ctx->static_root, leaf, 0, ctx);
        for (node = leaf; w->nodes[node].parent >= 0; node = w->nodes[node].parent) {
            int parent = w->nodes[node].parent;
            if (node == w->nodes[parent].A) {
                ai2c_mark_leaf_query(w, w->nodes[parent].B, leaf, 1, ctx);
            } else {
                ai2c_mark_leaf_query(w, w->nodes[parent].A, leaf, 0, ctx);
            }
        }
    } else {
        int pair = w->nodes[leaf].pairs;
        while (pair >= 0) {
            if (leaf == w->pairs[pair].b_leaf) {
                if (ctx->use_func) {
                    w->pairs[pair].id = ai2c_collide_shapes(w, w->nodes[w->pairs[pair].a_leaf].obj, w->nodes[leaf].obj, w->pairs[pair].id);
                }
                pair = w->pairs[pair].b_next;
            } else {
                pair = w->pairs[pair].a_next;
            }
        }
    }
}

AI2C_HD static inline void ai2c_mark_subtree(AI2CWorld *w, int subtree, const AI2CMarkContext *ctx) {
    if (ai2c_node_is_leaf(w, subtree)) {
        ai2c_mark_leaf(w, subtree, ctx);
    } else {
        ai2c_mark_subtree(w, w->nodes[subtree].A, ctx);
        ai2c_mark_subtree(w, w->nodes[subtree].B, ctx);
    }
}

AI2C_HD static inline int ai2c_leaf_update(AI2CWorld *w, AI2CTree *tree, int leaf) {
    int shape = w->nodes[leaf].obj;
    AI2CBB bb = w->shapes[shape].bb;
    if (!ai2c_bb_contains(w->nodes[leaf].bb, bb)) {
        int root = tree->root;
        w->nodes[leaf].bb = ai2c_tree_get_bb(w, tree, shape);
        root = ai2c_subtree_remove(w, root, leaf);
        tree->root = ai2c_subtree_insert(w, root, leaf);
        ai2c_pairs_clear(w, leaf);
        w->nodes[leaf].stamp = w->dyn_tree.stamp;
        return 1;
    }
    return 0;
}

/* cpBBTreeInsert with LeafAddPairs and IncrementStamp. */
AI2C_HD static inline int ai2c_tree_insert(AI2CWorld *w, AI2CTree *tree, int shape) {
    int leaf = ai2c_node_alloc(w);
    AI2CMarkContext ctx;
    if (leaf < 0) return AI2C_ERR_CAPACITY;
    w->nodes[leaf].obj = shape;
    w->nodes[leaf].bb = ai2c_tree_get_bb(w, tree, shape);
    w->nodes[leaf].parent = -1;
    w->nodes[leaf].stamp = 0;
    w->nodes[leaf].pairs = -1;
    w->leaf_of_shape[shape] = leaf;
    if (!tree->is_static) ai2c_leaf_set_insert(w, shape);
    tree->root = ai2c_subtree_insert(w, tree->root, leaf);
    w->nodes[leaf].stamp = w->dyn_tree.stamp;
    if (tree->is_static) {
        if (w->dyn_tree.root >= 0) {
            ctx.tree_is_dynamic = 1; ctx.static_root = -1; ctx.use_func = 0;
            ai2c_mark_leaf_query(w, w->dyn_tree.root, leaf, 1, &ctx);
        }
    } else {
        ctx.tree_is_dynamic = 1; ctx.static_root = w->static_tree.root; ctx.use_func = 0;
        ai2c_mark_leaf(w, leaf, &ctx);
    }
    w->dyn_tree.stamp++;
    return AI2C_OK;
}

/* cpHashSet model for the dynamic tree's leaves (cpHashSetNew(0, ...): size 5, resize at entries >= size). */
AI2C_HD static inline unsigned int ai2c_next_prime(unsigned int n) {
    /* Chipmunk's prime.h table prefix; written without a static array so it is device-safe. */
    if (n < 5u) return 5u;
    if (n < 13u) return 13u;
    if (n < 23u) return 23u;
    if (n < 47u) return 47u;
    if (n < 97u) return 97u;
    if (n < 193u) return 193u;
    return 389u;
}

AI2C_HD static inline int ai2c_leaf_set_min_rank(const AI2CWorld *w, unsigned int size, unsigned int bin) {
    int min_rank = 0;
    int s;
    for (s = 0; s < w->shape_count; ++s) {
        if (w->leaf_set_member[s] && (unsigned int)(w->shapes[s].hashid % size) == bin && w->leaf_set_rank[s] < min_rank) min_rank = w->leaf_set_rank[s];
    }
    return min_rank;
}

/* Emit the shapes in cpHashSetEach order: bins ascending, chain order (rank ascending) within a bin. */
AI2C_HD static inline int ai2c_leaf_set_order(const AI2CWorld *w, int *out) {
    unsigned int bin;
    int count = 0;
    for (bin = 0; bin < w->leaf_set_size; ++bin) {
        for (;;) {
            int best = -1;
            int s;
            for (s = 0; s < w->shape_count; ++s) {
                int already = 0, k;
                if (!w->leaf_set_member[s] || (unsigned int)(w->shapes[s].hashid % w->leaf_set_size) != bin) continue;
                for (k = 0; k < count; ++k) if (out[k] == s) { already = 1; break; }
                if (already) continue;
                if (best < 0 || w->leaf_set_rank[s] < w->leaf_set_rank[best]) best = s;
            }
            if (best < 0) break;
            out[count++] = best;
        }
    }
    return count;
}

AI2C_HD static inline void ai2c_leaf_set_resize(AI2CWorld *w) {
    int order[AI2C_MAX_SHAPES];
    int count = ai2c_leaf_set_order(w, order);
    unsigned int new_size = ai2c_next_prime(w->leaf_set_size + 1u);
    int i;
    /* cpHashSetResize walks old bins in order and pushes each element at the head of its new chain. */
    for (i = 0; i < w->shape_count; ++i) w->leaf_set_rank[i] = 0;
    w->leaf_set_size = new_size;
    for (i = 0; i < count; ++i) {
        int s = order[i];
        unsigned int bin = (unsigned int)(w->shapes[s].hashid % new_size);
        w->leaf_set_rank[s] = ai2c_leaf_set_min_rank(w, new_size, bin) - 1;
    }
}

AI2C_HD static inline void ai2c_leaf_set_insert(AI2CWorld *w, int shape) {
    unsigned int bin = (unsigned int)(w->shapes[shape].hashid % w->leaf_set_size);
    w->leaf_set_rank[shape] = ai2c_leaf_set_min_rank(w, w->leaf_set_size, bin) - 1;
    w->leaf_set_member[shape] = 1;
    w->leaf_set_entries++;
    if (w->leaf_set_entries >= w->leaf_set_size) ai2c_leaf_set_resize(w);
}

AI2C_HD static inline void ai2c_leaf_set_remove(AI2CWorld *w, int shape) {
    if (!w->leaf_set_member[shape]) return;
    w->leaf_set_member[shape] = 0;
    w->leaf_set_entries--;
}

AI2C_HD static inline void ai2c_reindex_query(AI2CWorld *w) {
    AI2CMarkContext ctx;
    int order[AI2C_MAX_SHAPES];
    int count, i;
    if (w->dyn_tree.root < 0) return;
    count = ai2c_leaf_set_order(w, order);
    for (i = 0; i < count; ++i) (void)ai2c_leaf_update(w, &w->dyn_tree, w->leaf_of_shape[order[i]]);
    ctx.tree_is_dynamic = 1; ctx.static_root = w->static_tree.root; ctx.use_func = 1;
    ai2c_mark_subtree(w, w->dyn_tree.root, &ctx);
    w->dyn_tree.stamp++;
}

/* ------------------------------------------------------------- space step */

AI2C_HD static inline int ai2c_arbiter_lookup(AI2CWorld *w, int a, int b) {
    int i;
    for (i = 0; i < w->arbiter_count; ++i) {
        AI2CArbiter *arb = &w->arbiters[i];
        if (arb->active && ((arb->a == a && arb->b == b) || (arb->a == b && arb->b == a))) return i;
    }
    for (i = 0; i < AI2C_MAX_ARBITERS; ++i) {
        AI2CArbiter *arb = &w->arbiters[i];
        if (!arb->active) {
            arb->active = 1;
            arb->a = a; arb->body_a = w->shapes[a].body;
            arb->b = b; arb->body_b = w->shapes[b].body;
            arb->swapped = 0;
            arb->e = 0.0; arb->u = 0.0;
            arb->surface_vr = ai2c_vzero(); arb->n = ai2c_vzero();
            arb->count = 0;
            arb->stamp = 0;
            arb->state = AI2C_STATE_FIRST_COLLISION;
            if (i >= w->arbiter_count) w->arbiter_count = i + 1;
            return i;
        }
    }
    w->last_error = AI2C_ERR_CAPACITY;
    return -1;
}

AI2C_HD static inline unsigned int ai2c_collide_shapes(AI2CWorld *w, int a, int b, unsigned int id) {
    const AI2CShape *sa = &w->shapes[a];
    const AI2CShape *sb = &w->shapes[b];
    AI2CCollisionInfo info;
    int slot, h, begin_false;
    AI2CArbiter *arb;
    if (!ai2c_bb_intersects(sa->bb, sb->bb) || sa->body == sb->body || ai2c_filter_reject(sa, sb)) return id;
    info = ai2c_collide(w, a, b, id, w->scratch_contacts);
    if (info.count == 0) return info.id;
    slot = ai2c_arbiter_lookup(w, info.a, info.b);
    if (slot < 0) return info.id;
    arb = &w->arbiters[slot];
    ai2c_arbiter_update(w, arb, &info);
    h = ai2c_find_handler(w, sa->collision_type, sb->collision_type);
    begin_false = (h >= 0) ? w->handler_begin_false[h] : 0;
    if (arb->state == AI2C_STATE_FIRST_COLLISION && begin_false) {
        arb->state = AI2C_STATE_IGNORE;
    }
    if (arb->state != AI2C_STATE_IGNORE
        && !(sa->sensor || sb->sensor)
        && !(w->bodies[sa->body].m == (double)INFINITY && w->bodies[sb->body].m == (double)INFINITY)) {
        if (w->solve_count < AI2C_MAX_ARBITERS) w->solve[w->solve_count++] = slot; else w->last_error = AI2C_ERR_CAPACITY;
    } else {
        arb->count = 0;
        if (arb->state != AI2C_STATE_IGNORE) arb->state = AI2C_STATE_NORMAL;
    }
    arb->stamp = w->stamp;
    return info.id;
}

AI2C_HD static inline void ai2c_step_coefficients(const AI2CWorld *w, double dt, double *bias_coef, double *damping_dt) {
    *bias_coef = 1.0 - pow(w->bias, dt);
    *damping_dt = pow(w->damping, dt);
}

AI2C_HD static inline int ai2c_world_step(AI2CWorld *w, double dt, double bias_coef, double damping_dt) {
    double prev_dt;
    double slop;
    double dt_coef;
    int i, k;
    if (w == 0) return AI2C_ERR_NULL;
    if (dt == 0.0) return AI2C_OK;
    w->last_error = AI2C_OK;
    w->stamp++;
    prev_dt = w->curr_dt;
    w->curr_dt = dt;
    for (i = 0; i < w->solve_count; ++i) w->arbiters[w->solve[i]].state = AI2C_STATE_NORMAL;
    w->solve_count = 0;
    /* Integrate positions (dynamic and kinematic bodies, add order). */
    for (i = 0; i < w->body_count; ++i) {
        if (w->body_active[i] && w->bodies[i].type != AI2C_BODY_STATIC) ai2c_body_update_position(&w->bodies[i], dt);
    }
    /* Cache dynamic shape data, then find colliding pairs. */
    for (i = 0; i < w->shape_count; ++i) {
        if (w->shape_active[i] && w->bodies[w->shapes[i].body].type != AI2C_BODY_STATIC) (void)ai2c_shape_cache(&w->shapes[i], w->bodies[w->shapes[i].body].tf);
    }
    ai2c_reindex_query(w);
    /* Clear out old cached arbiters (cpSpaceArbiterSetFilter). */
    for (i = 0; i < w->arbiter_count; ++i) {
        AI2CArbiter *arb = &w->arbiters[i];
        unsigned int ticks;
        if (!arb->active) continue;
        ticks = w->stamp - arb->stamp;
        if (w->bodies[arb->body_a].type == AI2C_BODY_STATIC && w->bodies[arb->body_b].type == AI2C_BODY_STATIC) continue;
        if (ticks >= 1u && arb->state != AI2C_STATE_CACHED) arb->state = AI2C_STATE_CACHED;
        if (ticks >= w->persistence) { arb->count = 0; arb->active = 0; }
    }
    /* Prestep. */
    slop = w->slop;
    for (i = 0; i < w->solve_count; ++i) ai2c_arbiter_prestep(w, &w->arbiters[w->solve[i]], dt, slop, bias_coef);
    /* Integrate velocities. */
    for (i = 0; i < w->body_count; ++i) {
        if (w->body_active[i] && w->bodies[i].type != AI2C_BODY_STATIC) ai2c_body_update_velocity(&w->bodies[i], w->gravity, damping_dt, dt);
    }
    /* Apply cached impulses. */
    dt_coef = (prev_dt == 0.0 ? 0.0 : dt / prev_dt);
    for (i = 0; i < w->solve_count; ++i) ai2c_arbiter_apply_cached_impulse(w, &w->arbiters[w->solve[i]], dt_coef);
    /* Run the impulse solver. */
    for (k = 0; k < w->iterations; ++k) {
        for (i = 0; i < w->solve_count; ++i) ai2c_arbiter_apply_impulse(w, &w->arbiters[w->solve[i]]);
    }
    return w->last_error;
}

/* Reset both trees and replay shape insertion in add order using the given add-time
 * bounding boxes and body velocities (so the initial fat bounds match the real space). */
AI2C_HD static inline int ai2c_world_build_trees(AI2CWorld *w, const int *add_order, const AI2CBB *add_bb, const AI2CVec *add_v, int count) {
    int i;
    for (i = 0; i < AI2C_MAX_NODES; ++i) w->nodes[i].active = 0;
    for (i = 0; i < AI2C_MAX_PAIRS; ++i) w->pairs[i].active = 0;
    for (i = 0; i < AI2C_MAX_SHAPES; ++i) { w->leaf_of_shape[i] = -1; w->leaf_set_member[i] = 0; w->leaf_set_rank[i] = 0; }
    w->leaf_set_size = AI2C_LEAF_HASH_INITIAL_SIZE;
    w->leaf_set_entries = 0;
    w->dyn_tree.root = -1; w->dyn_tree.stamp = 0; w->dyn_tree.is_static = 0;
    w->static_tree.root = -1; w->static_tree.stamp = 0; w->static_tree.is_static = 1;
    for (i = 0; i < count; ++i) {
        int shape = add_order[i];
        AI2CTree *tree;
        AI2CBB saved_bb;
        AI2CVec saved_v;
        int rc;
        if (shape < 0 || shape >= w->shape_count) return AI2C_ERR_INVALID;
        tree = (w->bodies[w->shapes[shape].body].type == AI2C_BODY_STATIC) ? &w->static_tree : &w->dyn_tree;
        /* Temporarily present the add-time bb/velocity so GetBB matches the real insert. */
        saved_bb = w->shapes[shape].bb;
        saved_v = w->bodies[w->shapes[shape].body].v;
        w->shapes[shape].bb = add_bb[i];
        w->bodies[w->shapes[shape].body].v = add_v[i];
        rc = ai2c_tree_insert(w, tree, shape);
        w->shapes[shape].bb = saved_bb;
        w->bodies[w->shapes[shape].body].v = saved_v;
        if (rc != AI2C_OK) return rc;
        w->shape_active[shape] = 1;
        w->body_active[w->shapes[shape].body] = 1;
    }
    return AI2C_OK;
}

/* cpSpaceAddBody + cpSpaceAddShape for a shape configured after reset (weapons). The caller has set
 * the body state and shape configuration; cpShapeUpdate runs here before the spatial insert. */
AI2C_HD static inline int ai2c_world_add_shape(AI2CWorld *w, int shape) {
    AI2CTree *tree;
    if (shape < 0 || shape >= w->shape_count) return AI2C_ERR_INVALID;
    if (w->shape_active[shape]) return AI2C_ERR_INVALID;
    w->body_active[w->shapes[shape].body] = 1;
    (void)ai2c_shape_cache(&w->shapes[shape], w->bodies[w->shapes[shape].body].tf);
    tree = (w->bodies[w->shapes[shape].body].type == AI2C_BODY_STATIC) ? &w->static_tree : &w->dyn_tree;
    w->shape_active[shape] = 1;
    return ai2c_tree_insert(w, tree, shape);
}

/* cpSpaceRemoveShape: filter arbiters touching the shape, then cpBBTreeRemove.
 * Each filtered arbiter is also deleted from the solve array with
 * cpArrayDeleteObj semantics (the last entry moves into the hole).  Chipmunk
 * visits the arbiters in cached-hash-set bin order; this port visits them in
 * slot order, which is identical whenever a single arbiter is removed. */
AI2C_HD static inline int ai2c_world_remove_shape(AI2CWorld *w, int shape) {
    AI2CTree *tree;
    int leaf, i, j;
    if (shape < 0 || shape >= w->shape_count || !w->shape_active[shape]) return AI2C_ERR_INVALID;
    for (i = 0; i < w->arbiter_count; ++i) {
        AI2CArbiter *arb = &w->arbiters[i];
        if (arb->active && (arb->a == shape || arb->b == shape)) {
            if (arb->state != AI2C_STATE_CACHED) arb->state = AI2C_STATE_INVALIDATED;
            arb->active = 0;
            arb->count = 0;
            for (j = 0; j < w->solve_count; ++j) {
                if (w->solve[j] == i) {
                    w->solve_count -= 1;
                    w->solve[j] = w->solve[w->solve_count];
                    break;
                }
            }
        }
    }
    tree = (w->bodies[w->shapes[shape].body].type == AI2C_BODY_STATIC) ? &w->static_tree : &w->dyn_tree;
    leaf = w->leaf_of_shape[shape];
    if (leaf >= 0) {
        if (!tree->is_static) ai2c_leaf_set_remove(w, shape);
        tree->root = ai2c_subtree_remove(w, tree->root, leaf);
        ai2c_pairs_clear(w, leaf);
        ai2c_node_recycle(w, leaf);
        w->leaf_of_shape[shape] = -1;
    }
    w->shape_active[shape] = 0;
    return AI2C_OK;
}

AI2C_HD static inline int ai2c_world_remove_body(AI2CWorld *w, int body) {
    if (body < 0 || body >= w->body_count) return AI2C_ERR_INVALID;
    w->body_active[body] = 0;
    return AI2C_OK;
}

#endif /* AI2_NATIVE_NAVIGATION_CONTACT_H */
