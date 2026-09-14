#ifndef AI2_GAME_CAPSULE_H
#define AI2_GAME_CAPSULE_H

/* Frozen environment.py c0116fc7... CapsuleCollider, including pygame.Rect's
 * truncation toward zero and strict circle tangency. Compile with contraction
 * disabled (-ffp-contract=off / --fmad=false) for CPython arithmetic parity.
 * Nonfinite or out-of-int32-range rectangle geometry fails closed. Callers
 * requiring positive dimensions must validate that policy separately. */
#include <limits.h>
#include <math.h>

#ifdef __CUDACC__
#define AI2GC_HD __host__ __device__
#else
#define AI2GC_HD
#endif

typedef struct AI2GCapsule { double x, y, width, height; } AI2GCapsule;
typedef struct AI2GCRect { double left, top, right, bottom; int width, height; } AI2GCRect;
typedef struct AI2GCParts {
    double x1, y1, x2, y2, radius;
    int circle;
    AI2GCRect rect;
} AI2GCParts;

AI2GC_HD static inline double ai2gc_min(double a, double b) { return a < b ? a : b; }
AI2GC_HD static inline double ai2gc_max(double a, double b) { return a > b ? a : b; }
AI2GC_HD static inline int ai2gc_int_range(double v) {
    return isfinite(v) && v >= (double)INT_MIN && v <= (double)INT_MAX;
}

AI2GC_HD static inline int ai2gc_parts(const AI2GCapsule *c, AI2GCParts *p) {
    double left, top, width, height;
    if (!c || !isfinite(c->x) || !isfinite(c->y) ||
        !isfinite(c->width) || !isfinite(c->height)) return 0;
    p->radius = ai2gc_min(c->width, c->height) / 2.0;
    p->circle = c->width == c->height;
    p->x1 = p->x2 = c->x;
    p->y1 = p->y2 = c->y;
    /* Circle branches never consume this rectangle. Initialize it explicitly
     * so GCC's inlined dataflow analysis can prove every field initialized. */
    p->rect.left = p->rect.top = p->rect.right = p->rect.bottom = 0.0;
    p->rect.width = p->rect.height = 0;
    if (p->circle) return 1;
    if (c->width < c->height) {
        double offset = c->height / 2.0 - p->radius;
        left = c->x - c->width / 2.0;
        top = c->y - offset;
        width = c->width;
        height = c->height - 2.0 * p->radius;
        p->y1 = c->y - offset;
        p->y2 = c->y + offset;
    } else {
        double offset = c->width / 2.0 - p->radius;
        left = c->x - offset;
        top = c->y - c->height / 2.0;
        width = c->width - 2.0 * p->radius;
        height = c->height;
        p->x1 = c->x - offset;
        p->x2 = c->x + offset;
    }
    if (!ai2gc_int_range(left) || !ai2gc_int_range(top) ||
        !ai2gc_int_range(width) || !ai2gc_int_range(height)) return 0;
    p->rect.left = (int)left;
    p->rect.top = (int)top;
    p->rect.width = (int)width;
    p->rect.height = (int)height;
    p->rect.right = p->rect.left + p->rect.width;
    p->rect.bottom = p->rect.top + p->rect.height;
    return ai2gc_int_range(p->rect.right) && ai2gc_int_range(p->rect.bottom);
}

AI2GC_HD static inline int ai2gc_circles(double x1, double y1, double r1,
                                       double x2, double y2, double r2) {
    double dx = x1 - x2, dy = y1 - y2;
    return sqrt(dx * dx + dy * dy) < r1 + r2;
}

AI2GC_HD static inline int ai2gc_circle_rect(double x, double y, double radius,
                                           const AI2GCRect *r) {
    double closest_x = ai2gc_max(r->left, ai2gc_min(x, r->right));
    double closest_y = ai2gc_max(r->top, ai2gc_min(y, r->bottom));
    double dx = x - closest_x, dy = y - closest_y;
    return sqrt(dx * dx + dy * dy) < radius;
}

AI2GC_HD static inline int ai2gc_rects(const AI2GCRect *a, const AI2GCRect *b) {
    if (!a->width || !a->height || !b->width || !b->height) return 0;
    return ai2gc_min(a->left, a->right) < ai2gc_max(b->left, b->right) &&
           ai2gc_max(a->left, a->right) > ai2gc_min(b->left, b->right) &&
           ai2gc_min(a->top, a->bottom) < ai2gc_max(b->top, b->bottom) &&
           ai2gc_max(a->top, a->bottom) > ai2gc_min(b->top, b->bottom);
}

AI2GC_HD static inline int ai2g_capsule_intersects(const AI2GCapsule *a,
                                                 const AI2GCapsule *b) {
    AI2GCParts p, q;
    if (!ai2gc_parts(a, &p) || !ai2gc_parts(b, &q)) return 0;
    if (ai2gc_circles(p.x1, p.y1, p.radius, q.x1, q.y1, q.radius)) return 1;
    if (p.circle && q.circle) return 0;
    if (!q.circle && ai2gc_circles(p.x1, p.y1, p.radius, q.x2, q.y2, q.radius)) return 1;
    if (!p.circle && ai2gc_circles(p.x2, p.y2, p.radius, q.x1, q.y1, q.radius)) return 1;
    if (!p.circle && !q.circle &&
        (ai2gc_circles(p.x2, p.y2, p.radius, q.x2, q.y2, q.radius) || ai2gc_rects(&p.rect, &q.rect))) return 1;
    if (!q.circle && (ai2gc_circle_rect(p.x1, p.y1, p.radius, &q.rect) ||
                     (!p.circle && ai2gc_circle_rect(p.x2, p.y2, p.radius, &q.rect)))) return 1;
    if (!p.circle && (ai2gc_circle_rect(q.x1, q.y1, q.radius, &p.rect) ||
                     (!q.circle && ai2gc_circle_rect(q.x2, q.y2, q.radius, &p.rect)))) return 1;
    return 0;
}
#endif
