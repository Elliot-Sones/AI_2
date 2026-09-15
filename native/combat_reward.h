#ifndef AI2_COMBAT_REWARD_H
#define AI2_COMBAT_REWARD_H

/* Combat curriculum reward wrapper for the raw full-game engine.
 *
 * This layer reads post-step raw state and raw terminal flags, then produces a
 * versioned term breakdown. It does not step physics, consume RNG, mutate
 * observations, or replace the raw reward path.
 */
#include "game_episode.h"

#define AI2CR_VERSION 1
#define AI2CR_TERMS 14
#define AI2CR_ERR_CONFIG 501
#define AI2CR_ERR_STATE 502

enum {
    AI2CR_TERM_APPROACH = 0,
    AI2CR_TERM_ATTACK_START = 1,
    AI2CR_TERM_CONTACT = 2,
    AI2CR_TERM_DAMAGE_DEALT = 3,
    AI2CR_TERM_DAMAGE_TAKEN = 4,
    AI2CR_TERM_NET_DAMAGE = 5,
    AI2CR_TERM_KO = 6,
    AI2CR_TERM_WIN = 7,
    AI2CR_TERM_CLEAN_WIN = 8,
    AI2CR_TERM_TIMEOUT = 9,
    AI2CR_TERM_OFFSTAGE = 10,
    AI2CR_TERM_WALKOFF = 11,
    AI2CR_TERM_RECOVERY = 12,
    AI2CR_TERM_WEAPON = 13
};

typedef struct {
    double term[AI2CR_TERMS];
    uint32_t enabled_mask;
    double fade;
} AI2CRCoefficients;
/* Sign contract:
 * - Most raw terms are nonnegative facts/counts/magnitudes. Coefficient signs
 *   encode desirability: damage_taken/offstage/walkoff/timeout may be negative.
 * - Intrinsically signed or comparative terms require nonnegative magnitudes:
 *   approach delta, zero-sum KO/win, and weapon pickup/drop.
 * - Legacy config loaders must call ai2cr_normalize_legacy_term() per mapped
 *   term. This preserves negative penalties while normalizing historical
 *   phase1 knockout:-3 to +3 so the death-reward double negative cannot return.
 */

typedef struct {
    int32_t version, initialized, learner_slot, tick;
    int32_t initial_stocks[2], last_stocks[2], last_weapon[2], last_offstage[2], clean_stock[2];
    double last_x[2], last_y[2], last_distance;
    double last_damage_done[2], last_damage_taken[2];
    double base_damage_done[2], base_damage_taken[2];
    uint64_t last_attack_starts[2], last_attack_contacts[2], last_damage_events[2];
    double returns[2], term_totals[2][AI2CR_TERMS];
} AI2CRState;

typedef struct {
    int32_t version, learner_slot, tick, terminated, truncated, win_mask;
    double terms[2][AI2CR_TERMS], total[2];
    double approach_delta, distance;
    int32_t attack_starts[2], contacts[2], damage_events[2], ko_events[2];
    int32_t weapon_pickups[2], weapon_drops[2], offstage[2], walkoff[2], recovery[2];
} AI2CRBreakdown;

typedef struct {
    int32_t version, learner_slot, length, terminated, truncated, win_mask;
    int32_t stocks[2];
    double returns[2], term_totals[2][AI2CR_TERMS];
    double damage_done[2], damage_taken[2];
    int32_t clean_win[2], timeout;
} AI2CREpisodeRecord;

AI2N_HD static inline double ai2cr_px(const AI2GState *g, int i) {
    const AI2NPlayer *p = &g->scene.players[i];
    return (p->body >= 0 && p->body < g->scene.world.body_count)
        ? ai2n_body_position(&g->scene.world.bodies[p->body]).x
        : p->ms.x;
}

AI2N_HD static inline double ai2cr_py(const AI2GState *g, int i) {
    const AI2NPlayer *p = &g->scene.players[i];
    return (p->body >= 0 && p->body < g->scene.world.body_count)
        ? ai2n_body_position(&g->scene.world.bodies[p->body]).y
        : p->ms.y;
}

AI2N_HD static inline double ai2cr_distance(double ax, double ay, double bx, double by) {
    const double dx = ax - bx, dy = ay - by;
    return sqrt(dx * dx + dy * dy);
}

AI2N_HD static inline int ai2cr_offstage(const AI2GState *g, int i) {
    const double half_w = floor(g->scene.mcfg.stage_width_tiles / 2.0);
    const double half_h = floor(g->scene.mcfg.stage_height_tiles / 2.0);
    const double x = ai2cr_px(g, i), y = ai2cr_py(g, i);
    return fabs(x) > half_w || fabs(y) > half_h;
}

AI2N_HD static inline int ai2cr_term_requires_magnitude(int term) {
    return term == AI2CR_TERM_APPROACH ||
        term == AI2CR_TERM_KO ||
        term == AI2CR_TERM_WIN ||
        term == AI2CR_TERM_WEAPON;
}

AI2N_HD static inline double ai2cr_normalize_legacy_term(int term, double value) {
    return ai2cr_term_requires_magnitude(term) && value < 0.0 ? -value : value;
}

AI2N_HD static inline int ai2cr_coefficients_valid(const AI2CRCoefficients *c) {
    int i;
    if (!c || !isfinite(c->fade) || c->fade < 0.0 || c->fade > 1.0 ||
        (c->enabled_mask & ~((1u << AI2CR_TERMS) - 1u))) return 0;
    for (i = 0; i < AI2CR_TERMS; ++i) {
        if (!isfinite(c->term[i]) || (ai2cr_term_requires_magnitude(i) && c->term[i] < 0.0))
            return 0;
    }
    return 1;
}

AI2N_HD static inline int ai2cr_game_valid(const AI2GState *g) {
    int i;
    if (!g || g->initialized != 1 || !isfinite(g->scene.mcfg.stage_width_tiles) ||
        !isfinite(g->scene.mcfg.stage_height_tiles)) return 0;
    for (i = 0; i < 2; ++i) {
        if (!isfinite(ai2cr_px(g, i)) || !isfinite(ai2cr_py(g, i)) ||
            !isfinite(g->combat[i].damage_done) || !isfinite(g->combat[i].damage_taken_total) ||
            g->scene.players[i].ms.stocks < INT32_MIN || g->scene.players[i].ms.stocks > INT32_MAX)
            return 0;
    }
    return 1;
}

AI2N_HD static inline void ai2cr_add(AI2CRBreakdown *b, int player, int term, double value) {
    b->terms[player][term] += value;
}

AI2N_HD static inline void ai2cr_add_zero_sum(AI2CRBreakdown *b, int positive_player, int term, double value) {
    ai2cr_add(b, positive_player, term, value);
    ai2cr_add(b, 1 - positive_player, term, -value);
}

AI2N_HD static inline int ai2cr_reset(AI2CRState *s, const AI2GState *g, int learner_slot) {
    int i;
    if (!s || !ai2cr_game_valid(g) || (learner_slot != 0 && learner_slot != 1)) return AI2CR_ERR_CONFIG;
    ai2n_zero(s, sizeof(*s));
    s->version = AI2CR_VERSION;
    s->initialized = 1;
    s->learner_slot = learner_slot;
    for (i = 0; i < 2; ++i) {
        s->initial_stocks[i] = s->last_stocks[i] = g->scene.players[i].ms.stocks;
        s->last_weapon[i] = g->scene.players[i].weapon;
        s->last_offstage[i] = ai2cr_offstage(g, i);
        s->clean_stock[i] = 1;
        s->last_x[i] = ai2cr_px(g, i);
        s->last_y[i] = ai2cr_py(g, i);
        s->last_damage_done[i] = g->combat[i].damage_done;
        s->last_damage_taken[i] = g->combat[i].damage_taken_total;
        s->base_damage_done[i] = g->combat[i].damage_done;
        s->base_damage_taken[i] = g->combat[i].damage_taken_total;
        s->last_attack_starts[i] = g->combat[i].attack_starts;
        s->last_attack_contacts[i] = g->combat[i].attack_contacts;
        s->last_damage_events[i] = g->combat[i].damage_events;
    }
    s->last_distance = ai2cr_distance(s->last_x[0], s->last_y[0], s->last_x[1], s->last_y[1]);
    return 0;
}

AI2N_HD static inline void ai2cr_count_events(const AI2GState *g, AI2CRBreakdown *b) {
    int source, k;
    for (source = 0; source < 2; ++source) {
        const AI2GCombatPlayer *c = &g->combat[source];
        const int n = c->event_count < 0 ? 0 :
            (c->event_count > AI2G_PLAYER_EVENT_CAPACITY ? AI2G_PLAYER_EVENT_CAPACITY : c->event_count);
        for (k = 0; k < n; ++k) {
            const AI2GPlayerEvent *e = &c->events[k];
            if (e->kind == AI2G_EVENT_KO && e->target >= 0 && e->target < 2) b->ko_events[e->target]++;
        }
    }
}

AI2N_HD static inline int ai2cr_step(AI2CRState *s, const AI2GState *g, const AI2GStepResult *raw,
                                    const AI2CRCoefficients *c, AI2CRBreakdown *b) {
    int i, term;
    double x[2], y[2], distance, delta[2], dealt[2], taken[2];
    int clean_stock[2];
    AI2CRBreakdown next;
    if (!s || !b || !raw || s->initialized != 1 || s->version != AI2CR_VERSION ||
        !ai2cr_game_valid(g) || !ai2cr_coefficients_valid(c)) return AI2CR_ERR_CONFIG;
    for (i = 0; i < 2; ++i) {
        if (g->combat[i].attack_starts < s->last_attack_starts[i] ||
            g->combat[i].attack_contacts < s->last_attack_contacts[i] ||
            g->combat[i].damage_events < s->last_damage_events[i])
            return AI2CR_ERR_STATE;
        if (g->combat[i].attack_starts - s->last_attack_starts[i] > (uint64_t)INT32_MAX ||
            g->combat[i].attack_contacts - s->last_attack_contacts[i] > (uint64_t)INT32_MAX ||
            g->combat[i].damage_events - s->last_damage_events[i] > (uint64_t)INT32_MAX)
            return AI2CR_ERR_STATE;
    }
    ai2n_zero(&next, sizeof(next));
    next.version = AI2CR_VERSION;
    next.learner_slot = s->learner_slot;
    next.tick = s->tick + 1;
    next.terminated = raw->terminated;
    next.truncated = raw->truncated;
    next.win_mask = raw->win_mask;
    for (i = 0; i < 2; ++i) {
        x[i] = ai2cr_px(g, i);
        y[i] = ai2cr_py(g, i);
        clean_stock[i] = s->clean_stock[i];
    }
    distance = ai2cr_distance(x[0], y[0], x[1], y[1]);
    next.distance = distance;
    next.approach_delta = s->last_distance - distance;
    ai2cr_add(&next, 0, AI2CR_TERM_APPROACH, next.approach_delta);
    ai2cr_add(&next, 1, AI2CR_TERM_APPROACH, next.approach_delta);
    ai2cr_count_events(g, &next);
    for (i = 0; i < 2; ++i) {
        next.attack_starts[i] = (int32_t)(g->combat[i].attack_starts - s->last_attack_starts[i]);
        next.contacts[i] = (int32_t)(g->combat[i].attack_contacts - s->last_attack_contacts[i]);
        next.damage_events[i] = (int32_t)(g->combat[i].damage_events - s->last_damage_events[i]);
        dealt[i] = g->combat[i].damage_done - s->last_damage_done[i];
        taken[i] = g->combat[i].damage_taken_total - s->last_damage_taken[i];
        delta[i] = dealt[i] - taken[i];
        if (next.attack_starts[i]) ai2cr_add(&next, i, AI2CR_TERM_ATTACK_START, (double)next.attack_starts[i]);
        if (next.contacts[i]) ai2cr_add(&next, i, AI2CR_TERM_CONTACT, (double)next.contacts[i]);
        ai2cr_add(&next, i, AI2CR_TERM_DAMAGE_DEALT, dealt[i]);
        ai2cr_add(&next, i, AI2CR_TERM_DAMAGE_TAKEN, taken[i]);
        ai2cr_add(&next, i, AI2CR_TERM_NET_DAMAGE, delta[i] > 0.0 ? delta[i] : 0.0);
        if (next.ko_events[i]) ai2cr_add_zero_sum(&next, 1 - i, AI2CR_TERM_KO, (double)next.ko_events[i]);
        if (g->scene.players[i].ms.stocks < s->initial_stocks[i]) clean_stock[i] = 0;
        next.offstage[i] = ai2cr_offstage(g, i);
        next.walkoff[i] = !s->last_offstage[i] && next.offstage[i];
        next.recovery[i] = s->last_offstage[i] && !next.offstage[i];
        if (next.offstage[i]) ai2cr_add(&next, i, AI2CR_TERM_OFFSTAGE, 1.0);
        if (next.walkoff[i]) ai2cr_add(&next, i, AI2CR_TERM_WALKOFF, 1.0);
        if (next.recovery[i]) ai2cr_add(&next, i, AI2CR_TERM_RECOVERY, 1.0);
        next.weapon_pickups[i] = s->last_weapon[i] == 0 && g->scene.players[i].weapon != 0;
        next.weapon_drops[i] = s->last_weapon[i] != 0 && g->scene.players[i].weapon == 0;
        if (next.weapon_pickups[i]) ai2cr_add(&next, i, AI2CR_TERM_WEAPON, 1.0);
        if (next.weapon_drops[i]) ai2cr_add(&next, i, AI2CR_TERM_WEAPON, -1.0);
    }
    if (raw->terminated) {
        for (i = 0; i < 2; ++i) if (raw->win_mask & (1 << i)) {
            const double episode_dealt = g->combat[i].damage_done - s->base_damage_done[i];
            const double episode_taken = g->combat[i].damage_taken_total - s->base_damage_taken[i];
            ai2cr_add_zero_sum(&next, i, AI2CR_TERM_WIN, 1.0);
            if (episode_dealt > episode_taken) ai2cr_add(&next, i, AI2CR_TERM_CLEAN_WIN, 1.0);
        }
    } else if (raw->truncated) {
        ai2cr_add(&next, 0, AI2CR_TERM_TIMEOUT, 1.0);
        ai2cr_add(&next, 1, AI2CR_TERM_TIMEOUT, 1.0);
    }
    for (i = 0; i < 2; ++i) {
        for (term = 0; term < AI2CR_TERMS; ++term) {
            const double value = (c->enabled_mask & (1u << term)) ? next.terms[i][term] * c->term[term] * c->fade : 0.0;
            if (!isfinite(value)) return AI2CR_ERR_CONFIG;
            next.total[i] += value;
        }
    }
    *b = next;
    s->tick = next.tick;
    for (i = 0; i < 2; ++i) {
        s->clean_stock[i] = clean_stock[i];
        s->returns[i] += next.total[i];
        for (term = 0; term < AI2CR_TERMS; ++term) s->term_totals[i][term] += next.terms[i][term];
        s->last_x[i] = x[i];
        s->last_y[i] = y[i];
        s->last_stocks[i] = g->scene.players[i].ms.stocks;
        s->last_weapon[i] = g->scene.players[i].weapon;
        s->last_offstage[i] = b->offstage[i];
        s->last_damage_done[i] = g->combat[i].damage_done;
        s->last_damage_taken[i] = g->combat[i].damage_taken_total;
        s->last_attack_starts[i] = g->combat[i].attack_starts;
        s->last_attack_contacts[i] = g->combat[i].attack_contacts;
        s->last_damage_events[i] = g->combat[i].damage_events;
    }
    s->last_distance = distance;
    return 0;
}

AI2N_HD static inline int ai2cr_episode_record(const AI2CRState *s, const AI2GState *g,
                                              const AI2GStepResult *raw, AI2CREpisodeRecord *r) {
    int i, term;
    if (!s || !r || !raw || s->initialized != 1 || s->version != AI2CR_VERSION ||
        !ai2cr_game_valid(g)) return AI2CR_ERR_CONFIG;
    ai2n_zero(r, sizeof(*r));
    r->version = AI2CR_VERSION;
    r->learner_slot = s->learner_slot;
    r->length = s->tick;
    r->terminated = raw->terminated;
    r->truncated = raw->truncated;
    r->win_mask = raw->win_mask;
    r->timeout = raw->truncated && !raw->terminated;
    for (i = 0; i < 2; ++i) {
        r->stocks[i] = g->scene.players[i].ms.stocks;
        r->returns[i] = s->returns[i];
        r->damage_done[i] = g->combat[i].damage_done;
        r->damage_taken[i] = g->combat[i].damage_taken_total;
        r->clean_win[i] = raw->terminated && (raw->win_mask & (1 << i)) &&
            (g->combat[i].damage_done - s->base_damage_done[i] >
             g->combat[i].damage_taken_total - s->base_damage_taken[i]);
        for (term = 0; term < AI2CR_TERMS; ++term) r->term_totals[i][term] = s->term_totals[i][term];
    }
    return 0;
}

#endif
