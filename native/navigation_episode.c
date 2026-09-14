#include "navigation_episode.h"

#include <stddef.h>

/* C ABI exports for the Python driver (native_navigation_episode.py). */

int ai2n_episode_size(void) { return (int)sizeof(AI2NEpisode); }
int ai2n_result_size(void) { return (int)sizeof(AI2NStepResult); }
int ai2n_draw_size(void) { return (int)sizeof(AI2NDraw); }
int ai2n_rng_size(void) { return (int)sizeof(AI2NRng); }
int ai2n_config_size(void) { return (int)sizeof(AI2NConfig); }
int ai2n_player_size(void) { return (int)sizeof(AI2NPlayer); }
int ai2n_spawner_size(void) { return (int)sizeof(AI2NSpawner); }
int ai2n_wrapper_size(void) { return (int)sizeof(AI2NWrapper); }
int ai2n_movement_state_size(void) { return (int)sizeof(AI2MovementState); }
int ai2n_movement_config_size(void) { return (int)sizeof(AI2MovementConfig); }
int ai2n_positions_count(void) { return AI2N_POSITIONS_COUNT; }
const char *ai2n_positions_sha256(void) { return AI2N_POSITIONS_SHA256; }

/* Field offsets so the ctypes mirror can be checked against the compiled layout. */
int ai2n_offsets(int *out, int capacity) {
    const int values[] = {
        (int)offsetof(AI2NEpisode, world), (int)offsetof(AI2NEpisode, mcfg), (int)offsetof(AI2NEpisode, cfg),
        (int)offsetof(AI2NEpisode, players), (int)offsetof(AI2NEpisode, spawners), (int)offsetof(AI2NEpisode, wrap),
        (int)offsetof(AI2NEpisode, waypoint1), (int)offsetof(AI2NEpisode, platform_body), (int)offsetof(AI2NEpisode, steps),
        (int)offsetof(AI2NEpisode, shape_id_counter), (int)offsetof(AI2NEpisode, obs),
        (int)offsetof(AI2NPlayer, input_state), (int)offsetof(AI2NPlayer, body), (int)offsetof(AI2NPlayer, damage),
        (int)offsetof(AI2NWrapper, target_valid), (int)offsetof(AI2NWrapper, falls),
        (int)offsetof(AI2NStepResult, reward), (int)offsetof(AI2NStepResult, error),
        (int)offsetof(AI2NRng, draws), (int)offsetof(AI2NRng, s), (int)offsetof(AI2NRng, error_b),
        (int)offsetof(AI2NConfig, settle_speed), (int)offsetof(AI2NConfig, damping_dt),
    };
    const int n = (int)(sizeof(values) / sizeof(values[0]));
    int i;
    if (out == 0 || capacity < n) return -n;
    for (i = 0; i < n; ++i) out[i] = values[i];
    return n;
}

/* Host-side step coefficients with runtime libm calls (never constant folded). */
int ai2n_config_finalize(AI2NConfig *cfg, int use_recorded_bias) {
    volatile double base = (double)(1.0f - 0.1f);
    volatile double dt;
    if (cfg == 0) return AI2N_ERR_NULL;
    dt = cfg->dt;
    if (!use_recorded_bias) cfg->collision_bias = pow(base, 60.0);
    cfg->bias_coef = 1.0 - pow(cfg->collision_bias, dt);
    cfg->damping_dt = pow(1.0, dt);
    return AI2N_ERR_NONE;
}

int ai2n_reset_export(AI2NEpisode *ep, const AI2NConfig *cfg, AI2NRng *rng) { return ai2n_reset(ep, cfg, rng); }
int ai2n_reset_scene_export(AI2NEpisode *ep, const AI2NConfig *cfg, AI2NRng *rng) { return ai2n_reset_scene(ep, cfg, rng); }

int ai2n_step_export(AI2NEpisode *ep, const double *action, AI2NRng *rng, const AI2NPlatformTable *table, AI2NStepResult *out) { return ai2n_step(ep, action, rng, table, out); }

int ai2n_platform_table_size(void) { return (int)sizeof(AI2NPlatformTable); }

void ai2n_platform_table_build_export(AI2NPlatformTable *table, double dt) { ai2n_platform_table_build(table, dt); }

int ai2n_observe_export(AI2NEpisode *ep, float *out) { return ai2n_observe(ep, out); }

int ai2n_is_on_floor_export(AI2NEpisode *ep, int player) { return ai2n_is_on_floor(ep, player); }

void ai2n_rng_replay_export(AI2NRng *rng, const AI2NDraw *draws, int count) { ai2n_rng_replay(rng, draws, count); }

void ai2n_rng_seed_export(AI2NRng *rng, unsigned long long seed, unsigned long long lane) { ai2n_rng_seed(rng, seed, lane); }

double ai2n_fma_probe_export(double a, double b, double c) { return ai2n_fma_probe(a, b, c); }

double ai2n_rng_request_export(AI2NRng *rng, int kind, double a, double b) { return ai2n_rng_request(rng, kind, a, b); }
