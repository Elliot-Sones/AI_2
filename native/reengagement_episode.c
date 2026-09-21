#include "reengagement_episode.h"
#include <stddef.h>

size_t ai2b2_scenario_size(void) { return sizeof(AI2B2Scenario); }
size_t ai2b2_state_size(void) { return sizeof(AI2B2State); }
size_t ai2b2_result_size(void) { return sizeof(AI2B2StepResult); }
size_t ai2b2_diagnostics_size(void) { return sizeof(AI2B2Diagnostics); }
size_t ai2b2_player_size(void) { return sizeof(AI2B2PlayerSnapshot); }
size_t ai2b2_player_event_size(void) { return sizeof(AI2GPlayerEvent); }

#define AI2B2_FIELD_TABLE(name) \
    size_t ai2b2_##name##_field_count(void) { \
        return sizeof(ai2b2_##name##_offsets) / sizeof(ai2b2_##name##_offsets[0]); \
    } \
    size_t ai2b2_##name##_field_offset(size_t index) { \
        return index < ai2b2_##name##_field_count() ? ai2b2_##name##_offsets[index] : (size_t)-1; \
    }

static const size_t ai2b2_scenario_offsets[] = {
    offsetof(AI2B2Scenario, seed),
    offsetof(AI2B2Scenario, player_slot),
    offsetof(AI2B2Scenario, direction),
    offsetof(AI2B2Scenario, gap),
    offsetof(AI2B2Scenario, midpoint)
};
AI2B2_FIELD_TABLE(scenario)

static const size_t ai2b2_player_offsets[] = {
    offsetof(AI2B2PlayerSnapshot, x),
    offsetof(AI2B2PlayerSnapshot, y),
    offsetof(AI2B2PlayerSnapshot, vx),
    offsetof(AI2B2PlayerSnapshot, vy),
    offsetof(AI2B2PlayerSnapshot, damage),
    offsetof(AI2B2PlayerSnapshot, damage_done),
    offsetof(AI2B2PlayerSnapshot, damage_taken),
    offsetof(AI2B2PlayerSnapshot, clearance),
    offsetof(AI2B2PlayerSnapshot, player_slot),
    offsetof(AI2B2PlayerSnapshot, facing),
    offsetof(AI2B2PlayerSnapshot, stocks),
    offsetof(AI2B2PlayerSnapshot, state_id),
    offsetof(AI2B2PlayerSnapshot, grounded),
    offsetof(AI2B2PlayerSnapshot, fully_supported),
    offsetof(AI2B2PlayerSnapshot, stun_time_stored),
    offsetof(AI2B2PlayerSnapshot, attack_id)
};
AI2B2_FIELD_TABLE(player)

static const size_t ai2b2_event_offsets[] = {
    offsetof(AI2GPlayerEvent, kind),
    offsetof(AI2GPlayerEvent, source),
    offsetof(AI2GPlayerEvent, target),
    offsetof(AI2GPlayerEvent, move),
    offsetof(AI2GPlayerEvent, power),
    offsetof(AI2GPlayerEvent, cast),
    offsetof(AI2GPlayerEvent, hitbox),
    offsetof(AI2GPlayerEvent, phase_player),
    offsetof(AI2GPlayerEvent, event_sequence),
    offsetof(AI2GPlayerEvent, attack_id),
    offsetof(AI2GPlayerEvent, amount)
};
AI2B2_FIELD_TABLE(event)

static const size_t ai2b2_diagnostics_offsets[] = {
    offsetof(AI2B2Diagnostics, seed),
    offsetof(AI2B2Diagnostics, scenario_id_hash),
    offsetof(AI2B2Diagnostics, first_hit_attack_id),
    offsetof(AI2B2Diagnostics, second_hit_attack_id),
    offsetof(AI2B2Diagnostics, hit_attack_id),
    offsetof(AI2B2Diagnostics, player_slot),
    offsetof(AI2B2Diagnostics, direction),
    offsetof(AI2B2Diagnostics, tick),
    offsetof(AI2B2Diagnostics, outcome),
    offsetof(AI2B2Diagnostics, first_hit_tick),
    offsetof(AI2B2Diagnostics, second_hit_tick),
    offsetof(AI2B2Diagnostics, relocation_tick),
    offsetof(AI2B2Diagnostics, pursuit_tick),
    offsetof(AI2B2Diagnostics, pursuit_direction),
    offsetof(AI2B2Diagnostics, terminal_reason),
    offsetof(AI2B2Diagnostics, success),
    offsetof(AI2B2Diagnostics, sequence_completed),
    offsetof(AI2B2Diagnostics, failure_category),
    offsetof(AI2B2Diagnostics, achieved_first_hit),
    offsetof(AI2B2Diagnostics, achieved_relocation),
    offsetof(AI2B2Diagnostics, achieved_pursuit),
    offsetof(AI2B2Diagnostics, achieved_second_hit),
    offsetof(AI2B2Diagnostics, achieved_safe_resolution),
    offsetof(AI2B2Diagnostics, attributed_damage_event_count),
    offsetof(AI2B2Diagnostics, attributed_damage_attack_ids),
    offsetof(AI2B2Diagnostics, attributed_damage_ticks),
    offsetof(AI2B2Diagnostics, attributed_damage_sources),
    offsetof(AI2B2Diagnostics, attributed_damage_targets),
    offsetof(AI2B2Diagnostics, attributed_damage_moves),
    offsetof(AI2B2Diagnostics, attributed_damage_amounts),
    offsetof(AI2B2Diagnostics, initial_stocks),
    offsetof(AI2B2Diagnostics, light_request_ticks),
    offsetof(AI2B2Diagnostics, light_press_count),
    offsetof(AI2B2Diagnostics, max_light_hold),
    offsetof(AI2B2Diagnostics, attack_ticks),
    offsetof(AI2B2Diagnostics, movement_request_ticks),
    offsetof(AI2B2Diagnostics, renewed_movement_request_ticks),
    offsetof(AI2B2Diagnostics, attack_start_count),
    offsetof(AI2B2Diagnostics, attack_starts),
    offsetof(AI2B2Diagnostics, attack_contacts),
    offsetof(AI2B2Diagnostics, damage_events),
    offsetof(AI2B2Diagnostics, initial_learner_x),
    offsetof(AI2B2Diagnostics, initial_target_x),
    offsetof(AI2B2Diagnostics, spawn_gap),
    offsetof(AI2B2Diagnostics, spawn_midpoint),
    offsetof(AI2B2Diagnostics, initial_gap),
    offsetof(AI2B2Diagnostics, max_initial_approach_progress),
    offsetof(AI2B2Diagnostics, relocation_gap),
    offsetof(AI2B2Diagnostics, relocation_learner_x),
    offsetof(AI2B2Diagnostics, relocation_target_x),
    offsetof(AI2B2Diagnostics, gap_closed),
    offsetof(AI2B2Diagnostics, learner_progress),
    offsetof(AI2B2Diagnostics, reward_totals),
    offsetof(AI2B2Diagnostics, min_clearance),
    offsetof(AI2B2Diagnostics, unsupported_ticks),
    offsetof(AI2B2Diagnostics, falls),
    offsetof(AI2B2Diagnostics, stock_changes),
    offsetof(AI2B2Diagnostics, relocation_pre_target),
    offsetof(AI2B2Diagnostics, relocation_post_target),
    offsetof(AI2B2Diagnostics, players)
};
AI2B2_FIELD_TABLE(diagnostics)

static const size_t ai2b2_state_offsets[] = {
    offsetof(AI2B2State, game),
    offsetof(AI2B2State, scenario),
    offsetof(AI2B2State, diagnostics),
    offsetof(AI2B2State, initial_stocks),
    offsetof(AI2B2State, initialized),
    offsetof(AI2B2State, error),
    offsetof(AI2B2State, previous_light),
    offsetof(AI2B2State, current_light_hold),
    offsetof(AI2B2State, second_stage),
    offsetof(AI2B2State, relocated),
    offsetof(AI2B2State, first_attack_id),
    offsetof(AI2B2State, second_attack_id),
    offsetof(AI2B2State, last_second_duplicate_id)
};
AI2B2_FIELD_TABLE(state)

static const size_t ai2b2_result_offsets[] = {
    offsetof(AI2B2StepResult, obs),
    offsetof(AI2B2StepResult, reward),
    offsetof(AI2B2StepResult, reward_terms),
    offsetof(AI2B2StepResult, raw_action),
    offsetof(AI2B2StepResult, executed_action),
    offsetof(AI2B2StepResult, terminated),
    offsetof(AI2B2StepResult, truncated),
    offsetof(AI2B2StepResult, error),
    offsetof(AI2B2StepResult, pre_players),
    offsetof(AI2B2StepResult, diagnostics)
};
AI2B2_FIELD_TABLE(result)

#define AI2B2_OFFSET_EXPORT(prefix, type, field) \
    size_t ai2b2_offset_##prefix##_##field(void) { return offsetof(type, field); }

AI2B2_OFFSET_EXPORT(scenario, AI2B2Scenario, seed)
AI2B2_OFFSET_EXPORT(scenario, AI2B2Scenario, player_slot)
AI2B2_OFFSET_EXPORT(scenario, AI2B2Scenario, direction)
AI2B2_OFFSET_EXPORT(scenario, AI2B2Scenario, gap)
AI2B2_OFFSET_EXPORT(scenario, AI2B2Scenario, midpoint)

AI2B2_OFFSET_EXPORT(player, AI2B2PlayerSnapshot, x)
AI2B2_OFFSET_EXPORT(player, AI2B2PlayerSnapshot, y)
AI2B2_OFFSET_EXPORT(player, AI2B2PlayerSnapshot, attack_id)

AI2B2_OFFSET_EXPORT(diagnostics, AI2B2Diagnostics, seed)
AI2B2_OFFSET_EXPORT(diagnostics, AI2B2Diagnostics, scenario_id_hash)
AI2B2_OFFSET_EXPORT(diagnostics, AI2B2Diagnostics, first_hit_attack_id)
AI2B2_OFFSET_EXPORT(diagnostics, AI2B2Diagnostics, second_hit_attack_id)
AI2B2_OFFSET_EXPORT(diagnostics, AI2B2Diagnostics, terminal_reason)
AI2B2_OFFSET_EXPORT(diagnostics, AI2B2Diagnostics, achieved_safe_resolution)
AI2B2_OFFSET_EXPORT(diagnostics, AI2B2Diagnostics, attributed_damage_attack_ids)
AI2B2_OFFSET_EXPORT(diagnostics, AI2B2Diagnostics, players)

AI2B2_OFFSET_EXPORT(state, AI2B2State, game)
AI2B2_OFFSET_EXPORT(state, AI2B2State, scenario)
AI2B2_OFFSET_EXPORT(state, AI2B2State, diagnostics)
AI2B2_OFFSET_EXPORT(state, AI2B2State, first_attack_id)
AI2B2_OFFSET_EXPORT(state, AI2B2State, second_attack_id)
AI2B2_OFFSET_EXPORT(state, AI2B2State, last_second_duplicate_id)

AI2B2_OFFSET_EXPORT(result, AI2B2StepResult, obs)
AI2B2_OFFSET_EXPORT(result, AI2B2StepResult, reward)
AI2B2_OFFSET_EXPORT(result, AI2B2StepResult, reward_terms)
AI2B2_OFFSET_EXPORT(result, AI2B2StepResult, diagnostics)

int ai2b2_reset_export(AI2B2State *s, const AI2NConfig *cfg, AI2NRng *rng, double scale,
                       const AI2B2Scenario *scenario, AI2B2StepResult *out) {
    return ai2b2_reset(s, cfg, rng, scale, scenario, out);
}
int ai2b2_step_export(AI2B2State *s, const double *action, AI2NRng *rng,
                      const AI2GPlatformTable *table, AI2B2StepResult *out) {
    return ai2b2_step(s, action, rng, table, out);
}
int ai2b2_step_injected_event_export(AI2B2State *s, const double *action, AI2NRng *rng,
                                     const AI2GPlatformTable *table, int stream,
                                     const AI2GPlayerEvent *event, AI2B2StepResult *out) {
    return ai2b2_step_core(s, action, rng, table, out, stream, event);
}
int ai2b2_debug_prepare_boundary_export(AI2B2State *s, const AI2GPlatformTable *table,
                                        int mode, int elapsed_after_step,
                                        int tick_after_step) {
    return ai2b2_debug_prepare_boundary(s, table, mode, elapsed_after_step, tick_after_step);
}
int ai2b2_validate_event_export(const AI2B2State *s, const AI2GPlayerEvent *e) {
    int rc;
    if (!s || !e) return AI2N_ERR_NULL;
    if (!ai2b2_state_valid(s)) return AI2B2_ERR_STATE;
    rc = ai2b2_event_validity(s, e);
    return rc == 1 ? 0 : rc;
}
