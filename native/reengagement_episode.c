#include "reengagement_episode.h"
#include <stddef.h>

size_t ai2b2_scenario_size(void) { return sizeof(AI2B2Scenario); }
size_t ai2b2_state_size(void) { return sizeof(AI2B2State); }
size_t ai2b2_result_size(void) { return sizeof(AI2B2StepResult); }
size_t ai2b2_diagnostics_size(void) { return sizeof(AI2B2Diagnostics); }
size_t ai2b2_player_size(void) { return sizeof(AI2B2PlayerSnapshot); }
size_t ai2b2_player_event_size(void) { return sizeof(AI2GPlayerEvent); }

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
int ai2b2_validate_event_export(const AI2B2State *s, const AI2GPlayerEvent *e) {
    int rc;
    if (!s || !e) return AI2N_ERR_NULL;
    if (!ai2b2_state_valid(s)) return AI2B2_ERR_STATE;
    rc = ai2b2_event_validity(s, e);
    return rc == 1 ? 0 : rc;
}
