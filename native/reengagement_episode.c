#include "reengagement_episode.h"
#include <stddef.h>

size_t ai2b2_scenario_size(void) { return sizeof(AI2B2Scenario); }
size_t ai2b2_state_size(void) { return sizeof(AI2B2State); }
size_t ai2b2_result_size(void) { return sizeof(AI2B2StepResult); }
size_t ai2b2_diagnostics_size(void) { return sizeof(AI2B2Diagnostics); }
size_t ai2b2_player_size(void) { return sizeof(AI2B2PlayerSnapshot); }
int ai2b2_reset_export(AI2B2State *s, const AI2NConfig *cfg, AI2NRng *rng, double scale,
                       const AI2B2Scenario *scenario, AI2B2StepResult *out) {
    return ai2b2_reset(s, cfg, rng, scale, scenario, out);
}
int ai2b2_step_export(AI2B2State *s, const double *action, AI2NRng *rng,
                      const AI2GPlatformTable *table, AI2B2StepResult *out) {
    return ai2b2_step(s, action, rng, table, out);
}
