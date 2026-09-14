#include "contact_episode.h"
#include <stddef.h>

size_t ai2b1_scenario_size(void) { return sizeof(AI2B1Scenario); }
size_t ai2b1_state_size(void) { return sizeof(AI2B1State); }
size_t ai2b1_result_size(void) { return sizeof(AI2B1StepResult); }
size_t ai2b1_diagnostics_size(void) { return sizeof(AI2B1Diagnostics); }
size_t ai2b1_player_size(void) { return sizeof(AI2B1PlayerSnapshot); }
int ai2b1_reset_export(AI2B1State *s, const AI2NConfig *cfg, AI2NRng *rng, double scale,
                     const AI2B1Scenario *scenario, AI2B1StepResult *out) {
    return ai2b1_reset(s, cfg, rng, scale, scenario, out);
}
int ai2b1_step_export(AI2B1State *s, const double *action, AI2NRng *rng,
                    const AI2GPlatformTable *table, AI2B1StepResult *out) {
    return ai2b1_step(s, action, rng, table, out);
}
