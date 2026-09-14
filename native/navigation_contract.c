#include "navigation_contract.h"

int navigation_contract_step_one(
    const NavigationConfigC *config,
    const NavigationStateC *state,
    const NavigationPostPhysicsC *post,
    NavigationStepResultC *out) {
    return navigation_contract_step_core(config, state, post, out);
}
