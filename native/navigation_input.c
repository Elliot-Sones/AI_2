#include "navigation_input.h"

void ai2_input_init(int32_t *state) {
    if (state) ai2_input_init_core(state);
}

int ai2_input_step(const double *action, const int32_t *before, int32_t *after) {
    return ai2_input_step_core(action, before, after);
}
