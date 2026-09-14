#ifndef AI2_NAVIGATION_INPUT_H
#define AI2_NAVIGATION_INPUT_H

/* PlayerInputHandler/HorizontalState only. No movement, physics, or mask.
 * Actions arrive after any curriculum mask; each value > 0.5 is pressed.
 * ABI v1: 35 int32 values, offsets below. Inputs may alias outputs exactly.
 * On invalid input return nonzero without changing output. */
#include <math.h>
#include <stdint.h>

#ifdef __CUDACC__
#define AI2_INPUT_HD __host__ __device__
#else
#define AI2_INPUT_HD
#endif

enum {
    AI2_INPUT_BUTTONS = 10, AI2_INPUT_WIDTH = 35,
    AI2_INPUT_HELD = 0, AI2_INPUT_PRESSED = 10, AI2_INPUT_RELEASED = 20,
    AI2_INPUT_VERTICAL = 30, AI2_INPUT_HORIZONTAL = 31,
    AI2_INPUT_NO_HORIZONTAL = 32, AI2_INPUT_HORIZONTAL_STATE = 33,
    AI2_INPUT_LAST_DIRECTION = 34
};

/* Original enum: NONE=0, LEFT=1, RIGHT=2, LEFT_RIGHT=3, RIGHT_LEFT=4. */
AI2_INPUT_HD static inline int32_t ai2_input_remove(int32_t state, int left) {
    if (left) {
        if (state == 1) return 0;
        if (state == 3 || state == 4) return 2;
    } else {
        if (state == 2) return 0;
        if (state == 3 || state == 4) return 1;
    }
    return state;
}

AI2_INPUT_HD static inline int32_t ai2_input_stack(int32_t state, int left) {
    if (state == 0) return left ? 1 : 2;
    if (state == 1 && !left) return 3;
    if (state == 2 && left) return 4;
    return state;
}

AI2_INPUT_HD static inline void ai2_input_init_core(int32_t *state) {
    for (int i = 0; i < AI2_INPUT_WIDTH; ++i) state[i] = 0;
    state[AI2_INPUT_NO_HORIZONTAL] = 1;
}

AI2_INPUT_HD static inline int ai2_input_step_core(
        const double *action, const int32_t *before, int32_t *after) {
    if (!action || !before || !after) return 1;
    int32_t horizontal = before[AI2_INPUT_HORIZONTAL_STATE];
    if (horizontal < 0 || horizontal > 4) return 2;
    for (int i = 0; i < AI2_INPUT_BUTTONS; ++i) {
        if (!isfinite(action[i])) return 3;
        if (before[i] != 0 && before[i] != 1) return 4;
    }
    /* Read persistent fields before writing, including for in-place operation. */
    int32_t last_direction = before[AI2_INPUT_LAST_DIRECTION];
    for (int i = 0; i < AI2_INPUT_BUTTONS; ++i) {
        int32_t previous = before[i];
        int32_t held = action[i] > 0.5;
        after[AI2_INPUT_HELD + i] = held;
        after[AI2_INPUT_PRESSED + i] = !previous && held;
        after[AI2_INPUT_RELEASED + i] = previous && !held;
    }
    after[AI2_INPUT_VERTICAL] = after[0] - after[2];
    after[AI2_INPUT_HORIZONTAL] = after[3] - after[1];
    after[AI2_INPUT_NO_HORIZONTAL] = !after[3] && !after[1];
    /* Preserve the frozen release-left/release-right/press-right/press-left order. */
    if (after[AI2_INPUT_RELEASED + 1]) horizontal = ai2_input_remove(horizontal, 1);
    if (after[AI2_INPUT_RELEASED + 3]) horizontal = ai2_input_remove(horizontal, 0);
    if (after[AI2_INPUT_PRESSED + 3]) horizontal = ai2_input_stack(horizontal, 0);
    if (after[AI2_INPUT_PRESSED + 1]) horizontal = ai2_input_stack(horizontal, 1);
    after[AI2_INPUT_HORIZONTAL_STATE] = horizontal;
    /* The original update does not assign last_direction. */
    after[AI2_INPUT_LAST_DIRECTION] = last_direction;
    return 0;
}

#endif
