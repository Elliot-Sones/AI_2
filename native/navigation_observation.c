#include "navigation_observation.h"
#include <stddef.h>
int ai2_observation_cpu(const double *p, const double *s, const double *platform,
                        float *out, int *errors, size_t lanes) {
    if (!p || !s || !platform || !out || !errors || lanes == 0 || lanes > 4096) return 1;
    for (size_t i = 0; i < lanes; ++i) {
        errors[i] = ai2_obs_pack(p + i * 32, s + i * 12, platform + i * 4, out + i * 64);
        if (errors[i]) for (int j = 0; j < 64; ++j) out[i * 64 + j] = NAN;
    }
    return 0;
}
