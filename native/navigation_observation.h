#ifndef AI2_NAVIGATION_OBSERVATION_H
#define AI2_NAVIGATION_OBSERVATION_H
#include <math.h>
#ifdef __CUDACC__
#define AI2_OBS_HD __host__ __device__
#else
#define AI2_OBS_HD
#endif

/* Player.get_obs field order, not normalized observation-space bounds.
 * p is two 16-field rows, controlled player then opponent. Swap for agent 1.
 * Field 6 is reserved zero; airborne is
 * derived from grounded. Damage field 12 is unscaled. Spawners retain the
 * controller's first-four order; type 0 means inactive, 1 normal, 2/3 dropped.
 * Contact/support facts and spawner lifecycle are inputs, not simulated here.
 */
AI2_OBS_HD static inline double ai2_obs_clamp(double x, double low, double high) {
    return fmax(low, fmin(high, x));
}
AI2_OBS_HD static inline int ai2_obs_pack(const double *p, const double *s,
                                        const double *platform, float *out) {
    for (int i = 0; i < 32; ++i) if (!isfinite(p[i])) return 1;
    for (int i = 0; i < 12; ++i) if (!isfinite(s[i])) return 1;
    for (int i = 0; i < 4; ++i) if (!isfinite(platform[i])) return 1;
    for (int player = 0; player < 2; ++player) {
        const double *row = p + 16 * player;
        if ((row[4] != -1 && row[4] != 1) || (row[5] != 0 && row[5] != 1) || row[6] != 0) return 2;
    }
    for (int i = 0; i < 4; ++i) {
        double type = s[3 * i + 2];
        if (type != 0 && type != 1 && type != 2 && type != 3) return 2;
    }
    for (int player = 0; player < 2; ++player) {
        const double *row = p + 16 * player;
        float *obs = out + 32 * player;
        for (int i = 0; i < 16; ++i) obs[i] = (float)row[i];
        obs[0] = (float)ai2_obs_clamp(row[0], -18, 18);
        obs[1] = (float)ai2_obs_clamp(row[1], -7, 7);
        obs[2] = (float)ai2_obs_clamp(row[2], -10, 10);
        obs[3] = (float)ai2_obs_clamp(row[3], -10, 10);
        obs[4] = row[4] == 1 ? 1.0f : 0.0f;
        obs[6] = row[5] == 1 ? 0.0f : 1.0f;
        obs[12] = (float)(row[12] / 700.0);
        for (int i = 0; i < 4; ++i) {
            int j = 3 * i;
            obs[16 + j] = s[j + 2] == 0 ? 0.0f : (float)ai2_obs_clamp(s[j], -18, 18);
            obs[17 + j] = s[j + 2] == 0 ? 0.0f : (float)ai2_obs_clamp(s[j + 1], -7, 7);
            obs[18 + j] = (float)s[j + 2];
        }
        obs[28] = (float)ai2_obs_clamp(platform[0], -18, 18);
        obs[29] = (float)ai2_obs_clamp(platform[1], -7, 7);
        obs[30] = (float)ai2_obs_clamp(platform[2], -10, 10);
        obs[31] = (float)ai2_obs_clamp(platform[3], -10, 10);
    }
    return 0;
}
#endif
