/*
 * Training lanes for the closed-loop navigation tick (milestone 4, part 1).
 *
 * One lane is one SB3 environment slot: the episode state, its own random
 * stream and the Monitor bookkeeping (episode return and length).  The lane
 * step reproduces what the frozen training stack does around the wrapper:
 *
 *   Monitor.step      appends float(reward) and reports the sum at episode end
 *   VecEnv.step_wait  casts the reward to float32, sets TimeLimit.truncated to
 *                     "truncated and not terminated", keeps the final
 *                     observation as terminal_observation and resets the
 *                     environment inside the same step
 *
 * Frame stacking (VecFrameStack) stays outside as a tensor operation on the
 * collector side.  The same function bodies run on the host
 * (navigation_lanes.c, a loop over lanes) and on the device
 * (navigation_lanes.cu, one thread per lane), so the CPU library is the
 * reference for the CUDA lanes.
 */
#ifndef AI2_NAVIGATION_LANES_H
#define AI2_NAVIGATION_LANES_H

#include "navigation_episode.h"

typedef struct AI2NLane {
    AI2NEpisode ep;
    AI2NRng rng;
    double ep_return;   /* Monitor: running sum of float(reward), left to right */
    int32_t ep_length;  /* Monitor: steps in the current episode */
    int32_t episodes;   /* completed episodes since the last seeded reset */
    int32_t error;      /* first error code; a lane with an error is frozen */
    int32_t active;     /* 1 once a reset has produced an observation */
    int32_t pad[2];
} AI2NLane;

/* One completed episode, appended by the step that ended it. */
typedef struct AI2NEpisodeRecord {
    int32_t lane;
    int32_t length;
    double ret;
    int32_t terminated;
    int32_t truncated;
    int32_t success;
    int32_t fell;
    int32_t targets;
    int32_t horizon;
    int32_t episode_index;
    int32_t pad;
} AI2NEpisodeRecord;

#if defined(__CUDA_ARCH__)
#define AI2N_LANE_FETCH_INC(p) atomicAdd((p), 1)
#else
static inline int ai2n_lane_fetch_inc_host(int *p) { const int v = *p; *p = v + 1; return v; }
#define AI2N_LANE_FETCH_INC(p) ai2n_lane_fetch_inc_host(p)
#endif

/* A frozen lane repeats its last observation with zero reward and no done flag. */
AI2N_HD static inline void ai2n_lane_hold(const AI2NLane *lane, float *obs, float *terminal, float *reward, uint8_t *terminated, uint8_t *truncated) {
    int i;
    for (i = 0; i < AI2N_OBS_WIDTH; ++i) { obs[i] = lane->ep.obs[i]; terminal[i] = lane->ep.obs[i]; }
    *reward = 0.0f;
    *terminated = 0;
    *truncated = 0;
}

/* Reset the episode with the lane's current random stream (seeded or replay). */
AI2N_HD static inline int ai2n_lane_begin(AI2NLane *lane, const AI2NConfig *cfg, float *obs) {
    int rc, i;
    lane->ep_return = 0.0;
    lane->ep_length = 0;
    lane->episodes = 0;
    lane->error = 0;
    lane->active = 0;
    rc = ai2n_reset(&lane->ep, cfg, &lane->rng);
    if (rc == AI2N_ERR_NONE && lane->rng.error) rc = AI2N_ERR_RNG;
    if (rc) {
        lane->error = rc;
        for (i = 0; i < AI2N_OBS_WIDTH; ++i) obs[i] = 0.0f;
        return rc;
    }
    for (i = 0; i < AI2N_OBS_WIDTH; ++i) obs[i] = lane->ep.obs[i];
    lane->active = 1;
    return AI2N_ERR_NONE;
}

AI2N_HD static inline int ai2n_lane_reset(AI2NLane *lane, const AI2NConfig *cfg, unsigned long long seed, unsigned long long stream, float *obs) {
    ai2n_rng_seed(&lane->rng, seed, stream);
    return ai2n_lane_begin(lane, cfg, obs);
}

/*
 * One collector step for one lane.  `action` is the float32 clipped action
 * the policy produced; `obs` receives the observation the policy sees next
 * (the reset observation when the episode ended), `terminal` the wrapper's
 * final observation of this step, `reward` the float32 reward, `terminated`
 * the wrapper's terminated flag and `truncated` SB3's TimeLimit.truncated.
 */
AI2N_HD static inline int ai2n_lane_step(AI2NLane *lane, int lane_index, const AI2NConfig *cfg, const AI2NPlatformTable *table,
                                         const float *action, float *obs, float *terminal, float *reward,
                                         uint8_t *terminated, uint8_t *truncated,
                                         AI2NEpisodeRecord *records, int record_capacity, int *record_count, int *record_overflow) {
    double act[AI2N_ACTION_WIDTH];
    AI2NStepResult out;
    int rc, i, done;
    if (lane->error) { ai2n_lane_hold(lane, obs, terminal, reward, terminated, truncated); return lane->error; }
    if (!lane->active) {
        lane->error = AI2N_ERR_CONFIG;
        ai2n_lane_hold(lane, obs, terminal, reward, terminated, truncated);
        return lane->error;
    }
    for (i = 0; i < AI2N_ACTION_WIDTH; ++i) act[i] = (double)action[i];
    rc = ai2n_step(&lane->ep, act, &lane->rng, table, &out);
    if (rc) { lane->error = rc; ai2n_lane_hold(lane, obs, terminal, reward, terminated, truncated); return rc; }
    /* Monitor.step: self.rewards.append(float(reward)); ep_rew = sum(self.rewards) at the end. */
    lane->ep_return += out.reward;
    lane->ep_length += 1;
    for (i = 0; i < AI2N_OBS_WIDTH; ++i) terminal[i] = out.obs[i];
    *reward = (float)out.reward; /* VecEnv buf_rews is float32 */
    *terminated = (uint8_t)(out.terminated != 0);
    *truncated = (uint8_t)(out.truncated != 0 && out.terminated == 0); /* info["TimeLimit.truncated"] */
    done = (out.terminated != 0 || out.truncated != 0) ? 1 : 0;
    if (!done) {
        for (i = 0; i < AI2N_OBS_WIDTH; ++i) obs[i] = out.obs[i];
        return AI2N_ERR_NONE;
    }
    {
        const int idx = AI2N_LANE_FETCH_INC(record_count);
        if (idx >= 0 && idx < record_capacity) {
            AI2NEpisodeRecord *r = &records[idx];
            r->lane = lane_index;
            r->length = lane->ep_length;
            r->ret = lane->ep_return;
            r->terminated = out.terminated;
            r->truncated = out.truncated;
            r->success = out.success;
            r->fell = lane->ep.wrap.fell_this_episode;
            r->targets = lane->ep.wrap.targets_this_episode;
            r->horizon = out.horizon;
            r->episode_index = lane->episodes;
            r->pad = 0;
        } else {
            (void)AI2N_LANE_FETCH_INC(record_overflow);
        }
    }
    lane->episodes += 1;
    lane->ep_return = 0.0;
    lane->ep_length = 0;
    /* VecEnv autoreset: the same step returns the next episode's first observation. */
    rc = ai2n_reset(&lane->ep, cfg, &lane->rng);
    if (rc == AI2N_ERR_NONE && lane->rng.error) rc = AI2N_ERR_RNG;
    if (rc) {
        lane->error = rc;
        lane->active = 0;
        for (i = 0; i < AI2N_OBS_WIDTH; ++i) obs[i] = out.obs[i];
        return rc;
    }
    for (i = 0; i < AI2N_OBS_WIDTH; ++i) obs[i] = lane->ep.obs[i];
    return AI2N_ERR_NONE;
}

#endif /* AI2_NAVIGATION_LANES_H */
