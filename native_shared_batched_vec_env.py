"""Native game patch plus shared-action transport and batched grouped RNG.

This opt-in composition is compatible with ``benchmark_campaign
--grouped-env-path``.  Cooperative constructors first wrap factories with the
native game patch, then run the shared-action transport with the batched RNG
worker.  Baseline grouped/native modules and frozen game source are unchanged.
"""

from __future__ import annotations

from native_grouped_vec_env import GroupedSubprocVecEnv as NativeGrouped
from shared_action_vec_env import SharedBatchedRngGroupedSubprocVecEnv


class GroupedSubprocVecEnv(NativeGrouped, SharedBatchedRngGroupedSubprocVecEnv):
    """NativeFactory -> shared-action transport -> batched RNG grouped worker."""
