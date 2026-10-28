"""Explicit opt-in composition of native game helpers and batched worker RNG.

Cooperative constructors first wrap the original env factories with the native
patch, then select the batched RNG worker, then construct the original grouped
shared-memory runner. No original runner or frozen game source is modified.
"""
from ai2.simulation.native_grouped_vec_env import GroupedSubprocVecEnv as NativeGrouped
from ai2.simulation.batched_rng_vec_env import BatchedRngGroupedSubprocVecEnv


class GroupedSubprocVecEnv(NativeGrouped, BatchedRngGroupedSubprocVecEnv):
    """NativeFactory -> batched RNG worker -> original grouped transport."""
