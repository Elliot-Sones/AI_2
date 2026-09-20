"""Curriculum metadata adapter for native navigation lanes."""
from __future__ import annotations

from typing import Any

import numpy as np

import native_lane_vec_env

try:
    from curriculum_ppo import TASK_MASKS
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by dependency-light local tests
    if exc.name not in {"torch", "stable_baselines3", "gymnasium"}:
        raise
    TASK_MASKS = {
        "navigation": (True, True, True, True, True, False, True, False, False, False),
        "combat": (True, True, True, True, True, False, True, True, False, False),
    }


NAVIGATION_VERSION = "bridge_0b_navigation-v1"
NAVIGATION_MASK = np.asarray(TASK_MASKS["navigation"], dtype=bool)


class NativeNavigationCurriculumEnv:
    curriculum_task = "navigation"
    curriculum_version = NAVIGATION_VERSION

    def __init__(self, env: Any) -> None:
        self.env = env

    @property
    def allowed_control_mask(self) -> np.ndarray:
        return NAVIGATION_MASK.copy()

    def host_has_timeouts(self) -> bool:
        return True

    def reset_tensor(self, seeds=None):
        return self.env.reset_tensor(seeds=seeds)

    def step_tensor(self, clipped_actions):
        return self.env.step_tensor(clipped_actions)

    def drain_episode_infos(self) -> list[dict[str, Any]]:
        infos = []
        for index, info in enumerate(self.env.drain_episode_infos()):
            if not isinstance(info, dict):
                raise ValueError(f"native navigation record {index} must be a dict")
            episode = info.get("episode")
            navigation = info.get("navigation")
            if not isinstance(episode, dict):
                raise ValueError(f"native navigation record {index} is missing an episode payload")
            if not isinstance(navigation, dict):
                raise ValueError(f"native navigation record {index} is missing a navigation payload")
            enriched = dict(info)
            enriched["episode"] = episode
            enriched["navigation"] = navigation
            enriched["curriculum_task"] = self.curriculum_task
            enriched["curriculum_version"] = self.curriculum_version
            enriched["allowed_control_mask"] = self.allowed_control_mask
            infos.append(enriched)
        return infos

    def close(self) -> None:
        self.env.close()

    def __getattr__(self, name: str):
        return getattr(self.env, name)


def build_navigation_curriculum_env(
    lanes,
    *,
    build_dir,
    device="cuda",
    seed=0,
    wrapper_cfg=None,
    record_capacity=None,
    observation_space=None,
    action_space=None,
    dt=1.0 / 30.0,
    collision_bias=None,
):
    env = native_lane_vec_env.build_cuda_env(
        lanes,
        build_dir=build_dir,
        device=device,
        seed=seed,
        wrapper_cfg=wrapper_cfg,
        record_capacity=record_capacity,
        observation_space=observation_space,
        action_space=action_space,
        dt=dt,
        collision_bias=collision_bias,
    )
    return NativeNavigationCurriculumEnv(env)
