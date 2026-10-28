from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from ai2.simulation.native_navigation_curriculum_env import (
    NAVIGATION_MASK,
    NAVIGATION_VERSION,
    NativeNavigationCurriculumEnv,
    build_navigation_curriculum_env,
)


class FakeRawNavigationEnv:
    def __init__(self, records=None):
        self.reset_result = object()
        self.step_result = object()
        self.records = list(records or [])
        self.closed = False
        self.num_envs = 3
        self.device = object()
        self.observation_space = object()
        self.action_space = object()

    def reset_tensor(self, seeds=None):
        self.last_reset_seeds = seeds
        return self.reset_result

    def step_tensor(self, actions):
        self.last_actions = actions
        return self.step_result

    def drain_episode_infos(self):
        records, self.records = self.records, []
        return records

    def close(self):
        self.closed = True


class NativeNavigationCurriculumEnvTests(unittest.TestCase):
    def test_declares_navigation_metadata_and_forwards_tensor_abi(self):
        raw = FakeRawNavigationEnv(
            [
                {
                    "episode": {"r": 1.25, "l": 17, "t": 0.5},
                    "navigation": {
                        "lane": 2,
                        "success": True,
                        "targets": 3,
                        "fell": False,
                        "horizon": False,
                        "terminated": True,
                        "truncated": False,
                        "episode_index": 9,
                    },
                }
            ]
        )
        env = NativeNavigationCurriculumEnv(raw)
        actions = object()

        self.assertEqual(env.curriculum_task, "navigation")
        self.assertEqual(env.curriculum_version, NAVIGATION_VERSION)
        np.testing.assert_array_equal(env.allowed_control_mask, NAVIGATION_MASK)
        self.assertTrue(env.host_has_timeouts())
        self.assertIs(env.reset_tensor(seeds=[1, 2, 3]), raw.reset_result)
        self.assertEqual(raw.last_reset_seeds, [1, 2, 3])
        self.assertIs(env.step_tensor(actions), raw.step_result)
        self.assertIs(raw.last_actions, actions)
        self.assertIs(env.num_envs, raw.num_envs)
        self.assertIs(env.device, raw.device)
        self.assertIs(env.observation_space, raw.observation_space)
        self.assertIs(env.action_space, raw.action_space)

        (info,) = env.drain_episode_infos()
        self.assertEqual(info["episode"], {"r": 1.25, "l": 17, "t": 0.5})
        self.assertEqual(
            info["navigation"],
            {
                "lane": 2,
                "success": True,
                "targets": 3,
                "fell": False,
                "horizon": False,
                "terminated": True,
                "truncated": False,
                "episode_index": 9,
            },
        )
        self.assertEqual(info["curriculum_task"], "navigation")
        self.assertEqual(info["curriculum_version"], NAVIGATION_VERSION)
        np.testing.assert_array_equal(info["allowed_control_mask"], NAVIGATION_MASK)
        self.assertEqual(env.drain_episode_infos(), [])

        env.close()
        self.assertTrue(raw.closed)

    def test_episode_records_fail_closed_when_navigation_payload_is_missing_or_malformed(self):
        bad_records = [
            {"episode": {"r": 0.0, "l": 1, "t": 0.0}},
            {"episode": {"r": 0.0, "l": 1, "t": 0.0}, "navigation": "not-a-dict"},
            {"navigation": {"lane": 0}},
        ]
        for record in bad_records:
            with self.subTest(record=record):
                env = NativeNavigationCurriculumEnv(FakeRawNavigationEnv([record]))
                with self.assertRaises(ValueError):
                    env.drain_episode_infos()

    def test_builder_delegates_to_existing_cuda_navigation_env(self):
        raw = FakeRawNavigationEnv()
        with patch('ai2.simulation.native_navigation_curriculum_env.native_lane_vec_env.build_cuda_env', return_value=raw) as build:
            env = build_navigation_curriculum_env(
                5,
                build_dir="/tmp/build",
                device="cuda:0",
                seed=11,
                wrapper_cfg={"time_limit": 123},
                record_capacity=42,
                observation_space="obs-space",
                action_space="act-space",
                dt=0.25,
                collision_bias=0.125,
            )
        self.assertIsInstance(env, NativeNavigationCurriculumEnv)
        self.assertIs(env.env, raw)
        build.assert_called_once_with(
            5,
            build_dir="/tmp/build",
            device="cuda:0",
            seed=11,
            wrapper_cfg={"time_limit": 123},
            record_capacity=42,
            observation_space="obs-space",
            action_space="act-space",
            dt=0.25,
            collision_bias=0.125,
        )


if __name__ == "__main__":
    unittest.main()
