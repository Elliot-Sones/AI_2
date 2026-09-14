"""Import and composition checks for native shared-action batched grouped VecEnv."""

from __future__ import annotations

import unittest

try:
    import gymnasium  # noqa: F401
    import stable_baselines3  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover - light local Python skips runtime tests.
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@unittest.skipIf(_IMPORT_ERROR is not None, f"missing grouped runtime dependency: {getattr(_IMPORT_ERROR, 'name', _IMPORT_ERROR)}")
class NativeSharedBatchedVecEnvTests(unittest.TestCase):
    def test_native_shared_batched_composes_native_wrapper_before_shared_batched_runner(self):
        import native_grouped_vec_env
        import native_shared_batched_vec_env
        import shared_action_vec_env

        grouped_cls = native_shared_batched_vec_env.GroupedSubprocVecEnv

        self.assertTrue(issubclass(grouped_cls, native_grouped_vec_env.GroupedSubprocVecEnv))
        self.assertTrue(issubclass(grouped_cls, shared_action_vec_env.SharedBatchedRngGroupedSubprocVecEnv))
        self.assertLess(
            grouped_cls.mro().index(native_grouped_vec_env.GroupedSubprocVecEnv),
            grouped_cls.mro().index(shared_action_vec_env.SharedBatchedRngGroupedSubprocVecEnv),
        )


if __name__ == "__main__":
    unittest.main()
