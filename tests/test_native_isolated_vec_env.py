"""Tests for native game patch + isolated RNG grouped composition."""

from __future__ import annotations

from pathlib import Path
import types
import unittest
from unittest.mock import patch


class NativeIsolatedVecEnvTests(unittest.TestCase):
    def test_composes_native_factory_before_isolated_grouped_env(self):
        import ai2.simulation.native_isolated_vec_env as module

        env_fns = [lambda: object(), lambda: object()]
        created = []

        def fake_native_factory(factory, *, source_path):
            created.append((factory, Path(source_path)))
            return ("native", factory, Path(source_path))

        class FakeIsolatedGrouped:
            def __init__(self, factories, *, group_size=4, start_method=None):
                self.factories = factories
                self.group_size = group_size
                self.start_method = start_method

        source = Path("/tmp/frozen/UTMIST-AI2-main/environment/environment.py")
        with patch('ai2.simulation.native_grouped_vec_env.native_factory', side_effect=fake_native_factory):
            with patch('ai2.simulation.native_game_patch.verify_source_hash', return_value="ok") as verify:
                with patch('ai2.simulation.isolated_rng_vec_env.GroupedSubprocVecEnv', FakeIsolatedGrouped):
                    env = module.GroupedSubprocVecEnv(
                        env_fns,
                        group_size=8,
                        start_method="spawn",
                        native_source_path=source,
                    )

        verify.assert_called_once_with(source)
        self.assertEqual([entry[1] for entry in created], [source.resolve(), source.resolve()])
        self.assertEqual(env.factories, [("native", env_fns[0], source.resolve()), ("native", env_fns[1], source.resolve())])
        self.assertEqual(env.group_size, 8)
        self.assertEqual(env.start_method, "spawn")

    def test_default_source_path_uses_current_working_directory(self):
        import ai2.simulation.native_isolated_vec_env as module

        class FakeIsolatedGrouped:
            def __init__(self, factories, *, group_size=4, start_method=None):
                self.factories = factories

        expected = Path.cwd() / "UTMIST-AI2-main/environment/environment.py"
        with patch('ai2.simulation.native_grouped_vec_env.native_factory', return_value="wrapped") as native_factory:
            with patch('ai2.simulation.native_game_patch.verify_source_hash', return_value="ok") as verify:
                with patch('ai2.simulation.isolated_rng_vec_env.GroupedSubprocVecEnv', FakeIsolatedGrouped):
                    env = module.GroupedSubprocVecEnv([dict])

        verify.assert_called_once_with(expected)
        native_factory.assert_called_once_with(dict, source_path=expected.resolve())
        self.assertEqual(env.factories, ["wrapped"])


if __name__ == "__main__":
    unittest.main()
