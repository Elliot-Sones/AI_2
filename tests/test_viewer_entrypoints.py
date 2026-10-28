"""Viewer CLIs are inert on import and use training-compatible observations."""

import importlib
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VIEWERS = ("watch_navigation", "watch_games", "play_vs_agent")


class ViewerEntrypointTests(unittest.TestCase):
    def test_help_is_available_without_loading_training_or_pygame(self):
        for name in VIEWERS:
            with self.subTest(name=name):
                code = (
                    "import sys; from tools.visualization import " + name + " as viewer; "
                    "assert 'pygame' not in sys.modules; "
                    "assert 'ai2.training.train' not in sys.modules; "
                    "viewer.main(['--help'])"
                )
                result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                        text=True, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--model", result.stdout)

    def test_direct_script_help_works_outside_the_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in VIEWERS:
                with self.subTest(name=name):
                    result = subprocess.run(
                        [sys.executable, str(ROOT / "tools/visualization" / f"{name}.py"), "--help"],
                        cwd=directory, text=True, capture_output=True, timeout=30,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_frame_stack_matches_vec_frame_stack_initial_padding(self):
        viewer = importlib.import_module("tools.visualization.watch_games")
        stack = viewer.FrameStack(3, (2,))
        np.testing.assert_array_equal(stack.reset(np.array([1, 2])), [0, 0, 0, 0, 1, 2])
        np.testing.assert_array_equal(stack.step(np.array([3, 4])), [0, 0, 1, 2, 3, 4])

    def test_game_closes_environment_when_prediction_fails(self):
        viewer = importlib.import_module("tools.visualization.watch_games")
        env = Mock()
        env.reset.return_value = (np.ones(3), {})
        model = Mock()
        model.predict.side_effect = RuntimeError("prediction failed")
        with patch.object(viewer, "make_game_env", return_value=env):
            with self.assertRaisesRegex(RuntimeError, "prediction failed"):
                viewer.run_game(model, Mock(), 1, "random", save_video=False)
        env.close.assert_called_once()

    def test_game_consumes_single_agent_observations_and_scalar_actions(self):
        viewer = importlib.import_module("tools.visualization.watch_games")
        env = Mock()
        env.reset.return_value = (np.array([1., 2.]), {})
        env.step.return_value = (np.array([3., 4.]), 0., True, False, {})
        stats = Mock(lives_left=3, damage_done=2, damage_taken=1)
        env.unwrapped.raw_env.get_stats.return_value = stats
        model = Mock()
        model.predict.return_value = (np.zeros(10), None)
        with patch.object(viewer, "make_game_env", return_value=env):
            viewer.run_game(model, Mock(), 1, "random", save_video=False, frame_stack=2)
        np.testing.assert_array_equal(model.predict.call_args.args[0], [0., 0., 1., 2.])
        np.testing.assert_array_equal(env.step.call_args.args[0], np.zeros(10))
        env.close.assert_called_once()

    def test_human_match_uses_training_stack_and_closes_on_quit(self):
        viewer = importlib.import_module("tools.visualization.play_vs_agent")
        env = Mock()
        env.reset.return_value = (np.array([1., 2.]), {})
        env.step.return_value = (np.array([3., 4.]), 0., True, False, {})
        human_stats, ai_stats = object(), object()
        env.unwrapped.raw_env.get_stats.side_effect = [human_stats, ai_stats]
        pygame = Mock(QUIT=1, KEYDOWN=2, K_ESCAPE=3)
        pygame.event.get.return_value = []
        model = Mock()
        model.predict.return_value = (np.zeros(10), None)
        with patch.dict(sys.modules, {"pygame": pygame}), \
                patch.object(viewer, "make_game_env", return_value=env) as make_env:
            result = viewer.play_match(model, frame_stack=2, seconds=1)
        self.assertEqual(result, (human_stats, ai_stats))
        self.assertIsInstance(make_env.call_args.args[0], viewer.UserInputAgent)
        np.testing.assert_array_equal(model.predict.call_args.args[0], [0., 0., 1., 2.])
        env.close.assert_called_once()
        pygame.quit.assert_called_once()

    def test_human_escape_closes_without_predicting(self):
        viewer = importlib.import_module("tools.visualization.play_vs_agent")
        env = Mock()
        env.reset.return_value = (np.array([1., 2.]), {})
        pygame = Mock(QUIT=1, KEYDOWN=2, K_ESCAPE=3)
        pygame.event.get.return_value = [SimpleNamespace(type=2, key=3)]
        model = Mock()
        with patch.dict(sys.modules, {"pygame": pygame}), \
                patch.object(viewer, "make_game_env", return_value=env):
            viewer.play_match(model, seconds=1)
        model.predict.assert_not_called()
        env.close.assert_called_once()
        pygame.quit.assert_called_once()

    def test_non_positive_games_are_rejected_before_runtime_imports(self):
        for name in ("watch_navigation", "watch_games"):
            with self.subTest(name=name):
                viewer = importlib.import_module(f"tools.visualization.{name}")
                with self.assertRaises(SystemExit) as raised:
                    viewer.main(["--model", "unused.zip", "--games", "0"])
                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
