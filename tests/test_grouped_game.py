"""Real frozen-game parity gate for GroupedSubprocVecEnv.

This test intentionally avoids policy training.  It builds the phase-0b
navigation environment through the same remote train_job.build_env entrypoint as
the benchmark campaign, drives stock SubprocVecEnv and GroupedSubprocVecEnv
with identical fixed action batches, and fails on the first behavioral
divergence with tick/env/field detail.

Local developer machines without the frozen source tree or RL dependencies skip
at runtime.  Authorized benchmark hosts should set either:

  TRAIN_JOB=/ROOT/scripts/train_job.py

or:

  AI2_CAMPAIGN_SOURCE=/ROOT/variants/corrected
"""

from __future__ import annotations

import copy
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock


PHASE_KEY = "0b"
ACTION_SEED = 20260906
ENV_SEED = 20260906
DEFAULT_STEPS = 1000
SHORT_TIMEOUT_SECONDS = 0.2


@contextmanager
def _cwd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _import_required_modules():
    try:
        import numpy as np
        import yaml
        from grouped_vec_env import GroupedSubprocVecEnv
    except ModuleNotFoundError as exc:
        raise unittest.SkipTest(f"missing runtime dependency for frozen game parity: {exc.name}") from exc

    return np, yaml, GroupedSubprocVecEnv


def _resolve_runtime_paths() -> tuple[Path, Path, Path]:
    explicit = os.environ.get("TRAIN_JOB")
    explicit_source = os.environ.get("AI2_CAMPAIGN_SOURCE")

    source_root = Path(explicit_source).expanduser().resolve() if explicit_source else None
    if source_root is not None and not source_root.is_dir():
        raise AssertionError(f"AI2_CAMPAIGN_SOURCE does not exist: {source_root}")
    if source_root is not None and not (source_root / "train.py").is_file():
        raise AssertionError(f"AI2_CAMPAIGN_SOURCE must point at frozen source containing train.py: {source_root}")

    if explicit:
        train_job = Path(explicit).expanduser().resolve()
        if not train_job.is_file():
            raise AssertionError(f"TRAIN_JOB does not exist: {train_job}")
    elif source_root is not None:
        candidate = source_root.parents[1] / "scripts" / "train_job.py"
        if not candidate.is_file():
            raise AssertionError(f"could not find sibling root scripts/train_job.py for {source_root}")
        train_job = candidate.resolve()
    else:
        raise unittest.SkipTest("set TRAIN_JOB or AI2_CAMPAIGN_SOURCE to run frozen game parity")

    if source_root is None:
        candidate = train_job.parents[1] / "variants" / "corrected"
        if not (candidate / "train.py").is_file():
            raise AssertionError(
                "TRAIN_JOB was supplied without AI2_CAMPAIGN_SOURCE and "
                f"could not find frozen source at {candidate}"
            )
        source_root = candidate.resolve()

    explicit_config = os.environ.get("AI2_CONFIG")
    if explicit_config:
        raw_config_path = Path(explicit_config).expanduser()
        config_path = raw_config_path if raw_config_path.is_absolute() else source_root / raw_config_path
        config_path = config_path.resolve()
        if not config_path.is_file():
            raise AssertionError(f"AI2_CONFIG does not exist: {config_path}")
    else:
        config_candidates = [source_root / "control.yaml", source_root / "config.yaml"]
        config_path = next((candidate for candidate in config_candidates if candidate.is_file()), None)
        if config_path is None:
            raise AssertionError(f"could not find control.yaml or config.yaml under {source_root}")
        config_path = config_path.resolve()

    return train_job, source_root, config_path


def _import_train_job(train_job_path: Path):
    train_job_dir = str(train_job_path.parent)
    source_root = str(train_job_path.parents[1])
    for path in (train_job_dir, source_root):
        if path not in sys.path:
            sys.path.insert(0, path)

    spec = importlib.util.spec_from_file_location("train_job", train_job_path)
    if spec is None or spec.loader is None:
        raise unittest.SkipTest(f"could not import train_job from {train_job_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_job"] = module
    sys.modules["campaign_target_train_job"] = module
    spec.loader.exec_module(module)
    return module


def _load_yaml_config(config_path: Path, yaml_module) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml_module.safe_load(handle)


def _phase_lookup(params: dict, phase_key: str) -> dict:
    phases = params.get("phases", {})
    if phase_key in phases:
        return phases[phase_key]
    for key, phase in phases.items():
        if str(key) == phase_key:
            return phase
    raise unittest.SkipTest(f"phase {phase_key!r} not found in runtime config")


def _prepare_phase0b_params(params: dict, n_envs: int, *, timeout_seconds=SHORT_TIMEOUT_SECONDS) -> dict:
    adjusted = copy.deepcopy(params)
    adjusted.setdefault("curriculum", {})["phase"] = PHASE_KEY
    adjusted["frame_stack"] = 4
    adjusted.setdefault("environment_settings", {})["n_envs"] = int(n_envs)
    adjusted["environment_settings"]["resolution"] = adjusted["environment_settings"].get("resolution", "LOW")
    phase = _phase_lookup(adjusted, PHASE_KEY)
    if phase.get("type") != "navigation":
        raise AssertionError(f"phase {PHASE_KEY!r} must be navigation, got {phase.get('type')!r}")
    if timeout_seconds is not None:
        phase.setdefault("navigation", {})["time_limit_seconds"] = timeout_seconds
    phase["opponents"] = {"constant_agent": 1.0}
    adjusted.setdefault("opponent_history", {})["enabled"] = bool(
        adjusted.get("opponent_history", {}).get("enabled", True)
    )
    return adjusted


def _load_train_module(train_job, source_root: Path):
    with _cwd(source_root):
        return train_job.load_runtime_train_module()


def _build_env(train_job, train_module, params: dict, n_envs: int, grouped_cls=None, group_size: int | None = None):
    phase_config = train_module.PhaseConfig(params, PHASE_KEY)
    if not phase_config.is_navigation:
        raise AssertionError(f"phase {PHASE_KEY!r} unexpectedly built as non-navigation")

    if grouped_cls is None:
        return train_job.build_env(train_module, phase_config, params, n_envs, ENV_SEED, "cuda")

    original_subproc = train_job.SubprocVecEnv

    def grouped_factory(env_fns, start_method=None):
        return grouped_cls(env_fns, group_size=group_size, start_method=start_method or "spawn")

    with mock.patch.object(train_job, "SubprocVecEnv", grouped_factory):
        try:
            return train_job.build_env(train_module, phase_config, params, n_envs, ENV_SEED, "cuda")
        finally:
            train_job.SubprocVecEnv = original_subproc


def _make_action_plan(action_space, n_envs: int, steps: int, np_module):
    if getattr(action_space, "dtype", None) != np_module.float32:
        raise AssertionError(f"expected Box float32 action space, got dtype={getattr(action_space, 'dtype', None)}")

    shape = tuple(action_space.shape)
    low = np_module.asarray(action_space.low, dtype=np_module.float32)
    high = np_module.asarray(action_space.high, dtype=np_module.float32)
    rng = np_module.random.default_rng(ACTION_SEED)
    batch_shape = (int(n_envs), *shape)
    actions = []
    if np_module.all(np_module.isfinite(low)) and np_module.all(np_module.isfinite(high)):
        for _ in range(steps):
            actions.append(rng.uniform(low, high, size=batch_shape).astype(np_module.float32))
    else:
        for _ in range(steps):
            actions.append(rng.normal(0.0, 0.25, size=batch_shape).astype(np_module.float32))
    return actions


def _sanitize(value, np_module, path=()):
    if isinstance(value, np_module.ndarray):
        return np_module.array(value, copy=True)
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if key == "t" and path and path[-1] == "episode":
                continue
            cleaned[key] = _sanitize(item, np_module, (*path, str(key)))
        return cleaned
    if isinstance(value, (list, tuple)):
        return [_sanitize(item, np_module, path) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return copy.deepcopy(value)


def _capture_trace(env, actions, np_module):
    records = []
    try:
        obs = env.reset()
        records.append(
            {
                "tick": "reset",
                "obs": np_module.array(obs, copy=True),
                "reset_infos": _sanitize(getattr(env, "reset_infos", []), np_module),
            }
        )
        for tick, action_batch in enumerate(actions):
            obs, rewards, dones, infos = env.step(action_batch)
            records.append(
                {
                    "tick": tick,
                    "obs": np_module.array(obs, copy=True),
                    "reward": np_module.array(rewards, copy=True),
                    "done": np_module.array(dones, copy=True),
                    "infos": _sanitize(infos, np_module),
                    "reset_infos": _sanitize(getattr(env, "reset_infos", []), np_module),
                }
            )
    finally:
        env.close()
    return records


def _array_mismatch(expected, actual, np_module):
    if expected.shape != actual.shape:
        return f"shape expected={expected.shape} actual={actual.shape}"
    if expected.dtype != actual.dtype:
        return f"dtype expected={expected.dtype} actual={actual.dtype}"
    equal = expected == actual
    if expected.dtype.kind == "f":
        equal = equal | (np_module.isnan(expected) & np_module.isnan(actual))
    if bool(np_module.all(equal)):
        return None
    flat_index = int(np_module.flatnonzero(~equal.reshape(-1))[0])
    index = np_module.unravel_index(flat_index, expected.shape)
    return f"index={tuple(int(i) for i in index)} expected={expected[index]!r} actual={actual[index]!r}"


def _fail(tick, env_idx, field, detail):
    raise AssertionError(f"first divergence tick={tick} env={env_idx} field={field}: {detail}")


def _compare_arrays_by_env(expected, actual, tick, field, np_module):
    expected = np_module.asarray(expected)
    actual = np_module.asarray(actual)
    mismatch = _array_mismatch(expected, actual, np_module)
    if mismatch is None:
        return
    if expected.ndim > 0 and actual.ndim > 0 and expected.shape[:1] == actual.shape[:1]:
        for env_idx in range(expected.shape[0]):
            env_mismatch = _array_mismatch(expected[env_idx], actual[env_idx], np_module)
            if env_mismatch is not None:
                _fail(tick, env_idx, field, env_mismatch)
    _fail(tick, "all", field, mismatch)


def _compare_value(expected, actual, tick, env_idx, field, np_module):
    if isinstance(expected, np_module.ndarray) or isinstance(actual, np_module.ndarray):
        expected_array = np_module.asarray(expected)
        actual_array = np_module.asarray(actual)
        mismatch = _array_mismatch(expected_array, actual_array, np_module)
        if mismatch is not None:
            _fail(tick, env_idx, field, mismatch)
        return

    if isinstance(expected, dict) or isinstance(actual, dict):
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            _fail(tick, env_idx, field, f"type expected={type(expected).__name__} actual={type(actual).__name__}")
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            _fail(tick, env_idx, field, f"keys expected={sorted(expected_keys)} actual={sorted(actual_keys)}")
        for key in sorted(expected_keys):
            _compare_value(expected[key], actual[key], tick, env_idx, f"{field}.{key}", np_module)
        return

    if isinstance(expected, list) or isinstance(actual, list):
        if not isinstance(expected, list) or not isinstance(actual, list):
            _fail(tick, env_idx, field, f"type expected={type(expected).__name__} actual={type(actual).__name__}")
        if len(expected) != len(actual):
            _fail(tick, env_idx, field, f"length expected={len(expected)} actual={len(actual)}")
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            _compare_value(expected_item, actual_item, tick, index, f"{field}[{index}]", np_module)
        return

    if expected != actual:
        _fail(tick, env_idx, field, f"expected={expected!r} actual={actual!r}")


def _compare_traces(expected, actual, np_module):
    if len(expected) != len(actual):
        raise AssertionError(f"trace length expected={len(expected)} actual={len(actual)}")
    for expected_record, actual_record in zip(expected, actual):
        tick = expected_record["tick"]
        if tick != actual_record["tick"]:
            _fail(tick, "all", "tick", f"expected={tick!r} actual={actual_record['tick']!r}")
        _compare_arrays_by_env(expected_record["obs"], actual_record["obs"], tick, "obs", np_module)
        _compare_value(
            expected_record.get("reset_infos", []),
            actual_record.get("reset_infos", []),
            tick,
            "all",
            "reset_infos",
            np_module,
        )
        if tick == "reset":
            continue
        _compare_arrays_by_env(expected_record["reward"], actual_record["reward"], tick, "reward", np_module)
        _compare_arrays_by_env(expected_record["done"], actual_record["done"], tick, "done", np_module)
        _compare_value(expected_record["infos"], actual_record["infos"], tick, "all", "infos", np_module)


class FrozenGroupedGameParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        cls.np, yaml_module, cls.grouped_cls = _import_required_modules()
        cls.train_job_path, cls.source_root, cls.config_path = _resolve_runtime_paths()
        cls.train_job = _import_train_job(cls.train_job_path)
        cls.base_params = _load_yaml_config(cls.config_path, yaml_module)
        cls.train_module = _load_train_module(cls.train_job, cls.source_root)
        cls.steps = int(os.environ.get("AI2_GROUPED_GAME_STEPS", DEFAULT_STEPS))

    def _assert_grouped_matches_stock(self, *, n_envs: int, group_size: int, timeout_seconds=SHORT_TIMEOUT_SECONDS):
        if n_envs > 8:
            raise AssertionError("frozen grouped parity test is capped at 8 game processes")

        params = _prepare_phase0b_params(self.base_params, n_envs, timeout_seconds=timeout_seconds)

        with _cwd(self.source_root):
            stock_env = _build_env(self.train_job, self.train_module, params, n_envs)
            try:
                action_plan = _make_action_plan(stock_env.action_space, n_envs, self.steps, self.np)
            except Exception:
                stock_env.close()
                raise
            stock_trace = _capture_trace(stock_env, action_plan, self.np)

            grouped_env = _build_env(
                self.train_job,
                self.train_module,
                params,
                n_envs,
                grouped_cls=self.grouped_cls,
                group_size=group_size,
            )
            grouped_trace = _capture_trace(grouped_env, action_plan, self.np)

        _compare_traces(stock_trace, grouped_trace, self.np)

    def test_phase0b_grouped2_matches_stock_subproc_for_fixed_actions_and_short_timeouts(self):
        self._assert_grouped_matches_stock(n_envs=4, group_size=2)

    def test_phase0b_grouped4_matches_stock_subproc_for_fixed_actions_and_short_timeouts(self):
        self._assert_grouped_matches_stock(n_envs=8, group_size=4)

    def test_phase0b_grouped2_matches_stock_subproc_for_fixed_actions_and_control_timeouts(self):
        self._assert_grouped_matches_stock(n_envs=4, group_size=2, timeout_seconds=None)

    def test_phase0b_grouped4_matches_stock_subproc_for_fixed_actions_and_control_timeouts(self):
        self._assert_grouped_matches_stock(n_envs=8, group_size=4, timeout_seconds=None)

    def test_phase0b_grouped8_matches_stock_subproc_for_fixed_actions_and_short_timeouts(self):
        self._assert_grouped_matches_stock(n_envs=8, group_size=8)

    def test_phase0b_grouped8_matches_stock_subproc_for_fixed_actions_and_control_timeouts(self):
        self._assert_grouped_matches_stock(n_envs=8, group_size=8, timeout_seconds=None)


class RuntimePathResolutionTest(unittest.TestCase):
    def _layout(self, tmp_path: Path):
        root = tmp_path / "payload"
        source = root / "variants" / "corrected"
        scripts = root / "scripts"
        source.mkdir(parents=True)
        scripts.mkdir(parents=True)
        train_job = scripts / "train_job.py"
        train_job.write_text("# train job\n", encoding="utf-8")
        (source / "train.py").write_text("# frozen train\n", encoding="utf-8")
        control = source / "control.yaml"
        config = source / "config.yaml"
        control.write_text("curriculum: {phase: 0b}\n", encoding="utf-8")
        config.write_text("curriculum: {phase: old}\n", encoding="utf-8")
        return root, source, train_job, control, config

    def test_explicit_train_job_and_source_are_honored_independently(self):
        with tempfile.TemporaryDirectory() as tmp:
            _root, source, train_job, control, _config = self._layout(Path(tmp))
            with mock.patch.dict(os.environ, {"TRAIN_JOB": str(train_job), "AI2_CAMPAIGN_SOURCE": str(source)}, clear=True):
                resolved_train_job, resolved_source, resolved_config = _resolve_runtime_paths()

        self.assertEqual(resolved_train_job, train_job.resolve())
        self.assertEqual(resolved_source, source.resolve())
        self.assertEqual(resolved_config, control.resolve())

    def test_source_can_resolve_sibling_root_train_job_and_prefers_control_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            _root, source, train_job, control, _config = self._layout(Path(tmp))
            with mock.patch.dict(os.environ, {"AI2_CAMPAIGN_SOURCE": str(source)}, clear=True):
                resolved_train_job, resolved_source, resolved_config = _resolve_runtime_paths()

        self.assertEqual(resolved_train_job, train_job.resolve())
        self.assertEqual(resolved_source, source.resolve())
        self.assertEqual(resolved_config, control.resolve())

    def test_relative_ai2_config_resolves_inside_frozen_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            _root, source, train_job, control, _config = self._layout(Path(tmp))
            with mock.patch.dict(
                os.environ,
                {"TRAIN_JOB": str(train_job), "AI2_CAMPAIGN_SOURCE": str(source), "AI2_CONFIG": "control.yaml"},
                clear=True,
            ):
                _resolved_train_job, _resolved_source, resolved_config = _resolve_runtime_paths()

        self.assertEqual(resolved_config, control.resolve())

    def test_explicit_bad_runtime_paths_fail_instead_of_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            _root, source, _train_job, _control, _config = self._layout(Path(tmp))
            missing_train_job = Path(tmp) / "missing_train_job.py"
            missing_source = Path(tmp) / "missing_source"
            missing_config = Path(tmp) / "missing_control.yaml"

            with mock.patch.dict(os.environ, {"TRAIN_JOB": str(missing_train_job), "AI2_CAMPAIGN_SOURCE": str(source)}, clear=True):
                with self.assertRaises(AssertionError):
                    _resolve_runtime_paths()

            with mock.patch.dict(os.environ, {"AI2_CAMPAIGN_SOURCE": str(missing_source)}, clear=True):
                with self.assertRaises(AssertionError):
                    _resolve_runtime_paths()

            with mock.patch.dict(os.environ, {"AI2_CAMPAIGN_SOURCE": str(source), "AI2_CONFIG": str(missing_config)}, clear=True):
                with self.assertRaises(AssertionError):
                    _resolve_runtime_paths()


class RuntimePhaseConfigTest(unittest.TestCase):
    def test_phase0b_params_preserve_active_sequence_target_count(self):
        params = {
            "phases": {
                PHASE_KEY: {
                    "type": "navigation",
                    "navigation": {
                        "multi_target": 3,
                        "completion_mode": "sequence",
                        "time_limit_seconds": 20,
                    },
                    "opponents": {"constant_agent": 1.0},
                }
            }
        }

        adjusted = _prepare_phase0b_params(params, n_envs=4)

        self.assertEqual(adjusted["phases"][PHASE_KEY]["navigation"]["multi_target"], 3)
        self.assertEqual(adjusted["phases"][PHASE_KEY]["navigation"]["completion_mode"], "sequence")
        self.assertEqual(adjusted["phases"][PHASE_KEY]["navigation"]["time_limit_seconds"], SHORT_TIMEOUT_SECONDS)

    def test_phase0b_params_can_preserve_control_timeout(self):
        params = {
            "phases": {
                PHASE_KEY: {
                    "type": "navigation",
                    "navigation": {
                        "multi_target": 3,
                        "completion_mode": "sequence",
                        "time_limit_seconds": 20,
                    },
                }
            }
        }

        adjusted = _prepare_phase0b_params(params, n_envs=4, timeout_seconds=None)

        self.assertEqual(adjusted["phases"][PHASE_KEY]["navigation"]["time_limit_seconds"], 20)


class RuntimeCwdTest(unittest.TestCase):
    def test_parity_runner_holds_frozen_source_cwd_for_build_and_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_root = (Path(tmp) / "variants" / "corrected").resolve()
            source_root.mkdir(parents=True)

            case = FrozenGroupedGameParityTest(
                methodName="test_phase0b_grouped2_matches_stock_subproc_for_fixed_actions_and_short_timeouts"
            )
            case.source_root = source_root
            case.base_params = {"phases": {PHASE_KEY: {"type": "navigation", "navigation": {}}}}
            case.train_job = object()
            case.train_module = object()
            case.grouped_cls = object()
            case.steps = 1
            case.np = object()

            calls = []
            fake_env = types.SimpleNamespace(action_space=object(), close=lambda: None)

            def fake_build(_train_job, _train_module, _params, _n_envs, grouped_cls=None, group_size=None):
                calls.append(("build", grouped_cls is not None, Path.cwd()))
                return fake_env

            def fake_capture(_env, _actions, _np_module):
                calls.append(("capture", None, Path.cwd()))
                return []

            with mock.patch(__name__ + "._build_env", side_effect=fake_build), mock.patch(
                __name__ + "._make_action_plan", return_value=["action"]
            ), mock.patch(__name__ + "._capture_trace", side_effect=fake_capture), mock.patch(
                __name__ + "._compare_traces"
            ):
                case._assert_grouped_matches_stock(n_envs=4, group_size=2)

        self.assertEqual(
            calls,
            [
                ("build", False, source_root),
                ("capture", None, source_root),
                ("build", True, source_root),
                ("capture", None, source_root),
            ],
        )


if __name__ == "__main__":
    unittest.main()
