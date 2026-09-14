import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import contextmanager
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "benchmark_policy_rollout_parity.py"


def load_module():
    spec = importlib.util.spec_from_file_location("benchmark_policy_rollout_parity", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_policy_rollout_parity"] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def preserved_import_state(*module_names):
    original_modules = {name: sys.modules.get(name) for name in module_names}
    missing = {name for name in module_names if name not in sys.modules}
    original_path = list(sys.path)
    try:
        yield
    finally:
        sys.path[:] = original_path
        for name in module_names:
            if name in missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original_modules[name]


class TinyArray:
    def __init__(self, payload, dtype="float32", shape=None):
        self.payload = payload
        self.dtype = dtype
        self.shape = shape or (len(payload),)

    def tolist(self):
        return self.payload

    @property
    def nbytes(self):
        return len(json.dumps(self.payload).encode("utf-8"))


class FakeNumpy:
    ndarray = TinyArray


class PolicyRolloutParityTests(unittest.TestCase):
    def test_sanitize_excludes_only_monitor_episode_wallclock_t(self):
        module = load_module()
        info = {
            "episode": {"r": 1.25, "l": 7, "t": 123.456},
            "terminal_observation": TinyArray([1, 2, 3]),
            "t": "keep-top-level",
        }

        sanitized = module.sanitize(info, FakeNumpy)

        self.assertEqual(sanitized["episode"], {"r": 1.25, "l": 7})
        self.assertEqual(sanitized["terminal_observation"]["sha256"], hashlib.sha256(b"[1,2,3]").hexdigest())
        self.assertEqual(sanitized["t"], "keep-top-level")

    def test_first_divergence_reports_tick_env_field_and_array_detail(self):
        module = load_module()
        left = {
            "tick": 3,
            "obs": [
                [0.0, 1.0],
                [2.0, 3.0],
            ],
            "reward": [0.0, 1.0],
            "done": [False, False],
            "infos": [{}, {}],
            "reset_infos": [{}, {}],
        }
        right = {
            "tick": 3,
            "obs": [
                [0.0, 1.0],
                [2.0, 4.0],
            ],
            "reward": [0.0, 1.0],
            "done": [False, False],
            "infos": [{}, {}],
            "reset_infos": [{}, {}],
        }

        diff = module.first_divergence([left], [right])

        self.assertEqual(diff["tick"], 3)
        self.assertEqual(diff["env"], 1)
        self.assertEqual(diff["field"], "obs")
        self.assertIn("index=(1, 1)", diff["detail"])

    def test_first_divergence_ignores_episode_t_but_catches_other_info_fields(self):
        module = load_module()
        base = {
            "tick": 0,
            "obs": [[0.0]],
            "reward": [0.0],
            "done": [False],
            "infos": [{"episode": {"r": 1.0, "l": 5, "t": 1.0}, "x": 1}],
            "reset_infos": [{}],
        }
        same_except_t = json.loads(json.dumps(base))
        same_except_t["infos"][0]["episode"]["t"] = 2.0

        self.assertIsNone(module.first_divergence([base], [same_except_t]))

        changed = json.loads(json.dumps(same_except_t))
        changed["infos"][0]["x"] = 2
        diff = module.first_divergence([base], [changed])

        self.assertEqual(diff["field"], "infos[0].x")

    def test_first_divergence_treats_tuple_and_list_sequences_as_same_infos(self):
        module = load_module()
        stock = {
            "tick": 0,
            "obs": [[0.0], [1.0]],
            "reward": [0.0, 1.0],
            "done": [False, True],
            "infos": (
                {"episode": {"r": 1.0, "l": 5, "t": 1.0}, "terminal_observation": TinyArray([1, 2, 3])},
                {"nested": ({"value": 7},)},
            ),
            "reset_infos": ({"seed": 11}, {"seed": 12}),
        }
        grouped = {
            "tick": 0,
            "obs": [[0.0], [1.0]],
            "reward": [0.0, 1.0],
            "done": [False, True],
            "infos": [
                {"episode": {"r": 1.0, "l": 5, "t": 2.0}, "terminal_observation": TinyArray([1, 2, 3])},
                {"nested": [{"value": 7}]},
            ],
            "reset_infos": [{"seed": 11}, {"seed": 12}],
        }

        self.assertIsNone(module.first_divergence([stock], [grouped]))

        grouped["infos"][1]["nested"][0]["value"] = 8
        diff = module.first_divergence([stock], [grouped])

        self.assertEqual(diff["field"], "infos[1].nested[0].value")

    def test_action_sampling_summary_distinguishes_natural_from_forced_same_actions(self):
        module = load_module()
        natural_stock = TinyArray([[1.0, 2.0], [3.0, 4.0]])
        natural_grouped = TinyArray([[1.0, 2.5], [3.0, 4.0]])
        shared = TinyArray([[9.0, 8.0], [7.0, 6.0]])

        summary = module.action_sampling_summary(natural_stock, natural_grouped, shared)

        self.assertFalse(summary["natural_policy_actions_equal"])
        self.assertEqual(summary["natural_action_diff"]["index"], [0, 1])
        self.assertEqual(summary["shared_action"]["shape"], [2, 2])
        self.assertEqual(summary["shared_action"]["sha256"], hashlib.sha256(b"[[9.0,8.0],[7.0,6.0]]").hexdigest())

    def test_build_provenance_records_hashes_and_runtime_scope(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "control.yaml"
            checkpoint = tmp_path / "selected.zip"
            grouped = tmp_path / "grouped_vec_env.py"
            train_job = tmp_path / "train_job.py"
            source = tmp_path / "source"
            for path in (config, checkpoint, grouped, train_job):
                path.write_text(path.name, encoding="utf-8")
            source.mkdir()
            args = argparse.Namespace(
                config=config,
                checkpoint=checkpoint,
                grouped_env_path=grouped,
                train_job=train_job,
                source=source,
                output=tmp_path / "out",
                device="cuda",
                phase="0b",
                n_envs=8,
                steps=1000,
                group_size=4,
                seed=20260906,
            )

            payload = module.build_provenance(args)

        self.assertEqual(payload["diagnostic"], "policy_rollout_parity")
        self.assertEqual(payload["runtime"]["device"], "cuda")
        self.assertEqual(payload["runtime"]["steps"], 1000)
        self.assertTrue(payload["files"]["checkpoint"]["exists"])
        self.assertIn("sha256", payload["files"]["grouped_env"])

    def test_seed_vec_envs_applies_same_explicit_seed_before_initial_reset(self):
        module = load_module()

        class FakeVecEnv:
            def __init__(self):
                self.seeds = []

            def seed(self, seed):
                self.seeds.append(seed)

        stock = FakeVecEnv()
        grouped = FakeVecEnv()

        module.seed_vec_envs(stock, grouped, 123)

        self.assertEqual(stock.seeds, [123])
        self.assertEqual(grouped.seeds, [123])

    def test_train_job_loader_delegates_to_campaign_stable_importer(self):
        module = load_module()
        train_job_path = Path("/tmp/runtime/scripts/train_job.py")
        sentinel = object()

        with mock.patch.object(module.benchmark_campaign, "import_train_job_module", return_value=sentinel) as importer:
            loaded = module.load_train_job_module(train_job_path)

        self.assertIs(loaded, sentinel)
        importer.assert_called_once_with(train_job_path)

    def test_grouped_env_loader_imports_by_real_filename_for_spawn(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            grouped_path = tmp_path / "grouped_vec_env.py"
            grouped_path.write_text("class GroupedSubprocVecEnv:\n    pass\n", encoding="utf-8")
            with preserved_import_state("grouped_vec_env", "campaign_grouped_vec_env"):
                sys.modules.pop("grouped_vec_env", None)
                sys.modules.pop("campaign_grouped_vec_env", None)

                grouped_cls = module.load_grouped_env_class(grouped_path)

                self.assertEqual(grouped_cls.__name__, "GroupedSubprocVecEnv")
                self.assertEqual(grouped_cls.__module__, "grouped_vec_env")
                self.assertIn("grouped_vec_env", sys.modules)


if __name__ == "__main__":
    unittest.main()
