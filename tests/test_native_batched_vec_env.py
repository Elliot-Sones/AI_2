import importlib
from pathlib import Path
import unittest
from unittest.mock import patch


class CombinedWorkerTests(unittest.TestCase):
    def test_native_factories_and_batched_rng_worker_are_both_installed(self):
        import native_grouped_vec_env as native
        import batched_rng_vec_env as rng
        import grouped_vec_env as original
        combined = importlib.import_module("native_batched_vec_env")
        events = []
        def construct(_self, factories, **kwargs):
            events.append((factories, kwargs, original._worker))
        worker_before = original._worker
        with patch.object(original.GroupedSubprocVecEnv, "__init__", construct), patch("native_game_patch.verify_source_hash"):
            combined.GroupedSubprocVecEnv([dict], group_size=4, start_method="spawn", native_source_path=Path("/tmp/frozen.py"))
        factories, kwargs, worker = events[0]
        self.assertIsInstance(factories[0], native.NativeFactory)
        self.assertIs(factories[0].factory, dict)
        self.assertEqual(kwargs, {"group_size": 4, "start_method": "spawn"})
        self.assertIs(worker, rng._batched_worker)
        self.assertIs(original._worker, worker_before)


if __name__ == "__main__":
    unittest.main()
