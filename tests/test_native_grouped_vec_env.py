import importlib
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch


class NativeGroupedFactoryTests(unittest.TestCase):
    def module(self):
        return importlib.import_module('ai2.simulation.native_grouped_vec_env')

    def test_factory_installs_after_original_creation_and_passes_pinned_path(self):
        module = self.module()
        env = types.SimpleNamespace(close=lambda: None)
        events = []
        def factory():
            events.append("created")
            return env
        def install(value, **kwargs):
            events.append((value, kwargs))
        path = Path("/tmp/frozen/environment.py")
        wrapped = module.native_factory(factory, source_path=path)
        with patch('ai2.simulation.native_game_patch.install_native_game_patch', side_effect=install):
            self.assertIs(wrapped(), env)
        self.assertEqual(events, ["created", (env, {"source_path": path})])

    def test_failed_install_closes_constructed_environment(self):
        module = self.module()
        closed = []
        env = types.SimpleNamespace(close=lambda: closed.append(True))
        wrapped = module.native_factory(lambda: env, source_path=Path("/tmp/frozen.py"))
        with patch('ai2.simulation.native_game_patch.install_native_game_patch', side_effect=ValueError("hash")):
            with self.assertRaisesRegex(ValueError, "hash"):
                wrapped()
        self.assertEqual(closed, [True])

    def test_factory_is_pickleable_without_live_native_handles(self):
        module = self.module()
        import pickle
        wrapped = module.native_factory(dict, source_path=Path("/tmp/frozen.py"))
        restored = pickle.loads(pickle.dumps(wrapped))
        self.assertEqual(restored.source_path, Path("/tmp/frozen.py"))
        self.assertIs(restored.factory, dict)

    def test_close_restores_patch_even_when_original_close_raises(self):
        module = self.module()
        events = []
        def close():
            events.append("close")
            raise RuntimeError("close failed")
        env = types.SimpleNamespace(close=close)
        active = types.SimpleNamespace(restore=lambda: events.append("restore"))
        with patch('ai2.simulation.native_game_patch.install_native_game_patch', return_value=active):
            wrapped = module.native_factory(lambda: env, source_path=Path("/tmp/frozen.py"))()
        with self.assertRaisesRegex(RuntimeError, "close failed"):
            wrapped.close()
        self.assertEqual(events, ["close", "restore"])


if __name__ == "__main__":
    unittest.main()
