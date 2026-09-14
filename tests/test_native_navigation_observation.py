"""Observation packing is compared with the actual frozen Player.get_obs AST."""
import ast
from enum import IntEnum
import importlib
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

import numpy as np

FROZEN = Path(os.environ.get("AI2_REPLAY_SOURCE", "/Users/elliot18/Projects/ml-research/AI_2-learning-validation-20260905/baselines/navigation-repair-20260905/payload/variants/corrected")) / "UTMIST-AI2-main/environment/environment.py"


def oracle(row, spawners, platform):
    tree = ast.parse(FROZEN.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Player")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "get_obs")
    class Facing(IntEnum):
        LEFT = -1
        RIGHT = 1
    class DroppedWeaponSpawner:
        pass
    namespace = {"Facing": Facing, "DroppedWeaponSpawner": DroppedWeaponSpawner}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(FROZEN), "exec"), namespace)
    entries = []
    for x, y, weapon_type in spawners:
        entry = DroppedWeaponSpawner() if weapon_type in (2, 3) else NS()
        entry.world_pos = (x, y)
        entry.active_weapon = object() if weapon_type else None
        entry.weapon_name = {2: "Spear", 3: "Hammer"}.get(weapon_type, "Punch")
        entries.append(entry)
    state = type("ActualState", (), {})()
    for index, field in ((7, "jumps_left"), (9, "recoveries_left"), (10, "dodge_timer"), (11, "stun_frames"), (14, "move_type")):
        setattr(state, field, row[index])
    player = NS(body=NS(position=NS(x=row[0], y=row[1]), velocity=NS(x=row[2], y=row[3])),
                facing=Facing(row[4]), is_on_floor=lambda: bool(row[5]), state=state,
                state_mapping={"ActualState": row[8]}, damage=row[12], stocks=row[13],
                weapon="held", weapon_mapping={"held": row[15]},
                env=NS(weapon_controller=NS(spawners=entries)),
                PLATFORM=NS(body=NS(position=NS(x=platform[0], y=platform[1]), velocity=NS(x=platform[2], y=platform[3]))))
    return namespace["get_obs"](player)


def fixtures(n):
    rng = np.random.default_rng(982)
    players = rng.uniform(-30, 30, (n, 2, 16))
    players[:, :, 4] = rng.choice([-1, 1], (n, 2))
    players[:, :, 5] = rng.integers(0, 2, (n, 2))
    players[:, :, 6] = 0  # Reserved; airborne is derived from grounded.
    spawners = rng.uniform(-30, 30, (n, 4, 3))
    spawners[:, :, 2] = rng.integers(0, 4, (n, 4))
    platform = rng.uniform(-30, 30, (n, 4))
    return players, spawners, platform


class ObservationTests(unittest.TestCase):
    def test_compiled_c_matches_frozen_get_obs_including_inactive_and_dropped_weapons(self):
        module = importlib.import_module("native_navigation_observation")
        p, s, platform = fixtures(32)
        with tempfile.TemporaryDirectory() as tmp:
            core = module.load_reference(Path(tmp))
            actual = core.pack_numpy(p, s, platform)
        expected = np.asarray([oracle(p[i, 0], s[i], platform[i]) + oracle(p[i, 1], s[i], platform[i]) for i in range(len(p))], dtype=np.float32)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.shape, (32, 64))
        with tempfile.TemporaryDirectory() as tmp:
            swapped = module.load_reference(Path(tmp)).pack_numpy(np.ascontiguousarray(p[:, ::-1]), s, platform)
        np.testing.assert_array_equal(swapped, np.concatenate([expected[:, 32:], expected[:, :32]], axis=1))

    def test_bad_shapes_dtype_nonfinite_and_reserved_fields_fail(self):
        module = importlib.import_module("native_navigation_observation")
        with tempfile.TemporaryDirectory() as tmp:
            core = module.load_reference(Path(tmp))
            p, s, platform = fixtures(2)
            with self.assertRaises(ValueError):
                core.pack_numpy(p.astype(np.float32), s, platform)
            with self.assertRaises(ValueError):
                core.pack_numpy(p[:, :, :15], s, platform)
            for index, value in ((0, float("nan")), (4, 0), (5, 2), (6, 1)):
                altered = p.copy()
                altered[1, 0, index] = value
                with self.subTest(index=index), self.assertRaisesRegex(ValueError, "lane 1"):
                    core.pack_numpy(altered, s, platform)

    def test_cuda_oracle_permutation_isolation_and_nondefault_stream(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Torch unavailable")
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        module = importlib.import_module("native_navigation_observation")
        with tempfile.TemporaryDirectory() as tmp:
            core = module.load_reference(Path(tmp), cuda=True)
            p, s, platform = fixtures(256)
            expected = module.load_reference(Path(tmp)).pack_numpy(p, s, platform)
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                inputs = [torch.as_tensor(a, device="cuda") for a in (p, s, platform)]
                out, errors = core.pack_cuda(*inputs)
                perm = torch.randperm(256, device="cuda")
                permuted, perm_errors = core.pack_cuda(*(a[perm] for a in inputs))
                inputs[0][3, 0, 0] = float("nan")
                invalid, invalid_errors = core.pack_cuda(*inputs)
            stream.synchronize()
            np.testing.assert_array_equal(out.cpu().numpy(), expected)
            self.assertTrue(torch.equal(permuted, out[perm]))
            self.assertEqual(errors.sum().item(), 0)
            self.assertEqual(perm_errors.sum().item(), 0)
            self.assertEqual(invalid_errors.nonzero().flatten().tolist(), [3])
            self.assertTrue(torch.isnan(invalid[3]).all())
            self.assertTrue(torch.equal(invalid[:3], out[:3]))
            with self.assertRaises(ValueError):
                core.pack_cuda(inputs[0].float(), *inputs[1:])


if __name__ == "__main__":
    unittest.main()
