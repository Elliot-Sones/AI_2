"""Native input decoding compared to the actual frozen input-handler source."""
import ast
import ctypes
from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FROZEN = Path(os.environ.get('AI2_FROZEN_STAGE_SOURCE', str(
    ROOT.parent / 'AI_2-learning-validation-20260905/baselines/navigation-repair-20260905/'
    'payload/variants/corrected/UTMIST-AI2-main/environment/environment.py')))
WIDTH = 35


def frozen_handler():
    tree = ast.parse(FROZEN.read_text())
    names = {'Facing', 'KeyStatus', 'HorizontalState', 'PlayerInputHandler'}
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    if len(nodes) != 4:
        raise AssertionError('Frozen input classes missing')
    namespace = {'Enum': Enum, 'dataclass': dataclass, 'np': np, '__name__': __name__}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(FROZEN), 'exec'), namespace)
    return namespace['PlayerInputHandler']()


def oracle_values(handler):
    return (
        [int(handler.key_status[k].held) for k in handler.key_names]
        + [int(handler.key_status[k].just_pressed) for k in handler.key_names]
        + [int(handler.key_status[k].just_released) for k in handler.key_names]
        + [int(handler.raw_vertical), int(handler.raw_horizontal), int(handler.no_horizontal),
           handler.horizontal_state.value, handler.last_direction]
    )


class NativeInputTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = ROOT / 'native/navigation_input.c'
        if not source.is_file():
            return
        cls.build = tempfile.TemporaryDirectory(prefix='ai2-input-')
        cls.addClassCleanup(cls.build.cleanup)
        path = Path(cls.build.name) / 'input.so'
        command = [os.environ.get('CC', 'cc'), '-std=c99', '-O2', '-ffp-contract=off',
                   '-Wall', '-Wextra', '-Werror', '-fPIC', '-shared', str(source), '-o', str(path)]
        subprocess.run(command, check=True, capture_output=True, timeout=60)
        cls.lib = ctypes.CDLL(str(path))
        cls.lib.ai2_input_init.argtypes = [ctypes.POINTER(ctypes.c_int32)]
        cls.lib.ai2_input_init.restype = None
        cls.lib.ai2_input_step.argtypes = [ctypes.POINTER(ctypes.c_double),
                                         ctypes.POINTER(ctypes.c_int32),
                                         ctypes.POINTER(ctypes.c_int32)]
        cls.lib.ai2_input_step.restype = ctypes.c_int

    def require_lib(self):
        self.assertTrue(hasattr(self, 'lib'), 'native input implementation is missing')

    def run_sequence(self, sequence, *, inplace=False):
        self.require_lib()
        before = (ctypes.c_int32 * WIDTH)()
        self.lib.ai2_input_init(before)
        handler = frozen_handler()
        self.assertEqual(list(before), oracle_values(handler))
        for tick, action in enumerate(sequence):
            after = before if inplace else (ctypes.c_int32 * WIDTH)()
            inputs = (ctypes.c_double * 10)(*action)
            snapshot = list(before)
            self.assertEqual(self.lib.ai2_input_step(inputs, before, after), 0)
            handler.update(np.array(action, dtype=np.float64))
            self.assertEqual(list(after), oracle_values(handler), f'tick={tick}')
            self.assertEqual(list(inputs), list(action))
            if not inplace:
                self.assertEqual(list(before), snapshot)
            before = after

    def test_pressed_held_released_and_threshold(self):
        self.run_sequence([[x] * 10 for x in (0.0, 0.5, np.nextafter(0.5, 1.0), 1.0, -2.0, 2.0, 0.0)])

    def test_simultaneous_directions_preserve_press_priority(self):
        sequence = []
        for pressed in ((), (1,), (1, 3), (3,), (), (3,), (1, 3), (1,), (), (1, 3), ()):
            sequence.append([float(i in pressed) for i in range(10)])
        self.run_sequence(sequence)

    def test_all_buttons_and_long_history_match_original(self):
        rng = random.Random(74)
        self.run_sequence([[rng.choice([0.0, 0.5, 0.500001, 1.0]) for _ in range(10)] for _ in range(2048)])

    def test_inplace_update_preserves_previous_button_edges(self):
        rng = random.Random(19)
        self.run_sequence([[float(rng.randrange(2)) for _ in range(10)] for _ in range(128)], inplace=True)

    def test_invalid_action_does_not_mutate_outputs(self):
        self.require_lib()
        before = (ctypes.c_int32 * WIDTH)()
        self.lib.ai2_input_init(before)
        for invalid in (float('nan'), float('inf'), -float('inf')):
            action = (ctypes.c_double * 10)(*([0.0] * 9 + [invalid]))
            after = (ctypes.c_int32 * WIDTH)(*([77] * WIDTH))
            self.assertNotEqual(self.lib.ai2_input_step(action, before, after), 0)
            self.assertEqual(list(after), [77] * WIDTH)

    def test_invalid_horizontal_state_does_not_mutate_outputs(self):
        self.require_lib()
        action = (ctypes.c_double * 10)()
        before = (ctypes.c_int32 * WIDTH)()
        before[33] = 5
        after = (ctypes.c_int32 * WIDTH)(*([77] * WIDTH))
        self.assertNotEqual(self.lib.ai2_input_step(action, before, after), 0)
        self.assertEqual(list(after), [77] * WIDTH)

    def test_cuda_lanes_and_nondefault_stream_match_cpu(self):
        self.require_lib()
        try:
            import torch
        except ImportError:
            self.skipTest('Torch unavailable; CUDA validation must run remotely')
        if not torch.cuda.is_available() or not shutil.which('nvcc'):
            self.skipTest('CUDA runtime/compiler unavailable')
        path = Path(self.build.name) / 'input_cuda.so'
        subprocess.run(['nvcc', '-O2', '--fmad=false', '--shared', '-Xcompiler', '-fPIC',
                        str(ROOT / 'native/navigation_input.cu'), '-o', str(path)],
                       check=True, capture_output=True, timeout=120)
        lib = ctypes.CDLL(str(path))
        lib.ai2_input_step_cuda.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int, ctypes.c_void_p]
        lib.ai2_input_step_cuda.restype = ctypes.c_int
        rng = np.random.default_rng(591)
        for lanes in (1, 256, 1024, 4096):
            state = np.zeros((lanes, WIDTH), dtype=np.int32)
            state[:, 32] = 1
            # Independent lane histories; each iteration compares real CPU C outputs.
            for tick in range(8):
                actions = rng.choice([0.0, 0.5, 0.500001, 1.0], (lanes, 10))
                expected = np.empty_like(state)
                for lane in range(lanes):
                    result = self.lib.ai2_input_step(
                        actions[lane].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        state[lane].ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                        expected[lane].ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
                    self.assertEqual(result, 0)
                order = rng.permutation(lanes)
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    a = torch.tensor(actions[order], device='cuda', dtype=torch.float64)
                    b = torch.tensor(state[order], device='cuda', dtype=torch.int32)
                    out = torch.empty_like(b)
                    errors = torch.empty(lanes, device='cuda', dtype=torch.int32)
                    status = lib.ai2_input_step_cuda(a.data_ptr(), b.data_ptr(), out.data_ptr(),
                                                    errors.data_ptr(), lanes, stream.cuda_stream)
                    self.assertEqual(status, 0)
                stream.synchronize()
                self.assertEqual(errors.cpu().tolist(), [0] * lanes)
                np.testing.assert_array_equal(out.cpu().numpy(), expected[order])
                np.testing.assert_array_equal(b.cpu().numpy(), state[order])
                state = expected

        # Bad lanes must not corrupt their output or their valid neighbors.
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            a = torch.zeros((3, 10), device='cuda', dtype=torch.float64)
            a[0, 9] = float('nan')
            b = torch.zeros((3, WIDTH), device='cuda', dtype=torch.int32)
            b[:, 32] = 1
            b[1, 33] = 5
            out = torch.full_like(b, 77)
            errors = torch.empty(3, device='cuda', dtype=torch.int32)
            self.assertEqual(lib.ai2_input_step_cuda(
                a.data_ptr(), b.data_ptr(), out.data_ptr(), errors.data_ptr(),
                3, stream.cuda_stream), 0)
        stream.synchronize()
        self.assertEqual(errors.cpu().tolist(), [3, 2, 0])
        self.assertEqual(out[:2].cpu().tolist(), [[77] * WIDTH] * 2)
        self.assertEqual(out[2].cpu().tolist(), oracle_values(frozen_handler()))


if __name__ == '__main__':
    unittest.main()
