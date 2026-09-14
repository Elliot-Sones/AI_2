import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validate_native_navigation.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class ValidateNativeNavigationTests(unittest.TestCase):
    def test_parser_defaults_are_bounded_and_document_postphysics_scope(self):
        module = load_module()

        args = module.parse_args(["capture", "--source-root", str(ROOT), "--config", "control.yaml", "--output", "capture.jsonl"])

        self.assertEqual(args.steps, 1200)
        self.assertEqual(args.seeds, "45,46,47")
        self.assertEqual(args.frame_stack, 1)
        self.assertEqual(args.controller, "adaptive")
        self.assertEqual(module.CONTRACT_CLI_SCOPE, "postphysics_navigation_rule_reducer_only")

    def test_source_provenance_requires_existing_files_and_hashes(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = create_source_files(root)

            proof = module.source_provenance(
                source_root=files["source_root"],
                config=files["config"],
                wrapper_source=files["wrapper_source"],
            )

            self.assertEqual(proof["source_root"], str(files["source_root"].resolve()))
            self.assertEqual(proof["config"]["sha256"], sha256_file(files["config"]))
            self.assertEqual(proof["wrapper_source"]["sha256"], sha256_file(files["wrapper_source"]))
            self.assertEqual(proof["positions"]["sha256"], sha256_file(files["positions"]))
            self.assertIn("native_navigation_contract", proof)

            files["config"].unlink()
            with self.assertRaisesRegex(RuntimeError, "config"):
                module.source_provenance(source_root=files["source_root"], config=files["config"], wrapper_source=files["wrapper_source"])

            files["config"].write_text("config\n")
            files["positions"].unlink()
            with self.assertRaisesRegex(RuntimeError, "positions"):
                module.source_provenance(source_root=files["source_root"], config=files["config"], wrapper_source=files["wrapper_source"])

    def test_verify_refuses_empty_records_before_native_execution(self):
        module = load_module()
        capture = FakeCapture([], metadata={"type": "metadata"})
        native = FakeNative()

        with tempfile.TemporaryDirectory() as tmp:
            receipt = Path(tmp) / "receipt.json"
            code = module.verify_capture(Path("records.jsonl"), receipt, native_module=native, capture_module=capture, require_cuda=False)

            self.assertEqual(code, 1)
            payload = json.loads(receipt.read_text())
            self.assertEqual(payload["status"], "failed")
            self.assertEqual(payload["reason"], "zero_records")
            self.assertEqual(Path(payload["capture_path"]), (Path.cwd() / "records.jsonl").resolve())
            self.assertIn("build_dir", payload)
            self.assertEqual(payload["records"], 0)
            self.assertEqual(native.cpu_steps, 0)

    def test_verify_returns_nonzero_and_writes_first_cpu_difference(self):
        module = load_module()
        records = [record("case-0", 0), record("case-1", 1)]
        capture = FakeCapture(records)
        native = FakeNative(cpu_bad_index=1)

        with tempfile.TemporaryDirectory() as tmp:
            receipt = Path(tmp) / "receipt.json"
            code = module.verify_capture(Path("records.jsonl"), receipt, native_module=native, capture_module=capture, require_cuda=False)

            self.assertEqual(code, 1)
            payload = json.loads(receipt.read_text())
            self.assertEqual(payload["status"], "failed")
            self.assertEqual(payload["cpu"]["passed"], False)
            self.assertEqual(payload["cpu"]["first_difference"]["tick"], 1)
            self.assertEqual(payload["first_difference"]["backend"], "cpu")

    def test_verify_fails_closed_on_nonfinite_native_output(self):
        module = load_module()
        capture = FakeCapture([record("case-0", 0)])
        native = FakeNative(nonfinite_index=0)

        with tempfile.TemporaryDirectory() as tmp:
            receipt = Path(tmp) / "receipt.json"
            code = module.verify_capture(Path("records.jsonl"), receipt, native_module=native, capture_module=capture, require_cuda=False)

            self.assertEqual(code, 1)
            payload = json.loads(receipt.read_text())
            self.assertEqual(payload["status"], "failed")
            self.assertIn("non-finite", payload["reason"])

    def test_verify_refuses_cuda_without_torch_cuda_and_nvcc(self):
        module = load_module()
        records = [record("case-0", 0)]
        capture = FakeCapture(records)
        native = FakeNative(cuda_available=False)

        with tempfile.TemporaryDirectory() as tmp:
            receipt = Path(tmp) / "receipt.json"
            code = module.verify_capture(Path("records.jsonl"), receipt, native_module=native, capture_module=capture, require_cuda=True)

            self.assertEqual(code, 1)
            payload = json.loads(receipt.read_text())
            self.assertEqual(payload["status"], "failed")
            self.assertIn("refusing CPU fallback", payload["reason"])
            self.assertEqual(native.cuda_steps, 0)

    def test_verify_cuda_batches_preserve_order_with_padding(self):
        module = load_module()
        records = [record(f"case-{index}", index) for index in range(3)]
        capture = FakeCapture(records)
        native = FakeNative(cuda_available=True)

        with tempfile.TemporaryDirectory() as tmp:
            receipt = Path(tmp) / "receipt.json"
            code = module.verify_capture(
                Path("records.jsonl"),
                receipt,
                native_module=native,
                capture_module=capture,
                require_cuda=True,
                cuda_batch_sizes=(2, 4),
                torch_module=FakeTorch(),
                nvcc_path="/usr/local/cuda/bin/nvcc",
            )

            self.assertEqual(code, 0)
            payload = json.loads(receipt.read_text())
            self.assertEqual(payload["status"], "passed")
            self.assertIn("native_sources", payload)
            self.assertIn("navigation_contract.cu", payload["native_sources"])
            self.assertEqual([item["batch_size"] for item in payload["cuda"]], [2, 4])
            self.assertEqual(native.cuda_lanes, [2, 2, 4])
            self.assertEqual(payload["cuda"][0]["compared_records"], 3)
            self.assertTrue(payload["cuda"][1]["passed"])

    def test_verify_resolves_paths_before_helpers_can_change_cwd(self):
        module = load_module()
        original_cwd = Path.cwd()
        records = [record("case-0", 0)]
        native = FakeNative()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = FakeCapture(records, chdir_on_read=root)
            receipt = Path("relative_verify_receipt.json")
            capture_path = root / "records.jsonl"
            build_dir = Path("relative_build_dir")

            os.chdir(root)
            try:
                code = module.verify_capture(
                    capture_path.name,
                    receipt,
                    native_module=native,
                    capture_module=capture,
                    build_dir=build_dir,
                    require_cuda=False,
                )
            finally:
                os.chdir(original_cwd)

            payload = json.loads((root / receipt).read_text())
            self.assertEqual(code, 0)
            self.assertEqual(Path.cwd(), original_cwd)
            self.assertEqual(Path(payload["capture_path"]), capture_path.resolve())
            self.assertEqual(Path(payload["build_dir"]), (root / build_dir).resolve())
            self.assertTrue((root / receipt).exists())

    def test_verify_fails_closed_when_capture_helpers_are_missing(self):
        module = load_module()
        capture = argparse.Namespace(read_jsonl=lambda _path: ({"type": "metadata"}, [record("case-0", 0)]))

        with tempfile.TemporaryDirectory() as tmp:
            receipt = Path(tmp) / "receipt.json"
            code = module.verify_capture(Path("records.jsonl"), receipt, native_module=FakeNative(), capture_module=capture, require_cuda=False)

            self.assertEqual(code, 1)
            payload = json.loads(receipt.read_text())
            self.assertIn("native_case_from_record", payload["reason"])
            self.assertEqual(Path(payload["capture_path"]), (Path.cwd() / "records.jsonl").resolve())

    def test_capture_rejects_more_than_three_seeds(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            files = create_source_files(Path(tmp))
            args = module.parse_args(
                [
                    "capture",
                    "--source-root",
                    str(files["source_root"]),
                    "--config",
                    str(files["config"]),
                    "--output",
                    str(Path(tmp) / "capture.jsonl"),
                    "--seeds",
                    "1,2,3,4",
                ]
            )

            with self.assertRaisesRegex(module.ValidationError, "at most 3 seeds"):
                module.capture_validation_records(args)

    def test_capture_uses_exact_wrapper_source_and_records_seed_total(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = create_source_files(root)
            explicit_wrapper = files["source_root"] / "explicit_train.py"
            explicit_wrapper.write_text("explicit wrapper\n")
            output = root / "capture.jsonl"
            replay = FakeReplay()
            capture = FakeCaptureWriter()
            args = module.parse_args(
                [
                    "capture",
                    "--source-root",
                    str(files["source_root"]),
                    "--config",
                    str(files["config"]),
                    "--wrapper-source",
                    str(explicit_wrapper),
                    "--output",
                    str(output),
                    "--steps",
                    "2",
                    "--seeds",
                    "7,8",
                    "--controller",
                    "scripted",
                ]
            )

            with temporary_modules(benchmark_replay=replay, navigation_contract_capture=capture):
                result = module.capture_validation_records(args)

            self.assertEqual(result["records"], 4)
            self.assertEqual([call["wrapper_source"] for call in replay.build_calls], [explicit_wrapper.resolve(), explicit_wrapper.resolve()])
            metadata = capture.metadata
            self.assertEqual(metadata["extra"]["action_count"], 4)
            self.assertEqual(metadata["extra"]["seeds"], [7, 8])
            self.assertEqual(metadata["extra"]["steps_per_seed"], 2)
            self.assertEqual(metadata["extra"]["provenance"]["wrapper_source"]["sha256"], sha256_file(explicit_wrapper))
            self.assertEqual(metadata["extra"]["truthy_observed_events"], [])
            self.assertEqual([item["seed"] for item in capture.records], [7, 7, 8, 8])

    def test_capture_writes_terminal_record_before_resetting_whole_env(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = create_source_files(root)
            replay = FakeReplay()
            capture = FakeCaptureWriter(terminate_ticks={0})
            args = module.parse_args(
                [
                    "capture",
                    "--source-root",
                    str(files["source_root"]),
                    "--config",
                    str(files["config"]),
                    "--output",
                    str(root / "capture.jsonl"),
                    "--steps",
                    "2",
                    "--seeds",
                    "7",
                    "--controller",
                    "scripted",
                ]
            )

            with temporary_modules(benchmark_replay=replay, navigation_contract_capture=capture):
                module.capture_validation_records(args)

            env = replay.envs[0]
            self.assertEqual([item["tick"] for item in capture.records], [0, 1])
            self.assertEqual(capture.records[0]["expected"]["terminated"], True)
            self.assertEqual(env.reset_count, 2)
            self.assertEqual(capture.capture_order[0], ("capture", 0, 0))
            self.assertEqual(capture.capture_order[1], ("capture", 1, 1))

    def test_capture_resolves_relative_paths_before_loader_changes_cwd(self):
        module = load_module()
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = create_source_files(root)
            output = Path("relative_capture.jsonl")
            replay = FakeReplay(chdir_to=files["source_root"])
            capture = FakeCaptureWriter()
            os.chdir(root)
            try:
                args = module.parse_args(
                    [
                        "capture",
                        "--source-root",
                        str(files["source_root"]),
                        "--config",
                        files["config"].name,
                        "--output",
                        str(output),
                        "--steps",
                        "1",
                        "--seeds",
                        "7",
                        "--controller",
                        "scripted",
                    ]
                )

                with temporary_modules(benchmark_replay=replay, navigation_contract_capture=capture):
                    result = module.capture_validation_records(args)
            finally:
                os.chdir(original_cwd)

            self.assertEqual(Path.cwd(), original_cwd)
            self.assertEqual(replay.build_calls[0]["config"], files["config"].resolve())
            self.assertEqual(Path(result["output"]), (root / output).resolve())
            self.assertTrue((root / output).exists())
            self.assertFalse((files["source_root"] / output).exists())
            self.assertEqual(capture.metadata["extra"]["truthy_observed_events"], [])


def load_module():
    spec = importlib.util.spec_from_file_location("validate_native_navigation", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["validate_native_navigation"] = module
    spec.loader.exec_module(module)
    return module


def create_source_files(root):
    source_root = root / "source"
    wrapper_source = source_root / "train.py"
    env = source_root / "UTMIST-AI2-main/environment/environment.py"
    positions = source_root / "positions.json"
    config = root / "control.yaml"
    env.parent.mkdir(parents=True)
    source_root.mkdir(parents=True, exist_ok=True)
    wrapper_source.write_text("wrapper\n")
    env.write_text("environment\n")
    positions.write_text("{}\n")
    config.write_text("config\n")
    return {"source_root": source_root, "wrapper_source": wrapper_source, "config": config, "positions": positions}


def record(case_id, tick):
    return {"case_id": case_id, "tick": tick, "expected": {"value": tick}, "input": {"executed_action": [0.0] * 10}}


class FakeCapture:
    def __init__(self, records, metadata=None, *, chdir_on_read=None):
        self.records = list(records)
        self.metadata = metadata or {"type": "metadata"}
        self.chdir_on_read = chdir_on_read

    def read_jsonl(self, _path):
        if self.chdir_on_read is not None:
            os.chdir(self.chdir_on_read)
        return self.metadata, list(self.records)

    def native_case_from_record(self, item):
        return "config", {"tick": item["tick"]}, {"tick": item["tick"]}

    def native_result_for_record(self, item, result):
        return {"value": result["tick"]}

    def compare_native_results(self, records, native_results, *, abs_tol=1e-9):
        records = list(records)
        native_results = list(native_results)
        if not records:
            return {"passed": False, "reason": "zero_records", "compared_records": 0, "first_difference": None}
        for index, (left, right) in enumerate(zip(records, native_results)):
            if left["expected"] != right:
                return {
                    "passed": False,
                    "reason": "value_mismatch",
                    "compared_records": index + 1,
                    "first_difference": {"tick": left["tick"], "field": "value", "expected": left["expected"], "actual": right},
                }
        return {"passed": True, "reason": "match", "compared_records": len(records), "first_difference": None}


class FakeNative:
    def __init__(self, *, cpu_bad_index=None, nonfinite_index=None, cuda_available=True):
        self.cpu_bad_index = cpu_bad_index
        self.nonfinite_index = nonfinite_index
        self.cuda_available = cuda_available
        self.cpu_steps = 0
        self.cuda_steps = 0
        self.cuda_lanes = []

    def load_cpu_reference(self, _build_dir):
        native = self

        class Cpu:
            def step(self, _config, state, _post):
                native.cpu_steps += 1
                if state["tick"] == native.cpu_bad_index:
                    return {"tick": 999}
                if state["tick"] == native.nonfinite_index:
                    return {"tick": float("nan")}
                return {"tick": state["tick"]}

        return Cpu()

    def can_run_cuda_contract(self):
        return self.cuda_available

    def load_cuda_contract(self, _build_dir):
        native = self

        class Cuda:
            def step(self, tensors):
                native.cuda_steps += 1
                native.cuda_lanes.append(len(tensors["states"]))
                return {"states": list(tensors["states"])}

        return Cuda()

    def tensors_from_cases(self, config, states, posts, *, torch, device):
        return {"config": config, "states": list(states), "posts": list(posts), "lanes": len(states)}

    def unpack_cuda_results(self, result):
        return list(result["states"])


class FakeTorch:
    __version__ = "fake-torch"

    class cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def get_device_name(_index):
            return "fake-cuda"


class FrozenOpponentWrapper:
    def __init__(self):
        self.action_space = object()


class FakeVecEnv:
    def __init__(self):
        self.envs = [FrozenOpponentWrapper()]
        self.reset_count = 0
        self.closed = False

    def reset(self):
        self.reset_count += 1

    def close(self):
        self.closed = True


class FakeReplay:
    def __init__(self, *, chdir_to=None):
        self.build_calls = []
        self.envs = []
        self.chdir_to = chdir_to

    def build_navigation_env(self, source_root, config, wrapper_source, seed, frame_stack=4):
        if self.chdir_to is not None:
            os.chdir(self.chdir_to)
        env = FakeVecEnv()
        self.envs.append(env)
        self.build_calls.append(
            {
                "source_root": Path(source_root).resolve(),
                "config": Path(config).resolve(),
                "wrapper_source": Path(wrapper_source).resolve(),
                "seed": seed,
                "frame_stack": frame_stack,
            }
        )
        return env, {}, types.SimpleNamespace(phase_key="0b")

    def navigation_fixture_labels(self, steps):
        return [f"scripted_{index}" for index in range(steps)]

    def scripted_navigation_actions(self, _space, labels):
        return [[index] for index, _label in enumerate(labels)]


class FakeCaptureWriter:
    def __init__(self, *, terminate_ticks=None):
        self.terminate_ticks = set(terminate_ticks or ())
        self.metadata = None
        self.records = []
        self.capture_order = []

    def capture_step(self, _wrapper, action, *, tick, case_id):
        self.capture_order.append(("capture", tick, action[0] if action else None))
        terminated = tick in self.terminate_ticks
        return {
            "type": "record",
            "tick": tick,
            "case_id": case_id,
            "input": {"executed_action": list(action)},
            "expected": {"terminated": terminated, "truncated": False},
            "observed_events": ["terminal_event"] if terminated else [],
        }

    def default_metadata(self, *, source_train_path, case_source, config_path=None, extra=None, **_kwargs):
        return {
            "type": "metadata",
            "source": {"train_path": str(source_train_path), "train_sha256": sha256_file(source_train_path)},
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "case_source": case_source,
            "extra": extra or {},
        }

    def write_jsonl(self, path, metadata, records):
        self.metadata = metadata
        self.records = list(records)
        Path(path).write_text("\n".join(json.dumps(item, sort_keys=True) for item in [metadata, *self.records]) + "\n")
        return len(self.records)


class temporary_modules:
    def __init__(self, **modules):
        self.modules = modules
        self.previous = {}

    def __enter__(self):
        for name, module in self.modules.items():
            self.previous[name] = sys.modules.get(name)
            sys.modules[name] = module

    def __exit__(self, *_exc):
        for name, previous in self.previous.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def sha256_file(path):
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
