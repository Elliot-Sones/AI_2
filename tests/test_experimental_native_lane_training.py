from dataclasses import asdict
import importlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def module():
    return importlib.import_module('ai2.training.experimental_native_lane_training')


def test_defaults_are_unchanged_baseline():
    options = module().OptimizationOptions()
    assert not any(asdict(options).values())
    assert options.training_algorithm == "installed stable_baselines3.PPO.train"


def test_compile_requires_explicit_deferred_metrics():
    with pytest.raises(ValueError, match="deferred metrics"):
        module().OptimizationOptions(compile_loss=True).validate()


def test_options_reject_non_boolean_flags():
    with pytest.raises(ValueError, match="boolean"):
        module().OptimizationOptions(optimizer_cpu_steps=1).validate()


def test_parse_independent_and_combined_options():
    m = module()
    parser = m.build_parser()
    args = parser.parse_args(["--output", "/tmp/ai2-options-test", "--optimizer-cpu-steps", "--deferred-metrics", "--deferred-timeouts", "--boundary-validation"])
    opts = m.options_from_args(args)
    assert opts.optimizer_cpu_steps and opts.deferred_metrics
    assert opts.deferred_timeouts and opts.boundary_validation
    assert not opts.compile_loss and not opts.compile_gae
    assert opts.training_algorithm != "installed stable_baselines3.PPO.train"


def test_driver_rejects_cpu_and_other_experiment_paths_before_launch():
    m = module()
    parser = m.build_parser()
    args = parser.parse_args(["--output", "/tmp/ai2-options-test", "--device", "cpu"])
    with pytest.raises(ValueError, match="CUDA"):
        m.validate_experiment(args)
    args.device = "cuda"
    args.resident_buffer = True
    with pytest.raises(ValueError, match="resident"):
        m.validate_experiment(args)


def test_source_provenance_includes_every_experiment_input():
    m = module()
    assert {"ppo_optimizer_counters.py", "ppo_update_optimizations.py", "tensor_ppo.py", "tensor_rollout_buffer.py", "experimental_native_lane_training.py"} <= set(m.EXPERIMENT_FILES)


def test_plain_baseline_class_is_original_tensor_ppo():
    from ai2.training.tensor_ppo import TensorPPO
    assert module().make_ppo_class(module().OptimizationOptions()) is TensorPPO

