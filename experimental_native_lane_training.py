#!/usr/bin/env python3
"""Independent opt-in native PPO ablations; all-off is the existing baseline.

Simulation, checkpoint, campaign timing and learning workload are unchanged.
Compiled arms are numerical experiments, not assumed bitwise-equivalent wins.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import benchmark_native_lane_training as native

ROOT = Path(__file__).resolve().parent
EXPERIMENT_FILES = tuple(dict.fromkeys((*native.CANDIDATE_FILES,
    "experimental_native_lane_training.py", "ppo_optimizer_counters.py",
    "ppo_update_optimizations.py", "profile_native_lane_training.py")))


@dataclass(frozen=True)
class OptimizationOptions:
    optimizer_cpu_steps: bool = False
    deferred_metrics: bool = False
    deferred_timeouts: bool = False
    boundary_validation: bool = False
    compile_loss: bool = False
    compile_gae: bool = False

    def validate(self) -> None:
        if any(type(value) is not bool for value in asdict(self).values()):
            raise ValueError("optimization flags must be boolean")
        if self.compile_loss and not self.deferred_metrics:
            raise ValueError("compile_loss requires deferred metrics")

    @property
    def training_algorithm(self) -> str:
        if not self.deferred_metrics:
            return "installed stable_baselines3.PPO.train"
        return "SB3-compatible deferred-metrics PPO; compiled loss" if self.compile_loss else "SB3-compatible deferred-metrics PPO; eager loss"


def options_from_args(args: Any) -> OptimizationOptions:
    options = OptimizationOptions(**{name: getattr(args, name) for name in OptimizationOptions.__dataclass_fields__})
    options.validate()
    return options


def optimizer_device_census(model: Any) -> dict[str, Any]:
    groups = []
    for group in model.policy.optimizer.param_groups:
        states = []
        for parameter in group["params"]:
            state = model.policy.optimizer.state.get(parameter, {})
            states.append({"parameter_device": str(parameter.device),
                           "state": {name: {"device": str(value.device), "dtype": str(value.dtype), "shape": list(value.shape)}
                                     for name, value in state.items() if hasattr(value, "device")}})
        groups.append({"capturable": bool(group.get("capturable", False)),
                       "fused": group.get("fused"), "foreach": group.get("foreach"), "parameters": states})
    return {"normalized_counter_count": getattr(model, "_normalized_optimizer_counters", 0),
            "num_timesteps": int(model.num_timesteps), "n_updates": int(model._n_updates), "groups": groups}


def make_ppo_class(options: OptimizationOptions, *, on_loaded: Any = None) -> type:
    from tensor_ppo import TensorPPO

    options.validate()
    if not any(asdict(options).values()):
        return TensorPPO

    class ExperimentalPPO(TensorPPO):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._configure_optimizations()

        def _configure_optimizations(self) -> None:
            self.defer_timeout_values = options.deferred_timeouts
            self.rollout_buffer.validation_mode = "rollout" if options.boundary_validation else "step"
            self.rollout_buffer.compile_gae = options.compile_gae
            if options.optimizer_cpu_steps:
                from ppo_optimizer_counters import normalize_adam_step_devices
                changed = normalize_adam_step_devices(self.policy.optimizer)
                self._normalized_optimizer_counters = getattr(self, "_normalized_optimizer_counters", 0) + changed
            if options.compile_loss and not hasattr(self, "_experimental_loss_function"):
                from ppo_update_optimizations import compile_ppo_loss
                self._experimental_loss_function = compile_ppo_loss()

        @classmethod
        def from_sb3(cls, sb3_model: Any, tensor_env: Any) -> Any:
            model = super().from_sb3(sb3_model, tensor_env)
            # set_parameters in from_sb3 restores optimizer tensors after __init__.
            model._configure_optimizations()
            if on_loaded is not None:
                on_loaded(model)
            return model

        def train(self) -> None:
            if options.deferred_metrics:
                from ppo_update_optimizations import deferred_metrics_train
                deferred_metrics_train(self, loss_function=getattr(self, "_experimental_loss_function", None))
            else:
                super().train()
            if options.compile_loss or options.compile_gae:
                from torch._dynamo.utils import counters
                if not hasattr(self, "_experimental_compile_history"):
                    self._experimental_compile_history = []
                self._experimental_compile_history.append({str(group): dict(values) for group, values in counters.items() if values})

        def _excluded_save_params(self) -> list[str]:
            return super()._excluded_save_params() + ["_experimental_loss_function"]

    return ExperimentalPPO


def build_parser():
    parser = native.campaign.build_parser()
    parser.description = __doc__
    parser.add_argument("--native-lanes", action="store_true", default=True)
    parser.add_argument("--lane-device", choices=("cuda",), default="cuda")
    parser.add_argument("--lane-build-dir", default=str(native.DEFAULT_BUILD_DIR))
    parser.add_argument("--lane-record-capacity", type=int, default=None)
    for name in OptimizationOptions.__dataclass_fields__:
        parser.add_argument("--" + name.replace("_", "-"), action="store_true")
    return parser


def validate_experiment(args: Any) -> OptimizationOptions:
    options = options_from_args(args)
    if str(args.device).split(":", 1)[0] != "cuda":
        raise ValueError("experiments require actual CUDA policy and training")
    if args.resident_buffer or args.rollout_policy_graph or args.mode != "baseline":
        raise ValueError("experiments require baseline timing, no resident-buffer or external policy-graph substitution")
    if (options.compile_loss or options.compile_gae) and args.warmup < 1:
        raise ValueError("compiled experiments require at least one warmup cycle")
    return options


def run(args: Any) -> int:
    options = validate_experiment(args)
    output = native.campaign.validate_output_path(Path(args.output))
    output.mkdir(parents=True, exist_ok=True)
    loaded_models = []
    loaded_censuses = []

    def record_loaded(model):
        loaded_models.append(model)
        loaded_censuses.append(optimizer_device_census(model))

    ppo_class = make_ppo_class(options, on_loaded=record_loaded)
    record = native.provenance(Path(args.lane_build_dir), lane_device="cuda", policy_device=args.device)
    record.update({
        "schema": "ai2.native_ppo_ablations.v1", "optimizations": asdict(options),
        "training_algorithm": options.training_algorithm,
        "scope": "Native CUDA navigation lanes with independently selectable collector and PPO optimizations",
        "candidate": {name: native.campaign.file_record(ROOT / name) for name in EXPERIMENT_FILES},
        "numerical_equivalence": "must be measured; compiled or differently batched arms are not assumed exact",
        "compile_startup": "lazy compilation is intended to finish during warmup but may recompile; inspect per-update compiler counters and report warmup plus full-driver wall separately",
    })
    native.campaign.write_json(output / "native_lane_provenance.json", record)
    original_import = native.campaign.import_train_job_module
    with ExitStack() as stack:
        def import_job(path):
            return stack.enter_context(native.native_lane_job_context(
                original_import(path), build_dir=Path(args.lane_build_dir),
                lane_device="cuda", ppo_class=ppo_class,
                record_capacity=args.lane_record_capacity))
        native.campaign.import_train_job_module = import_job
        try:
            return native.campaign.run_campaign(args)
        finally:
            native.campaign.import_train_job_module = original_import
            from torch.distributions import Distribution
            census = {"loaded": loaded_censuses, "final": [optimizer_device_census(model) for model in loaded_models],
                      "distribution_validation": Distribution._validate_args,
                      "baseline_note": "All-off uses the unchanged TensorPPO class; its loaded-state census is in the independent qualification receipt."}
            if options.compile_loss or options.compile_gae:
                from torch._dynamo.utils import counters
                census["compiler_counters"] = {str(group): dict(values) for group, values in counters.items() if values}
                histories = [getattr(model, "_experimental_compile_history", []) for model in loaded_models]
                census["compiler_history_per_update"] = histories
                census["measured_recompile_cycles"] = [i for history in histories for i in range(max(1, args.warmup), len(history)) if history[i] != history[i - 1]]
                census["compile_headline_eligible"] = bool(histories) and all(len(history) == args.warmup + args.cycles for history in histories) and not census["measured_recompile_cycles"]
            native.campaign.write_json(output / "optimization_runtime.json", census)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = native.campaign.validate_args(parser, parser.parse_args(argv))
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
