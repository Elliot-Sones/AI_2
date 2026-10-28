"""CUDA-only transfer receipts; this module never performs a learning update.

CPU tests cover the pure state comparisons. Real checkpoint inference, sampling
and save/reload qualification all fail closed unless every model is on CUDA.
The caller persists returned JSON-safe receipts with its normal atomic writer.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import tempfile

import numpy as np
import torch as th


PARENT_STEPS = 10027008
PARENT_PARAMETERS = 10249749
CALIBRATION_STEPS = 32768
PROTECTED_ROWS = (0, 1, 2, 3, 4, 5, 6, 8, 9)
OUTPUT_PARAMETERS = ("action_net.weight", "action_net.bias", "log_std")
COMBAT_MASK = (True, True, True, True, True, False, True, True, False, False)


def _finite_state(state):
    for name, tensor in state.items():
        if not th.isfinite(tensor).all():
            raise ValueError(f"Parameter must be finite: {name}")


def _fingerprint(tensor):
    data = tensor.detach().contiguous().cpu()
    return {"sha256": hashlib.sha256(data.numpy().tobytes()).hexdigest(),
            "shape": list(data.shape), "dtype": str(data.dtype)}


def _fingerprints(state):
    _finite_state(state)
    return {name: _fingerprint(tensor) for name, tensor in state.items()}


def protected_fingerprints(state):
    """Hash all inherited tensors, except light row7; exclude new combat tensors."""
    _finite_state(state)
    selected = {}
    for name, tensor in state.items():
        if name.startswith("combat_") or name == "allowed_control_mask":
            continue
        selected[name] = tensor[list(PROTECTED_ROWS)] if name in OUTPUT_PARAMETERS else tensor
    return {name: _fingerprint(tensor) for name, tensor in selected.items()}


def verify_protected_fingerprints(state, expected):
    actual = protected_fingerprints(state)
    if actual.keys() != expected.keys():
        raise ValueError("Protected parameter names changed during calibration")
    changed = [name for name in actual if actual[name] != expected[name]]
    if changed:
        raise ValueError("Protected calibration parameters changed: " + ", ".join(changed))
    return True


def compare_transfer_states(parent_state, derived_state):
    """Exact tensor comparison, also usable with tiny synthetic CPU tensors."""
    _finite_state(parent_state)
    _finite_state(derived_state)
    for name, parent in parent_state.items():
        if name not in derived_state:
            raise ValueError(f"Derived state is missing parent parameter: {name}")
        derived = derived_state[name]
        if parent.shape != derived.shape:
            raise ValueError(f"Parent parameter shape changed: {name}")
        if name in OUTPUT_PARAMETERS:
            parent, derived = parent[list(PROTECTED_ROWS)], derived[list(PROTECTED_ROWS)]
        if not th.equal(parent, derived):
            raise ValueError(f"Inherited parameter changed during transfer: {name}")
    for name in ("action_net.weight", "action_net.bias"):
        if not th.equal(derived_state[name][7], th.zeros_like(derived_state[name][7])):
            raise ValueError(f"Light mean must initialize to zero: {name}")
    std = derived_state["log_std"][7]
    if not th.equal(std, th.full_like(std, math.log(0.3))):
        raise ValueError("Light log standard deviation must initialize to log(0.3)")
    mask = th.as_tensor(COMBAT_MASK, device=derived_state["allowed_control_mask"].device)
    if not th.equal(derived_state["allowed_control_mask"], mask):
        raise ValueError("Derived B1 mask differs from the approved controls")
    combat_pairs = []
    for name, value in parent_state.items():
        if name.startswith("mlp_extractor.value_net."):
            other = name.replace("mlp_extractor.value_net.", "combat_value_features.", 1)
        elif name.startswith("value_net."):
            other = "combat_" + name
        else:
            continue
        if other not in derived_state or value.shape != derived_state[other].shape:
            raise ValueError(f"New combat critic is missing or incompatible: {other}")
        combat_pairs.append((value, derived_state[other]))
    if not combat_pairs or all(th.equal(left, right) for left, right in combat_pairs):
        raise ValueError("Combat critic must be fresh, not copied parent values")
    return {"movement_bit_exact": True, "navigation_critic_bit_exact": True,
            "combat_critic_fresh": True, "changed_actor_rows": [7],
            "light_mean": 0.0, "light_std": 0.3, "allowed_control_mask": list(COMBAT_MASK)}


def press_rate_receipt(presses, *, sample_count=100000):
    """Predeclared five-binomial-standard-deviation interval around N(0,.3)."""
    if type(sample_count) is not int or sample_count <= 0 or type(presses) is not int or not 0 <= presses <= sample_count:
        raise ValueError("Valid press count and positive integer sample count required")
    expected = 0.5 * math.erfc(0.5 / (0.3 * math.sqrt(2)))
    standard_error = math.sqrt(expected * (1 - expected) / sample_count)
    observed = presses / sample_count
    z_score = (observed - expected) / standard_error
    if abs(z_score) > 5:
        raise ValueError(f"Calibrated light press rate is outside predeclared five-sigma bound: z={z_score}")
    return {"sample_count": sample_count, "presses": presses, "observed_probability": observed,
            "expected_probability": expected, "binomial_standard_error": standard_error,
            "z_score": z_score, "z_limit": 5, "passed": True}


def _require_cuda(*models):
    if not th.cuda.is_available():
        raise RuntimeError("Real checkpoint qualification requires CUDA")
    for model in models:
        parameters = tuple(model.policy.parameters())
        if not parameters or any(parameter.device.type != "cuda" for parameter in parameters):
            raise RuntimeError("Every qualification policy parameter must be on CUDA")


def _observations(model, observations):
    if observations is None:
        values = th.linspace(-1, 1, 16 * 256, device=model.device).reshape(16, 256)
        source = "fixed synthetic CUDA schema probes; not behavioral game evidence"
    else:
        values = th.as_tensor(observations, dtype=th.float32, device=model.device)
        source = "caller-provided observation bank"
    if values.ndim != 2 or values.shape[1] != 256 or not th.isfinite(values).all():
        raise ValueError("Qualification observations must be finite [N,256]")
    return values, source


def _critic_metadata(model):
    groups = {"actor": [], "navigation": [], "combat": []}
    for name, parameter in model.policy.named_parameters():
        if name.startswith("combat_"):
            groups["combat"].append((name, parameter))
        elif name.startswith(("mlp_extractor.value_net.", "value_net.")):
            groups["navigation"].append((name, parameter))
        else:
            groups["actor"].append((name, parameter))
    pointers = {key: {p.data_ptr() for _, p in group} for key, group in groups.items()}
    if pointers["navigation"] & (pointers["combat"] | pointers["actor"]) or pointers["combat"] & pointers["actor"]:
        raise ValueError("Actor and task critics share parameter storage")
    return {key: {"parameters": sum(p.numel() for _, p in group),
                  "tensors": _fingerprints(dict(group))} for key, group in groups.items()}


def _parent_metadata(parent):
    import stable_baselines3
    if stable_baselines3.__version__ != "2.5.0":
        raise RuntimeError("Parent qualification requires stable_baselines3 2.5.0")
    architecture = parent.policy.net_arch
    actor = architecture["pi"] if isinstance(architecture, dict) else architecture
    critic = architecture["vf"] if isinstance(architecture, dict) else architecture
    parameters = sum(p.numel() for p in parent.policy.parameters())
    if actor != [1536, 1536, 1536] or critic != actor or parameters != PARENT_PARAMETERS:
        raise ValueError("Parent architecture or parameter count differs from Large-145")
    if parent.num_timesteps != PARENT_STEPS:
        raise ValueError("Parent saved transition count differs from 10027008")
    if parent.observation_space.shape != (256,) or parent.action_space.shape != (10,):
        raise ValueError("Parent observation/action schema differs from 256/10")
    if parent.observation_space.dtype != np.float32:
        raise ValueError("Parent observation dtype must be float32")
    if not np.all(parent.action_space.low == 0) or not np.all(parent.action_space.high == 1):
        raise ValueError("Parent action bounds must be [0,1]")
    return {"parameters": parameters, "actor_architecture": actor, "critic_architecture": critic,
            "num_timesteps": parent.num_timesteps, "observation_size": 256, "action_size": 10,
            "sb3": stable_baselines3.__version__, "torch": th.__version__, "cuda": th.version.cuda,
            "gpu": th.cuda.get_device_name(parent.device)}


def preflight_transfer(parent, derived, *, checkpoint_path=None, observations=None, sample_count=100000, seed=145):
    """Return (public receipt, protected snapshot) before the counted calibration."""
    _require_cuda(parent, derived)
    metadata = _parent_metadata(parent)
    checkpoint = None
    if checkpoint_path is not None:
        from train_combat_curriculum import validate_parent
        checkpoint = validate_parent(checkpoint_path)
    if derived.task != "combat" or not derived.calibration or not derived.policy.calibration:
        raise ValueError("Transfer must begin in combat calibration mode")
    if derived.num_timesteps != PARENT_STEPS or derived.parent_num_timesteps != PARENT_STEPS:
        raise ValueError("Derived transfer counters must equal the unchanged parent")
    if derived.policy.optimizer.state:
        raise ValueError("Transfer optimizer is not fresh")
    if any(parameter.requires_grad for parameter in derived.policy.mlp_extractor.policy_net.parameters()):
        raise ValueError("Actor features must be frozen during calibration")
    original_parent = _fingerprints(parent.policy.state_dict())
    transfer = compare_transfer_states(parent.policy.state_dict(), derived.policy.state_dict())
    critics = {"parent_navigation": _critic_metadata(parent)["navigation"],
               "derived": _critic_metadata(derived)}
    obs, observation_source = _observations(derived, observations)
    with th.no_grad():
        distribution = derived.policy.get_distribution(obs)
        means = distribution.distribution.mean[:, 7]
        if not th.equal(means, th.zeros_like(means)):
            raise ValueError("Actual CUDA light means are not zero")
        generator = th.Generator(device=derived.device).manual_seed(seed)
        samples = th.randn(sample_count, device=derived.device, generator=generator)
        samples = samples * derived.policy.log_std[7].exp() + means[0]
        sampled = press_rate_receipt(int((samples > 0.5).sum().item()), sample_count=sample_count)
        if bool((distribution.get_actions(deterministic=True)[:, 7] > 0.5).any()):
            raise ValueError("Initialized deterministic policy unexpectedly presses light")
    if _fingerprints(parent.policy.state_dict()) != original_parent:
        raise ValueError("Parent parameters changed during transfer qualification")
    protected = protected_fingerprints(derived.policy.state_dict())
    trainable = {name: tensor for name, tensor in derived.policy.state_dict().items() if name.startswith("combat_")}
    trainable.update({name + "[7]": derived.policy.state_dict()[name][7] for name in OUTPUT_PARAMETERS})
    snapshot = {"schema": "curriculum-calibration-snapshot-v1", "parent_num_timesteps": PARENT_STEPS,
                "start_num_timesteps": derived.num_timesteps, "expected_calibration_transitions": CALIBRATION_STEPS,
                "protected_parameters": protected, "calibrated_parameters_before": _fingerprints(trainable),
                "parent_parameters": original_parent}
    receipt = {"schema": "curriculum-transfer-qualification-v1", "passed": True, "parent": metadata,
               "parent_checkpoint": checkpoint, "checkpoint_hash_verified_here": checkpoint is not None,
               "transfer": transfer, "critics": critics, "optimizer_state_entries": 0,
               "optimizer_provenance": derived.optimizer_provenance, "sampled_light": sampled,
               "sampling_seed": seed, "deterministic_light_pressed": False,
               "observation_source": observation_source, "observation_count": len(obs),
               "parent_unchanged": True, "training_updates_performed": 0}
    return receipt, snapshot


def _outputs_by_task(model, observations):
    policy = model.policy
    prior_task, prior_calibration, prior_training = policy.task, policy.calibration, policy.training
    outputs = {}
    try:
        policy.set_training_mode(False)
        with th.no_grad():
            for task in ("navigation", "combat"):
                policy.set_task(task, calibration=False)
                outputs[task] = {
                    "means": policy.get_distribution(observations).distribution.mean.detach().clone(),
                    "values": policy.predict_values(observations).detach().clone(),
                    "log_std": policy.log_std.detach().clone(),
                }
    finally:
        policy.set_task(prior_task, calibration=prior_calibration)
        policy.set_training_mode(prior_training)
    return outputs


def _nested_equal(left, right):
    if isinstance(left, th.Tensor):
        return isinstance(right, th.Tensor) and th.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(_nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(_nested_equal(a, b) for a, b in zip(left, right))
    return left == right


def compare_calibration_state(state, snapshot, *, num_timesteps, parent_num_timesteps):
    """Pure comparison of the recorded calibration boundary and protected state."""
    if snapshot.get("schema") != "curriculum-calibration-snapshot-v1":
        raise ValueError("Invalid calibration snapshot schema")
    if parent_num_timesteps != snapshot["parent_num_timesteps"]:
        raise ValueError("Calibration parent counter changed")
    transitions = num_timesteps - snapshot["start_num_timesteps"]
    if transitions != CALIBRATION_STEPS or transitions != snapshot["expected_calibration_transitions"]:
        raise ValueError("Calibration must contain exactly 32768 counted transitions")
    verify_protected_fingerprints(state, snapshot["protected_parameters"])
    mask = th.as_tensor(COMBAT_MASK, device=state["allowed_control_mask"].device)
    if not th.equal(state["allowed_control_mask"], mask):
        raise ValueError("Allowed-control mask changed during calibration")
    before = snapshot["calibrated_parameters_before"]
    light_changed = any(_fingerprint(state[name][7]) != before[name + "[7]"] for name in OUTPUT_PARAMETERS)
    combat_changed = any(_fingerprint(tensor) != before[name] for name, tensor in state.items() if name.startswith("combat_"))
    if not light_changed or not combat_changed:
        raise ValueError("Calibration did not update both light and combat critic")
    return {"calibration_transitions": transitions, "parent_num_timesteps": parent_num_timesteps,
            "cumulative_num_timesteps": num_timesteps, "protected_parameters_bit_exact": True,
            "light_updated": light_changed, "combat_critic_updated": combat_changed}


def validate_calibration(derived, snapshot, *, observations=None, reload_path=None, stage="bridge_b1_contact"):
    """Verify one counted calibration; export contact metadata unless explicitly overridden."""
    _require_cuda(derived)
    if derived.task != "combat" or not derived.calibration or not derived.policy.calibration:
        raise ValueError("Validate calibration before switching to joint training")
    state_receipt = compare_calibration_state(
        derived.policy.state_dict(), snapshot, num_timesteps=derived.num_timesteps,
        parent_num_timesteps=derived.parent_num_timesteps,
    )
    if not derived.policy.optimizer.state:
        raise ValueError("Calibration optimizer has no update state")
    obs, observation_source = _observations(derived, observations)
    from curriculum_ppo import CurriculumPPO, load_actor
    with tempfile.TemporaryDirectory(prefix="ai2-cuda-qualification-") as directory:
        checkpoint = Path(reload_path) if reload_path is not None else Path(directory) / "calibration.zip"
        if reload_path is None:
            derived.save(checkpoint)
        restored = CurriculumPPO.load(checkpoint, device=derived.device)
        _require_cuda(restored)
        if not _nested_equal(derived.policy.state_dict(), restored.policy.state_dict()):
            raise ValueError("Saved/reloaded CUDA policy tensors differ")
        if not _nested_equal(derived.policy.optimizer.state_dict(), restored.policy.optimizer.state_dict()):
            raise ValueError("Saved/reloaded CUDA optimizer states differ")
        for name in ("task", "calibration", "arm", "num_timesteps", "parent_num_timesteps"):
            if getattr(derived, name) != getattr(restored, name):
                raise ValueError(f"Saved/reloaded calibration metadata differs: {name}")
        expected = _outputs_by_task(derived, obs)
        actual = _outputs_by_task(restored, obs)
        if not _nested_equal(expected, actual):
            raise ValueError("Saved/reloaded actor outputs or task critic values differ")
        actor_path = Path(directory) / "actor.pt"
        derived.export_actor(actor_path, stage=stage)
        actor, actor_metadata = load_actor(actor_path, device=derived.device)
        with th.no_grad():
            if not th.equal(actor(obs), expected["combat"]["means"]) or not th.equal(actor.log_std, derived.policy.log_std):
                raise ValueError("Exported CUDA actor outputs differ")
    return {"schema": "curriculum-calibration-qualification-v1", "passed": True,
            **state_receipt,
            "critics": _critic_metadata(derived), "optimizer_state_entries": len(derived.policy.optimizer.state),
            "reload_policy_bit_exact": True, "reload_optimizer_bit_exact": True,
            "reload_outputs_by_task": {task: _fingerprints(outputs) for task, outputs in actual.items()},
            "actor_export_bit_exact": True, "actor_export_metadata": actor_metadata,
            "observation_source": observation_source, "observation_count": len(obs),
            "training_updates_performed": 0}
