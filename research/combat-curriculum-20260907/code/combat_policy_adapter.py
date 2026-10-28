"""Strict SB3 policy bridge from the original game's 64-value observations."""

from pathlib import Path
import sys

import numpy as np


RAW_OBSERVATION_SIZE = 64
STACK_SIZE = 4
ACTION_SIZE = 10


def _validate_spaces(observation_space, action_space, observation_size):
    if getattr(observation_space, "shape", None) != (observation_size,):
        raise ValueError(f"Expected observation shape ({observation_size},)")
    if getattr(action_space, "shape", None) != (ACTION_SIZE,):
        raise ValueError(f"Expected action shape ({ACTION_SIZE},)")
    low = np.asarray(getattr(action_space, "low", None))
    high = np.asarray(getattr(action_space, "high", None))
    if (
        low.shape != (ACTION_SIZE,)
        or high.shape != (ACTION_SIZE,)
        or not np.all(low == 0)
        or not np.all(high == 1)
    ):
        raise ValueError("Expected action bounds [0, 1] for all ten actions")


def _is_cuda(device):
    return getattr(device, "type", str(device).split(":")[0]) == "cuda"


def _require_cuda(model):
    policy = getattr(model, "policy", None)
    if not _is_cuda(getattr(model, "device", None)):
        raise RuntimeError("Combat checkpoint model must be on CUDA")
    if not _is_cuda(getattr(policy, "device", None)):
        raise RuntimeError("Combat checkpoint policy must be on CUDA")
    parameters = getattr(policy, "parameters", None)
    if not callable(parameters):
        raise RuntimeError("Cannot verify CUDA policy parameters")
    count = 0
    for parameter in parameters():
        count += 1
        if not _is_cuda(getattr(parameter, "device", None)):
            raise RuntimeError("Every combat policy parameter must be on CUDA")
    if not count:
        raise RuntimeError("Cannot verify CUDA policy without parameters")
    buffers = getattr(policy, "buffers", None)
    if callable(buffers):
        for buffer in buffers():
            if not _is_cuda(getattr(buffer, "device", None)):
                raise RuntimeError("Every combat policy buffer must be on CUDA")


def _finite_vector(values, size, name):
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.array(values, dtype=np.float32, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite numeric vector") from error
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape ({size},)")
    return result


class SB3CombatPolicy:
    """Own one player's four-frame history while allowing shared model weights.

    ``require_cuda=False`` is for lightweight test doubles. Real SB3 algorithms
    always undergo CUDA checks, even when that flag is false.
    """

    def __init__(self, model, deterministic=True, require_cuda=True):
        _validate_spaces(
            getattr(model, "observation_space", None),
            getattr(model, "action_space", None),
            RAW_OBSERVATION_SIZE * STACK_SIZE,
        )
        # An actual SB3 instance implies this module is already imported. Avoid
        # importing Torch/SB3 just to construct the dependency-free test double.
        sb3_base = sys.modules.get("stable_baselines3.common.base_class")
        actual_sb3 = sb3_base is not None and isinstance(model, sb3_base.BaseAlgorithm)
        if require_cuda or actual_sb3:
            _require_cuda(model)
        self.model = model
        self.deterministic = deterministic
        self.reset()

    def reset(self):
        """Start a fresh episode with the same zero padding as VecFrameStack."""
        self._history = np.zeros(RAW_OBSERVATION_SIZE * STACK_SIZE, dtype=np.float32)

    def get_env_info(self, env):
        """Validate the original game's raw spaces and clear episode history."""
        observation_space = getattr(env, "observation_space", None)
        action_space = getattr(env, "action_space", None)
        if callable(observation_space):
            observation_space = observation_space(0)
        if callable(action_space):
            action_space = action_space(0)
        _validate_spaces(observation_space, action_space, RAW_OBSERVATION_SIZE)
        self.reset()

    def predict(self, raw_obs):
        """Append one raw frame and return ten finite, clipped float32 actions."""
        observation = _finite_vector(raw_obs, RAW_OBSERVATION_SIZE, "Observation")
        stacked = np.concatenate((self._history[RAW_OBSERVATION_SIZE:], observation))
        # Keep private history independent even if a model retains or mutates its input.
        prediction = self.model.predict(stacked.copy(), deterministic=self.deterministic)
        if not isinstance(prediction, tuple) or len(prediction) != 2:
            raise ValueError("Model.predict must return an (action, state) tuple")
        action = _finite_vector(prediction[0], ACTION_SIZE, "Action")
        self._history = stacked
        return np.clip(action, 0.0, 1.0)


def load_cuda_checkpoint(path):
    """Load and verify one checkpoint; callers may cache the returned model."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Combat checkpoint not found: {path}")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to load combat checkpoints")

    from stable_baselines3 import PPO

    model = PPO.load(str(path), device="cuda")
    _validate_spaces(
        getattr(model, "observation_space", None),
        getattr(model, "action_space", None),
        RAW_OBSERVATION_SIZE * STACK_SIZE,
    )
    _require_cuda(model)
    return model
