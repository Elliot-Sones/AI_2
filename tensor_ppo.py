"""CUDA-native rollout collection seam for SB3 PPO.

``TensorPPO`` deliberately leaves ``stable_baselines3.PPO.train()`` as the
algorithm owner.  Only rollout collection is device-native; the inherited train
method is called through a narrow stats proxy because SB3 computes
``explained_variance`` from NumPy arrays at the end of each update.  That proxy
copies only rollout ``values`` and ``returns`` once per train update for logging.

This is not a drop-in callback compatibility layer: per-step callback locals use
CUDA tensors.  Hybrid CPU-game adapters may expose already-host-resident infos
through ``host_step_infos()``/``host_step_dones()`` for checkpoint/logging
callbacks; pure tensor environments may instead expose episode dictionaries
through ``drain_episode_infos()``, which are copied to SB3's info buffers once at
rollout end.
"""

from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
import numbers
import random
import time
from typing import Any, Optional, Sequence

import numpy as np

try:  # Keep local unittest discovery working on dependency-light interpreters.
    from stable_baselines3 import PPO as _SB3PPO
except Exception:  # pragma: no cover - exercised by Python 3.14 local runtime
    _SB3PPO = object  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class TensorStep:
    observations: Any
    rewards: Any
    terminated: Any
    truncated: Any
    terminal_observations: Any = None


class TensorVecEnv:
    """Adapter for device-native vector environments.

    The wrapped env must provide:

    - ``reset_tensor(seeds=None) -> CUDA float32 [n_envs, obs_dim]``
    - ``step_tensor(clipped_actions) -> TensorStep``
    - optional ``host_step_infos()``/``host_step_dones()`` for hybrid CPU envs
    - optional ``host_timeout_indices()`` for hybrid CPU envs to omit CUDA
      terminal observations when no lanes timed out
    - optional ``drain_episode_infos()`` called once at rollout end for host logs
    """

    def __init__(self, env: Any, *, device: Any) -> None:
        import torch

        self.env = env
        self.device = self._canonical_cuda_device(torch.device(device), torch)
        if self.device.type != "cuda":
            raise ValueError("TensorVecEnv is CUDA-only; no CPU fallback")
        raw_num_envs = getattr(env, "num_envs")
        if isinstance(raw_num_envs, bool) or not isinstance(raw_num_envs, numbers.Integral):
            raise ValueError("num_envs must be a strict positive integer")
        self.num_envs = int(raw_num_envs)
        self.observation_space = getattr(env, "observation_space")
        self.action_space = getattr(env, "action_space")
        self._validate_spaces()

    def _validate_spaces(self) -> None:
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if type(self.observation_space).__name__ != "Box" or type(self.action_space).__name__ != "Box":
            raise ValueError("TensorVecEnv supports Box observation/action spaces only")
        if np.dtype(self.observation_space.dtype) != np.float32 or np.dtype(self.action_space.dtype) != np.float32:
            raise ValueError("TensorVecEnv supports float32 observation/action spaces only")
        if len(tuple(self.observation_space.shape)) != 1:
            raise ValueError("TensorVecEnv supports 1D observations only")
        if tuple(self.action_space.shape) != (10,):
            raise ValueError("TensorVecEnv supports 10D actions only")

    def _canonical_cuda_device(self, device: Any, torch: Any) -> Any:
        if device.type != "cuda" or device.index is not None:
            return device
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cuda", 0)

    @property
    def obs_shape(self) -> tuple[int, ...]:
        return tuple(int(dim) for dim in self.observation_space.shape)

    def reset_tensor(self, seeds: Optional[Sequence[int]] = None) -> Any:
        obs = self.env.reset_tensor(seeds=seeds)
        self._validate_cuda_float32("reset observations", obs, (self.num_envs, *self.obs_shape))
        return obs

    def step_tensor(self, clipped_actions: Any) -> TensorStep:
        self._validate_cuda_float32("clipped_actions", clipped_actions, (self.num_envs, 10))
        step = self.env.step_tensor(clipped_actions)
        self.validate_step(step)
        return step

    def drain_episode_infos(self) -> list[dict[str, Any]]:
        drain = getattr(self.env, "drain_episode_infos", None)
        if drain is None:
            return []
        infos = drain()
        if infos is None:
            return []
        return list(infos)

    def host_step_infos(self) -> list[dict[str, Any]]:
        host_infos = getattr(self.env, "host_step_infos", None)
        if host_infos is None:
            return []
        infos = host_infos()
        if infos is None:
            return []
        return list(infos)

    def host_step_dones(self) -> Optional[np.ndarray]:
        host_dones = getattr(self.env, "host_step_dones", None)
        if host_dones is None:
            return None
        dones = host_dones()
        if dones is None:
            return None
        array = np.asarray(dones, dtype=bool)
        if tuple(array.shape) != (self.num_envs,):
            raise ValueError(f"host_step_dones shape must be ({self.num_envs},)")
        return array

    def host_has_timeouts(self) -> Optional[bool]:
        host_has = getattr(self.env, "host_has_timeouts", None)
        if host_has is not None:
            value = host_has()
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError("host_has_timeouts must return bool")
            return bool(value)
        indices = self.host_timeout_indices()
        if indices is None:
            return None
        return bool(indices.size)

    def host_timeout_indices(self) -> Optional[np.ndarray]:
        host_indices = getattr(self.env, "host_timeout_indices", None)
        if host_indices is None:
            return None
        indices = host_indices()
        if indices is None:
            return None
        array = np.asarray(indices)
        if array.dtype.kind not in {"i", "u"} or array.dtype == np.dtype(bool):
            raise ValueError("host_timeout_indices must be an integer array")
        if len(array.shape) != 1:
            raise ValueError("host_timeout_indices must be 1D")
        if array.size and (int(array.min()) < 0 or int(array.max()) >= self.num_envs):
            raise ValueError(f"host_timeout_indices must be in [0, {self.num_envs})")
        return np.ascontiguousarray(array, dtype=np.int64)

    def seed(self, seed: Optional[int] = None) -> list[Optional[int]]:
        if seed is None:
            seed = int(np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
            raise ValueError("seed must be an integer or None")
        seeds = [int(seed) + idx for idx in range(self.num_envs)]
        seed_method = getattr(self.env, "seed", None)
        if seed_method is not None:
            returned = seed_method(seeds)
            if returned is not None:
                return list(returned)
        return seeds

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if close is not None:
            close()

    def reset(self) -> Any:
        raise RuntimeError("TensorVecEnv has no CPU reset fallback; use reset_tensor()")

    def step_async(self, actions: Any) -> None:
        raise RuntimeError("TensorVecEnv has no CPU step fallback; use step_tensor()")

    def step_wait(self) -> Any:
        raise RuntimeError("TensorVecEnv has no CPU step fallback; use step_tensor()")

    def validate_step(self, step: TensorStep) -> None:
        self._validate_cuda_float32("observations", step.observations, (self.num_envs, *self.obs_shape))
        self._validate_cuda_float32("rewards", step.rewards, (self.num_envs,))
        self._validate_cuda_bool("terminated", step.terminated)
        self._validate_cuda_bool("truncated", step.truncated)
        if step.terminal_observations is None:
            host_has_timeouts = self.host_has_timeouts()
            if host_has_timeouts is not False:
                raise ValueError("terminal_observations may be None only when host reports no timeouts")
        else:
            self._validate_cuda_float32("terminal_observations", step.terminal_observations, (self.num_envs, *self.obs_shape))

    def _validate_cuda_float32(self, name: str, tensor: Any, shape: tuple[int, ...]) -> None:
        import torch

        if not torch.is_tensor(tensor):
            raise ValueError(f"{name} must be a CUDA torch.Tensor")
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be CUDA; no CPU fallback")
        if tensor.device != self.device:
            raise ValueError(f"{name} must be on {self.device}")
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be float32")
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} shape must be {shape}, got {tuple(tensor.shape)}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    def _validate_cuda_bool(self, name: str, tensor: Any) -> None:
        import torch

        if not torch.is_tensor(tensor):
            raise ValueError(f"{name} must be a CUDA torch.Tensor")
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be CUDA; no CPU fallback")
        if tensor.device != self.device:
            raise ValueError(f"{name} must be on {self.device}")
        if tensor.dtype != torch.bool:
            raise ValueError(f"{name} must be bool")
        if tuple(tensor.shape) != (self.num_envs,):
            raise ValueError(f"{name} shape must be ({self.num_envs},), got {tuple(tensor.shape)}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    def as_sb3_vec_env(self) -> Any:
        from stable_baselines3.common.vec_env import VecEnv

        adapter = self

        class _TensorVecEnvShim(VecEnv):
            def __init__(self) -> None:
                super().__init__(adapter.num_envs, adapter.observation_space, adapter.action_space)

            def reset(self) -> Any:
                raise RuntimeError("TensorVecEnv shim has no CPU reset fallback")

            def step_async(self, actions: Any) -> None:
                raise RuntimeError("TensorVecEnv shim has no CPU step fallback")

            def step_wait(self) -> Any:
                raise RuntimeError("TensorVecEnv shim has no CPU step fallback")

            def close(self) -> None:
                adapter.close()

            def get_attr(self, attr_name: str, indices: Any = None) -> list[Any]:
                return [getattr(adapter.env, attr_name) for _ in self._get_indices(indices)]

            def set_attr(self, attr_name: str, value: Any, indices: Any = None) -> None:
                for _ in self._get_indices(indices):
                    setattr(adapter.env, attr_name, value)

            def env_method(self, method_name: str, *method_args: Any, indices: Any = None, **method_kwargs: Any) -> list[Any]:
                return [getattr(adapter.env, method_name)(*method_args, **method_kwargs) for _ in self._get_indices(indices)]

            def env_is_wrapped(self, wrapper_class: type, indices: Any = None) -> list[bool]:
                return [False for _ in self._get_indices(indices)]

            def seed(self, seed: Optional[int] = None) -> list[Optional[int]]:
                seeds = adapter.seed(seed)
                self._seeds = list(seeds)
                return seeds

        return _TensorVecEnvShim()


class _TensorRolloutStatsProxy:
    """Expose NumPy values/returns only for SB3 train end-of-update logging."""

    def __init__(self, rollout_buffer: Any) -> None:
        self._rollout_buffer = rollout_buffer
        self.values = rollout_buffer.values.detach().to("cpu").numpy()
        self.returns = rollout_buffer.returns.detach().to("cpu").numpy()

    def get(self, *args: Any, **kwargs: Any) -> Any:
        return self._rollout_buffer.get(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._rollout_buffer, name)


class _DeferredTimeoutValues:
    """Own rollout-local snapshots until accepted rewards have been corrected.

    The policy stays in evaluation mode with unchanged weights during collection.
    Selecting timeout rows synchronizes once at the rollout boundary; inference
    and gathers then use bounded batches. Different batch shapes can change the
    critic's floating-point rounding relative to per-step full-batch inference.
    """

    def __init__(self, n_steps: int, env: TensorVecEnv, batch_size: int) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, numbers.Integral) or batch_size <= 0:
            raise ValueError("terminal_value_batch_size must be a strict positive integer")
        self.batch_size = int(batch_size)
        self.shape = (n_steps, env.num_envs, *env.obs_shape)
        self.observations = None
        self.timeout_masks = None

    def __enter__(self) -> "_DeferredTimeoutValues":
        return self

    def __exit__(self, *exc: Any) -> None:
        # Callbacks retain their locals, including this object. Release the large
        # tensors explicitly on success, cancellation, and exceptions.
        self.observations = None
        self.timeout_masks = None

    def add(self, step_index: int, terminal_observations: Any, timeout_mask: Any) -> None:
        import torch

        if self.observations is None:
            self.observations = torch.empty(self.shape, dtype=torch.float32, device=terminal_observations.device)
            self.timeout_masks = torch.zeros(self.shape[:2], dtype=torch.bool, device=timeout_mask.device)
        self.observations[step_index].copy_(terminal_observations.detach())
        self.timeout_masks[step_index].copy_(timeout_mask)

    def apply(self, policy: Any, rollout_buffer: Any, n_steps: int, gamma: float) -> None:
        import torch

        if self.observations is None or n_steps == 0:
            return
        indices = torch.nonzero(self.timeout_masks[:n_steps].reshape(-1), as_tuple=False).flatten()
        observations = self.observations.reshape(-1, *self.shape[2:])
        rewards = rollout_buffer.rewards.reshape(-1)
        with torch.no_grad():
            for start in range(0, indices.numel(), self.batch_size):
                batch_indices = indices[start : start + self.batch_size]
                terminal_values = policy.predict_values(observations.index_select(0, batch_indices)).flatten()
                if tuple(terminal_values.shape) != tuple(batch_indices.shape):
                    raise ValueError("terminal values shape must match the timeout batch")
                if terminal_values.device != rewards.device or terminal_values.dtype != rewards.dtype:
                    raise ValueError("terminal values must match rollout rewards device and dtype")
                corrected_rewards = rewards.index_select(0, batch_indices) + gamma * terminal_values
                rollout_buffer._accumulate_finite_check(corrected_rewards)
                rewards.index_copy_(0, batch_indices, corrected_rewards)


class TensorPPO(_SB3PPO):  # type: ignore[misc,valid-type]
    """PPO with CUDA-native rollout collection and inherited SB3 train math.

    ``defer_timeout_values`` is an opt-in experiment for passive logging or
    checkpoint callbacks and a policy whose weights stay frozen in evaluation
    mode throughout collection. Step callbacks still receive raw environment
    rewards, but earlier ``rollout_buffer.rewards`` remain unbootstrapped until
    the rollout boundary (or callback cancellation). Callbacks that consume
    those intermediate buffer rewards or mutate the policy are unsupported by
    this option. ``terminal_value_batch_size`` bounds selected critic batches.
    """

    defer_timeout_values = False
    terminal_value_batch_size = 8192

    def __init__(self, policy: Any, env: TensorVecEnv, *args: Any, **kwargs: Any) -> None:
        if _SB3PPO is object:
            raise ImportError("stable_baselines3 is required for TensorPPO")
        import torch
        from tensor_rollout_buffer import TensorRolloutBuffer

        if not isinstance(env, TensorVecEnv):
            raise ValueError("TensorPPO requires a TensorVecEnv")
        device = env._canonical_cuda_device(torch.device(kwargs.get("device", env.device)), torch)
        if device.type != "cuda":
            raise ValueError("TensorPPO is CUDA-only; no CPU fallback")
        if device != env.device:
            raise ValueError(f"TensorPPO device {device} must match TensorVecEnv device {env.device}")
        if kwargs.get("use_sde", False):
            raise ValueError("TensorPPO does not support SDE")
        self.tensor_vec_env = env
        kwargs["device"] = device
        kwargs["rollout_buffer_class"] = TensorRolloutBuffer
        kwargs.setdefault("rollout_buffer_kwargs", {})
        super().__init__(policy, env.as_sb3_vec_env(), *args, **kwargs)
        if getattr(self.policy, "squash_output", False):
            raise ValueError("TensorPPO supports unsquashed Gaussian Box policies only")
        self._action_low_tensor = torch.as_tensor(self.action_space.low, dtype=torch.float32, device=self.device)
        self._action_high_tensor = torch.as_tensor(self.action_space.high, dtype=torch.float32, device=self.device)

    @classmethod
    def from_sb3(cls, sb3_model: Any, tensor_env: TensorVecEnv) -> "TensorPPO":
        """Convert an already-loaded SB3 PPO into TensorPPO on ``tensor_env``.

        This preserves the checkpoint's policy/optimizer tensors and PPO
        hyperparameters while replacing only rollout collection.
        """

        if not isinstance(tensor_env, TensorVecEnv):
            raise ValueError("from_sb3 requires env=TensorVecEnv")
        kwargs = {
            "learning_rate": getattr(sb3_model, "learning_rate"),
            "n_steps": sb3_model.n_steps,
            "batch_size": sb3_model.batch_size,
            "n_epochs": sb3_model.n_epochs,
            "gamma": sb3_model.gamma,
            "gae_lambda": sb3_model.gae_lambda,
            "clip_range": sb3_model.clip_range,
            "clip_range_vf": sb3_model.clip_range_vf,
            "normalize_advantage": sb3_model.normalize_advantage,
            "ent_coef": sb3_model.ent_coef,
            "vf_coef": sb3_model.vf_coef,
            "max_grad_norm": sb3_model.max_grad_norm,
            "use_sde": sb3_model.use_sde,
            "sde_sample_freq": sb3_model.sde_sample_freq,
            "target_kl": sb3_model.target_kl,
            "stats_window_size": sb3_model._stats_window_size,
            "tensorboard_log": sb3_model.tensorboard_log,
            "policy_kwargs": dict(sb3_model.policy_kwargs),
            "verbose": sb3_model.verbose,
            "seed": sb3_model.seed,
            "device": tensor_env.device,
        }
        import torch

        py_state = random.getstate()
        np_state = np.random.get_state()
        cpu_state = torch.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            model = cls(sb3_model.policy_class, tensor_env, **kwargs)
            model.set_parameters(sb3_model.get_parameters(), exact_match=True, device=tensor_env.device)
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
        for attr in (
            "num_timesteps",
            "_n_updates",
            "_episode_num",
            "_total_timesteps",
            "_num_timesteps_at_start",
            "_current_progress_remaining",
        ):
            if hasattr(sb3_model, attr):
                setattr(model, attr, getattr(sb3_model, attr))
        if sb3_model.ep_info_buffer is not None:
            model.ep_info_buffer = deque(sb3_model.ep_info_buffer, maxlen=sb3_model.ep_info_buffer.maxlen)
        if sb3_model.ep_success_buffer is not None:
            model.ep_success_buffer = deque(sb3_model.ep_success_buffer, maxlen=sb3_model.ep_success_buffer.maxlen)
        return model

    def _excluded_save_params(self) -> list[str]:
        excluded = super()._excluded_save_params()
        return excluded + [
            "tensor_vec_env",
            "_action_low_tensor",
            "_action_high_tensor",
            "rollout_buffer_class",
            "rollout_buffer_kwargs",
            "_last_obs",
            "_last_episode_starts",
        ]

    @classmethod
    def load(cls, path: Any, env: Optional[TensorVecEnv] = None, device: Any = "auto", **kwargs: Any) -> "TensorPPO":
        """Load an SB3 PPO checkpoint and convert it for TensorVecEnv collection."""

        if _SB3PPO is object:
            raise ImportError("stable_baselines3 is required for TensorPPO")
        if not isinstance(env, TensorVecEnv):
            raise ValueError("TensorPPO.load requires env=TensorVecEnv")
        if device != "auto" and not str(device).startswith("cuda"):
            raise ValueError("TensorPPO.load is CUDA-only; use standard PPO.load for CPU deployment")
        load_device = env.device if device == "auto" else device
        stock_model = _SB3PPO.load(path, env=env.as_sb3_vec_env(), device=load_device, **kwargs)
        return cls.from_sb3(stock_model, env)

    def train(self) -> None:
        original_buffer = self.rollout_buffer
        self.rollout_buffer = _TensorRolloutStatsProxy(original_buffer)
        try:
            super().train()
        finally:
            self.rollout_buffer = original_buffer

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: Any = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ) -> tuple[int, Any]:
        import torch
        from stable_baselines3.common import utils

        self.start_time = time.time_ns()

        if self.ep_info_buffer is None or reset_num_timesteps:
            self.ep_info_buffer = deque(maxlen=self._stats_window_size)
            self.ep_success_buffer = deque(maxlen=self._stats_window_size)

        if self.action_noise is not None:
            self.action_noise.reset()

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps

        if reset_num_timesteps or self._last_obs is None:
            self._last_obs = self.tensor_vec_env.reset_tensor()
            self._last_episode_starts = torch.ones((self.tensor_vec_env.num_envs,), dtype=torch.float32, device=self.device)

        if not self._custom_logger:
            self._logger = utils.configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)

        callback = self._init_callback(callback, progress_bar)
        return total_timesteps, callback

    def _ensure_info_buffers(self) -> None:
        if self.ep_info_buffer is None:
            self.ep_info_buffer = deque(maxlen=self._stats_window_size)
        if self.ep_success_buffer is None:
            self.ep_success_buffer = deque(maxlen=self._stats_window_size)

    def collect_rollouts(self, env: Any, callback: Any, rollout_buffer: Any, n_rollout_steps: int) -> bool:
        import torch

        del env
        if getattr(self, "use_sde", False):
            raise ValueError("TensorPPO does not support SDE")
        self.policy.set_training_mode(False)
        timeout_context = (
            _DeferredTimeoutValues(n_rollout_steps, self.tensor_vec_env, self.terminal_value_batch_size)
            if self.defer_timeout_values else nullcontext(None)
        )

        with timeout_context as deferred_timeouts:
            n_steps = 0
            rollout_buffer.reset()
            if self._last_obs is None:
                self._last_obs = self.tensor_vec_env.reset_tensor()
                self._last_episode_starts = torch.ones((self.tensor_vec_env.num_envs,), dtype=torch.float32, device=self.device)

            callback.on_rollout_start()

            last_dones = torch.zeros((self.tensor_vec_env.num_envs,), dtype=torch.bool, device=self.device)
            new_obs = self._last_obs
            while n_steps < n_rollout_steps:
                pre_action_obs = self._last_obs.clone()
                with torch.no_grad():
                    actions, values, log_probs = self.policy(pre_action_obs)
                raw_actions = actions.reshape((self.tensor_vec_env.num_envs, 10))
                clipped_actions = torch.clamp(raw_actions, self._action_low_tensor, self._action_high_tensor)

                step = self.tensor_vec_env.step_tensor(clipped_actions)
                self.num_timesteps += self.tensor_vec_env.num_envs
                new_obs = step.observations
                rewards = step.rewards
                dones = step.terminated | step.truncated
                infos = self.tensor_vec_env.host_step_infos()
                host_dones = self.tensor_vec_env.host_step_dones()
                callback.update_locals(locals())
                if not callback.on_step():
                    if deferred_timeouts is not None:
                        deferred_timeouts.apply(self.policy, rollout_buffer, n_steps, self.gamma)
                    return False
                if infos:
                    self._ensure_info_buffers()
                    self._update_info_buffer(infos, host_dones)

                timeout_mask = step.truncated & ~step.terminated
                host_timeout_indices = self.tensor_vec_env.host_timeout_indices()
                if host_timeout_indices is None:
                    should_bootstrap_timeouts = self.tensor_vec_env.host_has_timeouts()
                else:
                    should_bootstrap_timeouts = bool(host_timeout_indices.size)
                if should_bootstrap_timeouts is not False:
                    if deferred_timeouts is not None:
                        deferred_timeouts.add(n_steps, step.terminal_observations, timeout_mask)
                    else:
                        with torch.no_grad():
                            terminal_values = self.policy.predict_values(step.terminal_observations).flatten()
                        rewards = torch.where(timeout_mask, rewards + self.gamma * terminal_values, rewards)

                episode_starts = self._last_episode_starts
                if episode_starts.dtype != torch.float32:
                    episode_starts = episode_starts.to(dtype=torch.float32)
                rollout_buffer.add(pre_action_obs, raw_actions, rewards, episode_starts, values, log_probs)
                new_obs = new_obs.clone()
                self._last_obs = new_obs
                self._last_episode_starts = dones.to(dtype=torch.float32)
                last_dones = dones
                n_steps += 1

            if deferred_timeouts is not None:
                deferred_timeouts.apply(self.policy, rollout_buffer, n_steps, self.gamma)
                if n_steps:
                    rewards = rollout_buffer.rewards[n_steps - 1]
            with torch.no_grad():
                last_values = self.policy.predict_values(new_obs)
            rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=last_dones.to(dtype=torch.float32))

            drained_infos = self.tensor_vec_env.drain_episode_infos()
            if drained_infos:
                self._ensure_info_buffers()
                self._update_info_buffer(drained_infos, np.ones((len(drained_infos),), dtype=bool))

            callback.update_locals(locals())
            callback.on_rollout_end()
            return True
