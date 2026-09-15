"""SB3 2.5 curriculum PPO: one shared actor and independent task critics.

The runner selects an environment and task at rollout boundaries, then calls
``learn(n_steps * n_envs, reset_num_timesteps=False)``. Each environment transition
must identify ``curriculum_task`` and its ten-entry ``allowed_control_mask``.
All PPO likelihoods use raw Gaussian samples; only execution/predict is masked.
"""

from __future__ import annotations

import copy
import math
import weakref
from pathlib import Path

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.distributions import BernoulliDistribution, DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.utils import obs_as_tensor


TASK_MASKS = {
    "navigation": (True, True, True, True, True, False, True, False, False, False),
    "combat": (True, True, True, True, True, False, True, True, False, False),
}
PPO_DEFAULTS = dict(
    learning_rate=0.0001, n_steps=512, batch_size=8192, n_epochs=4,
    gamma=0.99, gae_lambda=0.95, clip_range=0.1, target_kl=0.01,
    vf_coef=0.5, max_grad_norm=0.5,
)
ACTION_HEADS = ("gaussian", "bernoulli_10")


def _require_finite_tensor(name, tensor):
    if not th.isfinite(tensor).all():
        raise ValueError(f"{name} must be finite")


def _require_finite_module(module, label):
    for name, value in module.state_dict().items():
        if th.is_floating_point(value):
            _require_finite_tensor(f"{label}.{name}", value)


def _task_mask(task):
    if task not in TASK_MASKS:
        raise ValueError(f"Unknown curriculum task: {task!r}")
    return np.asarray(TASK_MASKS[task], dtype=bool)


class MaskedDiagGaussianDistribution(DiagGaussianDistribution):
    """Full ten-output samples, with likelihood and entropy on active controls."""

    def __init__(self, action_dim, mask):
        super().__init__(action_dim)
        self.mask = th.as_tensor(mask, dtype=th.bool)
        if self.mask.shape != (action_dim,) or not self.mask.any():
            raise ValueError("Expected a nonempty action mask matching action_dim")

    def log_prob(self, actions):
        return self.distribution.log_prob(actions)[..., self.mask.to(actions.device)].sum(-1)

    def proba_distribution(self, mean_actions, log_std):
        if not th.isfinite(mean_actions).all() or not th.isfinite(log_std).all():
            raise ValueError("Gaussian parameters must be finite")
        std = log_std.exp()
        if not th.isfinite(std).all() or not (std > 0).all():
            raise ValueError("Gaussian standard deviations must be finite and positive")
        return super().proba_distribution(mean_actions, log_std)

    def entropy(self):
        entropy = self.distribution.entropy()
        return entropy[..., self.mask.to(entropy.device)].sum(-1)


class MaskedBernoulliDistribution(BernoulliDistribution):
    """Ten keyboard buttons, with inactive controls forced out of PPO math."""

    def __init__(self, action_dim, mask):
        super().__init__(action_dim)
        self.mask = th.as_tensor(mask, dtype=th.bool)
        if self.mask.shape != (action_dim,) or not self.mask.any():
            raise ValueError("Expected a nonempty action mask matching action_dim")

    def _mask_for(self, tensor):
        return self.mask.to(tensor.device)

    def _validate_actions(self, actions):
        if not th.isfinite(actions).all():
            raise ValueError("Bernoulli actions must be finite")
        if actions.shape[-1] != self.action_dims:
            raise ValueError("Bernoulli actions must have ten controls")
        zeros = th.zeros((), dtype=actions.dtype, device=actions.device)
        ones = th.ones((), dtype=actions.dtype, device=actions.device)
        if not bool(((actions == zeros) | (actions == ones)).all()):
            raise ValueError("Bernoulli actions must be exact 0/1 buttons")
        inactive = ~self._mask_for(actions)
        if bool((actions[..., inactive] != zeros).any()):
            raise ValueError("Bernoulli inactive actions must be zero")

    def log_prob(self, actions):
        self._validate_actions(actions)
        return self.distribution.log_prob(actions)[..., self._mask_for(actions)].sum(-1)

    def proba_distribution(self, action_logits):
        _require_finite_tensor("Bernoulli logits", action_logits)
        return super().proba_distribution(action_logits)

    def entropy(self):
        entropy = self.distribution.entropy()
        return entropy[..., self._mask_for(entropy)].sum(-1)

    def sample(self):
        return self.distribution.sample() * self._mask_for(self.distribution.logits).to(self.distribution.logits.dtype)

    def mode(self):
        logits = self.distribution.logits
        return (logits > 0).to(logits.dtype) * self._mask_for(logits).to(logits.dtype)


class _ProtectedAdam(th.optim.Adam):
    """Prevent frozen output slices from drifting under previously saved moments."""

    def __init__(self, params, *, policy, **kwargs):
        super().__init__(params, **kwargs)
        self._policy_ref = weakref.ref(policy)

    def step(self, closure=None):
        if closure is not None:
            raise ValueError("Curriculum optimizer does not support closures")
        policy = self._policy_ref()
        finite_gradients = [
            th.isfinite(parameter.grad).all()
            for parameter in policy.parameters() if parameter.grad is not None
        ]
        if finite_gradients and not th.stack(finite_gradients).all():
            raise ValueError("Optimizer gradients must be finite")
        inactive = ~policy.trainable_rows
        protected = (policy.action_net.weight, policy.action_net.bias, policy.log_std)
        snapshots = []
        with th.no_grad():
            for parameter in protected:
                snapshots.append(parameter[inactive].clone())
                for value in self.state.get(parameter, {}).values():
                    if isinstance(value, th.Tensor) and value.shape == parameter.shape:
                        value[inactive] = 0
        result = super().step()
        with th.no_grad():
            for parameter, original in zip(protected, snapshots):
                parameter[inactive] = original
        return result


class CurriculumPolicy(ActorCriticPolicy):
    """The inherited critic is navigation; combat has a fresh independent MLP."""

    def __init__(self, *args, action_head="gaussian", **kwargs):
        if action_head not in ACTION_HEADS:
            raise ValueError(f"action_head must be one of {ACTION_HEADS}, got {action_head!r}")
        self.action_head = action_head
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        if (
            not isinstance(self.observation_space, spaces.Box)
            or self.observation_space.shape != (256,)
            or self.observation_space.dtype != np.float32
            or not isinstance(self.action_space, spaces.Box)
            or self.action_space.shape != (10,)
            or not np.all(self.action_space.low == 0)
            or not np.all(self.action_space.high == 1)
        ):
            raise ValueError("Curriculum requires float32 256 observations and ten [0,1] actions")
        if self.use_sde or self.squash_output or type(self.features_extractor) is not FlattenExtractor:
            raise ValueError("Curriculum supports only unsquashed Gaussian MLP policies with flat features")
        super()._build(lr_schedule)
        self.combat_value_features = copy.deepcopy(self.mlp_extractor.value_net)
        self.combat_value_net = copy.deepcopy(self.value_net)
        self.combat_value_features.apply(lambda module: self.init_weights(module, gain=math.sqrt(2)))
        self.combat_value_net.apply(lambda module: self.init_weights(module, gain=1))
        self.register_buffer("allowed_control_mask", th.as_tensor(TASK_MASKS["combat"]))
        self.task = "combat"
        self.calibration = False
        for parameter in (self.action_net.weight, self.action_net.bias, self.log_std):
            parameter.register_hook(self._mask_output_gradient)
        self.optimizer = _ProtectedAdam(
            self.parameters(), policy=self, lr=lr_schedule(1), **self.optimizer_kwargs,
        )
        self.set_task("combat")

    @property
    def active_indices(self):
        return tuple(np.flatnonzero(TASK_MASKS[self.task]).tolist())

    @property
    def trainable_rows(self):
        mask = self.allowed_control_mask.clone()
        if self.calibration:
            mask[:] = False
            mask[7] = True
        return mask

    def _mask_output_gradient(self, grad):
        shape = (10,) + (1,) * (grad.ndim - 1)
        return grad * self.trainable_rows.reshape(shape)

    def set_task(self, task, calibration=False):
        mask = _task_mask(task)
        if calibration and task != "combat":
            raise ValueError("Light calibration requires the combat task")
        self.task, self.calibration = task, bool(calibration)
        self.allowed_control_mask.copy_(th.as_tensor(mask, device=self.device))
        if self.action_head == "gaussian":
            self.action_dist = MaskedDiagGaussianDistribution(10, mask)
        elif self.action_head == "bernoulli_10":
            self.action_dist = MaskedBernoulliDistribution(10, mask)
        else:
            raise ValueError(f"Unsupported action_head: {self.action_head!r}")
        for parameter in self.mlp_extractor.policy_net.parameters():
            parameter.requires_grad_(not calibration)
            parameter.grad = None
        for module in (self.mlp_extractor.value_net, self.value_net):
            for parameter in module.parameters():
                parameter.requires_grad_(task == "navigation")
                parameter.grad = None
        for module in (self.combat_value_features, self.combat_value_net):
            for parameter in module.parameters():
                parameter.requires_grad_(task == "combat")
                parameter.grad = None

    def forward(self, obs, deterministic=False):
        distribution = self.get_distribution(obs)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions, self.predict_values(obs), distribution.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        distribution = self.get_distribution(obs)
        return self.predict_values(obs), distribution.log_prob(actions), distribution.entropy()

    def predict_values(self, obs):
        if self.task == "navigation":
            return super().predict_values(obs)
        features = BasePolicy.extract_features(self, obs, self.vf_features_extractor)
        return self.combat_value_net(self.combat_value_features(features))

    def predict(self, *args, **kwargs):
        actions, state = super().predict(*args, **kwargs)
        if self.action_head == "gaussian":
            return actions * np.asarray(TASK_MASKS[self.task], dtype=actions.dtype), state
        return self.decode_actions(th.as_tensor(actions)).cpu().numpy().astype(actions.dtype, copy=False), state

    def decode_actions(self, actions):
        tensor = actions if th.is_tensor(actions) else th.as_tensor(actions, dtype=th.float32, device=self.device)
        mask = self.allowed_control_mask.to(tensor.device).to(tensor.dtype)
        if self.action_head == "gaussian":
            return tensor.clamp(0, 1) * mask
        if self.action_head == "bernoulli_10":
            zeros = th.zeros((), dtype=tensor.dtype, device=tensor.device)
            ones = th.ones((), dtype=tensor.dtype, device=tensor.device)
            if not bool(((tensor == zeros) | (tensor == ones)).all()):
                raise ValueError("Bernoulli actions must be exact 0/1 buttons")
            return tensor * mask
        raise ValueError(f"Unsupported action_head: {self.action_head!r}")

    def action_contract(self):
        if self.action_head == "gaussian":
            return {
                "action_head": "gaussian",
                "policy_action_domain": "gaussian_raw_float32_box_10",
                "executed_action_domain": "float32_box_10_clipped_masked",
                "deterministic_decode": "raw_mean_then_clip_and_mask",
                "log_prob_controls": "active_controls_only",
            }
        if self.action_head == "bernoulli_10":
            return {
                "action_head": "bernoulli_10",
                "policy_action_domain": "independent_bernoulli_logits_10",
                "executed_action_domain": "float32_binary_10_masked",
                "deterministic_decode": "logit_greater_than_zero_else_unpressed",
                "log_prob_controls": "active_controls_only",
                "unused_parameters": ["log_std retained only for Box(10) ABI/checkpoint compatibility"],
            }
        raise ValueError(f"Unsupported action_head: {self.action_head!r}")

    def normalized_entropy(self, entropy):
        if self.action_head != "bernoulli_10":
            return None
        return entropy / (len(self.active_indices) * math.log(2.0))


class TaskRolloutBuffer(RolloutBuffer):
    """Homogeneous buffer retaining execution and task metadata beside raw actions."""

    def reset(self):
        super().reset()
        self.tasks = np.full((self.buffer_size, self.n_envs), "", dtype="U10")
        self.allowed_control_masks = np.zeros((self.buffer_size, self.n_envs, 10), dtype=bool)
        self.executed_actions = np.zeros((self.buffer_size, self.n_envs, 10), dtype=np.float32)
        self.task = None
        self.consumed = False

    def add(self, *args, task, mask, executed_actions):
        expected = _task_mask(task)
        if self.task is not None and self.task != task:
            raise ValueError("Cannot mix tasks within a rollout")
        if not np.array_equal(mask, expected):
            raise ValueError("Wrong allowed-control mask for rollout task")
        self.task = task
        self.tasks[self.pos] = task
        self.allowed_control_masks[self.pos] = mask
        self.executed_actions[self.pos] = executed_actions
        super().add(*args)

    def validate(self, task, mask):
        if not self.full or self.consumed:
            raise ValueError("A full, unconsumed rollout is required")
        if self.task != task or not np.all(self.tasks == task):
            raise ValueError("Wrong task or mixed task identities in rollout")
        if not np.all(self.allowed_control_masks == np.asarray(mask)):
            raise ValueError("Rollout mask differs from the current policy mask")
        for name in ("observations", "actions", "rewards", "values", "log_probs", "advantages", "returns"):
            if not np.isfinite(getattr(self, name)).all():
                raise ValueError(f"Rollout {name} must be finite")


class CurriculumPPO(PPO):
    """PPO with fixed task selection, labeled rollouts and protected transfer."""

    def __init__(self, policy=CurriculumPolicy, env=None, *, arm="E1", task="combat", calibration=True,
                 action_head="gaussian", **kwargs):
        if arm not in ("E1", "E2"):
            raise ValueError("arm must be E1 or E2")
        if action_head not in ACTION_HEADS:
            raise ValueError(f"action_head must be one of {ACTION_HEADS}, got {action_head!r}")
        _task_mask(task)
        self.arm, self.task, self.calibration = arm, task, bool(calibration)
        self.action_head = action_head
        self.parent_num_timesteps = 0
        self.optimizer_provenance = "fresh Adam; navigation parent moments not inherited"
        self.task_history = []
        self._collecting = False
        settings = {**PPO_DEFAULTS, **kwargs}
        if settings.get("_init_setup_model") is False:
            # SB3 load restores the saved rollout geometry before _setup_model.
            # Its temporary constructor must not warn about the pilot defaults.
            settings["batch_size"] = 2
        settings["rollout_buffer_class"] = TaskRolloutBuffer
        settings["ent_coef"] = 0.001 if calibration or task == "navigation" or arm == "E1" else 0.003
        policy_kwargs = dict(settings.get("policy_kwargs") or {})
        if "action_head" in policy_kwargs and policy_kwargs["action_head"] != action_head:
            raise ValueError("policy_kwargs action_head disagrees with learner action_head")
        policy_kwargs["action_head"] = action_head
        settings["policy_kwargs"] = policy_kwargs
        super().__init__(policy, env, **settings)

    def _setup_model(self):
        super()._setup_model()
        if not isinstance(self.policy, CurriculumPolicy):
            raise ValueError("CurriculumPPO requires CurriculumPolicy")
        self._collecting = False
        self.policy.set_task(self.task, self.calibration)

    @classmethod
    def from_parent(cls, parent_model, env, *, arm="E1", task="combat", calibration=True, **kwargs):
        parent = parent_model.policy
        if isinstance(parent, CurriculumPolicy):
            raise ValueError("Use load to clone a curriculum state; from_parent is navigation transfer only")
        if type(parent) is not ActorCriticPolicy or parent.use_sde or parent.squash_output:
            raise ValueError("Unsupported parent policy: expected SB3 Gaussian ActorCriticPolicy")
        policy_kwargs = dict(
            net_arch=copy.deepcopy(parent.net_arch), activation_fn=parent.activation_fn,
            ortho_init=parent.ortho_init, features_extractor_class=parent.features_extractor_class,
            features_extractor_kwargs=copy.deepcopy(parent.features_extractor_kwargs),
            share_features_extractor=parent.share_features_extractor,
            normalize_images=parent.normalize_images,
            optimizer_kwargs={"eps": 1e-5},
        )
        if "policy_kwargs" in kwargs:
            raise ValueError("Parent actor architecture must not be overridden")
        kwargs.setdefault("device", parent_model.device)
        action_head = kwargs.pop("action_head", "gaussian")
        model = cls(env=env, arm=arm, task=task, calibration=calibration,
                    action_head=action_head, policy_kwargs=policy_kwargs, **kwargs)
        state = model.policy.state_dict()
        for name, value in parent.state_dict().items():
            if name not in state or state[name].shape != value.shape:
                raise ValueError(f"Incompatible parent parameter: {name}")
            if action_head == "bernoulli_10" and name.startswith(("action_net.", "log_std")):
                continue
            state[name] = value.detach().clone()
        model.policy.load_state_dict(state)
        with th.no_grad():
            if action_head == "gaussian":
                model.policy.action_net.weight[7].zero_()
                model.policy.action_net.bias[7].zero_()
                model.policy.log_std[7] = math.log(0.3)
            else:
                model.policy.action_net.weight.zero_()
                model.policy.action_net.bias.zero_()
                model.policy.log_std.zero_()
                model.optimizer_provenance = (
                    "fresh neutral Bernoulli action head; parent trunk/value copied; "
                    "fresh Adam optimizer not transferred"
                )
        model.num_timesteps = model.parent_num_timesteps = int(parent_model.num_timesteps)
        model.task_history.append(dict(task=task, calibration=calibration, num_timesteps=model.num_timesteps))
        return model

    def set_task(self, task, calibration=False):
        _task_mask(task)
        if calibration and task != "combat":
            raise ValueError("Light calibration requires combat")
        buffer = self.rollout_buffer
        if self._collecting or (buffer.pos and not buffer.consumed):
            raise ValueError("Cannot change task or mask before the current rollout is consumed")
        self.policy.set_task(task, calibration)
        self.task, self.calibration = task, bool(calibration)
        self.ent_coef = 0.001 if calibration or task == "navigation" or self.arm == "E1" else 0.003
        self.task_history.append(dict(task=task, calibration=calibration, num_timesteps=self.num_timesteps))

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        """SB3 collection with explicit masks and fail-closed transition labels."""
        if self._last_obs is None:
            raise ValueError("Call learn/_setup_learn before collection")
        if not isinstance(rollout_buffer, TaskRolloutBuffer) or n_rollout_steps != rollout_buffer.buffer_size:
            raise ValueError("Collection requires one full curriculum rollout")
        if rollout_buffer.pos and not rollout_buffer.consumed:
            raise ValueError("Previous rollout has not been consumed")
        task, mask = self.task, _task_mask(self.task)
        if self.policy.task != task or not np.array_equal(self.policy.allowed_control_mask.cpu(), mask):
            raise ValueError("Policy task/mask differs from rollout task")
        self.policy.set_training_mode(False)
        rollout_buffer.reset()
        self._collecting = True
        try:
            callback.on_rollout_start()
            for _ in range(n_rollout_steps):
                with th.no_grad():
                    actions, values, log_probs = self.policy(obs_as_tensor(self._last_obs, self.device))
                actions = actions.cpu().numpy()
                if not np.isfinite(actions).all() or not th.isfinite(values).all() or not th.isfinite(log_probs).all():
                    raise ValueError("Policy actions, values and log probabilities must be finite")
                executed_actions = self.policy.decode_actions(actions).cpu().numpy()
                new_obs, rewards, dones, infos = env.step(executed_actions)
                self.num_timesteps += env.num_envs
                if not np.isfinite(new_obs).all() or not np.isfinite(rewards).all():
                    raise ValueError("Environment observations and rewards must be finite")
                for info in infos:
                    if info.get("curriculum_task") != task:
                        raise ValueError("Environment transition has missing or wrong curriculum task")
                    if not np.array_equal(info.get("allowed_control_mask"), mask):
                        raise ValueError("Environment transition has missing or wrong allowed-control mask")
                callback.update_locals(locals())
                if not callback.on_step():
                    return False
                if self.task != task or self.policy.task != task or not np.array_equal(self.policy.allowed_control_mask.cpu(), mask):
                    raise ValueError("Task/mask changed during rollout")
                self._update_info_buffer(infos, dones)
                for index, done in enumerate(dones):
                    if done and infos[index].get("terminal_observation") is not None and infos[index].get("TimeLimit.truncated", False):
                        terminal_obs = self.policy.obs_to_tensor(infos[index]["terminal_observation"])[0]
                        with th.no_grad():
                            terminal_value = self.policy.predict_values(terminal_obs)[0]
                        rewards[index] += self.gamma * terminal_value.item()
                rollout_buffer.add(
                    self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs,
                    task=task, mask=mask, executed_actions=executed_actions,
                )
                self._last_obs, self._last_episode_starts = new_obs, dones
            with th.no_grad():
                values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
            rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
            callback.update_locals(locals())
            callback.on_rollout_end()
            return True
        finally:
            self._collecting = False

    def train(self):
        if self.policy.task != self.task or self.policy.calibration != self.calibration:
            raise ValueError("Policy task/calibration differs from learner task")
        if not np.array_equal(self.policy.allowed_control_mask.cpu(), _task_mask(self.task)):
            raise ValueError("Policy mask differs from the canonical task mask")
        self.rollout_buffer.validate(self.task, self.policy.allowed_control_mask.cpu().numpy())
        # SB3 ratios, entropy and approximate KL all use the same masked log-prob.
        super().train()
        self.rollout_buffer.consumed = True
        self.logger.record(f"train/{self.task}/value_loss", self.logger.name_to_value["train/value_loss"])
        self.logger.record(f"train/{self.task}/explained_variance", self.logger.name_to_value["train/explained_variance"])
        entropy = -float(self.logger.name_to_value["train/entropy_loss"])
        self.logger.record("train/active_entropy_sum", entropy)
        self.logger.record("train/active_entropy_mean", entropy / len(self.policy.active_indices))
        normalized_entropy = self.policy.normalized_entropy(th.as_tensor(entropy))
        self.logger.record(
            "train/active_entropy_normalized",
            -1.0 if normalized_entropy is None else float(normalized_entropy.item()),
        )
        self.logger.record("train/action_head_id", ACTION_HEADS.index(self.policy.action_head))
        self.logger.record("train/action_entropy_scale_id", 0 if normalized_entropy is None else 1)
        self.logger.record("train/active_entropy_coefficient", self.ent_coef)

    def export_actor(self, path, *, stage="bridge_b1"):
        """Write actor-only tensors and metadata; load with :func:`load_actor`."""
        _require_finite_module(self.policy, "policy")
        actor = CurriculumActor(self.policy.net_arch, self.policy.activation_fn, action_head=self.policy.action_head)
        actor.features.load_state_dict(self.policy.mlp_extractor.policy_net.state_dict())
        actor.action_net.load_state_dict(self.policy.action_net.state_dict())
        actor.log_std.data.copy_(self.policy.log_std.detach().cpu())
        _require_finite_module(actor, "actor")
        metadata = dict(
            schema="ai2-curriculum-actor-v2", stage=stage, task=self.task,
            action_head=self.policy.action_head, action_contract=self.policy.action_contract(),
            unused_parameters=(["log_std"] if self.policy.action_head == "bernoulli_10" else []),
            allowed_control_mask=list(TASK_MASKS[self.task]), observation_size=256,
            action_size=10, num_timesteps=self.num_timesteps,
            parent_num_timesteps=self.parent_num_timesteps,
            net_arch=copy.deepcopy(self.policy.net_arch),
            activation=self.policy.activation_fn.__name__,
        )
        th.save({"metadata": metadata, "state_dict": actor.state_dict()}, Path(path))
        return metadata


class CurriculumActor(th.nn.Module):
    """Small deployment surface: ten raw means and the learned Gaussian scales."""

    def __init__(self, net_arch, activation_fn=th.nn.Tanh, *, action_head="gaussian"):
        if action_head not in ACTION_HEADS:
            raise ValueError(f"action_head must be one of {ACTION_HEADS}, got {action_head!r}")
        super().__init__()
        self.action_head = action_head
        widths = net_arch["pi"] if isinstance(net_arch, dict) else net_arch
        layers, last = [], 256
        for width in widths:
            layers.extend((th.nn.Linear(last, width), activation_fn()))
            last = width
        self.features = th.nn.Sequential(*layers)
        self.action_net = th.nn.Linear(last, 10)
        self.log_std = th.nn.Parameter(th.zeros(10))
        self.mask = np.asarray(TASK_MASKS["combat"])

    def forward(self, observations):
        return self.action_net(self.features(observations))

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        tensor = th.as_tensor(observation, dtype=th.float32, device=self.log_std.device)
        _require_finite_tensor("Actor observation", tensor)
        with th.no_grad():
            logits_or_means = self(tensor)
            _require_finite_tensor("Actor logits/means", logits_or_means)
            if self.action_head == "gaussian":
                _require_finite_tensor("Actor Gaussian log_std", self.log_std)
                std = self.log_std.exp()
                _require_finite_tensor("Actor Gaussian std", std)
                if not bool((std > 0).all()):
                    raise ValueError("Actor Gaussian std must be finite and positive")
                actions = logits_or_means if deterministic else th.distributions.Normal(logits_or_means, std).sample()
                _require_finite_tensor("Actor Gaussian output", actions)
                decoded = actions.clamp(0, 1)
            else:
                decoded = (logits_or_means > 0).to(logits_or_means.dtype) if deterministic else th.distributions.Bernoulli(logits=logits_or_means).sample()
                _require_finite_tensor("Actor Bernoulli output", decoded)
        return decoded.cpu().numpy() * self.mask, state


def load_actor(path, *, device="cpu", expected_action_head=None):
    payload = th.load(Path(path), map_location=device, weights_only=True)
    metadata = payload["metadata"]
    if metadata.get("schema") == "ai2-curriculum-actor-v1":
        action_head = "gaussian"
    elif metadata.get("schema") == "ai2-curriculum-actor-v2":
        action_head = metadata.get("action_head")
    else:
        raise ValueError("Unsupported curriculum actor schema")
    if action_head not in ACTION_HEADS:
        raise ValueError("Unsupported curriculum actor action head")
    if expected_action_head is not None and action_head != expected_action_head:
        raise ValueError("Exported actor action head differs from expected action head")
    activations = {"Tanh": th.nn.Tanh, "ReLU": th.nn.ReLU, "ELU": th.nn.ELU, "LeakyReLU": th.nn.LeakyReLU}
    if metadata["activation"] not in activations:
        raise ValueError("Unsupported exported actor activation")
    actor = CurriculumActor(metadata["net_arch"], activations[metadata["activation"]], action_head=action_head).to(device)
    actor.load_state_dict(payload["state_dict"])
    actor.mask = _task_mask(metadata["task"])
    if metadata["allowed_control_mask"] != actor.mask.tolist():
        raise ValueError("Exported actor task and mask do not match")
    return actor.eval(), metadata
