"""Opt-in SB3 2.5.0 PPO updates with deferred metric readback.

``deferred_metrics_train(model)`` operates on the real rollout buffer, including
CUDA ``TensorRolloutBuffer`` instances. The policy, distribution validation,
minibatch iterator, optimizer, gradient clipping, and early-stop timing remain
owned by their existing implementations. Only logging scalar transfers move to
the update boundary when ``target_kl`` is unset.

``compile_ppo_loss()`` is a separate, experimental fixed-shape tensor-math arm.
It does not compile policy evaluation or change distribution validation. Compile
is lazy: measure the first forward/backward update separately from steady state.
Different minibatch shapes or scalar schedule values can trigger recompilation.
Inductor fusion can change floating-point results; validate that arm separately.

The objective and update loop are adapted from Stable-Baselines3 2.5.0 PPO.train
(source SHA-256 7c8d09940123242f0ac401c5ec2cc858b8bf3339f571c222382f048ffd8306ae).

The MIT License
Copyright (c) 2019 Antonin Raffin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
import hashlib
from typing import Any


SB3_VERSION = "2.5.0"
SB3_TRAIN_SHA256 = "7c8d09940123242f0ac401c5ec2cc858b8bf3339f571c222382f048ffd8306ae"


@lru_cache(maxsize=1)
def _source_metadata(version: str, train_method: Callable[..., Any]) -> tuple[str, str, bool]:
    import hashlib
    import inspect

    digest = hashlib.sha256(inspect.getsource(train_method).encode()).hexdigest()
    return version, digest, version == SB3_VERSION and digest == SB3_TRAIN_SHA256


def sb3_train_source_metadata() -> dict[str, Any]:
    """Report the installed algorithm source; unsupported variants cannot train."""
    import stable_baselines3

    version, digest, supported = _source_metadata(stable_baselines3.__version__, stable_baselines3.PPO.train)
    return {"version": version, "train_sha256": digest, "supported": supported}


def _diagnostic_array(value: Any) -> Any:
    import numpy as np

    detach = getattr(value, "detach", None)
    if detach is not None:
        value = detach()
    cpu = getattr(value, "cpu", None)
    if cpu is not None:
        value = cpu()
    numpy = getattr(value, "numpy", None)
    if numpy is not None:
        value = numpy()
    return np.ascontiguousarray(value)


def diagnostic_array_summary(value: Any) -> dict[str, Any]:
    """Small host summary for opt-in PPO update diagnostics."""
    import numpy as np

    array = _diagnostic_array(value)
    if array.size == 0:
        return {"shape": list(array.shape), "dtype": str(array.dtype), "mean": None,
                "std": None, "min": None, "max": None}
    numeric = array.astype(np.float64, copy=False)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "mean": float(numeric.mean()),
        "std": float(numeric.std()),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
    }


def diagnostic_minibatch_digest(**arrays: Any) -> str:
    """Stable digest of the sampled minibatch without storing sample payloads."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = _diagnostic_array(arrays[name])
        digest.update(name.encode("utf8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _diagnostic_float(value: Any) -> float:
    item = getattr(value, "item", None)
    if item is not None:
        value = item()
    return float(value)


def _tensor_cuda_device_indexes(*values: Any) -> list[int]:
    import torch as th

    devices = set()
    for value in values:
        if th.is_tensor(value) and value.device.type == "cuda":
            devices.add(th.cuda.current_device() if value.device.index is None else value.device.index)
    return sorted(devices)


def _reject_stateful_post_step_kl_modules(policy: Any) -> None:
    import torch as th

    for module in policy.modules():
        if module is policy or not module.training:
            continue
        if isinstance(module, th.nn.modules.batchnorm._BatchNorm):
            raise RuntimeError(
                "post-step KL diagnostics do not support training BatchNorm modules; "
                "the extra policy evaluation would mutate running statistics"
            )


def post_step_approx_kl(policy: Any, rollout_data: Any, actions: Any) -> Any:
    """Measure KL after an optimizer step without consuming caller RNG state."""
    import torch as th

    _reject_stateful_post_step_kl_modules(policy)
    devices = _tensor_cuda_device_indexes(rollout_data.observations, actions, rollout_data.old_log_prob)
    with th.random.fork_rng(devices=devices):
        with th.no_grad():
            _values, post_log_prob, _entropy = policy.evaluate_actions(rollout_data.observations, actions)
            log_ratio = post_log_prob - rollout_data.old_log_prob
            return th.mean((th.exp(log_ratio) - 1) - log_ratio)


def _emit_update_diagnostic(
    sink: Callable[[dict[str, Any]], Any] | None,
    *,
    epoch: int,
    minibatch_in_epoch: int,
    minibatch_ordinal: int,
    rollout_data: Any,
    evaluate_actions_actions: Any,
    log_prob: Any,
    policy_loss: Any,
    value_loss: Any,
    entropy_loss: Any,
    clip_fraction: Any,
    approx_kl_div: Any,
    target_kl: float | None,
    optimizer_step_executed: bool,
    early_stop: bool,
    gradient_norm: Any,
    learning_rate: float,
    approx_kl_host: Any = None,
    post_step_approx_kl_host: Any = None,
) -> None:
    if sink is None:
        return
    if approx_kl_host is None:
        approx_kl_host = approx_kl_div.detach().cpu().numpy()
    actions = rollout_data.actions
    old_values = rollout_data.old_values
    old_log_prob = rollout_data.old_log_prob
    row = {
        "epoch": int(epoch),
        "minibatch_in_epoch": int(minibatch_in_epoch),
        "minibatch_ordinal": int(minibatch_ordinal),
        "pre_step_approx_kl": _diagnostic_float(approx_kl_host),
        "post_step_approx_kl": None if post_step_approx_kl_host is None else _diagnostic_float(post_step_approx_kl_host),
        "target_kl": None if target_kl is None else float(target_kl),
        "target_kl_threshold": None if target_kl is None else float(1.5 * target_kl),
        "optimizer_step_executed": bool(optimizer_step_executed),
        "early_stop": bool(early_stop),
        "policy_loss": _diagnostic_float(policy_loss.detach()),
        "value_loss": _diagnostic_float(value_loss.detach()),
        "entropy_loss": _diagnostic_float(entropy_loss.detach()),
        "entropy": -_diagnostic_float(entropy_loss.detach()),
        "clip_fraction": _diagnostic_float(clip_fraction.detach()),
        "gradient_norm": None if gradient_norm is None else _diagnostic_float(gradient_norm),
        "learning_rate": float(learning_rate),
        "sample_count": int(actions.shape[0]),
        "sample_digest": diagnostic_minibatch_digest(
            observations=rollout_data.observations,
            actions=actions,
            evaluate_actions_actions=evaluate_actions_actions,
            old_log_prob=old_log_prob,
            old_values=old_values,
            advantages=rollout_data.advantages,
            returns=rollout_data.returns,
        ),
        "actions": diagnostic_array_summary(actions),
        "evaluate_actions_actions": diagnostic_array_summary(evaluate_actions_actions),
        "log_prob": diagnostic_array_summary(log_prob),
        "old_log_prob": diagnostic_array_summary(old_log_prob),
        "old_values": diagnostic_array_summary(old_values),
    }
    sink(row)


def ppo_loss(
    values: Any,
    log_prob: Any,
    entropy: Any,
    advantages: Any,
    old_log_prob: Any,
    old_values: Any,
    returns: Any,
    clip_range: float,
    clip_range_vf: float | None,
    ent_coef: float,
    vf_coef: float,
    normalize_advantage: bool,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """SB3 objective and metric tensors, with no policy calls or host readback."""
    import torch as th
    from torch.nn import functional as F

    values = values.flatten()
    if normalize_advantage and len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    ratio = th.exp(log_prob - old_log_prob)
    policy_loss_1 = advantages * ratio
    policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
    clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float())

    if clip_range_vf is None:
        values_pred = values
    else:
        values_pred = old_values + th.clamp(values - old_values, -clip_range_vf, clip_range_vf)
    value_loss = F.mse_loss(returns, values_pred)

    if entropy is None:
        entropy_loss = -th.mean(-log_prob)
    else:
        entropy_loss = -th.mean(entropy)
    loss = policy_loss + ent_coef * entropy_loss + vf_coef * value_loss

    with th.no_grad():
        log_ratio = log_prob - old_log_prob
        approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio)
    return loss, policy_loss, value_loss, entropy_loss, clip_fraction, approx_kl_div


def compile_ppo_loss(*, backend: str | Callable[..., Any] = "inductor") -> Callable[..., Any]:
    """Return the lazy, full-graph loss compiler; compilation failures propagate.

    ``aot_eager`` is useful for source/algebra tests, but only ``inductor`` on the
    benchmark CUDA host exercises the proposed fused-compute performance arm.
    """
    import torch

    # Torch 2.4.1's Dynamo initialization globally disables Distribution
    # validation, even when the compiled function does not use distributions.
    # Preserve the caller's existing policy validation setting on both success
    # and failure. Policy evaluation itself always stays outside this graph.
    validate_args = torch.distributions.Distribution._validate_args
    try:
        return torch.compile(ppo_loss, backend=backend, fullgraph=True, dynamic=False)
    finally:
        torch.distributions.Distribution.set_default_validate_args(validate_args)


def deferred_metrics_train(model: Any, *, loss_function: Callable[..., Any] | None = None) -> None:
    """Run one PPO update, preserving SB3's objective and epoch semantics.

    Call with the actual tensor buffer, before TensorPPO's NumPy stats proxy is
    installed. For an optional compile experiment pass ``compile_ppo_loss()`` as
    ``loss_function`` and keep that callable across updates to reuse its cache.
    """
    import numpy as np
    import torch as th
    from gymnasium import spaces
    from stable_baselines3.common.utils import explained_variance

    metadata = sb3_train_source_metadata()
    if not metadata["supported"]:
        raise RuntimeError(f"deferred PPO update requires the validated SB3 {SB3_VERSION} train source; got {metadata}")
    compute_loss = ppo_loss if loss_function is None else loss_function
    model.policy.set_training_mode(True)
    model._update_learning_rate(model.policy.optimizer)
    clip_range = model.clip_range(model._current_progress_remaining)
    clip_range_vf = (None if model.clip_range_vf is None else
                     model.clip_range_vf(model._current_progress_remaining))
    diagnostic_sink = getattr(model, "update_diagnostic_sink", None)

    metric_rows = []
    continue_training = True
    minibatch_ordinal = 0
    for epoch in range(model.n_epochs):
        # SB3 reports KL only from the last attempted epoch, including the
        # minibatch that triggers an early stop without an optimizer step.
        last_epoch_start = len(metric_rows)
        for minibatch_in_epoch, rollout_data in enumerate(model.rollout_buffer.get(model.batch_size)):
            actions = rollout_data.actions
            if isinstance(model.action_space, spaces.Discrete):
                actions = rollout_data.actions.long().flatten()

            values, log_prob, entropy = model.policy.evaluate_actions(rollout_data.observations, actions)
            loss, policy_loss, value_loss, entropy_loss, clip_fraction, approx_kl_div = compute_loss(
                values, log_prob, entropy, rollout_data.advantages, rollout_data.old_log_prob,
                rollout_data.old_values, rollout_data.returns, clip_range, clip_range_vf,
                model.ent_coef, model.vf_coef, model.normalize_advantage,
            )
            metric_rows.append(th.stack((entropy_loss.detach(), policy_loss.detach(), value_loss.detach(),
                                         clip_fraction.detach(), approx_kl_div.detach())))

            approx_kl_host = None
            early_stop = False
            if model.target_kl is not None:
                # Preserve the exact SB3 host-side NumPy comparison, before
                # zero_grad/backward/step. This synchronization is intentional.
                approx_kl_host = approx_kl_div.detach().cpu().numpy()
                if approx_kl_host > 1.5 * model.target_kl:
                    continue_training = False
                    early_stop = True
                    _emit_update_diagnostic(
                        diagnostic_sink,
                        epoch=epoch,
                        minibatch_in_epoch=minibatch_in_epoch,
                        minibatch_ordinal=minibatch_ordinal,
                        rollout_data=rollout_data,
                        evaluate_actions_actions=actions,
                        log_prob=log_prob,
                        policy_loss=policy_loss,
                        value_loss=value_loss,
                        entropy_loss=entropy_loss,
                        clip_fraction=clip_fraction,
                        approx_kl_div=approx_kl_div,
                        target_kl=model.target_kl,
                        optimizer_step_executed=False,
                        early_stop=True,
                        gradient_norm=None,
                        learning_rate=model.policy.optimizer.param_groups[0]["lr"],
                        approx_kl_host=approx_kl_host,
                    )
                    minibatch_ordinal += 1
                    if model.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_host:.2f}")
                    break

            model.policy.optimizer.zero_grad()
            loss.backward()
            gradient_norm = th.nn.utils.clip_grad_norm_(model.policy.parameters(), model.max_grad_norm)
            model.policy.optimizer.step()
            post_step_kl_host = None
            if diagnostic_sink is not None:
                post_step_kl_host = post_step_approx_kl(model.policy, rollout_data, actions).detach().cpu().numpy()
            _emit_update_diagnostic(
                diagnostic_sink,
                epoch=epoch,
                minibatch_in_epoch=minibatch_in_epoch,
                minibatch_ordinal=minibatch_ordinal,
                rollout_data=rollout_data,
                evaluate_actions_actions=actions,
                log_prob=log_prob,
                policy_loss=policy_loss,
                value_loss=value_loss,
                entropy_loss=entropy_loss,
                clip_fraction=clip_fraction,
                approx_kl_div=approx_kl_div,
                target_kl=model.target_kl,
                optimizer_step_executed=True,
                early_stop=early_stop,
                gradient_norm=gradient_norm,
                learning_rate=model.policy.optimizer.param_groups[0]["lr"],
                approx_kl_host=approx_kl_host,
                post_step_approx_kl_host=post_step_kl_host,
            )
            minibatch_ordinal += 1

        model._n_updates += 1
        if not continue_training:
            break

    # Keep NumPy's original reduction precision: SB3's .item() lists become
    # float64, whereas its KL list contains NumPy float32 scalars. Preserve the
    # one-dimensional contiguous reduction order for exact metric parity.
    metrics = th.stack(metric_rows).cpu().numpy()
    entropy_losses = metrics[:, 0].astype(np.float64)
    pg_losses = metrics[:, 1].astype(np.float64)
    value_losses = metrics[:, 2].astype(np.float64)
    clip_fractions = metrics[:, 3].astype(np.float64)
    approx_kl_divs = np.ascontiguousarray(metrics[last_epoch_start:, 4])

    values = model.rollout_buffer.values
    returns = model.rollout_buffer.returns
    if th.is_tensor(values):
        values = values.detach().cpu().numpy()
    if th.is_tensor(returns):
        returns = returns.detach().cpu().numpy()
    explained_var = explained_variance(values.flatten(), returns.flatten())

    model.logger.record("train/entropy_loss", np.mean(entropy_losses))
    model.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
    model.logger.record("train/value_loss", np.mean(value_losses))
    model.logger.record("train/approx_kl", np.mean(approx_kl_divs))
    model.logger.record("train/clip_fraction", np.mean(clip_fractions))
    model.logger.record("train/loss", loss.item())
    model.logger.record("train/explained_variance", explained_var)
    if hasattr(model.policy, "log_std"):
        model.logger.record("train/std", th.exp(model.policy.log_std).mean().item())
    model.logger.record("train/n_updates", model._n_updates, exclude="tensorboard")
    model.logger.record("train/clip_range", clip_range)
    if model.clip_range_vf is not None:
        model.logger.record("train/clip_range_vf", clip_range_vf)
