#!/usr/bin/env python3
"""One finite CUDA contact-curriculum block, never a curriculum promotion.

The two minibatch arms start from the same calibrated policy. Simulation and
policy updates run on CUDA; original-game evaluation is a separate command.
Each invocation ends at an immutable, retrievable artifact inventory. It never
starts another block, rents a GPU, publishes telemetry, or deletes an instance.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import time
import traceback

from benchmark_native_combat_training import require_cuda, loss_metrics

ROOT = Path(__file__).resolve().parent
CONTACT_VERSION = 'bridge_b1_contact-v1'
QUALIFICATION_SCHEMA = 'ai2.native_contact_qualification.v1'
CHECKPOINT_SHA256 = '301ce3c72924f3e5c62f265f88292b8952e86e6d71d9a0552213cb8f3c6784b6'
CHECKPOINT_BYTES = 102673333
CHECKPOINT_STEPS = 10059776
ACTION_HEADS = ('gaussian', 'bernoulli_10')
ACTION_EXPERIMENT_ARMS = {
    'G_keep': {'action_head': 'gaussian', 'action_init_strategy': 'keep'},
    'G_neutral': {'action_head': 'gaussian', 'action_init_strategy': 'neutral'},
    'B_neutral': {'action_head': 'bernoulli_10', 'action_init_strategy': 'neutral'},
    'B_projected': {'action_head': 'bernoulli_10', 'action_init_strategy': 'projected'},
}
ACTION_AB_ARMS = tuple(ACTION_EXPERIMENT_ARMS)
ACTION_EXPERIMENT_INTERPRETATION = {
    'G_keep': 'practical warm-start baseline; preserves source action head and optimizer where same-class loading allows it',
    'B_projected': 'practical warm-start comparison; preserves source executed-button behavior approximately but uses a fresh optimizer',
    'G_neutral': 'causal representation arm; fresh Gaussian optimizer/head behavior-matched to B_neutral',
    'B_neutral': 'causal representation arm; fresh Bernoulli optimizer/head behavior-matched to G_neutral',
}
PENDING_ACTION_EXPERIMENTS = {
    'multidiscrete': 'pending separate experiment; excluded from this four-arm Gaussian/Bernoulli run',
}
REFERENCE_FILES = ('combat_curriculum.py', 'curriculum_navigation.py', 'curriculum_telemetry.py',
                   'curriculum_evaluation.py', 'evaluate_combat_curriculum.py', 'train_combat_curriculum.py')
CHECKS = ('original_cpu_exact', 'cpu_cuda_exact', 'learner_update_exact')


@dataclass(frozen=True)
class ContactBenchmarkConfig:
    lanes: int = 64
    n_steps: int = 512
    batch_size: int = 8192
    n_epochs: int = 4
    rollouts: int = 4
    seed: int = 145
    device: str = 'cuda'
    update_mode: str = 'deferred_metrics'
    action_head: str = 'gaussian'
    action_ab_arm: str = 'G_keep'
    learning_rate: float = .0001
    clip_range: float = .1
    target_kl: float = .01
    ent_coef: float = .001
    gamma: float = .99
    gae_lambda: float = .95

    def __post_init__(self):
        if any(type(getattr(self, name)) is not int for name in ('lanes', 'n_steps', 'n_epochs', 'batch_size')):
            raise ValueError('rollout geometry and minibatches require integer values')
        if (self.lanes, self.n_steps, self.n_epochs) != (64, 512, 4):
            raise ValueError('keep the original 64 lanes x 512 ticks / 4 epochs geometry')
        if self.batch_size not in (8192, 16384):
            raise ValueError('only the declared minibatch arms 8192 and 16384 are supported')
        if type(self.rollouts) is not int or self.rollouts <= 0:
            raise ValueError('a positive finite rollout count is mandatory')
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - self.lanes:
            raise ValueError('seed and all lane seeds must fit uint32')
        if self.device != 'cuda' and not (self.device.startswith('cuda:') and self.device[5:].isdigit()):
            raise ValueError('simulation and policy require CUDA; no CPU fallback')
        if self.update_mode not in ('sb3', 'deferred_metrics'):
            raise ValueError('unsupported update mode')
        if self.action_ab_arm not in ACTION_EXPERIMENT_ARMS:
            raise ValueError('unsupported action A/B arm')
        if self.action_head != ACTION_EXPERIMENT_ARMS[self.action_ab_arm]['action_head']:
            raise ValueError('action_head and action_ab_arm disagree')
        if (self.learning_rate, self.clip_range, self.target_kl, self.ent_coef, self.gamma, self.gae_lambda) != (
                .0001, .1, .01, .001, .99, .95):
            raise ValueError('the approved contact PPO settings must not change between arms')

    @property
    def transitions_per_rollout(self):
        return self.lanes * self.n_steps

    @property
    def game_ticks_per_rollout(self):
        return self.transitions_per_rollout

    @property
    def nominal_optimizer_steps(self):
        return self.n_epochs * self.transitions_per_rollout // self.batch_size


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    """Publish once, with fsync and no overwrite even under concurrent writers."""
    path = Path(path)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x', encoding='utf8') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def write_inventory(directory, *, block_id):
    """Use the existing cloud retriever's inventory/ACK protocol."""
    directory = Path(directory)
    files = []
    excluded = {'inventory.json', 'local-verified.json', 'preservation_receipt.json', 'preservation_ack.json'}
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            raise ValueError('symlinks are not valid preserved artifacts')
        if not path.is_file() or path.relative_to(directory).as_posix() in excluded:
            continue
        if '.tmp' in path.name:
            raise ValueError('unfinished artifact cannot be inventoried')
        files.append({'path': path.relative_to(directory).as_posix(),
                      'bytes': path.stat().st_size, 'sha256': sha256(path)})
    result = {'schema_version': 'curriculum-inventory-v1', 'block_id': block_id, 'files': files}
    atomic_json(directory / 'inventory.json', result)
    return result


def preserve_failed_block(directory, error, **details):
    """Seal a quiescent failed attempt, preserving partial bytes as incomplete.

    Callers must first close log streams and stop child writers. Temporary
    artifacts cannot enter the standard downloader inventory under their old
    names. Relocate them without overwriting anything, recording a reversible
    mapping; a recovered temporary ZIP is never called a valid checkpoint.
    """
    directory = Path(directory)
    if (directory / 'inventory.json').exists() or (directory / 'failure.json').exists():
        raise FileExistsError('failed attempt was already sealed')
    artifacts = []
    paths = sorted(directory.rglob('*'), key=lambda path: len(path.parts), reverse=True)
    for path in paths:
        if path.is_symlink():
            raise ValueError('cannot preserve a symlink as an incomplete artifact')
        if '.tmp' not in path.name:
            continue
        incomplete = directory / 'incomplete-artifacts'
        incomplete.mkdir(exist_ok=True)
        kind = 'directory' if path.is_dir() else 'file'
        target = incomplete / f'{kind}-{len(artifacts):06d}.incomplete'
        if target.exists():
            raise FileExistsError(target)
        original = path.relative_to(directory).as_posix()
        if path.is_dir():
            path.rename(target)
        else:
            os.link(path, target)
            path.unlink()
        record = {'original_path': original, 'preserved_path': target.relative_to(directory).as_posix(),
                  'kind': kind, 'status': 'incomplete', 'valid_checkpoint': False}
        if target.is_file():
            record.update(bytes=target.stat().st_size, sha256=sha256(target))
        artifacts.append(record)
    if artifacts:
        atomic_json(directory / 'incomplete-artifacts.json', {
            'schema_version': 'ai2.incomplete_artifact_recovery.v1', 'artifacts': artifacts})
    atomic_json(directory / 'failure.json', {'error': str(error), 'type': type(error).__name__,
        'traceback': traceback.format_exc(), 'status': 'failed_attempt_preserved', **details})
    return write_inventory(directory, block_id=directory.name)


def validate_deadline(deadline, *, now=None, reserve_seconds=120):
    now = time.time() if now is None else now
    if type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline <= 0:
        raise ValueError('a finite absolute deadline is mandatory')
    if now + reserve_seconds >= deadline:
        raise TimeoutError('deadline reached; preserve results without another update')


def validate_checkpoint(path):
    path = Path(path)
    if path.stat().st_size != CHECKPOINT_BYTES or sha256(path) != CHECKPOINT_SHA256:
        raise ValueError('the canonical calibrated checkpoint checksum/size is required')
    return {'sha256': CHECKPOINT_SHA256, 'bytes': CHECKPOINT_BYTES, 'num_timesteps': CHECKPOINT_STEPS}


def require_checkpoint_preservation(checkpoint, ack_path=None):
    checkpoint = Path(checkpoint)
    inventory_path = checkpoint.parent / 'inventory.json'
    ack_path = checkpoint.parent / 'preservation_ack.json' if ack_path is None else Path(ack_path)
    inventory = json.loads(inventory_path.read_text())
    ack = json.loads(ack_path.read_text())
    if (inventory.get('schema_version') != 'curriculum-inventory-v1'
            or ack.get('verified') is not True or ack.get('block_id') != inventory.get('block_id')
            or ack.get('inventory_sha256') != sha256(inventory_path)):
        raise ValueError('checkpoint requires its verified-local preservation ACK')
    listed = [row for row in inventory.get('files', []) if row.get('path') == checkpoint.name]
    if (len(listed) != 1 or checkpoint.is_symlink() or listed[0].get('sha256') != sha256(checkpoint)
            or listed[0].get('bytes') != checkpoint.stat().st_size):
        raise ValueError('checkpoint does not match its preserved inventory entry')
    return {'verified': True, 'checkpoint_sha256': listed[0]['sha256'],
            'inventory_sha256': sha256(inventory_path), 'preservation_ack_sha256': sha256(ack_path)}


def qualification_sources(reference_root, frozen_root):
    """Exact source set the coordinator must bind to actual passing evidence."""
    names = ('benchmark_native_contact_training.py', 'benchmark_native_combat_training.py',
             'evaluate_native_contact_training.py', 'native_curriculum_ppo.py', 'curriculum_ppo.py',
             'tensor_ppo.py', 'tensor_rollout_buffer.py', 'ppo_update_optimizations.py',
             'native_contact_episode.py', 'native_contact_lane_vec_env.py', 'native_game_episode.py',
             'native_navigation_episode.py', 'native_navigation_contact.py', 'native_lane_vec_env.py')
    paths = {name: ROOT / name for name in names}
    paths['native/navigation_spaces.json'] = ROOT / 'native/navigation_spaces.json'
    for suffix in ('*.h', '*.c', '*.cu'):
        paths.update({path.relative_to(ROOT).as_posix(): path for path in (ROOT / 'native').glob(suffix)})
    paths.update({f'reference/{name}': Path(reference_root) / name for name in REFERENCE_FILES})
    # Assets, raw source and all imported original-game helpers are frozen too.
    game_root = Path(frozen_root) / 'UTMIST-AI2-main'
    paths.update({f'oracle/{path.relative_to(game_root).as_posix()}': path
                  for path in game_root.rglob('*') if path.is_file()
                  and path.suffix in ('.py', '.json', '.gif', '.png') and '__pycache__' not in path.parts})
    if not (game_root / 'environment/environment.py').is_file():
        raise ValueError('missing frozen original game')
    for name in ('train.py', 'tracking.py'):
        paths[f'oracle/{name}'] = Path(frozen_root) / name
    return {name: sha256(path) for name, path in sorted(paths.items())}


def require_qualification(path, reference_root, frozen_root):
    path = Path(path)
    proof = json.loads(path.read_text())
    if (proof.get('schema_version') != QUALIFICATION_SCHEMA or proof.get('qualified') is not True
            or proof.get('stage_version') != CONTACT_VERSION
            or proof.get('source_sha256') != qualification_sources(reference_root, frozen_root)
            or proof.get('checkpoint_sha256') != CHECKPOINT_SHA256
            or any(proof.get('checks', {}).get(check) is not True for check in CHECKS)):
        raise ValueError('native contact qualification failed, stale, incomplete, or for a different source')
    evidence = proof.get('evidence', [])
    if not isinstance(evidence, list) or {row.get('check') for row in evidence} != set(CHECKS):
        raise ValueError('each qualification check requires actual evidence artifacts')
    for row in evidence:
        relative = Path(row.get('path', ''))
        if not relative.parts or relative.is_absolute() or '..' in relative.parts:
            raise ValueError('qualification evidence must be an explicit receipt-relative path')
        artifact = path.parent / relative
        if not artifact.is_file() or artifact.is_symlink() or sha256(artifact) != row.get('sha256'):
            raise ValueError('qualification evidence is missing or has changed')
    return proof


def validate_update_diagnostics(diagnostics, config):
    keys = ('attempted_epochs', 'optimizer_steps', 'kl_early_stop')
    if not isinstance(diagnostics, dict) or any(key not in diagnostics for key in keys):
        raise ValueError('actual PPO update work was not reported')
    epochs, steps, stopped = (diagnostics[key] for key in keys)
    if (type(epochs) is not int or not 1 <= epochs <= config.n_epochs
            or type(steps) is not int or not 0 <= steps <= config.nominal_optimizer_steps
            or type(stopped) is not bool
            or (not stopped and (epochs != config.n_epochs or steps != config.nominal_optimizer_steps))):
        raise ValueError('invalid actual minibatch/epoch/KL-stop diagnostics')
    return diagnostics


def completed_episode_infos(collector_locals):
    """Match NativeCurriculumPPO's actual callback contract; never lose records."""
    infos = collector_locals.get('infos')
    if not isinstance(infos, list) or any(
            not isinstance(info, dict) or 'episode' not in info
            or not isinstance(info.get('contact', {}).get('diagnostics'), dict) for info in infos):
        raise RuntimeError('native collector did not provide complete episode diagnostics')
    return list(infos)


def summarize_rollouts(rows):
    if not rows:
        raise ValueError('no completed training rollouts')
    for row in rows:
        if any(not math.isfinite(row[key]) or row[key] <= 0
               for key in ('collection_seconds', 'update_seconds', 'total_seconds')):
            raise ValueError('invalid timing receipt')
        if row['transitions'] != row['game_ticks']:
            raise ValueError('contact has exactly one learner transition per game tick')
    totals = {key: sum(row[key] for row in rows) for key in (
        'transitions', 'game_ticks', 'collection_seconds', 'update_seconds', 'total_seconds')}
    return {**totals, 'training_steps_per_second': totals['transitions'] / totals['total_seconds'],
            'collection_steps_per_second': totals['transitions'] / totals['collection_seconds'],
            'learner_transitions_per_game_tick': 1, 'mastery_established': False,
            'timing_includes': 'collection, GAE, native autoresets and PPO updates',
            'timing_excludes': 'startup, build, checkpoint writes, logging writes and original-game evaluation'}


def validate_runtime(model, cfg):
    for key in ('n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda', 'target_kl', 'ent_coef'):
        if getattr(model, key) != getattr(cfg, key):
            raise ValueError(f'loaded contact learner runtime drift: {key}')
    if model.n_envs != cfg.lanes or model.num_timesteps != CHECKPOINT_STEPS:
        raise ValueError('canonical checkpoint counter or lane count differs')
    if model.calibration or model.task != 'combat' or model.arm != 'E1':
        raise ValueError('the contact comparison requires joint, non-calibration E1 learning')
    for progress in (0., .5, 1.):
        if model.lr_schedule(progress) != cfg.learning_rate or model.clip_range(progress) != cfg.clip_range:
            raise ValueError('approved constant learning rate/clip schedules differ')
    if any(p.device.type != 'cuda' for p in model.policy.parameters()):
        raise RuntimeError('policy contains non-CUDA parameters')
    if getattr(model, 'action_head', 'gaussian') != cfg.action_head:
        raise ValueError('model action head differs from requested benchmark arm')


def action_press_metrics(model):
    """Summarize policy/executed key press rates once per rollout."""
    import torch
    buffer = model.rollout_buffer
    actions = getattr(buffer, 'actions', None)
    executed = getattr(buffer, 'executed_actions', None)
    if actions is None or executed is None:
        return {'action_metrics_available': False}
    mask = model.policy.allowed_control_mask.to(dtype=torch.bool, device=executed.device)
    executed_buttons = executed > 0.5
    policy_buttons = actions > 0.5
    with torch.no_grad():
        flat_obs = buffer.observations.reshape((-1, buffer.observations.shape[-1]))
        deterministic, _, _ = model.policy(flat_obs, deterministic=True)
        deterministic_executed = model.policy.decode_actions(deterministic.reshape((-1, 10))) > 0.5
    return {
        'action_metrics_available': True,
        'action_head': model.policy.action_head,
        'policy_sample_press_rate_all': float(policy_buttons.float().mean().item()),
        'policy_sample_press_rate_active': float(policy_buttons[..., mask].float().mean().item()),
        'policy_sample_press_rate_per_button': [float(value) for value in policy_buttons.float().mean(dim=(0, 1)).detach().cpu().tolist()],
        'executed_press_rate_all': float(executed_buttons.float().mean().item()),
        'executed_press_rate_active': float(executed_buttons[..., mask].float().mean().item()),
        'executed_press_rate_per_button': [float(value) for value in executed_buttons.float().mean(dim=(0, 1)).detach().cpu().tolist()],
        'deterministic_press_rate_all': float(deterministic_executed.float().mean().item()),
        'deterministic_press_rate_active': float(deterministic_executed[..., mask].float().mean().item()),
        'deterministic_press_rate_per_button': [float(value) for value in deterministic_executed.float().mean(dim=0).detach().cpu().tolist()],
        'attack_light_sample_rate': float(executed_buttons[..., 7].float().mean().item()),
        'attack_initiation_proxy': float((executed_buttons[..., 7] & ~torch.roll(executed_buttons[..., 7], shifts=1, dims=0)).float()[1:].mean().item())
            if executed_buttons.shape[0] > 1 else 0.0,
    }


def run_block(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('use a new output directory; never overwrite a block')
    cfg = ContactBenchmarkConfig(batch_size=args.batch_size, rollouts=args.rollouts,
                                 seed=args.seed, device=args.device, update_mode=args.update_mode,
                                 action_head=ACTION_EXPERIMENT_ARMS[args.action_ab_arm]['action_head'],
                                 action_ab_arm=args.action_ab_arm)
    validate_deadline(args.deadline_unix)
    checkpoint = validate_checkpoint(args.checkpoint)
    preservation = require_checkpoint_preservation(args.checkpoint, getattr(args, 'checkpoint_ack', None))
    proof = require_qualification(args.qualification, args.reference_root, args.frozen_root)
    device = require_cuda(cfg.device)
    import numpy as np
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from native_contact_lane_vec_env import build_contact_env
    from native_curriculum_ppo import NativeCurriculumPPO
    from tensor_ppo import TensorVecEnv

    os.environ['AI2_FROZEN_SOURCE_ROOT'] = str(Path(args.frozen_root).resolve())
    os.environ['WANDB_MODE'] = 'offline'
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    rows, raw, model, logger = [], None, None, None
    atomic_json(output / 'manifest.json', {'schema_version': 'ai2.native_contact_training.v1',
        'config': asdict(cfg), 'checkpoint': checkpoint, 'checkpoint_preservation': preservation,
        'action_experiment_arms': ACTION_EXPERIMENT_ARMS,
        'action_experiment_interpretation': ACTION_EXPERIMENT_INTERPRETATION,
        'pending_action_experiments': PENDING_ACTION_EXPERIMENTS,
        'qualification_sha256': sha256(args.qualification),
        'qualification': proof, 'deadline_unix': args.deadline_unix, 'started_unix': time.time(),
        'hardware': {'device': str(device), 'gpu': torch.cuda.get_device_name(device),
                     'torch': torch.__version__, 'cuda': torch.version.cuda, 'python': platform.python_version(),
                     'packages': {name: importlib.metadata.version(name) for name in (
                         'stable-baselines3', 'numpy', 'gymnasium', 'pymunk')}},
        'training_device': str(device), 'environment_device': str(device), 'wandb_mode': 'offline',
        'claim': 'bounded learning comparison, not mastery or phase promotion'})

    class ReceiptCallback(BaseCallback):
        def _on_step(self):
            # Wall-clock check only: no device readback or per-tick trace transfer.
            validate_deadline(args.deadline_unix, reserve_seconds=60)
            return True

        def _on_rollout_end(self):
            self.episodes = completed_episode_infos(self.locals)

    try:
        raw = build_contact_env(cfg.lanes, device=device, seed=cfg.seed,
            episodes_per_lane=cfg.n_steps * cfg.rollouts + 1,
            record_capacity=cfg.transitions_per_rollout)
        env = TensorVecEnv(raw, device=device)
        model = NativeCurriculumPPO.load_contact_checkpoint(args.checkpoint, env,
            expected_sha256=CHECKPOINT_SHA256, batch_size=cfg.batch_size,
            update_mode=cfg.update_mode, joint=True, action_ab_arm=cfg.action_ab_arm)
        validate_runtime(model, cfg)
        atomic_json(output / 'scenarios.json', raw.backend.scenario_metadata)
        logger = configure(str(output / 'ppo'), ['json', 'csv'])
        model.set_logger(logger)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        _, callback = model._setup_learn(cfg.rollouts * cfg.transitions_per_rollout,
                                         callback=ReceiptCallback(), reset_num_timesteps=False)
        callback.on_training_start(locals(), globals())
        for index in range(cfg.rollouts):
            validate_deadline(args.deadline_unix)
            before_count = model.num_timesteps
            before_steps = np.array(raw.backend.valid_steps, copy=True)
            torch.cuda.synchronize(device)
            begin = time.perf_counter()
            if not model.collect_rollouts(model.env, callback, model.rollout_buffer, cfg.n_steps):
                raise RuntimeError('cancelled/incomplete rollout is not a training result')
            torch.cuda.synchronize(device)
            collected = time.perf_counter()
            delta = np.asarray(raw.backend.valid_steps) - before_steps
            if (model.num_timesteps - before_count != cfg.transitions_per_rollout
                    or delta.shape != (cfg.lanes,) or not np.all(delta == cfg.n_steps)
                    or np.any(raw.backend.errors)):
                raise RuntimeError('native lane error or incorrect learner transition count')
            validate_deadline(args.deadline_unix, reserve_seconds=60)
            model._update_current_progress_remaining(model.num_timesteps, model._total_timesteps)
            model.train()
            torch.cuda.synchronize(device)
            updated = time.perf_counter()
            diagnostics = validate_update_diagnostics(model.last_update_diagnostics, cfg)
            row = {'rollout': index, 'transitions': cfg.transitions_per_rollout,
                'game_ticks': int(delta.sum()), 'cumulative_timesteps': model.num_timesteps,
                'collection_seconds': collected - begin, 'update_seconds': updated - collected,
                'total_seconds': updated - begin, 'actual_update': diagnostics,
                'nominal_optimizer_steps': cfg.nominal_optimizer_steps,
                'metrics': {**loss_metrics(model.logger.name_to_value), **action_press_metrics(model)},
                'episodes': callback.episodes, 'episode_count': len(callback.episodes)}
            atomic_json(output / f'rollout-{index:03d}.json', row)
            # Complete immutable checkpoint receipt after every rollout, not only final success.
            destination = output / f'rollout-{index:03d}.zip'
            temporary = destination.with_suffix('.tmp.zip')
            model.save(temporary)
            os.link(temporary, destination)
            temporary.unlink()
            atomic_json(output / f'rollout-{index:03d}-checkpoint.json', {
                'path': destination.name, 'sha256': sha256(destination), 'bytes': destination.stat().st_size,
                'num_timesteps': model.num_timesteps})
            model.logger.dump(step=model.num_timesteps)
            rows.append(row)
            print(json.dumps({'rollout': index, 'transitions': model.num_timesteps - CHECKPOINT_STEPS,
                'training_steps_per_second': row['transitions'] / row['total_seconds'],
                'actual_update': diagnostics, 'checkpoint': destination.name}), flush=True)
        callback.on_training_end()
        os.link(output / f'rollout-{cfg.rollouts - 1:03d}.zip', output / 'final.zip')
        result = {**summarize_rollouts(rows), 'status': 'trained_not_evaluated',
            'rollouts': rows, 'checkpoint_sha256': sha256(output / 'final.zip'),
            'wall_seconds_including_setup_logging_checkpoints': time.perf_counter() - started,
            'finished_unix': time.time(), 'next_required_action': 'retrieve and hash-verify this block locally'}
        atomic_json(output / 'result.json', result)
        logger.close()
        logger = None
        write_inventory(output, block_id=output.name)
        return result
    except BaseException as error:
        if logger is not None:
            logger.close()
            logger = None
        preserve_failed_block(output, error, completed_rollouts=len(rows),
            last_completed_checkpoint=f'rollout-{len(rows)-1:03d}.zip' if rows else None)
        raise
    finally:
        if logger is not None:
            logger.close()
        if raw is not None:
            raw.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'frozen-root', 'reference-root', 'qualification', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--deadline-unix', type=float, required=True)
    parser.add_argument('--checkpoint-ack', type=Path, help='defaults to calibrated checkpoint sibling preservation_ack.json')
    parser.add_argument('--batch-size', type=int, choices=(8192, 16384), default=8192)
    parser.add_argument('--rollouts', type=int, default=4)
    parser.add_argument('--seed', type=int, default=145)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--update-mode', choices=('sb3', 'deferred_metrics'), default='deferred_metrics')
    parser.add_argument('--action-ab-arm', choices=ACTION_AB_ARMS, default='G_keep')
    parser.add_argument('--action-head', choices=ACTION_HEADS, help='deprecated; use --action-ab-arm')
    args = parser.parse_args(argv)
    if args.action_head is not None and args.action_head != ACTION_EXPERIMENT_ARMS[args.action_ab_arm]['action_head']:
        parser.error('--action-head disagrees with --action-ab-arm')
    args.action_head = ACTION_EXPERIMENT_ARMS[args.action_ab_arm]['action_head']
    return args


if __name__ == '__main__':
    run_block(parse_args())
