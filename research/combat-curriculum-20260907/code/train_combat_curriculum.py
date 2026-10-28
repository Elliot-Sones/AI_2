"""Finite, CUDA-only B1 transfer pilot with immutable local-preservation gates.

Each invocation runs ONE calibration or training block. It never provisions a
machine, advances a curriculum stage, extends its budget, or destroys anything.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

PARENT_SHA256 = '0e2f64e02cd0f7d300554f7a6cb641065c75771f69f55c0885258dda32afc6cd'
PARENT_BYTES = 123047421
PARENT_STEPS = 10027008
CONTACT_METRIC_VERSION = 'bridge_b1_contact-v1'


class NavigationTraceWriter:
    """Separate schema: retain maintenance trajectories without pretending B1 events."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.temporary = self.directory / 'navigation.jsonl.gz.tmp'
        self.destination = self.directory / 'navigation.jsonl.gz'
        if self.destination.exists():
            raise FileExistsError(self.destination)
        self.stream = gzip.open(self.temporary, 'xt')
        self.frames = 0
        self.closed = False

    def append(self, trace):
        if self.closed or type(trace.get('tick')) is not int or len(trace.get('post_players', [])) != 2:
            raise ValueError('invalid navigation trajectory or closed writer')
        payload = json.dumps({**trace, 'schema_version': 'navigation-trace-v1'}, allow_nan=False)
        self.stream.write(payload + '\n')
        self.frames += 1

    def close(self):
        if not self.closed:
            self.stream.close()
            os.link(self.temporary, self.destination)
            self.temporary.unlink()
            atomic_json(self.directory / 'inventory.json', {
                'schema_version': 'navigation-trace-inventory-v1', 'frames': self.frames,
                'files': [{'path': self.destination.name, 'bytes': self.destination.stat().st_size,
                           'sha256': sha256(self.destination)}]})
            self.closed = True


@dataclass(frozen=True)
class PilotConfig:
    n_envs: int = 64
    n_steps: int = 512
    batch_size: int = 8192
    n_epochs: int = 4
    learning_rate: float = 0.0001
    clip_range: float = 0.10
    target_kl: float = 0.01
    parent_steps: int = PARENT_STEPS
    seed: int = 145

    def __post_init__(self):
        if (self.n_envs, self.n_steps, self.batch_size, self.n_epochs) != (64, 512, 8192, 4):
            raise ValueError('pilot geometry must remain the approved 64x512 / 8192 / 4')
        if self.parent_steps != PARENT_STEPS:
            raise ValueError('unexpected parent transition count')


def block_schedule():
    rows = [{'id': 'calibration', 'arm': 'E1', 'round': 0,
             'tasks': ['combat'], 'transitions': 32768}]
    for round_id, navigation in enumerate(('near', 'wide', 'near'), 1):
        for arm in (('E1', 'E2') if round_id % 2 else ('E2', 'E1')):
            rows.append({'id': f'{arm}-round{round_id}', 'arm': arm,
                         'round': round_id, 'tasks': ['combat'] * 4 + [navigation],
                         'transitions': 163840})
    return rows


def expected_parent_block(block_id):
    row = next(row for row in block_schedule() if row['id'] == block_id)
    if row['round'] == 0:
        return None, PARENT_STEPS
    parent_id = 'calibration' if row['round'] == 1 else f"{row['arm']}-round{row['round'] - 1}"
    return parent_id, PARENT_STEPS + 32768 + (row['round'] - 1) * 163840


def validate_runtime(model, cfg):
    expected = {key: getattr(cfg, key) for key in ('n_envs', 'n_steps', 'batch_size', 'n_epochs', 'target_kl')}
    expected.update(gamma=.99, gae_lambda=.95, vf_coef=.5, max_grad_norm=.5)
    for key, value in expected.items():
        if getattr(model, key) != value:
            raise ValueError(f'saved runtime drift: {key}')
    for progress in (0., .5, 1.):
        if model.lr_schedule(progress) != cfg.learning_rate or model.clip_range(progress) != cfg.clip_range:
            raise ValueError('learning rate and clipping must use the approved constant schedules')
    if any(group['lr'] != cfg.learning_rate for group in model.policy.optimizer.param_groups):
        raise ValueError('effective optimizer learning rate differs from the manifest')


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            result.update(chunk)
    return result.hexdigest()


def atomic_json(path, value):
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
    # Exclusive link publishes complete content without overwriting a receipt.
    os.link(temporary, path)
    temporary.unlink()


def validate_parent(path):
    path = Path(path)
    if path.stat().st_size != PARENT_BYTES or sha256(path) != PARENT_SHA256:
        raise ValueError('parent checksum/size mismatch; refusing alternate weights')
    return {'sha256': PARENT_SHA256, 'bytes': PARENT_BYTES, 'num_timesteps': PARENT_STEPS}


def validate_deadline(deadline, *, now=None, reserve_seconds=60):
    now = time.time() if now is None else now
    if not math.isfinite(deadline) or deadline <= 0:
        raise ValueError('a finite absolute deadline is mandatory')
    if now + reserve_seconds >= deadline:
        raise TimeoutError('deadline reached; preserve results without more updates')


def write_inventory(directory, *, block_id):
    directory = Path(directory)
    files = []
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            raise ValueError('symlinks are not valid preserved artifacts')
        if not path.is_file() or path.relative_to(directory).as_posix() in {
            'inventory.json', 'local-verified.json', 'preservation_receipt.json', 'preservation_ack.json'
        }:
            continue
        if path.name.endswith('.tmp'):
            raise ValueError('unfinished artifact cannot be inventoried')
        files.append({'path': path.relative_to(directory).as_posix(),
                      'bytes': path.stat().st_size, 'sha256': sha256(path)})
    result = {'schema_version': 'curriculum-inventory-v1', 'block_id': block_id, 'files': files}
    atomic_json(directory / 'inventory.json', result)
    return result


def require_preserved(directory, ack_path):
    directory = Path(directory)
    inventory = directory / 'inventory.json'
    body = json.loads(inventory.read_text())
    ack = json.loads(Path(ack_path).read_text())
    if (ack.get('verified') is not True or ack.get('block_id') != body['block_id']
            or ack.get('inventory_sha256') != sha256(inventory)):
        raise ValueError('missing or mismatched verified-local preservation receipt')


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('real checkpoint inference and learning require CUDA')
    value = (torch.ones(8, device='cuda') * 2).sum().item()
    if value != 16:
        raise RuntimeError('CUDA arithmetic preflight failed')
    return {'torch': torch.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0), 'cuda_tensor_check': value}


def require_contact_qualification(path, frozen_root):
    """Bind a scripted qualification receipt to this exact lesson and source."""
    if path is None:
        raise ValueError('contact qualification receipt is required before training')
    proof = json.loads(Path(path).read_text())
    game = Path(frozen_root) / 'UTMIST-AI2-main/environment/environment.py'
    adapter = Path(__file__).with_name('combat_curriculum.py')
    if (proof.get('qualified') is not True
            or proof.get('stage_version') != CONTACT_METRIC_VERSION
            or proof.get('source_sha256') != sha256(game)
            or proof.get('adapter_source_sha256') != sha256(adapter)):
        raise ValueError('contact qualification is failed, stale or for a different lesson')
    return proof


def make_vector_env(frozen_root, task, *, n_envs=64, seed=145):
    from functools import partial
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
    from stable_baselines3.common.monitor import Monitor
    from combat_curriculum import make_b1_contact_env
    from curriculum_navigation import make_navigation_env

    def create(rank):
        if task == 'combat':
            raw = make_b1_contact_env(frozen_root, seed=seed + rank, player_slot=rank % 2)
        else:
            raw = make_navigation_env(frozen_root, seed=seed + rank, variant=task, player_slot=rank % 2)
        return Monitor(raw)

    env = SubprocVecEnv([partial(create, rank) for rank in range(n_envs)], start_method='spawn')
    return VecFrameStack(env, n_stack=4)


def seed_rollout(env, paired_seed):
    """Restore paired scenario seeds after SB3 load seeds an attached env."""
    import numpy as np
    import torch

    random.seed(paired_seed)
    np.random.seed(paired_seed)
    torch.manual_seed(paired_seed)
    env.seed(paired_seed)


def run_block(args):
    # Imports and weights are deliberately after the hardware/deadline gates.
    validate_deadline(args.deadline_unix, reserve_seconds=120)
    qualification = require_contact_qualification(args.qualification, args.frozen_root)
    hardware = require_cuda()
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from curriculum_ppo import CurriculumPPO
    from curriculum_telemetry import AtomicTraceWriter
    from qualify_curriculum_cuda import preflight_transfer, validate_calibration

    cfg = PilotConfig()
    block = next(row for row in block_schedule() if row['id'] == args.block)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('use a new output directory; no checkpoint overwrites')
    if block['id'] != 'calibration':
        if args.previous_block is None or args.verified_ack is None:
            raise ValueError('continuation requires a verified-local previous block')
        require_preserved(args.previous_block, args.verified_ack)
        previous_id = json.loads((args.previous_block / 'inventory.json').read_text())['block_id']
        schedule = [row['id'] for row in block_schedule()]
        if previous_id != schedule[schedule.index(block['id']) - 1]:
            raise ValueError('the immediately preceding paired block must be locally verified')
        parent_id, expected_counter = expected_parent_block(block['id'])
        parent_dir = args.checkpoint.parent
        require_preserved(parent_dir, parent_dir / 'preservation_ack.json')
        parent_inventory = json.loads((parent_dir / 'inventory.json').read_text())
        listed = next((item for item in parent_inventory['files'] if item['path'] == args.checkpoint.name), None)
        if (parent_inventory['block_id'] != parent_id or listed is None
                or listed['sha256'] != sha256(args.checkpoint) or args.checkpoint.name != 'final.zip'):
            raise ValueError('checkpoint does not match the verified expected arm lineage')
    else:
        validate_parent(args.checkpoint)
        expected_counter = PARENT_STEPS
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    atomic_json(output / 'manifest.json', {
        'schema_version': 'b1-contact-pilot-v1', 'block': block, 'config': asdict(cfg),
        'metric_version': CONTACT_METRIC_VERSION,
        'qualification_sha256': sha256(args.qualification), 'qualification': qualification,
        'checkpoint_sha256': sha256(args.checkpoint), 'hardware': hardware,
        'deadline_unix': args.deadline_unix, 'frozen_root': str(args.frozen_root),
        'training_device': 'cuda', 'environment_device': 'cpu',
        'started_unix': time.time(), 'claim': 'B1 pilot, not confirmed mastery',
    })
    model = None
    calibration_snapshot = None
    rows = []
    training_started = time.perf_counter()
    for rollout_index, task in enumerate(block['tasks']):
        validate_deadline(args.deadline_unix, reserve_seconds=120)
        started = time.perf_counter()
        env = make_vector_env(str(args.frozen_root), task, seed=145 + block['round'] * 10000 + rollout_index * 1000)
        trace_writer = None
        try:
            learning_task = 'combat' if task == 'combat' else 'navigation'
            if model is None:
                if block['id'] == 'calibration':
                    parent = PPO.load(args.checkpoint, device='cuda')
                    if parent.num_timesteps != PARENT_STEPS:
                        raise ValueError('parent saved counter mismatch')
                    model = CurriculumPPO.from_parent(parent, env, arm='E1', task='combat', calibration=True, seed=cfg.seed)
                    receipt, calibration_snapshot = preflight_transfer(parent, model, checkpoint_path=args.checkpoint)
                    atomic_json(output / 'transfer-qualification.json', receipt)
                    atomic_json(output / 'calibration-snapshot.json', calibration_snapshot)
                    del parent
                else:
                    model = CurriculumPPO.load(args.checkpoint, env=env, device='cuda')
                    # Arm selection is explicit; API validated by learner tests.
                    model.arm = block['arm']
                    model.set_task(learning_task, calibration=False)
                if model.num_timesteps != expected_counter or model.parent_num_timesteps != PARENT_STEPS:
                    raise ValueError('saved checkpoint counters do not match this arm and round')
                model.set_logger(configure(str(output / 'ppo'), ['json', 'csv', 'stdout']))
            else:
                model.set_env(env, force_reset=True)
                model.set_task(learning_task, calibration=False)
            if any(p.device.type != 'cuda' for p in model.policy.parameters()):
                raise RuntimeError('policy has non-CUDA parameters')
            validate_runtime(model, cfg)
            paired_seed = 145 + block['round'] * 10000 + rollout_index * 1000
            seed_rollout(env, paired_seed)
            writer_type = AtomicTraceWriter if task == 'combat' else NavigationTraceWriter
            trace_writer = writer_type(output / f'rollout-{rollout_index:02d}-traces')
            episode_rows = []
            collect_started = time.perf_counter()

            class ReceiptCallback(BaseCallback):
                def _on_step(self):
                    validate_deadline(args.deadline_unix, reserve_seconds=60)
                    raw = self.locals['actions']
                    for rank, info in enumerate(self.locals['infos']):
                        trace = info.get('trace' if task == 'combat' else 'navigation_trace')
                        if trace is not None:
                            trace = dict(trace)
                            trace.update(env_index=rank, raw_action=raw[rank].tolist(),
                                         action_source='raw_gaussian_sample', rollout_index=rollout_index)
                            trace_writer.append(trace)
                        episode = info.get('episode_diagnostics')
                        if episode is not None:
                            episode_rows.append({'env_index': rank, **episode})
                    return True

                def _on_rollout_end(self):
                    self.collection_seconds = time.perf_counter() - collect_started

            callback = ReceiptCallback()
            before = model.num_timesteps
            model.learn(total_timesteps=32768, reset_num_timesteps=False, callback=callback)
            after_learning = time.perf_counter()
            if model.num_timesteps - before != 32768:
                raise RuntimeError('unexpected rollout transition count')
            trace_writer.close()
            trace_writer = None
            elapsed = time.perf_counter() - started
            row = {'rollout': rollout_index, 'task': task, 'transitions': 32768,
                   'paired_seed': paired_seed,
                   'cumulative_timesteps': model.num_timesteps,
                   'collection_seconds': callback.collection_seconds,
                   'update_seconds': after_learning - collect_started - callback.collection_seconds,
                   'wall_seconds_including_setup_logging': elapsed,
                   'steps_per_second_including_setup_logging': 32768 / elapsed,
                   'episode_count': len(episode_rows), 'episodes': episode_rows,
                   'learning_rate': model.policy.optimizer.param_groups[0]['lr'],
                   'active_indices': list(model.policy.active_indices),
                   'learned_action_std': model.policy.log_std.detach().exp().cpu().tolist(),
                   'ppo_metrics': {key: float(value) for key, value in model.logger.name_to_value.items()
                                   if isinstance(value, (float, int)) or getattr(value, 'shape', None) == ()}}
            # SB3 normally dumps before training; one-rollout calls would
            # otherwise omit the update that just completed.
            model.logger.dump(step=model.num_timesteps)
            rows.append(row)
            atomic_json(output / f'rollout-{rollout_index:02d}.json', row)
            # Immutable rollout snapshots let a downloader preserve progress even
            # when a later rollout, evaluation, or connection fails.
            temp = output / f'rollout-{rollout_index:02d}.tmp.zip'
            model.save(temp)
            os.replace(temp, output / f'rollout-{rollout_index:02d}.zip')
        finally:
            if trace_writer is not None:
                trace_writer.close()
            env.close()
    elapsed = time.perf_counter() - training_started
    model.save(output / 'final.tmp.zip')
    os.replace(output / 'final.tmp.zip', output / 'final.zip')
    if block['id'] == 'calibration':
        receipt = validate_calibration(model, calibration_snapshot, reload_path=output / 'final.zip')
        atomic_json(output / 'calibration-qualification.json', receipt)
    atomic_json(output / 'result.json', {'block': block, 'status': 'trained_not_qualified',
                'new_transitions': sum(row['transitions'] for row in rows),
                'training_wall_seconds': elapsed, 'rollouts': rows,
                'finished_unix': time.time()})
    write_inventory(output, block_id=block['id'])
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--block', choices=[r['id'] for r in block_schedule()], required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--frozen-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline-unix', type=float, required=True)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--previous-block', type=Path)
    parser.add_argument('--verified-ack', type=Path)
    return parser.parse_args(argv)


if __name__ == '__main__':
    raise SystemExit(run_block(parse_args()))
