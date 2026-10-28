#!/usr/bin/env python3
"""CUDA checkpoint reload + original Python-game infrastructure validation.

At most three fresh two-player games, at most 256 ticks each. Both players
use the same deterministic Gaussian mean policy, without action masks or reward
shaping. This is not a skill evaluation, curriculum qualification, or training.
"""
from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


from ai2.paths import REPO_ROOT as PROJECT_ROOT
from ai2.paths import source_path

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from ai2.training.benchmark_native_combat_training import (
    BACKENDS, PythonGameCPUEnv, SCHEMA as BENCHMARK_SCHEMA, policy_receipt, require_cuda,
)


SCHEMA = 'ai2.native_combat_checkpoint_validation.v1'
CLAIM = 'infrastructure validation only; not evidence of strong combat play'
ROOT = PROJECT_ROOT


@dataclass(frozen=True)
class ValidationConfig:
    seeds: tuple[int, ...] = (3, 7, 101)
    ticks: int = 256
    device: str = 'cuda'
    backend: str = 'native_cuda'

    def __post_init__(self):
        if type(self.ticks) is not int or not 1 <= self.ticks <= 256:
            raise ValueError('ticks must be an integer from 1 to 256')
        if (not 1 <= len(self.seeds) <= 3 or any(type(seed) is not int or not 0 <= seed < 2**32-1 for seed in self.seeds)
                or len(set(self.seeds)) != len(self.seeds)):
            raise ValueError('require 1 to 3 distinct uint32 seeds with room for the second player slot')
        if not isinstance(self.device, str) or not re.fullmatch(r'cuda(?::\d+)?', self.device):
            raise ValueError('policy load and inference require CUDA; no CPU fallback')
        if self.backend not in BACKENDS:
            raise ValueError('unsupported benchmark backend')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def expected_backend(document, backend, seeds):
    if document.get('schema') != BENCHMARK_SCHEMA:
        raise ValueError('unsupported benchmark result schema')
    rows = [row for row in document.get('backends', []) if row.get('backend') == backend]
    if len(rows) != 1:
        raise ValueError('require exactly one matching benchmark backend result')
    row = rows[0]
    if row.get('status') != 'passed':
        raise ValueError('selected benchmark backend must have passed')
    final = row.get('final_policy', {})
    for value in (final.get('sha256'), row.get('checkpoint_sha256')):
        if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
            raise ValueError('benchmark requires exact parameter and checkpoint SHA-256 receipts')
    if type(final.get('parameter_count')) is not int or final['parameter_count'] <= 0:
        raise ValueError('benchmark requires a positive final parameter count')
    config = document.get('config', {})
    if type(config.get('seed')) is not int or type(config.get('games')) is not int or config['games'] <= 0:
        raise ValueError('benchmark training seed and game count are required')
    training_seeds = {config['seed']+2*index for index in range(config['games'])}
    if training_seeds.intersection(seeds):
        raise ValueError('held-out seeds overlap initial benchmark game seeds')
    return row


def source_receipt(frozen_root):
    from ai2.simulation.native_attack_tables import build_tables
    source = Path(frozen_root)/'UTMIST-AI2-main/environment/environment.py'
    tables = build_tables(source.parent)  # Rejects altered frozen source/assets.
    helpers = ('validate_native_combat_checkpoint.py', 'benchmark_native_combat_training.py',
               'combat_episode_capture.py', 'native_attack_tables.py', 'native/navigation_spaces.json')
    return dict(environment_sha256=sha256(source), attack_assets_sha256=tables['asset_sha256'],
                helpers={name: sha256(source_path(ROOT, name)) for name in helpers})


def load_cuda_policy(checkpoint, device, *, ppo_class=None):
    """The guard must run before SB3 deserializes anything, even for direct use."""
    device = require_cuda(str(device))
    if ppo_class is None:
        from stable_baselines3 import PPO
        ppo_class = PPO
    from ai2.simulation.native_lane_vec_env import frozen_spaces
    model = ppo_class.load(str(checkpoint), device=device)
    observation_space, action_space = frozen_spaces()
    if model.observation_space != observation_space or model.action_space != action_space:
        raise ValueError('checkpoint must use frozen Box10 actions and four-frame float32[256] observations')
    if model.policy.squash_output:
        raise ValueError('benchmark policy must use unsquashed Gaussian actions')
    return model.policy


def _array(value, shape, name):
    value = np.asarray(value)
    if value.shape != shape:
        raise ValueError(f'{name} shape must be {shape}')
    if not np.isfinite(value).all():
        raise ValueError(f'{name} must be finite')
    return value


def player_fields(observations):
    """Own-player fields, newest 64-wide frame; source get_obs clips XY."""
    current = _array(observations, (2, 256), 'observations')[:, -64:]
    players = []
    for slot in (0, 1):
        row = current[slot]
        if row[13] != int(row[13]):
            raise ValueError('observed stocks must be integral')
        players.append(dict(player=slot, position_xy=row[:2].astype(float).tolist(),
                            damage_from_obs=float(row[12])*700., stocks=int(row[13]),
                            state_index=float(row[8]), move_type=float(row[14]), weapon_type=float(row[15])))
    return players


def trace_row(seed, tick, before, after, raw_actions, actions, rewards, terminated, truncated, infos):
    before, after = player_fields(before), player_fields(after)
    rewards = _array(rewards, (2,), 'rewards').astype(float)
    raw_actions = _array(raw_actions, (2, 10), 'raw actions')
    actions = _array(actions, (2, 10), 'clipped actions')
    flags = []
    for name, values in (('terminated', terminated), ('truncated', truncated)):
        values = _array(values, (2,), name)
        if not np.isin(values, (0, 1)).all() or values[0] != values[1]:
            raise ValueError('both player slots must share boolean game termination flags')
        flags.append(values.astype(bool))
    damage = [after[p]['damage_from_obs']-before[p]['damage_from_obs'] for p in (0, 1)]
    stocks = [after[p]['stocks']-before[p]['stocks'] for p in (0, 1)]
    return dict(seed=seed, tick=tick, before=before, after=after,
                raw_actions=raw_actions.astype(float).tolist(), actions=actions.astype(float).tolist(),
                rewards=rewards.tolist(), terminated=flags[0].tolist(), truncated=flags[1].tolist(),
                done=np.logical_or(*flags).tolist(), damage_delta_from_obs=damage,
                damage_increased=[value > 0 for value in damage], damage_decreased=[value < 0 for value in damage],
                stock_delta=stocks, stock_lost=[value < 0 for value in stocks], episode_infos=infos)


def cuda_actions(policy, observations, action_space, device, *, torch_module=None):
    if torch_module is None:
        import torch as torch_module
    torch = torch_module
    if tuple(observations.shape) != (2, 256) or observations.dtype != torch.float32:
        raise ValueError('require two float32 four-frame observation slots')
    inputs = observations.to(device)
    if not inputs.is_cuda or inputs.device != device:
        raise RuntimeError('policy input escaped CUDA device')
    if not bool(torch.isfinite(inputs).all().item()):
        raise ValueError('policy observations must be finite')
    with torch.inference_mode():
        raw, _, _ = policy(inputs, deterministic=True)
    if not raw.is_cuda or raw.device != device:
        raise RuntimeError('policy actions escaped CUDA device')
    if tuple(raw.shape) != (2, 10) or raw.dtype != torch.float32:
        raise RuntimeError('policy actions must be float32[2,10]')
    if not bool(torch.isfinite(raw).all().item()):
        raise RuntimeError('policy actions must be finite')
    # CPU transport/clipping only. No policy operation runs on CPU.
    actions = raw.detach().cpu().contiguous()
    sampled = actions.numpy().copy()
    np.clip(actions.numpy(), action_space.low, action_space.high, out=actions.numpy())
    return actions, sampled


def _evaluate_seed(policy, config, seed, frozen_root, device, path):
    env = PythonGameCPUEnv(1, frozen_root, seed=seed, workers=1)
    try:
        if env.num_envs != 2 or env.observation_space.shape != (256,) or env.action_space.shape != (10,):
            raise RuntimeError('original game wrapper shape mismatch')
        observations = env.reset_tensor([seed, seed+1])
        initial = player_fields(observations.numpy())
        returns = np.zeros(2, dtype=np.float64)
        stock_losses, damage_increases = [0, 0], [0, 0]
        infos = []
        with path.open('x') as handle:
            for tick in range(1, config.ticks+1):
                actions, raw = cuda_actions(policy, observations, env.action_space, device)
                step = env.step_tensor(actions)
                row = trace_row(seed, tick, observations.numpy(), step.terminal_observations.numpy(),
                                raw, actions.numpy(), step.rewards.numpy(), step.terminated.numpy(),
                                step.truncated.numpy(), env.drain_episode_infos())
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
                handle.flush()
                returns += row['rewards']
                for player in (0, 1):
                    stock_losses[player] += max(0, -row['stock_delta'][player])
                    damage_increases[player] += int(row['damage_increased'][player])
                infos.extend(row['episode_infos'])
                if any(row['done']):
                    break
                observations = step.observations
        return dict(seed=seed, game_ticks=tick, player_transitions=2*tick,
                    initial=initial, final=row['after'], returns=returns.tolist(),
                    terminated=row['terminated'], truncated=row['truncated'],
                    completed_episode=any(row['done']), capped_by_validation=not any(row['done']),
                    stock_losses=stock_losses, damage_increase_ticks=damage_increases, episode_infos=infos,
                    trace=path.name)
    finally:
        env.close()


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def run_validation(config, checkpoint, benchmark_result, frozen_root, output):
    # The verified original game's context holds cwd at frozen_root until close.
    checkpoint, benchmark_result, frozen_root, output = (
        Path(path).resolve() for path in (checkpoint, benchmark_result, frozen_root, output))
    if output.exists():
        raise ValueError('output directory must be new; existing artifacts are never overwritten')
    output.mkdir(parents=True)
    summary = dict(schema=SCHEMA, status='running', claim=CLAIM, config=asdict(config), games=[],
                   checkpoint=str(checkpoint.resolve()), benchmark_result=str(benchmark_result.resolve()),
                   action_mode='deterministic Gaussian means; clipped to original Box bounds; no masks',
                   observation_fields='newest frame offset 192: XY=0:2 (clipped game units), damage=12*700 (float32-derived), stocks=13',
                   termination='stop after first episode or tick cap; trace terminal observations precede wrapper autoreset',
                   seed_scope='distinct initial game seeds, disjoint from initial benchmark game seeds; generators differ')
    _write_json(output/'summary.json', summary)
    try:
        device = require_cuda(config.device)
        document = json.loads(benchmark_result.read_text())
        expected = expected_backend(document, config.backend, config.seeds)
        summary['benchmark_result_sha256'] = sha256(benchmark_result)
        summary['checkpoint_sha256'] = sha256(checkpoint)
        if summary['checkpoint_sha256'] != expected['checkpoint_sha256']:
            raise RuntimeError('checkpoint file hash does not match benchmark result')
        summary['source'] = source_receipt(frozen_root)
        for field in ('environment_sha256', 'attack_assets_sha256'):
            recorded = document.get('provenance', {}).get(field)
            if recorded is not None and recorded != summary['source'][field]:
                raise RuntimeError(f'benchmark {field} does not match original evaluation game')
        policy = load_cuda_policy(checkpoint, device)
        loaded = policy_receipt(policy, device)
        if loaded != expected['final_policy']:
            raise RuntimeError('loaded CUDA policy parameter hash/count does not match benchmark final policy')
        summary['loaded_policy'] = loaded
        policy.set_training_mode(False)
        for seed in config.seeds:
            summary['games'].append(_evaluate_seed(policy, config, seed, frozen_root, device, output/f'seed-{seed}.jsonl'))
            _write_json(output/'summary.json', summary)
        if policy_receipt(policy, device) != loaded:
            raise RuntimeError('policy parameters changed during validation')
        if sha256(checkpoint) != summary['checkpoint_sha256'] or sha256(benchmark_result) != summary['benchmark_result_sha256']:
            raise RuntimeError('checkpoint or benchmark result changed during validation')
        if source_receipt(frozen_root) != summary['source']:
            raise RuntimeError('evaluation source changed during validation')
        summary.update(status='passed', parameters_finite=True, actions_finite=True,
                       game_ticks=sum(row['game_ticks'] for row in summary['games']),
                       player_transitions=sum(row['player_transitions'] for row in summary['games']))
        return summary
    except BaseException as exc:
        summary.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        _write_json(output/'summary.json', summary)
        _write_json(output/'artifacts.json', dict(schema=SCHEMA, status=summary['status'],
                    sha256={path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file() and path.name != 'artifacts.json'}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'benchmark-result', 'frozen-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--backend', choices=BACKENDS, default='native_cuda')
    parser.add_argument('--seeds', type=int, nargs='+', default=[3, 7, 101])
    parser.add_argument('--ticks', type=int, default=256)
    args = parser.parse_args(argv)
    config = ValidationConfig(tuple(args.seeds), args.ticks, args.device, args.backend)
    result = run_validation(config, args.checkpoint, args.benchmark_result, args.frozen_root, args.output)
    print(json.dumps(dict(status=result['status'], claim=CLAIM, game_ticks=result['game_ticks'],
                         player_transitions=result['player_transitions'], output=str(args.output)), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
