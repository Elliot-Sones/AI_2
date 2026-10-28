"""Trace-first original-game evaluation; real policy inference is CUDA only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np

from train_combat_curriculum import (CONTACT_METRIC_VERSION, NavigationTraceWriter, atomic_json, require_cuda,
                                    sha256, validate_deadline, write_inventory)


class FrameHistory:
    def __init__(self):
        self.values = np.zeros(256, dtype=np.float32)

    def reset(self):
        self.values.fill(0)

    def push(self, observation):
        value = np.asarray(observation, dtype=np.float32)
        if value.shape != (64,) or not np.isfinite(value).all():
            raise ValueError('expected finite raw64 observation')
        self.values[:-64] = self.values[64:].copy()
        self.values[-64:] = value
        return self.values.copy()


def evaluation_cases(task, *, count_per_mode=20, historical=False, confirmation=False):
    if task not in {'combat', 'near', 'wide'} or count_per_mode <= 0 or count_per_mode % 2:
        raise ValueError('task and even positive per-mode count required')
    if historical and (task == 'combat' or count_per_mode > 100):
        raise ValueError('historical reproduction only covers 100 navigation seeds')
    if historical:
        first = 84001 if task == 'near' else 83001
    else:
        first = {'combat': 131000, 'near': 132000, 'wide': 133000}[task]
        first += 10000 if confirmation else 0
    rows = []
    for mode in ('deterministic', 'sampled'):
        for index in range(count_per_mode):
            seed = first + index
            slot = 0 if historical else index % 2
            rows.append({'id': f'{task}-{mode}-p{slot}-s{seed}', 'task': task,
                         'seed': seed, 'player_slot': slot, 'mode': mode,
                         'direction': 1 if (index // 2) % 2 else -1,
                         'bank': 'historical-development' if historical else ('confirmation' if confirmation else 'development'),
                         'metric_version': CONTACT_METRIC_VERSION if task == 'combat' else 'active_sequence_v3'})
    return rows


def run_case(env, action_fn, case, *, deadline_unix, trace_sink=None):
    validate_deadline(deadline_unix)
    random.seed(case['seed'])
    np.random.seed(case['seed'])
    observation, _ = env.reset(seed=case['seed'], options={'scenario': case})
    frames = FrameHistory()
    traces = []
    total_reward = 0.0
    started = time.perf_counter()
    digest = hashlib.sha256()
    maximum = 300 if case['task'] == 'combat' else 600
    for tick in range(1, maximum + 1):
        validate_deadline(deadline_unix, reserve_seconds=10)
        stacked = frames.push(observation)
        raw = np.asarray(action_fn(stacked, case['mode'] == 'deterministic'), dtype=np.float32).reshape(-1)
        if raw.shape != (10,) or not np.isfinite(raw).all():
            raise ValueError('policy returned nonfinite or non10 actions')
        digest.update(stacked.tobytes())
        digest.update(raw.tobytes())
        observation, reward, terminated, truncated, info = env.step(np.clip(raw, 0, 1))
        total_reward += float(reward)
        trace = info.get('trace' if case['task'] == 'combat' else 'navigation_trace')
        if trace is not None:
            trace = dict(trace)
            source = case.get('action_source', 'policy_mean' if case['mode'] == 'deterministic'
                              else 'raw_gaussian_sample')
            trace.update(raw_action=raw.tolist(), action_source=source,
                         scenario_id=case['id'], mode=case['mode'])
            traces.append(trace)
            if trace_sink is not None:
                trace_sink(trace)
        if terminated or truncated:
            diagnostics = dict(info.get('episode_diagnostics', {}))
            if not diagnostics:
                raise RuntimeError('terminal episode lacks behavioral diagnostics')
            if case['task'] == 'combat' and traces:
                from curriculum_telemetry import aggregate_episode
                reconstructed = aggregate_episode(traces, learner_slot=case['player_slot'],
                                                  scenario_id=case['id'], mode=case['mode'])
                if bool(diagnostics.get('success')) != reconstructed['success']:
                    raise RuntimeError('environment success and trace reconstruction disagree')
                reconstructed_reward = sum(reconstructed['reward_terms'].values())
                if not np.isclose(reconstructed_reward, total_reward, rtol=0, atol=1e-6):
                    raise RuntimeError('reward cannot be reconstructed from trace components')
                diagnostics = {**diagnostics, **reconstructed}
            return {**case, **diagnostics, 'ticks': tick, 'sim_seconds': tick / 30,
                    'rng_scheme': 'python_numpy_seed_v1',
                    'wall_seconds': time.perf_counter() - started, 'return': total_reward,
                    'terminated': bool(terminated), 'truncated': bool(truncated),
                    'policy_input_action_sha256': digest.hexdigest()}
    raise RuntimeError('environment exceeded its declared episode deadline')


def evaluate(args):
    validate_deadline(args.deadline_unix, reserve_seconds=120)
    hardware = require_cuda()
    import torch
    from stable_baselines3 import PPO
    from curriculum_ppo import CurriculumPPO
    from curriculum_telemetry import AtomicTraceWriter
    from combat_curriculum import make_b1_contact_env
    from curriculum_navigation import make_navigation_env

    output = Path(args.output)
    if output.exists():
        raise FileExistsError('evaluation outputs are immutable')
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    if args.parent:
        from train_combat_curriculum import validate_parent
        validate_parent(args.checkpoint)
        model = PPO.load(args.checkpoint, device='cuda')
    else:
        model = CurriculumPPO.load(args.checkpoint, device='cuda')
        model.set_task('combat' if args.task == 'combat' else 'navigation', calibration=False)
    if any(p.device.type != 'cuda' for p in model.policy.parameters()):
        raise RuntimeError('evaluation policy is not entirely CUDA')
    model.policy.set_training_mode(False)

    def action_fn(observation, deterministic):
        with torch.no_grad():
            tensor, _ = model.policy.obs_to_tensor(observation)
            distribution = model.policy.get_distribution(tensor)
            return distribution.get_actions(deterministic=deterministic).cpu().numpy()[0]

    cases = evaluation_cases(args.task, count_per_mode=args.episodes,
                             historical=args.historical, confirmation=args.confirmation)
    atomic_json(output / 'manifest.json', {'checkpoint_sha256': sha256(args.checkpoint),
                'task': args.task, 'hardware': hardware, 'cases': cases,
                'historical_reproduction': args.historical, 'no_learning': True})
    rows = []
    for case in cases:
        env = (make_b1_contact_env(str(args.frozen_root), seed=case['seed'], player_slot=case['player_slot'])
               if args.task == 'combat' else make_navigation_env(str(args.frozen_root),
                    seed=case['seed'], variant=args.task, player_slot=case['player_slot']))
        writer_type = AtomicTraceWriter if args.task == 'combat' else NavigationTraceWriter
        writer = writer_type(output / 'traces' / case['id'])
        try:
            np.random.seed(case['seed'])
            torch.manual_seed(case['seed'])
            row = run_case(env, action_fn, case, deadline_unix=args.deadline_unix,
                           trace_sink=writer.append)
            rows.append(row)
            atomic_json(output / 'episodes' / (case['id'] + '.json'), row)
            print(json.dumps({k: row.get(k) for k in ('id', 'outcome', 'success', 'ticks', 'return')}, allow_nan=False), flush=True)
        finally:
            writer.close()
            env.close()
    summary = {}
    for mode in ('deterministic', 'sampled'):
        subset = [row for row in rows if row['mode'] == mode]
        summary[mode] = {'episodes': len(subset),
                         'successes': sum(row.get('success') is True for row in subset),
                         'falls': sum(bool(row.get('own_stock_loss', row.get('fell', row.get('fall', False)))) for row in subset),
                         'mean_ticks': sum(row['ticks'] for row in subset) / len(subset)}
    source_digest = hashlib.sha256()
    for name in ('combat_curriculum.py', 'curriculum_navigation.py', 'curriculum_telemetry.py',
                 'curriculum_evaluation.py', 'evaluate_combat_curriculum.py'):
        source_digest.update(name.encode())
        source_digest.update((Path(__file__).resolve().parent / name).read_bytes())
    bank_digest = hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest()
    provenance = {'checkpoint_sha256': sha256(args.checkpoint), 'source_sha256': source_digest.hexdigest(),
                  'metric_version': cases[0]['metric_version'], 'check_id': output.name,
                  'measured_at_utc': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                  'scenario_bank_id': bank_digest}
    atomic_json(output / 'summary.json', {'task': args.task, 'metric_version': cases[0]['metric_version'],
                'verified_unchanged_parent': args.parent, 'provenance': provenance,
                'check_id': output.name, 'checkpoint_sha256': provenance['checkpoint_sha256'],
                'steps': model.num_timesteps, 'bank': cases[0]['bank'], 'modes': summary, 'episodes': rows})
    if args.task == 'combat':
        from curriculum_evaluation import render_markdown_report
        with (output / 'behavior-report.md').open('x') as handle:
            handle.write(render_markdown_report({'status': 'evaluated_not_promoted', 'episodes': rows}))
    write_inventory(output, block_id=output.name)
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--frozen-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', choices=('combat', 'near', 'wide'), required=True)
    parser.add_argument('--episodes', type=int, default=20, help='episodes per action mode')
    parser.add_argument('--deadline-unix', type=float, required=True)
    parser.add_argument('--parent', action='store_true')
    parser.add_argument('--historical', action='store_true')
    parser.add_argument('--confirmation', action='store_true')
    return parser.parse_args(argv)


if __name__ == '__main__':
    raise SystemExit(evaluate(parse_args()))
