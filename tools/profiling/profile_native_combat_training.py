#!/usr/bin/env python3
"""Optional CUDA-only raw combat/PPO attribution; NOT headline throughput.

Runs the unchanged benchmark with 128 games, 128 steps, MLP 512/512/256,
batch 4096, eight epochs, one warmup and one measured rollout. Profiler overhead
affects all timings, including benchmark/result.json. Use an unprofiled matched
run for performance claims. No gameplay, policy, or collector algorithm changes.
"""
from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
import time

import ai2.training.benchmark_native_combat_training as driver
import tools.profiling.profile_native_lane_training as regions


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    return parser


def source_hashes():
    paths = (Path(__file__).resolve(), Path(regions.__file__).resolve(), Path(driver.__file__).resolve())
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def run(args):
    config = driver.BenchmarkConfig(games=128, n_steps=128, batch_size=4096, n_epochs=8,
        net_arch=(512, 512, 256), warmup_rollouts=1, measured_rollouts=1,
        backends=('native_cuda',), device=args.device, seed=args.seed)
    import torch
    device = driver.require_cuda(config.device, torch_module=torch)
    activity = torch.profiler.ProfilerActivity
    if activity.CUDA not in torch.profiler.supported_activities():
        raise RuntimeError('Torch profiler CUDA activity support is required')
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('output must be new or empty; existing receipts are not overwritten')
    from stable_baselines3.common.policies import ActorCriticPolicy
    from ai2.training.tensor_ppo import TensorPPO, TensorVecEnv
    from ai2.training.tensor_rollout_buffer import TensorRolloutBuffer

    output.mkdir(parents=True, exist_ok=True)
    interpretation = [line for line in regions.INTERPRETATION if 'NativeLaneEnv.step_tensor' not in line]
    interpretation += [
        'native_environment_step wraps TensorVecEnv.step_tensor, including its existing action/output handling.',
        'All benchmark/ artifacts are instrumented diagnostics, not headline throughput or combat mastery evidence.',
        'The whole-driver trace includes startup and final checkpoint saving; named methods include warmup and measured rollouts.',
        'CPU/device tables are profiler aggregates; nested inclusive totals and annotation spans must not be summed.',
    ]
    report = dict(schema='ai2.raw_combat_training_profile.v1', status='running', diagnostic_only=True,
        headline_throughput_eligible=False, config=asdict(config), interpretation=interpretation,
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        gpu=torch.cuda.get_device_name(device), activities=['CPU', 'CUDA'],
        source_hashes=source_hashes(), benchmark_result='benchmark/result.json',
        record_shapes=False, with_stack=False, profile_memory=False)
    driver._write_json(output / 'profile.json', report)
    started, failure, stats, profiler = time.perf_counter(), None, None, None
    try:
        with torch.profiler.profile(activities=[activity.CPU, activity.CUDA], record_shapes=False,
                                    with_stack=False, profile_memory=False) as profiler:
            with regions.instrument_regions(policy_class=ActorCriticPolicy, env_class=TensorVecEnv,
                    buffer_class=TensorRolloutBuffer, ppo_class=TensorPPO,
                    record_function=torch.profiler.record_function) as stats:
                result = driver.run_benchmark(config, output / 'benchmark', Path(args.frozen_root))
                if result['status'] != 'passed':
                    raise RuntimeError('raw combat benchmark did not pass')
    except BaseException as exc:
        failure = exc
    report['diagnostic_driver_wall_seconds'] = time.perf_counter() - started
    try:
        if profiler is None:
            raise RuntimeError('profiler did not start')
        profiler.export_chrome_trace(str(output / 'trace.json'))
        averages = profiler.key_averages()
        if stats is not None:
            aggregate = regions.aggregate_regions(averages, stats)
            aggregate.pop('interpretation')
            report.update(aggregate)
        (output / 'cpu-table.txt').write_text(averages.table(sort_by='self_cpu_time_total', row_limit=100) + '\n')
        (output / 'device-table.txt').write_text(averages.table(sort_by='self_device_time_total', row_limit=100) + '\n')
        report['cuda_trace_event_count'] = sum(regions._device_type(event) == 'CUDA' for event in profiler.events())
        report['source_hashes_unchanged'] = source_hashes() == report['source_hashes']
        report['artifact_sha256'] = {name: driver._sha(output / name)
                                    for name in ('trace.json', 'cpu-table.txt', 'device-table.txt')}
        if not report['source_hashes_unchanged']:
            raise RuntimeError('profiling source changed during capture')
        if not report['cuda_trace_event_count']:
            raise RuntimeError('No CUDA device events captured; no CUDA profiling evidence')
        missing = [name for name in regions.REGIONS if stats is None or not stats[name]['calls']]
        if missing:
            raise RuntimeError(f'Expected profiler regions did not execute: {missing}')
    except BaseException as exc:
        report['artifact_error'] = f'{type(exc).__name__}: {exc}'
        if failure is None:
            failure = exc
    report['status'] = 'failed' if failure else 'passed'
    if failure:
        report['error'] = f'{type(failure).__name__}: {failure}'
    driver._write_json(output / 'profile.json', report)
    if failure:
        raise failure
    return 0


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
