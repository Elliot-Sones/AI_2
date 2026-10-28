#!/usr/bin/env python3
"""Evaluate a locally preserved native-trained policy in the original Python game.

Policy inference remains CUDA-only. The unchanged reference evaluator writes
frame-level XY/actions/attack/damage traces and reconstructs episode outcomes.
This wrapper adds a paired nearby-navigation retention panel against the real
unchanged navigation parent; it never advances the curriculum.
"""
from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


from ai2.paths import source_path

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ai2.training.benchmark_native_contact_training import (
    ROOT, atomic_json, preserve_failed_block, require_cuda, require_qualification, sha256,
    validate_deadline, write_inventory,
)

PARENT_SHA256 = '0e2f64e02cd0f7d300554f7a6cb641065c75771f69f55c0885258dda32afc6cd'
PARENT_BYTES = 123047421
RESULT_CONTRACT = {'status': 'evaluated_not_promoted', 'mastery_established': False,
                   'required_development_checks': 2, 'required_confirmation_episodes_per_mode': 100}


def require_preserved_checkpoint(checkpoint, ack_path):
    checkpoint = Path(checkpoint)
    directory = checkpoint.parent
    inventory_path = directory / 'inventory.json'
    inventory = json.loads(inventory_path.read_text())
    ack = json.loads(Path(ack_path).read_text())
    if (ack.get('verified') is not True or ack.get('block_id') != inventory.get('block_id')
            or ack.get('inventory_sha256') != sha256(inventory_path)
            or inventory.get('schema_version') != 'curriculum-inventory-v1'):
        raise ValueError('candidate checkpoint requires its hash-matched verified-local preservation ACK')
    matching = [row for row in inventory['files'] if row['path'] == checkpoint.name]
    if (len(matching) != 1 or checkpoint.is_symlink()
            or matching[0]['bytes'] != checkpoint.stat().st_size
            or matching[0]['sha256'] != sha256(checkpoint)):
        raise ValueError('candidate checkpoint does not match the preserved inventory')
    result_path = directory / 'result.json'
    result_entries = [row for row in inventory['files'] if row['path'] == result_path.name]
    if len(result_entries) != 1 or result_path.is_symlink() or not result_path.is_file():
        raise ValueError('result metadata does not match the preserved inventory')
    result_bytes = result_path.read_bytes()
    if (result_entries[0]['bytes'] != len(result_bytes)
            or result_entries[0]['sha256'] != hashlib.sha256(result_bytes).hexdigest()):
        raise ValueError('result metadata does not match the preserved inventory')
    result = json.loads(result_bytes)
    if result.get('status') != 'trained_not_evaluated' or result.get('checkpoint_sha256') != sha256(checkpoint):
        raise ValueError('candidate is not the completed final checkpoint of this block')
    return {'checkpoint_sha256': matching[0]['sha256'], 'inventory_sha256': sha256(inventory_path),
            'preservation_ack_sha256': sha256(ack_path)}


def evaluation_jobs(args):
    return [{'name': 'near-parent', 'task': 'near', 'checkpoint': args.navigation_parent, 'parent': True},
            {'name': 'contact-current', 'task': 'combat', 'checkpoint': args.checkpoint, 'parent': False},
            {'name': 'near-current', 'task': 'near', 'checkpoint': args.checkpoint, 'parent': False}]


def job_output(args, job):
    # Reference check_id is the output basename; keep it distinct across blocks.
    return Path(args.output) / f'{Path(args.output).name}-{job["name"]}'


def evaluation_command(args, job):
    command = [sys.executable, '-B', str(Path(args.reference_root) / 'evaluate_combat_curriculum.py'),
        '--checkpoint', str(job['checkpoint']), '--frozen-root', str(args.frozen_root),
        '--output', str(job_output(args, job)), '--task', job['task'],
        '--episodes', str(args.episodes), '--deadline-unix', str(args.deadline_unix)]
    if job['parent']:
        command.append('--parent')
    return command


def evaluate(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('evaluation outputs must be new and immutable')
    if args.episodes != 20:
        raise ValueError('this bounded comparison uses the existing 20-per-mode development panels')
    validate_deadline(args.deadline_unix)
    proof = require_qualification(args.qualification, args.reference_root, args.frozen_root)
    preserved = require_preserved_checkpoint(args.checkpoint, args.verified_ack)
    parent = Path(args.navigation_parent)
    if parent.stat().st_size != PARENT_BYTES or sha256(parent) != PARENT_SHA256:
        raise ValueError('nearby retention requires the exact unchanged navigation parent')
    require_cuda('cuda')
    # Original evaluator imports curriculum_ppo. Any reference copy must be the
    # same frozen policy module the adapter used, not an unqualified replacement.
    reference_policy = source_path(Path(args.reference_root), 'curriculum_ppo.py')
    if reference_policy.exists() and sha256(reference_policy) != sha256(source_path(ROOT, 'curriculum_ppo.py')):
        raise ValueError('reference evaluation policy source differs from the qualified adapter source')
    output.mkdir(parents=True, exist_ok=False)
    jobs = evaluation_jobs(args)
    atomic_json(output / 'manifest.json', {'schema_version': 'ai2.native_contact_evaluation.v1',
        **RESULT_CONTRACT, 'candidate': preserved, 'navigation_parent_sha256': PARENT_SHA256,
        'qualification_sha256': sha256(args.qualification), 'qualification': proof,
        'episodes_per_action_mode': args.episodes, 'simulation': 'original Python game',
        'inference_device': 'cuda', 'full_frame_and_event_traces': True,
        'commands': [evaluation_command(args, job) for job in jobs], 'started_unix': time.time()})
    environment = {**os.environ, 'WANDB_MODE': 'offline', 'OMP_NUM_THREADS': '1',
                   'SDL_VIDEODRIVER': 'dummy', 'SDL_AUDIODRIVER': 'dummy',
                   'PYTHONPATH': str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', '')}
    reports = {}
    started = time.perf_counter()
    try:
        for job in jobs:
            validate_deadline(args.deadline_unix)
            with (output / f'{job["name"]}.log').open('x', encoding='utf8') as log:
                result = subprocess.run(evaluation_command(args, job), env=environment,
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False,
                    timeout=max(1., args.deadline_unix - time.time() - 60))
            if result.returncode:
                raise RuntimeError(f'original-game {job["name"]} evaluation failed: exit {result.returncode}')
            reports[job['name']] = json.loads((job_output(args, job) / 'summary.json').read_text())
        source = Path(args.reference_root) / 'curriculum_evaluation.py'
        spec = importlib.util.spec_from_file_location('_native_contact_reference_gates', source)
        gates = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gates)
        result = {**RESULT_CONTRACT, 'panels': reports,
            'contact_fast_gate': gates.fast_gate([reports['contact-current']]),
            'nearby_retention': gates.retention_gate(reports['near-current'], reports['near-parent'], panel_kind='fast'),
            'wall_seconds': time.perf_counter() - started, 'finished_unix': time.time(),
            'interpretation': 'one contact development check is diagnostic, not mastery; retention is a fast alarm only'}
        atomic_json(output / 'result.json', result)
        write_inventory(output, block_id=output.name)
        return result
    except BaseException as error:
        # subprocess.run has exited or killed/waited for its timed-out child;
        # the surrounding with block has closed the stdout/stderr log stream.
        preserve_failed_block(output, error, completed_panels=list(reports))
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'navigation-parent', 'verified-ack', 'frozen-root',
                 'reference-root', 'qualification', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--deadline-unix', type=float, required=True)
    parser.add_argument('--episodes', type=int, default=20)
    return parser.parse_args(argv)


if __name__ == '__main__':
    evaluate(parse_args())
