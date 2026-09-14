#!/usr/bin/env python3
"""Balanced, CUDA-only continuation of the nine E6 model-size lineages.

One invocation is one common cumulative checkpoint. Dry-run is the default;
execution uses the existing finite subprocess supervisor. A new round must be
explicitly selected after reviewing learning, not started by an infinite loop.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import dataclasses
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import zipfile

import campaign_learning as learning

ARCHITECTURES = ((512,512,256), (1024,1024,1024), (1536,1536,1536))
SEEDS = (145,146,147)
CONTROLS = {'n_envs':64, 'n_steps':512, 'batch_size':8192, 'epochs':8}
ROLLOUT = 32768
PANELS = {'dev':{'near':84001,'wide':83001}, 'final':{'near':114001,'wide':113001}}
INITIAL_HASHES = (
    'ab79ed1b28cac7644428fe79ef4426cdba1e2a5c93c1299399ab3b95c6651f89',
    'a4f77cfc0c72c229fd98559136b2e014a8341891a7df2e41e7f7f58c1d59144b',
    '69306277009711552b8d981595d4e8e907ccb31b9a06043f0b3c57718016433d',
    '156906b96df2aa53e6b10a2af294be2f5e7854940b5b9c20f4fbc9fda89f4341',
    'eed042a87d5376b9e714d282c4320a4868b9977b6b78890bb91caad0d893fef2',
    'da708d62b100bfed834c3747175c44d45f3bd04bcbf2246bfe227dc2a7d7b609',
    '98a2dc3d17b8608a9632730b692a20a89fe3e504fec10ed691a92d474bb97935',
    '746b13ff27c572fdb3081a3784104d49587976f6064716467fefe57c52efdd47',
    '4d94803e125a2f363cd86ebec75bb37b4057d40d931c2d3c55be6c8ae35a03a8',
)


def rounded_target(target: int) -> int:
    if type(target) is not int or target <= 0:
        raise ValueError('target must be a positive integer')
    return ((target+ROLLOUT-1)//ROLLOUT)*ROLLOUT


def initial_records(root: Path) -> list[dict]:
    rows = []
    for arch in ARCHITECTURES:
        for seed in SEEDS:
            label = f"arch-{'-'.join(map(str,arch))}_seed{seed}"
            rows.append({'label':label, 'net_arch':list(arch), 'seed':seed,
                         'parameters':learning.architecture_parameter_count(256,10,arch),
                         'steps':2097152, 'updates':512,
                         'checkpoint':str(root/'inputs'/label/'final.zip'),
                         'checkpoint_sha256':INITIAL_HASHES[len(rows)],
                         'config':str(root/'variants'/label/'config.yaml')})
    return rows


def checkpoint_metadata(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read('data'))


def validate_inputs(rows: list[dict], *, read_files: bool = False) -> None:
    expected = {(arch,seed) for arch in ARCHITECTURES for seed in SEEDS}
    actual = {(tuple(row['net_arch']),row['seed']) for row in rows}
    if len(rows)!=9 or actual!=expected or len({row['label'] for row in rows})!=9:
        raise ValueError('require exactly the nine distinct matched architecture/seed arms')
    if len({(r['steps'],r['updates']) for r in rows})!=1:
        raise ValueError('all starting checkpoints must have equal steps and updates')
    baseline_config = None
    for row in rows:
        if row['steps'] <= 0 or row['steps'] % ROLLOUT or row['updates'] != row['steps']//ROLLOUT*8:
            raise ValueError('inputs must be completed-update checkpoints under E6 geometry')
        expected_count = learning.architecture_parameter_count(256,10,tuple(row['net_arch']))
        if row['parameters'] != expected_count:
            raise ValueError('parameter count does not match architecture')
        if read_files:
            checkpoint = Path(row['checkpoint'])
            if learning.sha256_file(checkpoint) != row['checkpoint_sha256']:
                raise ValueError(f"checkpoint hash mismatch: {row['label']}")
            data = checkpoint_metadata(checkpoint)
            expected_values = {'num_timesteps':row['steps'], '_n_updates':row['updates'],
                               'n_envs':64,'n_steps':512,'batch_size':8192,'n_epochs':8,'ent_coef':0.0}
            if any(data.get(key)!=value for key,value in expected_values.items()):
                raise ValueError(f"checkpoint control mismatch: {row['label']}")
            import yaml
            config = yaml.safe_load(Path(row['config']).read_text())
            ppo = config['ppo_settings']
            if config['policy_kwargs']['net_arch'] != row['net_arch'] or any(
                ppo.get(k)!=v for k,v in {'learning_rate':0.0003,'clip_range':0.2,'nav_ent_coef':0.0}.items()
            ):
                raise ValueError(f"configuration mismatch: {row['label']}")
            for phase, minimum, maximum in [('0a',1.5,3.),('0b',2.,6.)]:
                phase_config=config['phases'][phase]
                required={'min_dist':minimum,'max_dist':maximum,'fall_penalty':-3.,
                          'time_limit_seconds':20,'multi_target':3,'completion_mode':'sequence',
                          'control_mode':'movement','settle_seconds':.3,'post_goal_seconds':.5}
                if phase_config.get('type')!='navigation' or phase_config.get('opponents')!={'constant_agent':1.0} or any(
                    phase_config.get('navigation',{}).get(k)!=v for k,v in required.items()
                ):
                    raise ValueError(f"frozen curriculum mismatch: {row['label']} {phase}")
            normalized=copy.deepcopy(config)
            normalized['policy_kwargs'].pop('net_arch')
            if baseline_config is None:
                baseline_config=normalized
            elif normalized!=baseline_config:
                raise ValueError('all arm configs must match except net_arch')


def build_plan(root: Path, rows: list[dict], *, target: int, phase: str,
               panel: str, episodes: int, python: str = sys.executable,
               train_timeout: float = 1200, eval_timeout: float = 600,
               evaluate_only: bool = False) -> dict:
    validate_inputs(rows)
    end = rounded_target(target)
    if phase not in ('0a','0b') or panel not in PANELS or type(episodes) is not int or not 1<=episodes<=100:
        raise ValueError('unsupported phase/panel/episode count')
    start = rows[0]['steps']
    if evaluate_only and end!=start:
        raise ValueError('evaluate-only target must equal input checkpoint steps')
    if not evaluate_only and end<=start:
        raise ValueError('target must advance all models')
    if any(not math.isfinite(t) or t<=0 for t in (train_timeout,eval_timeout)):
        raise ValueError('timeouts must be positive and finite')
    output_root = root/'artifacts'/f'{phase}-steps{end}-{panel}{episodes}'
    commands = []
    # Rotate architecture order between seeds; no simultaneous GPU jobs.
    ordered = []
    for index,seed in enumerate(SEEDS):
        arches = ARCHITECTURES[index:]+ARCHITECTURES[:index]
        ordered.extend(next(r for r in rows if r['seed']==seed and tuple(r['net_arch'])==arch) for arch in arches)
    for row in ordered:
        source = Path(row['config']).parent
        output = output_root/row['label']/'train'
        config_hash = learning.sha256_file(Path(row['config'])) if Path(row['config']).is_file() else None
        argv = [python,str(root/'source/native_learning_runner.py'),
                '--source',str(source),'--train-job',str(root/'scripts/train_job.py'),
                '--lane-build-dir',str(root/'build'), '--config',row['config'],
                '--phase',phase,'--seed',str(row['seed']), '--steps',str(end-start),
                '--n-envs','64','--n-steps','512','--batch-size','8192','--epochs','8',
                '--device','cuda','--checkpoint',row['checkpoint'],'--output',str(output),
                '--expected-parameters',str(row['parameters']),'--max-seconds',str(train_timeout)]
        meta = {'output_root':str(root/'artifacts'), 'phase':phase,'seed':row['seed'],
                'net_arch':row['net_arch'],'expected_parameter_count':row['parameters'],
                'checkpoint':row['checkpoint'],'config_sha256':config_hash,
                'expected_starting_steps':start,'expected_ending_steps':end,
                'rounded_steps':end-start,'controls':dict(CONTROLS),'nav_ent_coef':0.0}
        if not evaluate_only:
            commands.append(dataclasses.asdict(learning.CommandSpec(row['label']+'-train','train',source,
                                                                   argv,output,train_timeout+30,meta)))
        checkpoint = Path(row['checkpoint']) if evaluate_only else output/'final.zip'
        eval_output = output.parent/'eval'
        eval_argv = [python,str(root/'scripts/evaluate_modes.py'),'--repo',str(source),
                     '--checkpoint',str(checkpoint),'--output',str(eval_output),'--device','cuda',
                     '--episodes',str(episodes),'--near-seed',str(PANELS[panel]['near']),
                     '--wide-seed',str(PANELS[panel]['wide'])]
        eval_meta = {'output_root':str(root/'artifacts'),'episodes':episodes,
                     'checkpoint':str(checkpoint),'config_sha256':config_hash}
        commands.append(dataclasses.asdict(learning.CommandSpec(row['label']+'-eval','eval',source,
                                                               eval_argv,eval_output,eval_timeout,eval_meta)))
    return learning.json_safe({'schema':'native-capacity-round-v1','root':root,'output':output_root,
                               'phase':phase,'target':end,'starting_steps':start,'panel':panel,
                               'episodes_per_panel':episodes,'evaluate_only':evaluate_only,'inputs':rows,'commands':commands})


def evaluation_record(path: Path, *, phase: str = '0a') -> dict:
    if phase not in ('0a','0b'):
        raise ValueError('unsupported mastery phase')
    mastery_phase=phase
    panels = {}
    for phase in ('0a','0b'):
        for mode in ('deterministic','sampled'):
            panels[f'{phase}-{mode}'] = json.loads((path/f'{phase}-{mode}.json').read_text())['summary']
    required_phases=('0a',) if mastery_phase=='0a' else ('0a','0b')
    required=[panels[f'{p}-{mode}'] for p in required_phases for mode in ('deterministic','sampled')]
    return {'panels':panels,'mastery_phase':mastery_phase,
            'candidate_mastery':all(p['success_rate']>=.9 and p['fall_rate']<=.05 for p in required),
            'mastery_confirmed':all(p['episodes']>=100 and p['success_rate']>=.9 and p['fall_rate']<=.05 for p in required)}


@contextmanager
def tracking_environment(root: Path):
    original=learning.command_env
    def environment(command):
        env=original(command)
        env.update(WANDB_MODE='offline',WANDB_DIR=str(root/'wandb'),
                   WANDB_RUN_GROUP='native-capacity-session11',WANDB_NAME=command.label,
                   SDL_AUDIODRIVER='dummy',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                   WANDB_NOTES=('Goal: equal-step model-size comparison, native navigation continuation of E6. '
                     f'Change: {command.label}; CUDA environment and PPO, unchanged64x512 rollout geometry. '
                     'Expect: measure completed-update learning and original-game success/falls, not human combat mastery. '
                     'Credentials remain local; offline training records are downloaded without cloud sync.'))
        return env
    learning.command_env=environment
    try:
        yield
    finally:
        learning.command_env=original


def validate_native_train_artifacts(command: learning.CommandSpec) -> dict:
    """Do not let a completed SB3 result bypass the native post-update gate."""
    evidence = learning.validate_train_artifacts(command)
    receipt_path = command.output / 'native_learning_receipt.json'
    try:
        receipt = json.loads(receipt_path.read_text())
        start = command.metadata['expected_starting_steps']
        end = command.metadata['expected_ending_steps']
        controls = command.metadata['controls']
        rollout = controls['n_envs'] * controls['n_steps']
        cycles, remainder = divmod(end - start, rollout)
        if cycles <= 0 or remainder:
            raise ValueError('invalid completed-cycle geometry')
        initial = checkpoint_metadata(Path(command.metadata['checkpoint']))
        initial_counters = {'num_timesteps': start, '_n_updates': initial['_n_updates']}
        final_counters = {'num_timesteps': end, '_n_updates': initial['_n_updates'] + cycles * controls['epochs']}
        expected = {'starting_steps': start, 'expected_final_steps': end,
                    'expected_additional_steps': end - start, 'transitions_per_cycle': rollout,
                    'cycles': cycles, 'completed_cycles': cycles,
                    'expected_final_updates': final_counters['_n_updates']}
        if (receipt.get('status') != 'completed' or receipt.get('error') is not None
                or receipt.get('final_completed_update_verified') is not True
                or any(type(receipt.get(key)) is not int or receipt[key] != value
                       for key, value in expected.items())):
            raise ValueError('receipt lacks the expected completed-cycle proof')
        if (receipt.get('initial_checkpoint_counters') != initial_counters
                or receipt.get('final_checkpoint_counters') != final_counters):
            raise ValueError('receipt checkpoint counters differ from expected updates')
        for name, expected_counters in [('initial.zip', initial_counters),
                                        ('latest.zip', final_counters), ('final.zip', final_counters)]:
            data = checkpoint_metadata(command.output / name)
            if any(type(data.get(key)) is not int or data[key] != value
                   for key, value in expected_counters.items()):
                raise ValueError(f'{name} is not the expected completed-update checkpoint')
        final = command.output / 'final.zip'
        recorded = receipt.get('final_checkpoint', {})
        if (recorded.get('path') != str(final.resolve())
                or recorded.get('sha256') != learning.sha256_file(final)):
            raise ValueError('final checkpoint differs from the native receipt')
    except (OSError, ValueError, KeyError, TypeError, AttributeError, zipfile.BadZipFile) as exc:
        raise RuntimeError(f'native completion receipt/checkpoint invalid for {command.label}: {exc}') from exc
    return {**evidence, 'native_learning_receipt': str(receipt_path),
            'completed_cycles': cycles, 'final_checkpoint_counters': final_counters}


def validate_job_output(command: learning.CommandSpec, round_output: Path) -> None:
    """Archival is restricted to one direct arm's literal train/eval paths."""
    path = command.output
    if (command.kind not in ('train', 'eval') or path.name != command.kind
            or path.parent.parent != round_output
            or command.label != f'{path.parent.name}-{command.kind}'):
        raise ValueError(f'job output is outside its exact round/arm path: {path}')
    for target in (path, path.with_suffix('.log'), path.parent / 'interrupted-attempts'):
        if target.is_symlink() or target.resolve() != target:
            raise ValueError(f'refusing symlink or noncanonical job output path: {target}')
    if path.exists() and not path.is_dir():
        raise ValueError(f'job output is not a directory: {path}')
    log = path.with_suffix('.log')
    if log.exists() and not log.is_file():
        raise ValueError(f'job log is not a regular file: {log}')


def archive_incomplete_job(command: learning.CommandSpec, round_output: Path) -> str | None:
    """Move exact retry targets, without deleting or following nested links."""
    validate_job_output(command, round_output)
    targets = [path for path in (command.output, command.output.with_suffix('.log')) if path.exists()]
    if not targets:
        return None
    archive_root = command.output.parent / 'interrupted-attempts'
    archive_root.mkdir(exist_ok=True)
    archive = Path(tempfile.mkdtemp(prefix=f'{command.kind}-', dir=archive_root))
    learning.write_json(archive / 'archive.json', {
        'label': command.label, 'reason': 'incomplete job retry from the original immutable inputs',
        'original_paths': [str(path) for path in targets], 'created_unix': time.time()})
    for path in targets:
        path.rename(archive / path.name)
    return str(archive)


def execute(plan: dict, deadline: float, max_new_jobs: int | None = None) -> dict:
    if max_new_jobs is not None and (type(max_new_jobs) is not int or max_new_jobs <= 0):
        raise ValueError('max_new_jobs must be a positive integer or None')
    if not math.isfinite(deadline):
        raise ValueError('execution requires a finite deadline')
    validate_inputs(plan['inputs'], read_files=True)
    output = Path(plan['output'])
    if not output.is_absolute() or output.resolve() != output:
        raise ValueError('round output must be an absolute nonsymlink path')
    commands = [learning.CommandSpec(**{**item, 'cwd': Path(item['cwd']), 'output': Path(item['output'])})
                for item in plan['commands']]
    for command in commands:
        validate_job_output(command, output)
    output.mkdir(parents=True, exist_ok=True)
    result_file = output/'round-result.json'
    plan_path = output/'plan.json'
    for path in (result_file, plan_path, output / 'execution_history'):
        if path.is_symlink():
            raise ValueError(f'refusing symlink journal path: {path}')
    if not plan_path.is_file() and any(output.iterdir()):
        raise RuntimeError('cannot resume without the original immutable plan')
    learning.write_bytes_if_new_or_same(plan_path,json.dumps(plan,indent=2,sort_keys=True).encode()+b'\n')
    previous = learning.archive_existing_execution_manifest(result_file)
    previous_result = json.loads(previous.read_text()) if previous else {}
    previous_records = {record['label']: record for record in previous_result.get('results', [])
                        if isinstance(record, dict) and 'label' in record
                        and record.get('status') in ('completed', 'skipped_completed', 'skipped_existing')}
    result = {'status':'running','results':[],'next_inputs':[],'evaluations':{},
              'started_unix':previous_result.get('started_unix', time.time()),
              'max_new_jobs':max_new_jobs, 'new_jobs_run':0,
              'previous_journal': str(previous) if previous else None, 'archived_attempts': []}
    tracking=tracking_environment(Path(plan['root']))
    tracking.__enter__()
    try:
        for command in commands:
            completed = (learning.is_completed_output(command.output) if command.kind == 'train'
                         else (command.output / 'summary.json').is_file())
            if command.kind == 'train' and completed:
                validate_native_train_artifacts(command)
            if command.kind == 'eval' and not plan['evaluate_only']:
                train = next((item for item in commands if item.kind == 'train'
                              and item.output / 'final.zip' == Path(command.metadata['checkpoint'])), None)
                if train is None:
                    raise RuntimeError('evaluation has no matching native training command')
                validate_native_train_artifacts(train)
            if not completed:
                if max_new_jobs is not None and result['new_jobs_run'] >= max_new_jobs:
                    result['status'] = 'paused_for_retrieval'
                    break
                learning.ensure_launch_safety(command)
                learning.assert_deadline_allows(now=time.time(), deadline=deadline,
                                               timeout=command.timeout_seconds, reserve=300)
                archive = archive_incomplete_job(command, output)
                if archive:
                    result['archived_attempts'].append(archive)
                    learning.write_json(result_file, result)
            record = learning.run_command(command,deadline=deadline,reserve=300)
            if record['status'] in ('skipped_completed', 'skipped_existing'):
                prior = previous_records.get(command.label, {})
                if prior.get('kind') == command.kind and prior.get('output') == str(command.output):
                    # Keep the original execution times/log while refreshing
                    # status and evidence from the revalidated artifacts.
                    record = {**prior, **record}
            else:
                result['new_jobs_run'] += 1
            result['results'].append(record)
            if record['status'] not in ('completed','skipped_completed','skipped_existing'):
                raise RuntimeError(f"job did not complete: {command.label}")
            if command.kind == 'train':
                record['native_evidence'] = validate_native_train_artifacts(command)
            if command.kind == 'eval':
                label = command.label.removesuffix('-eval')
                result['evaluations'][label] = evaluation_record(command.output,phase=plan['phase'])
                old = next(r for r in plan['inputs'] if r['label']==label)
                final = Path(command.metadata['checkpoint'])
                data = checkpoint_metadata(final)
                if data['num_timesteps']!=plan['target'] or data['_n_updates']!=plan['target']//ROLLOUT*8:
                    raise ValueError('final steps/update count mismatch')
                result['next_inputs'].append({**old,'checkpoint':str(final),
                    'checkpoint_sha256':learning.sha256_file(final),
                    'steps':data['num_timesteps'],'updates':data['_n_updates']})
            learning.write_json(result_file,result)
        else:
            result['status']='completed'
            validate_inputs(result['next_inputs'],read_files=True)
    except Exception as exc:
        result.update(status='incomplete',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        tracking.__exit__(None,None,None)
        result['ended_unix']=time.time()
        learning.write_json(result_file,result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--inputs',type=Path)
    parser.add_argument('--target',type=int,required=True)
    parser.add_argument('--phase',choices=('0a','0b'),default='0a')
    parser.add_argument('--panel',choices=tuple(PANELS),default='dev')
    parser.add_argument('--episodes',type=int,default=20)
    parser.add_argument('--deadline-unix',type=float)
    parser.add_argument('--max-new-jobs',type=int,
                        help='pause for retrieval after this many newly run commands; verified skips do not count')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--evaluate-only',action='store_true')
    args=parser.parse_args(argv)
    if args.max_new_jobs is not None and args.max_new_jobs <= 0:
        parser.error('--max-new-jobs must be a positive integer')
    root=args.root.resolve()
    rows=json.loads(args.inputs.read_text())['next_inputs'] if args.inputs else initial_records(root)
    plan=build_plan(root,rows,target=args.target,phase=args.phase,panel=args.panel,episodes=args.episodes,
                    evaluate_only=args.evaluate_only)
    if args.execute:
        if args.deadline_unix is None or not math.isfinite(args.deadline_unix):
            parser.error('execution requires finite --deadline-unix')
        result=execute(plan,args.deadline_unix,max_new_jobs=args.max_new_jobs)
        print(json.dumps({'status':result['status'],'output':plan['output'],'target':plan['target']}))
    else:
        print(json.dumps(plan,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
