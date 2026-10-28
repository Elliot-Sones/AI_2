
import importlib
import importlib.util
import copy
import json
from pathlib import Path
import tempfile
import time
import types
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]


class CapacityCampaignTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('baselines.native-capacity-session11.scripts.native_capacity_campaign'),
                             'native capacity campaign has not been implemented')
        native_capacity_campaign = importlib.import_module('baselines.native-capacity-session11.scripts.native_capacity_campaign')
        return native_capacity_campaign

    def test_rollout_boundary_targets_are_cumulative(self):
        m = self.module()
        self.assertEqual(m.rounded_target(5_000_000), 5_013_504)
        self.assertEqual(m.rounded_target(10_000_000), 10_027_008)
        self.assertEqual(m.rounded_target(20_000_000), 20_021_248)

    def test_plan_balances_architectures_and_preserves_e6_geometry(self):
        m = self.module()
        inputs = m.initial_records(Path('/gpu'))
        plan = m.build_plan(Path('/gpu'), inputs, target=5_000_000, phase='0a', panel='dev', episodes=20)
        self.assertEqual(len(plan['commands']), 18)
        jobs = [c for c in plan['commands'] if c['kind'] == 'train']
        self.assertEqual(len(jobs), 9)
        self.assertEqual({c['metadata']['expected_starting_steps'] for c in jobs}, {2_097_152})
        self.assertEqual({c['metadata']['expected_ending_steps'] for c in jobs}, {5_013_504})
        self.assertEqual({c['metadata']['rounded_steps'] for c in jobs}, {2_916_352})
        for c in jobs:
            self.assertEqual(c['metadata']['controls'], {'n_envs':64, 'n_steps':512, 'batch_size':8192, 'epochs':8})
            self.assertEqual(c['metadata']['nav_ent_coef'], 0.0)
            self.assertIn('--expected-parameters', c['argv'])
            self.assertEqual(c['argv'][c['argv'].index('--device')+1], 'cuda')

    def test_rejects_unbalanced_or_nonadvancing_inputs_and_unsupported_phase(self):
        m = self.module()
        rows = m.initial_records(Path('/gpu'))
        for bad in (rows[:-1], [dict(r, steps=2_621_440) if i==0 else r for i,r in enumerate(rows)]):
            with self.assertRaises(ValueError):
                m.build_plan(Path('/gpu'), bad, target=5_000_000, phase='0a', panel='dev', episodes=20)
        for phase, target in [('0c',5_000_000),('0a',2_000_000)]:
            with self.assertRaises(ValueError):
                m.build_plan(Path('/gpu'), rows, target=target, phase=phase, panel='dev', episodes=20)

    def test_runner_path_matches_campaign_source_layout(self):
        m = self.module()
        for packaged in (False, True):
            with self.subTest(packaged=packaged), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / 'source'
                source.mkdir()
                runner = source / 'native_learning_runner.py'
                if packaged:
                    (source / 'ai2/training').mkdir(parents=True)
                    (source / 'ai2/paths.py').touch()
                    runner = source / 'ai2/training/native_learning_runner.py'
                runner.write_text('# local runner fixture\n')
                plan = m.build_plan(root, m.initial_records(root), target=5_000_000,
                                    phase='0a', panel='dev', episodes=20)
                train_commands = [item for item in plan['commands'] if item['kind'] == 'train']
                self.assertEqual({item['argv'][1] for item in train_commands}, {str(runner)})

    def test_final_and_development_seed_panels_are_disjoint(self):
        m = self.module()
        ranges = [{start+i for start in m.PANELS[panel].values() for i in range(100)} for panel in ('dev','final')]
        self.assertFalse(ranges[0] & ranges[1])

    def test_tracking_is_scoped_offline_and_describes_actual_run(self):
        m=self.module()
        command=m.learning.CommandSpec('arch-512-seed145-train','train',Path('/gpu'),[],Path('/gpu/out'),1,{})
        original=m.learning.command_env
        with self.assertRaisesRegex(RuntimeError,'deliberate'):
            with m.tracking_environment(Path('/gpu')):
                env=m.learning.command_env(command)
                self.assertEqual(env['WANDB_MODE'],'offline')
                self.assertEqual(env['WANDB_DIR'],'/gpu/wandb')
                self.assertIn('native',env['WANDB_NOTES'])
                self.assertIn('continuation',env['WANDB_NOTES'])
                raise RuntimeError('deliberate')
        self.assertIs(m.learning.command_env,original)

    def test_summary_keeps_seed_scores_and_does_not_declare_mastery_from_reward(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            for phase in ('0a','0b'):
                for mode in ('deterministic','sampled'):
                    (p/f'{phase}-{mode}.json').write_text(json.dumps({'summary':{'episodes':20,'success_rate':0.5,'fall_rate':0.2,'timeout_rate':0.3,'navigation_targets_per_episode':2.0}}))
            got = m.evaluation_record(p)
            self.assertFalse(got['candidate_mastery'])
            self.assertEqual(got['panels']['0a-sampled']['success_rate'],0.5)

    def test_wide_mastery_requires_wide_success_and_near_retention(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            for phase in ('0a','0b'):
                for mode in ('deterministic','sampled'):
                    (p/f'{phase}-{mode}.json').write_text(json.dumps({'summary':{'episodes':100,'success_rate':1.0 if phase=='0a' else 0.0,'fall_rate':0.0}}))
            self.assertFalse(m.evaluation_record(p,phase='0b')['mastery_confirmed'])

    def test_evaluate_only_accepts_equal_steps_and_has_no_train_commands(self):
        m = self.module()
        rows=m.initial_records(Path('/gpu'))
        plan=m.build_plan(Path('/gpu'),rows,target=2097152,phase='0a',panel='final',episodes=100,evaluate_only=True)
        self.assertEqual(len(plan['commands']),9)
        self.assertEqual({c['kind'] for c in plan['commands']},{'eval'})
        self.assertEqual({c['metadata']['checkpoint'] for c in plan['commands']},{r['checkpoint'] for r in rows})

    def test_rejects_file_backed_different_ppo_and_navigation_configs(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            rows=m.initial_records(root)
            for row in rows:
                p=Path(row['checkpoint'])
                p.parent.mkdir(parents=True)
                with zipfile.ZipFile(p,'w') as z:
                    z.writestr('data',json.dumps({'num_timesteps':row['steps'],'_n_updates':row['updates'],
                        'n_envs':64,'n_steps':512,'batch_size':8192,'n_epochs':8,'ent_coef':0.0}))
                row['checkpoint_sha256']=m.learning.sha256_file(p)
                cfg={'policy_kwargs':{'net_arch':row['net_arch']},'frame_stack':4,'opponent_history':{'enabled':False},
                    'ppo_settings':{'learning_rate':.0003,'clip_range':.2,'nav_ent_coef':0,'gamma':.99},
                    'phases':{phase:{'type':'navigation','opponents':{'constant_agent':1.0},'navigation':{
                        'min_dist':minimum,'max_dist':maximum,'fall_penalty':-3.,'time_limit_seconds':20,
                        'multi_target':3,'completion_mode':'sequence','control_mode':'movement',
                        'settle_seconds':.3,'post_goal_seconds':.5}}
                        for phase,minimum,maximum in [('0a',1.5,3.),('0b',2.,6.)]}}
                c=Path(row['config'])
                c.parent.mkdir(parents=True)
                c.write_text(json.dumps(cfg))
            m.validate_inputs(rows,read_files=True)
            path=Path(rows[-1]['config'])
            original=path.read_text()
            for field,value in [('gamma',.5),('min_dist',9.)]:
                cfg=json.loads(original)
                dest=cfg['ppo_settings'] if field=='gamma' else cfg['phases']['0a']['navigation']
                dest[field]=value
                path.write_text(json.dumps(cfg))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    m.validate_inputs(rows,read_files=True)


class CapacityRestartTests(unittest.TestCase):
    """Exercise real planning/validation/journaling; only GPU subprocesses are fake."""

    def setUp(self):
        campaign = importlib.import_module('baselines.native-capacity-session11.scripts.native_capacity_campaign')
        self.m = campaign
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.rows = self.m.initial_records(self.root)
        for row in self.rows:
            self.write_checkpoint(Path(row['checkpoint']), row['steps'])
            row['checkpoint_sha256'] = self.m.learning.sha256_file(Path(row['checkpoint']))
            config = {'policy_kwargs': {'net_arch': row['net_arch']},
                      'ppo_settings': {'learning_rate': .0003, 'clip_range': .2, 'nav_ent_coef': 0},
                      'phases': {phase: {'type': 'navigation', 'opponents': {'constant_agent': 1.0},
                          'navigation': {'min_dist': low, 'max_dist': high, 'fall_penalty': -3.,
                              'time_limit_seconds': 20, 'multi_target': 3, 'completion_mode': 'sequence',
                              'control_mode': 'movement', 'settle_seconds': .3, 'post_goal_seconds': .5}}
                          for phase, low, high in [('0a', 1.5, 3.), ('0b', 2., 6.)]}}
            self.m.learning.write_json(Path(row['config']), config)
        self.plan = self.m.build_plan(self.root, self.rows, target=2097152 + 32768,
                                     phase='0a', panel='dev', episodes=20)
        self.m.learning.write_json(Path(self.plan['output']) / 'plan.json', self.plan)
        self.commands = [self.m.learning.CommandSpec(**{**item, 'cwd': Path(item['cwd']),
                            'output': Path(item['output'])}) for item in self.plan['commands']]
        self.launched = []
        self.fail_at = None
        self.deadline = time.time() + 10000
        self.addCleanup(patch.stopall)
        patch.object(self.m.learning.subprocess, 'Popen', side_effect=self.spawn).start()
        patch.object(self.m.learning.shutil, 'disk_usage',
                     return_value=types.SimpleNamespace(free=100 * 1024**3)).start()

    def write_checkpoint(self, path, steps, updates=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('data', json.dumps({'num_timesteps': steps,
                '_n_updates': steps // 32768 * 8 if updates is None else updates,
                'n_envs': 64, 'n_steps': 512, 'batch_size': 8192, 'n_epochs': 8, 'ent_coef': 0.0}))

    def completed_train(self, command):
        meta, output = command.metadata, command.output
        start, end = meta['expected_starting_steps'], meta['expected_ending_steps']
        for name, steps in [('initial.zip', start), ('latest.zip', end), ('final.zip', end)]:
            self.write_checkpoint(output / name, steps)
        self.m.learning.write_json(output / 'result.json', {
            'status': 'completed', 'error': None, 'stop_reason': None, 'wall_seconds': .1,
            'starting_steps': start, 'ending_steps': end, 'actual_steps': end - start,
            **{key + '_checkpoint': str(output / (key + '.zip')) for key in ('initial', 'latest', 'final')}})
        self.m.learning.write_json(output / 'metadata.json', {
            'seed': meta['seed'], 'checkpoint': meta['checkpoint'], 'config_sha256': meta['config_sha256'],
            'input_checkpoint_sha256': self.m.learning.sha256_file(Path(meta['checkpoint'])),
            'parameter_count': meta['expected_parameter_count'], 'n_envs': 64,
            'ppo_controls': {'net_arch': meta['net_arch'], 'n_steps': 512, 'batch_size': 8192,
                             'n_epochs': 8, 'ent_coef': 0, 'reset_num_timesteps': False}})
        self.m.learning.write_json(output / 'runtime_device.json', {
            'all_policy_parameters_cuda': True, 'policy_device': 'cuda:0'})
        cycles = (end - start) // 32768
        receipt = {'status': 'completed', 'error': None, 'final_completed_update_verified': True,
            'starting_steps': start, 'expected_final_steps': end, 'expected_additional_steps': end - start,
            'transitions_per_cycle': 32768, 'cycles': cycles, 'completed_cycles': cycles,
            'expected_final_updates': end // 32768 * 8,
            'initial_checkpoint_counters': {'num_timesteps': start, '_n_updates': start // 32768 * 8},
            'final_checkpoint_counters': {'num_timesteps': end, '_n_updates': end // 32768 * 8},
            'final_checkpoint': self.m.learning.file_record(output / 'final.zip')}
        self.m.learning.write_json(output / 'native_learning_receipt.json', receipt)
        return receipt

    def completed_eval(self, command):
        self.m.learning.write_json(command.output / 'metadata.json', {
            'training_performed': False, 'actual_policy_device': 'cuda:0',
            'checkpoint': command.metadata['checkpoint'],
            'checkpoint_sha256': self.m.learning.sha256_file(Path(command.metadata['checkpoint'])),
            'config_sha256': command.metadata['config_sha256']})
        for phase in ('0a', '0b'):
            for mode in ('deterministic', 'sampled'):
                self.m.learning.write_json(command.output / f'{phase}-{mode}.json', {'summary': {
                    'episodes': 20, 'success_rate': .5, 'fall_rate': .2, 'nav_falls': 4}})
        self.m.learning.write_json(command.output / 'summary.json', {})

    def spawn(self, argv, **kwargs):
        command = next(command for command in self.commands if command.argv == argv)
        self.launched.append(command.label)
        self.assertFalse(command.output.exists(), 'retry must not reuse a nonempty output directory')
        if len(self.launched) == self.fail_at:
            command.output.mkdir(parents=True)
            (command.output / 'partial.bin').write_bytes(b'preserve partial output')
            kwargs['stdout'].write('preserve interrupted log\n')
            code = 1
        else:
            (self.completed_train if command.kind == 'train' else self.completed_eval)(command)
            code = 0
        return types.SimpleNamespace(wait=lambda timeout: code)

    def test_rejects_incomplete_native_receipt_before_evaluation_or_reuse(self):
        command = self.commands[0]
        receipt = self.completed_train(command)
        receipt.update(status='incomplete', final_completed_update_verified=False)
        self.m.learning.write_json(command.output / 'native_learning_receipt.json', receipt)
        with self.assertRaisesRegex(RuntimeError, 'native.*(receipt|completion)'):
            self.m.execute(self.plan, self.deadline)
        self.assertEqual(self.launched, [])
        self.assertTrue((command.output / 'final.zip').is_file())

    def test_rejects_wrong_completed_cycle_count(self):
        command = self.commands[0]
        receipt = self.completed_train(command)
        receipt['completed_cycles'] = 0
        self.m.learning.write_json(command.output / 'native_learning_receipt.json', receipt)
        with self.assertRaisesRegex(RuntimeError, 'native.*(cycle|receipt|completion)'):
            self.m.execute(self.plan, self.deadline)
        self.assertEqual(self.launched, [])

    def test_missing_native_receipt_cannot_reuse_a_completed_sb3_result(self):
        command = self.commands[0]
        self.completed_train(command)
        receipt = command.output / 'native_learning_receipt.json'
        receipt.rename(receipt.with_suffix('.saved'))
        with self.assertRaisesRegex(RuntimeError, 'native.*receipt'):
            self.m.execute(self.plan, self.deadline)
        self.assertEqual(self.launched, [])

    def test_native_receipt_hash_is_bound_to_the_actual_final_checkpoint(self):
        command = self.commands[0]
        receipt = self.completed_train(command)
        receipt['final_checkpoint']['sha256'] = '0' * 64
        self.m.learning.write_json(command.output / 'native_learning_receipt.json', receipt)
        with self.assertRaisesRegex(RuntimeError, 'native.*checkpoint'):
            self.m.execute(self.plan, self.deadline)
        self.assertEqual(self.launched, [])

    def test_rejects_preupdate_checkpoint_even_with_a_passing_receipt(self):
        command = self.commands[0]
        self.completed_train(command)
        self.write_checkpoint(command.output / 'final.zip', self.plan['target'], updates=512)
        with self.assertRaisesRegex(RuntimeError, 'native.*(checkpoint|receipt|completion)'):
            self.m.execute(self.plan, self.deadline)
        self.assertEqual(self.launched, [])

    def test_resume_preserves_history_and_archives_only_incomplete_job_and_log(self):
        self.fail_at = 2  # One verified train finishes; its evaluation is interrupted.
        with self.assertRaisesRegex(RuntimeError, 'job did not complete'):
            self.m.execute(self.plan, self.deadline)
        result_path = Path(self.plan['output']) / 'round-result.json'
        previous = result_path.read_bytes()
        complete_bytes = (self.commands[0].output / 'final.zip').read_bytes()
        self.fail_at = None
        result = self.m.execute(self.plan, self.deadline)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(len(result['next_inputs']), 9)
        self.assertEqual(self.launched.count(self.commands[0].label), 1)
        self.assertEqual((self.commands[0].output / 'final.zip').read_bytes(), complete_bytes)
        history = list(Path(self.plan['output']).glob('execution_history/*.json'))
        self.assertIn(previous, [path.read_bytes() for path in history])
        archives = list(self.commands[1].output.parent.glob('interrupted-attempts/*'))
        self.assertEqual(len(archives), 1)
        self.assertEqual((archives[0] / 'eval' / 'partial.bin').read_bytes(), b'preserve partial output')
        self.assertEqual((archives[0] / 'eval.log').read_text(), 'preserve interrupted log\n')
        self.assertFalse((self.commands[0].output / 'partial.bin').exists())

    def test_plan_mismatch_leaves_existing_journal_and_outputs_untouched(self):
        self.fail_at = 1
        with self.assertRaises(RuntimeError):
            self.m.execute(self.plan, self.deadline)
        result_path = Path(self.plan['output']) / 'round-result.json'
        before = result_path.read_bytes()
        changed = copy.deepcopy(self.plan)
        changed['commands'][0]['timeout_seconds'] += 1
        with self.assertRaisesRegex(RuntimeError, 'differs|immutable'):
            self.m.execute(changed, self.deadline)
        self.assertEqual(result_path.read_bytes(), before)
        self.assertTrue((self.commands[0].output / 'partial.bin').is_file())
        self.assertFalse((Path(self.plan['output']) / 'execution_history').exists())

    def test_expired_deadline_does_not_move_partial_artifacts(self):
        self.fail_at = 1
        with self.assertRaises(RuntimeError):
            self.m.execute(self.plan, self.deadline)
        with self.assertRaisesRegex(RuntimeError, 'deadline'):
            self.m.execute(self.plan, time.time() - 1)
        self.assertTrue((self.commands[0].output / 'partial.bin').is_file())
        self.assertEqual(self.commands[0].output.with_suffix('.log').read_text(), 'preserve interrupted log\n')
        self.assertFalse((self.commands[0].output.parent / 'interrupted-attempts').exists())

    def test_repeated_interrupted_train_retries_keep_distinct_archives(self):
        for failure in (1, 2):
            self.fail_at = failure
            with self.assertRaisesRegex(RuntimeError, 'job did not complete'):
                self.m.execute(self.plan, self.deadline)
        self.fail_at = None
        self.assertEqual(self.m.execute(self.plan, self.deadline)['status'], 'completed')
        archives = list(self.commands[0].output.parent.glob('interrupted-attempts/*'))
        self.assertEqual(len(archives), 2)
        for archive in archives:
            self.assertEqual((archive / 'train' / 'partial.bin').read_bytes(), b'preserve partial output')
            self.assertEqual((archive / 'train.log').read_text(), 'preserve interrupted log\n')

    def test_stop_request_does_not_archive_an_interrupted_job(self):
        self.fail_at = 1
        with self.assertRaises(RuntimeError):
            self.m.execute(self.plan, self.deadline)
        (self.root / 'artifacts' / 'stop.request').write_text('stop')
        with self.assertRaisesRegex(RuntimeError, 'stop.request'):
            self.m.execute(self.plan, self.deadline)
        self.assertTrue((self.commands[0].output / 'partial.bin').is_file())
        self.assertFalse((self.commands[0].output.parent / 'interrupted-attempts').exists())

    def test_output_symlink_is_rejected_without_touching_its_target(self):
        command = self.commands[0]
        external = self.root / 'external'
        external.mkdir()
        (external / 'keep').write_text('unchanged')
        command.output.parent.mkdir(parents=True)
        command.output.symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.m.execute(self.plan, self.deadline)
        self.assertEqual((external / 'keep').read_text(), 'unchanged')
        self.assertTrue(command.output.is_symlink())
        self.assertEqual(self.launched, [])

    def test_completed_round_can_be_revalidated_without_launching_again(self):
        self.m.execute(self.plan, self.deadline)
        original_launches = list(self.launched)
        result = self.m.execute(self.plan, self.deadline)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(self.launched, original_launches)
        self.assertTrue(all(record['status'].startswith('skipped_') for record in result['results']))

    def test_max_new_jobs_rejects_nonpositive_or_noninteger_values_before_execution(self):
        for value in (True, False, 0, -1, 1.5, '1'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'positive integer'):
                self.m.execute(self.plan, self.deadline, max_new_jobs=value)
        self.assertEqual(self.launched, [])
        self.assertFalse((Path(self.plan['output']) / 'round-result.json').exists())

    def test_max_new_jobs_pauses_after_one_new_command_before_next_output(self):
        plan_path = Path(self.plan['output']) / 'plan.json'
        immutable_plan = plan_path.read_bytes()
        result = self.m.execute(self.plan, self.deadline, max_new_jobs=1)
        self.assertEqual(result['status'], 'paused_for_retrieval')
        self.assertEqual(result['new_jobs_run'], 1)
        self.assertEqual(self.launched, [self.commands[0].label])
        self.assertEqual(len(result['results']), 1)
        self.assertFalse(self.commands[1].output.exists())
        self.assertEqual(plan_path.read_bytes(), immutable_plan)
        saved = json.loads((Path(self.plan['output']) / 'round-result.json').read_text())
        self.assertEqual(saved['status'], 'paused_for_retrieval')

    def test_verified_skipped_commands_do_not_use_the_new_job_budget(self):
        self.completed_train(self.commands[0])
        self.completed_eval(self.commands[1])
        result = self.m.execute(self.plan, self.deadline, max_new_jobs=1)
        self.assertEqual(result['status'], 'paused_for_retrieval')
        self.assertEqual(self.launched, [self.commands[2].label])
        self.assertEqual([record['status'] for record in result['results']],
                         ['skipped_completed', 'skipped_existing', 'completed'])
        self.assertFalse(self.commands[3].output.exists())

    def test_limit_reports_completed_when_the_last_new_job_finishes_the_round(self):
        for command in self.commands[:-1]:
            (self.completed_train if command.kind == 'train' else self.completed_eval)(command)
        result = self.m.execute(self.plan, self.deadline, max_new_jobs=1)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(self.launched, [self.commands[-1].label])
        self.assertEqual(len(result['next_inputs']), 9)
        revalidated = self.m.execute(self.plan, self.deadline, max_new_jobs=1)
        self.assertEqual(revalidated['status'], 'completed')
        self.assertEqual(self.launched, [self.commands[-1].label])

    def test_paused_resume_revalidates_and_preserves_prior_job_timing_and_provenance(self):
        first = self.m.execute(self.plan, self.deadline, max_new_jobs=1)
        prior_record = first['results'][0]
        self.assertIn('started_unix', prior_record)
        self.assertIn('ended_unix', prior_record)
        result_path = Path(self.plan['output']) / 'round-result.json'
        previous = result_path.read_bytes()
        resumed = self.m.execute(self.plan, self.deadline, max_new_jobs=1)
        self.assertEqual(resumed['status'], 'paused_for_retrieval')
        self.assertEqual(self.launched, [self.commands[0].label, self.commands[1].label])
        reused = resumed['results'][0]
        for key in ('started_unix', 'ended_unix', 'returncode', 'log', 'native_evidence'):
            self.assertEqual(reused[key], prior_record[key])
        self.assertEqual(resumed['started_unix'], first['started_unix'])
        self.assertEqual(reused['status'], 'skipped_completed')
        self.assertEqual(Path(resumed['previous_journal']).read_bytes(), previous)
        receipt_path = self.commands[0].output / 'native_learning_receipt.json'
        receipt = json.loads(receipt_path.read_text())
        receipt['completed_cycles'] = 0
        self.m.learning.write_json(receipt_path, receipt)
        with self.assertRaisesRegex(RuntimeError, 'native.*receipt'):
            self.m.execute(self.plan, self.deadline, max_new_jobs=1)
        self.assertEqual(self.launched, [self.commands[0].label, self.commands[1].label])

    def test_cli_passes_max_new_jobs_to_execution(self):
        argv = ['--root', str(self.root), '--target', str(self.plan['target']), '--execute',
                '--deadline-unix', str(self.deadline), '--max-new-jobs', '1']
        with patch.object(self.m, 'initial_records', return_value=self.rows):
            self.assertEqual(self.m.main(argv), 0)
        self.assertEqual(self.launched, [self.commands[0].label])
        result = json.loads((Path(self.plan['output']) / 'round-result.json').read_text())
        self.assertEqual(result['status'], 'paused_for_retrieval')

    def test_resume_accepts_a_seeded_paused_journal_and_keeps_manual_job_times(self):
        command = self.commands[0]
        manual = self.m.learning.run_command(command, deadline=self.deadline, reserve=300)
        manual['native_evidence'] = self.m.validate_native_train_artifacts(command)
        journal = {'status': 'paused_for_retrieval', 'results': [manual],
                   'next_inputs': [], 'evaluations': {}, 'started_unix': manual['started_unix']}
        result_path = Path(self.plan['output']) / 'round-result.json'
        self.m.learning.write_json(result_path, journal)
        original = result_path.read_bytes()
        result = self.m.execute(self.plan, self.deadline, max_new_jobs=1)
        self.assertEqual(result['status'], 'paused_for_retrieval')
        self.assertEqual(result['new_jobs_run'], 1)
        self.assertEqual(self.launched, [command.label, self.commands[1].label])
        for key in ('started_unix', 'ended_unix', 'returncode', 'log', 'native_evidence'):
            self.assertEqual(result['results'][0][key], manual[key])
        self.assertEqual(Path(result['previous_journal']).read_bytes(), original)
        self.assertEqual(len(result['next_inputs']), 1)

    def test_cli_rejects_nonpositive_max_new_jobs_even_for_dry_run(self):
        for value in ('0', '-1', 'true', '1.5'):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                self.m.main(['--root', str(self.root), '--target', str(self.plan['target']),
                             '--max-new-jobs', value])
        self.assertEqual(self.launched, [])

    def test_missing_original_plan_does_not_adopt_existing_round_artifacts(self):
        self.completed_train(self.commands[0])
        plan_path = Path(self.plan['output']) / 'plan.json'
        plan_path.rename(plan_path.with_suffix('.saved'))
        with self.assertRaisesRegex(RuntimeError, 'original immutable plan'):
            self.m.execute(self.plan, self.deadline)
        self.assertEqual(self.launched, [])
        self.assertFalse(plan_path.exists())
        self.assertTrue((self.commands[0].output / 'final.zip').is_file())


if __name__ == '__main__':
    unittest.main()
