import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class ContactEvaluationTests(unittest.TestCase):
    def subject(self):
        self.assertIsNotNone(importlib.util.find_spec('evaluate_native_contact_training'),
                             'original-game native-policy evaluation wrapper is missing')
        import evaluate_native_contact_training
        return evaluate_native_contact_training

    def test_candidate_must_be_preserved_and_match_immutable_inventory(self):
        m = self.subject()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'final.zip'
            checkpoint.write_bytes(b'test checkpoint')
            m.atomic_json(root / 'result.json', {'status': 'trained_not_evaluated',
                'checkpoint_sha256': m.sha256(checkpoint)})
            m.write_inventory(root, block_id='block-a')
            ack = root / 'preservation_ack.json'
            ack.write_text(json.dumps({'verified': True, 'block_id': 'block-a',
                                       'inventory_sha256': m.sha256(root / 'inventory.json')}))
            m.require_preserved_checkpoint(checkpoint, ack)
            checkpoint.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                m.require_preserved_checkpoint(checkpoint, ack)

    def test_result_status_cannot_be_changed_after_preservation(self):
        m = self.subject()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'final.zip'
            checkpoint.write_bytes(b'test checkpoint')
            result = {'status': 'not_trained', 'checkpoint_sha256': m.sha256(checkpoint)}
            m.atomic_json(root / 'result.json', result)
            m.write_inventory(root, block_id='block-a')
            ack = root / 'preservation_ack.json'
            ack.write_text(json.dumps({'verified': True, 'block_id': 'block-a',
                'inventory_sha256': m.sha256(root / 'inventory.json')}))
            (root / 'result.json').write_text(json.dumps({**result, 'status': 'trained_not_evaluated'}))
            with self.assertRaisesRegex(ValueError, 'result.*preserved inventory'):
                m.require_preserved_checkpoint(checkpoint, ack)

    def test_result_requires_one_listed_regular_nonsymlink_file(self):
        m = self.subject()
        for defect in ('unlisted', 'duplicate', 'symlink', 'directory'):
            with self.subTest(defect=defect), TemporaryDirectory() as directory:
                root = Path(directory)
                checkpoint = root / 'final.zip'
                checkpoint.write_bytes(b'test checkpoint')
                result = root / 'result.json'
                m.atomic_json(result, {'status': 'trained_not_evaluated', 'checkpoint_sha256': m.sha256(checkpoint)})
                inventory = m.write_inventory(root, block_id='block-a')
                if defect == 'unlisted':
                    inventory['files'] = [row for row in inventory['files'] if row['path'] != 'result.json']
                elif defect == 'duplicate':
                    inventory['files'].append(next(row for row in inventory['files'] if row['path'] == 'result.json'))
                else:
                    target = root / 'actual-result.json'
                    result.rename(target)
                    if defect == 'symlink':
                        result.symlink_to(target)
                    else:
                        result.mkdir()
                (root / 'inventory.json').write_text(json.dumps(inventory))
                ack = root / 'preservation_ack.json'
                ack.write_text(json.dumps({'verified': True, 'block_id': 'block-a',
                    'inventory_sha256': m.sha256(root / 'inventory.json')}))
                with self.assertRaisesRegex(ValueError, 'result.*preserved inventory'):
                    m.require_preserved_checkpoint(checkpoint, ack)

    def test_commands_use_original_evaluator_with_cuda_contract_and_same_panels(self):
        m = self.subject()
        args = SimpleNamespace(reference_root=Path('/reference'), frozen_root=Path('/oracle'),
            output=Path('/output'), navigation_parent=Path('/parent.zip'), checkpoint=Path('/trained.zip'),
            episodes=20, deadline_unix=10000)
        jobs = m.evaluation_jobs(args)
        self.assertEqual([job['name'] for job in jobs], ['near-parent', 'contact-current', 'near-current'])
        self.assertTrue(jobs[0]['parent'])
        self.assertFalse(jobs[1]['parent'])
        commands = [m.evaluation_command(args, job) for job in jobs]
        self.assertTrue(all('/reference/evaluate_combat_curriculum.py' in command for command in commands))
        self.assertTrue(all(command[command.index('--episodes') + 1] == '20' for command in commands))
        self.assertIn('--parent', commands[0])
        self.assertNotIn('--parent', commands[1])
        self.assertTrue(all('--historical' not in command and '--confirmation' not in command for command in commands))

    def test_gate_never_treats_one_development_panel_as_mastery(self):
        m = self.subject()
        # A full gate delegates to the unchanged reference; this wrapper promises no promotion.
        self.assertFalse(m.RESULT_CONTRACT['mastery_established'])
        self.assertEqual(m.RESULT_CONTRACT['status'], 'evaluated_not_promoted')
        self.assertEqual(m.RESULT_CONTRACT['required_development_checks'], 2)
        self.assertEqual(m.RESULT_CONTRACT['required_confirmation_episodes_per_mode'], 100)

    def test_failed_partial_panel_keeps_completed_panel_logs_and_hash_inventory(self):
        m = self.subject()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / 'parent.zip'
            parent.write_bytes(b'fixture')
            proof = root / 'qualification.json'
            proof.write_text('{}')
            args = SimpleNamespace(output=root / 'evaluation', episodes=20, deadline_unix=10**12,
                navigation_parent=parent, checkpoint=root / 'candidate.zip', verified_ack=root / 'ack.json',
                qualification=proof, reference_root=root, frozen_root=root)
            jobs = m.evaluation_jobs(args)
            calls = []
            def run(command, **kwargs):
                calls.append(command)
                destination = Path(command[command.index('--output') + 1])
                destination.mkdir(parents=True)
                kwargs['stdout'].write('fixture subprocess log\n')
                if len(calls) == 1:
                    m.atomic_json(destination / 'summary.json', {'status': 'fixture complete panel'})
                    m.write_inventory(destination, block_id=destination.name)
                    return SimpleNamespace(returncode=0)
                (destination / 'trace.jsonl.gz.tmp').write_bytes(b'incomplete trace bytes')
                return SimpleNamespace(returncode=1)
            with patch.object(m, 'require_qualification', return_value={}), \
                 patch.object(m, 'require_preserved_checkpoint', return_value={}), \
                 patch.object(m, 'require_cuda'), patch.object(m, 'PARENT_BYTES', parent.stat().st_size), \
                 patch.object(m, 'PARENT_SHA256', m.sha256(parent)), patch.object(m.subprocess, 'run', side_effect=run):
                with self.assertRaisesRegex(RuntimeError, 'contact-current'):
                    m.evaluate(args)
            self.assertTrue((args.output / 'inventory.json').is_file(), 'failed evaluation lacks retrievable inventory')
            inventory = json.loads((args.output / 'inventory.json').read_text())
            paths = {row['path'] for row in inventory['files']}
            actual = {p.relative_to(args.output).as_posix() for p in args.output.rglob('*') if p.is_file()}
            self.assertEqual(paths | {'inventory.json'}, actual)
            self.assertIn(m.job_output(args, jobs[0]).relative_to(args.output).as_posix() + '/summary.json', paths)
            self.assertIn('contact-current.log', paths)
            self.assertIn('failure.json', paths)
            failure = json.loads((args.output / 'failure.json').read_text())
            self.assertEqual(failure['completed_panels'], ['near-parent'])
            for item in inventory['files']:
                self.assertEqual(item['sha256'], m.sha256(args.output / item['path']))
                self.assertEqual(item['bytes'], (args.output / item['path']).stat().st_size)
                self.assertNotIn('.tmp', item['path'])
            original_hash = m.sha256(args.output / 'inventory.json')
            with self.assertRaises(FileExistsError):
                m.evaluate(args)
            self.assertEqual(original_hash, m.sha256(args.output / 'inventory.json'))


if __name__ == '__main__':
    unittest.main()
