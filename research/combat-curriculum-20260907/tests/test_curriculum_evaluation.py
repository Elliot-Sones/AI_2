import copy
import importlib
import unittest


def panel(count=20, successes=18, falls=1, metric='bridge_b1-v1', prefix='case', *, check_id='check1', steps=163840, bank=None):
    rows = []
    for mode in ('deterministic', 'sampled'):
        for index in range(count):
            rows.append(dict(scenario_id=f'{prefix}-{index}', mode=mode, learner_slot=index % 2,
                             success=index < successes, fell=index >= count-falls,
                             evidence_valid=True, systemic_movement_lock=False, metric_version=metric,
                             training_qualified=True))
    return dict(episodes=rows, check_id=check_id, steps=steps, checkpoint_sha256='a'*64,
                bank=bank or ('confirmation' if count == 100 else 'development'))


def retention(count=100, *, parent=False, successes=None, falls=0, check_id=None):
    return dict(metric_version='overlap-v1', verified_unchanged_parent=parent,
                modes={mode: dict(successes=count if successes is None else successes, episodes=count, falls=falls)
                       for mode in ('deterministic', 'sampled')},
                provenance=dict(checkpoint_sha256=('a' if parent else 'b')*64,
                                source_sha256='c'*64, metric_version='overlap-v1',
                                measured_at_utc='2026-09-08T12:00:00Z' if parent else '2026-09-08T12:01:00Z',
                                check_id=check_id or ('parent' if parent else 'current'),
                                scenario_bank_id='near-confirmation' if count == 100 else 'near-development'))


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluation = importlib.import_module('curriculum_evaluation') if importlib.util.find_spec('curriculum_evaluation') else None

    def setUp(self):
        self.assertIsNotNone(self.evaluation, 'curriculum_evaluation implementation is missing')

    def test_fast_promotion_requires_two_consecutive_both_mode_balanced_panels(self):
        previous = panel(check_id='check0', steps=0)
        self.assertTrue(self.evaluation.fast_gate([previous, panel()])['passed'])
        self.assertFalse(self.evaluation.fast_gate([panel()])['passed'])
        self.assertFalse(self.evaluation.fast_gate([previous, panel(successes=17)])['passed'])
        self.assertFalse(self.evaluation.fast_gate([previous, panel(falls=2)])['passed'])
        rows = panel()
        rows['episodes'][0]['evidence_valid'] = False
        self.assertFalse(self.evaluation.fast_gate([previous, rows])['passed'])
        rows = panel()
        rows['episodes'][0]['systemic_movement_lock'] = True
        self.assertFalse(self.evaluation.fast_gate([previous, rows])['passed'])

    def test_confirmation_exact_boundaries_and_side_failures(self):
        self.assertTrue(self.evaluation.confirmation_gate(panel(100, 90, 5))['passed'])
        self.assertFalse(self.evaluation.confirmation_gate(panel(100, 89, 5))['passed'])
        self.assertFalse(self.evaluation.confirmation_gate(panel(100, 90, 6))['passed'])
        rows = panel(100, 100, 0)
        for mode in ('deterministic', 'sampled'):
            failures = [r for r in rows['episodes'] if r['mode'] == mode and r['learner_slot'] == 0][:11]
            for row in failures:
                row['success'] = False
        self.assertFalse(self.evaluation.confirmation_gate(rows)['passed'])

    def test_retention_uses_same_fresh_metric_and_five_percentage_points(self):
        baseline = retention(parent=True, successes=98, falls=1)
        baseline['modes']['sampled'].update(successes=97, falls=2)
        current = retention(successes=93, falls=5)
        current['modes']['sampled']['successes'] = 92
        self.assertTrue(self.evaluation.retention_gate(current, baseline, panel_kind='confirmation')['passed'])
        current['modes']['sampled']['successes'] = 91
        self.assertFalse(self.evaluation.retention_gate(current, baseline, panel_kind='confirmation')['passed'])
        current['metric_version'] = 'center-v0'
        with self.assertRaises(ValueError):
            self.evaluation.retention_gate(current, baseline, panel_kind='confirmation')

    def test_reused_or_unattested_fast_checks_do_not_pass(self):
        self.assertFalse(self.evaluation.fast_gate([panel(), panel()])['passed'])
        for key in ('check_id', 'checkpoint_sha256', 'steps'):
            second = panel(check_id='check2', steps=327680)
            second.pop(key)
            with self.subTest(key=key):
                self.assertFalse(self.evaluation.fast_gate([panel(), second])['passed'])
        self.assertFalse(self.evaluation.fast_gate([panel(), panel(check_id='check2', steps=1)])['passed'])

    def test_contact_metric_passes_only_in_homogeneous_panels_and_fast_checks(self):
        metric = 'bridge_b1_contact-v1'
        first = panel(metric=metric, check_id='check0', steps=0)
        second = panel(metric=metric)
        self.assertTrue(self.evaluation.fast_gate([first, second])['passed'])
        self.assertTrue(self.evaluation.confirmation_gate(panel(100, 90, 5, metric=metric))['passed'])
        mixed = panel(100, 90, 5, metric=metric)
        mixed['episodes'][0]['metric_version'] = 'bridge_b1-v1'
        self.assertFalse(self.evaluation.confirmation_gate(mixed)['passed'])
        self.assertFalse(self.evaluation.fast_gate([panel(check_id='check0', steps=0), second])['passed'])
        self.assertFalse(self.evaluation.confirmation_gate(panel(100, 90, 5, metric='unknown'))['passed'])
        self.assertTrue(self.evaluation.progression_status(fast_passed=True, confirmation_passed=True,
                         retention_passed=True, stage='bridge_b1_contact')['mastered'])

    def test_confirmation_rejects_development_or_missing_bank(self):
        for bank in ('development', '', None):
            check = panel(100, 90, 5)
            check['bank'] = bank
            self.assertFalse(self.evaluation.confirmation_gate(check)['passed'])
        self.assertTrue(self.evaluation.confirmation_gate(panel(100, 90, 5, bank='final'))['passed'])
        unqualified = panel(100, 90, 5)
        unqualified['episodes'][0]['training_qualified'] = False
        self.assertFalse(self.evaluation.confirmation_gate(unqualified)['passed'])
        unqualified['episodes'][0].pop('training_qualified')
        self.assertFalse(self.evaluation.confirmation_gate(unqualified)['passed'])

    def test_retention_requires_provenance_and_exact_panel_size(self):
        baseline, current = retention(parent=True), retention()
        for key in baseline['provenance']:
            bad = copy.deepcopy(baseline)
            bad['provenance'].pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.evaluation.retention_gate(current, bad, panel_kind='confirmation')
        for key, value in [('source_sha256', 'd'*64), ('scenario_bank_id', 'other'),
                           ('check_id', 'current'), ('measured_at_utc', '2026-09-08T12:02:00Z'),
                           ('measured_at_utc', '2026-08-08T12:00:00Z')]:
            bad = copy.deepcopy(baseline)
            bad['provenance'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.evaluation.retention_gate(current, bad, panel_kind='confirmation')
        for count in (1, 20):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.evaluation.retention_gate(retention(count), retention(count, parent=True), panel_kind='confirmation')

    def test_fast_retention_alarm_is_distinct_from_promotion(self):
        baseline, current = retention(20, parent=True), retention(20, successes=19, falls=1)
        result = self.evaluation.retention_gate(current, baseline, panel_kind='fast')
        self.assertTrue(result['passed'])
        self.assertFalse(result['promotion_qualified'])
        current['modes']['sampled']['successes'] = 18
        result = self.evaluation.retention_gate(current, baseline, panel_kind='fast')
        self.assertTrue(result['pause_updates'])
        self.assertFalse(result['passed'])
        baseline = retention(20, parent=True, successes=19)
        current = retention(20, successes=18, falls=2)
        self.assertTrue(self.evaluation.retention_gate(current, baseline, panel_kind='fast')['passed'])
        current['modes']['sampled']['successes'] = 17
        current['modes']['sampled']['falls'] = 3
        self.assertFalse(self.evaluation.retention_gate(current, baseline, panel_kind='fast')['passed'])

    def test_failed_gate_dictionaries_cannot_be_truthy_mastery(self):
        with self.assertRaises(ValueError):
            self.evaluation.progression_status(fast_passed={'passed': False}, confirmation_passed={'passed': False}, retention_passed={'passed': False})
        with self.assertRaises(ValueError):
            self.evaluation.progression_status(fast_passed=True, confirmation_passed=True, retention_passed=True, stage='typo')

    def test_seed_banks_disjoint_preserve_final_and_balance_sides_and_modes(self):
        banks = self.evaluation.build_scenario_banks(train_count=8, development_count=20, final_count=100, preserved_holdout_seeds=[1, 2, 3])
        sets = [{r['seed'] for r in banks[split]} for split in ('train', 'development', 'final')]
        self.assertTrue(all(not sets[i] & sets[j] for i in range(3) for j in range(i+1, 3)))
        self.assertEqual(banks['preserved_holdout_seeds'], [1, 2, 3])
        self.assertFalse(set.union(*sets) & {1,2,3})
        for split in ('development', 'final'):
            for mode in ('deterministic', 'sampled'):
                rows = [r for r in banks[split] if r['mode'] == mode]
                self.assertEqual(sum(r['learner_slot'] == 0 for r in rows), sum(r['learner_slot'] == 1 for r in rows))
        with self.assertRaises(ValueError):
            self.evaluation.validate_seed_banks(dict(train=[dict(seed=1)], development=[dict(seed=1)], final=[], preserved_holdout_seeds=[]))

    def test_stopping_plateau_and_progression_never_claim_mastery(self):
        for stop in ('budget_exhausted', 'plateau_review', 'retention_failed', 'paused'):
            result = self.evaluation.progression_status(fast_passed=True, confirmation_passed=True, retention_passed=True, stop_reason=stop)
            self.assertFalse(result['mastered'])
            self.assertEqual(result['status'], stop)
        self.assertEqual(self.evaluation.progression_status(fast_passed=True, confirmation_passed=False, retention_passed=True)['status'], 'confirmation_required')
        self.assertTrue(self.evaluation.progression_status(fast_passed=True, confirmation_passed=True, retention_passed=True)['mastered'])
        self.assertFalse(self.evaluation.progression_status(fast_passed=True, confirmation_passed=True, retention_passed=True, stage='bridge_b2', wider_mastered=False)['mastered'])
        checks = [dict(weaker_successes=12, first_hit_seconds=2., reengagement_seconds=2.) for _ in range(3)]
        self.assertTrue(self.evaluation.plateau_gate(checks)['review_required'])
        checks[-1]['first_hit_seconds'] = 1.
        self.assertFalse(self.evaluation.plateau_gate(checks)['review_required'])

    def test_report_contains_exact_failure_evidence_and_no_timeout_win_credit(self):
        row = dict(scenario_id='failure-1', mode='sampled', outcome='timeout', diagnoses=['started_but_no_hit'],
                   failures=[dict(tick=63, seconds=2.1, distance=4., learner={'x':1.,'y':0.}, target={'x':5.,'y':0.}, raw_action=[0.]*7+[1.,0.,0.], executed_action=[0.]*7+[1.,0.,0.])])
        report = self.evaluation.render_markdown_report(dict(episodes=[row], status='not_mastered'))
        self.assertIn('failure-1', report)
        self.assertIn('63', report)
        self.assertIn('2.100', report)
        self.assertIn('started_but_no_hit', report)
        self.assertFalse(self.evaluation.credited_elimination(dict(outcome='win', learner_damage=0., attribution='opponent_self_destruct')))
        self.assertFalse(self.evaluation.credited_elimination(dict(outcome='timeout', learner_damage=20., attribution='learner_hit_within_90_ticks')))
        for damage in (float('inf'), float('nan'), True, '20'):
            with self.subTest(damage=damage):
                self.assertFalse(self.evaluation.credited_elimination(dict(outcome='win', learner_damage=damage, attribution='learner_hit_within_90_ticks')))


if __name__ == '__main__':
    unittest.main()
