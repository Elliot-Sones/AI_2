"""Pure scenario construction, qualification gates, and readable evidence reports."""
import json
import math
from datetime import datetime

MODES = ('deterministic', 'sampled')
SUPPORTED_STAGE_METRICS = ('bridge_b1-v1', 'bridge_b1_contact-v1')


def _nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def _sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_seed_banks(banks):
    reserved = set(banks.get('preserved_holdout_seeds', []))
    splits = {split: {row['seed'] for row in banks[split]} for split in ('train', 'development', 'final')}
    for name, seeds in splits.items():
        if any(type(seed) is not int or seed < 0 for seed in seeds) or seeds & reserved:
            raise ValueError(f'{name} seed overlaps a preserved holdout or is invalid')
        for other, other_seeds in splits.items():
            if name != other and seeds & other_seeds:
                raise ValueError(f'Seed collision between {name} and {other}')
    return banks


def build_scenario_banks(*, train_count=64, development_count=20, final_count=100, preserved_holdout_seeds=(), base_seed=410000):
    """Counts are episodes per mode; paired modes reuse geometry, never cross splits."""
    counts = dict(train=train_count, development=development_count, final=final_count)
    if any(type(n) is not int or n < 2 or n % 2 for n in counts.values()):
        raise ValueError('Scenario counts must be positive even counts')
    if type(base_seed) is not int or base_seed < 0:
        raise ValueError('Invalid base seed')
    reserved = list(preserved_holdout_seeds)
    if any(type(seed) is not int or seed < 0 for seed in reserved) or len(reserved) != len(set(reserved)):
        raise ValueError('Invalid preserved holdout seeds')
    banks = dict(schema_version='b1-scenarios-v1', preserved_holdout_seeds=reserved,
                 development_note='Repeated development use is model selection; final cases remain held out.')
    seed = base_seed
    for split, count in counts.items():
        rows = []
        for index in range(count):
            while seed in reserved:
                seed += 1
            for mode in MODES:
                rows.append(dict(scenario_id=f'b1-{split}-{index:04d}', seed=seed, learner_slot=index % 2,
                                 mode=mode, stage='bridge_b1', split=split))
            seed += 1
        banks[split] = rows
    return validate_seed_banks(banks)


def _panel_gate(episodes, *, expected, successes, max_falls, side_minimum=0):
    reasons, metrics = [], {}
    if not isinstance(episodes, list):
        return dict(passed=False, reasons=['Panel must be an episode list'], modes={})
    try:
        json.dumps(episodes, allow_nan=False)
    except (ValueError, TypeError):
        return dict(passed=False, reasons=['Nonfinite or non-JSON evidence'], modes={})
    for row in episodes:
        if not isinstance(row, dict) or row.get('mode') not in MODES:
            reasons.append('Malformed episode or unknown action mode')
            continue
        if row.get('evidence_valid') is not True or row.get('systemic_movement_lock') is not False:
            reasons.append('Invalid evidence or systemic movement lock')
        if row.get('training_qualified') is not True:
            reasons.append('Stage geometry/source has not been qualified for training')
        if type(row.get('success')) is not bool or type(row.get('fell')) is not bool or row.get('learner_slot') not in (0, 1):
            reasons.append('Missing outcome/slot evidence')
        if row.get('success') and row.get('fell'):
            reasons.append('Success cannot coincide with own stock loss')
        if row.get('metric_version') not in SUPPORTED_STAGE_METRICS:
            reasons.append('Unexpected or missing stage metric version')
    panel_metrics = {row.get('metric_version') for row in episodes if isinstance(row, dict) and isinstance(row.get('metric_version'), str)}
    if len(panel_metrics) != 1:
        reasons.append('A panel must contain exactly one supported stage metric')
    for mode in MODES:
        rows = [r for r in episodes if isinstance(r, dict) and r.get('mode') == mode]
        ids = [r.get('scenario_id') for r in rows]
        if len(rows) != expected or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            reasons.append(f'{mode}: expected {expected} distinct scenarios')
        wins = sum(r.get('success') is True for r in rows)
        falls = sum(r.get('fell') is True for r in rows)
        sides = {str(slot): dict(episodes=sum(r.get('learner_slot') == slot for r in rows),
                                successes=sum(r.get('learner_slot') == slot and r.get('success') is True for r in rows)) for slot in (0, 1)}
        metrics[mode] = dict(episodes=len(rows), successes=wins, falls=falls, slots=sides)
        if wins < successes or falls > max_falls:
            reasons.append(f'{mode}: needs >= {successes}/{expected} success and <= {max_falls} falls')
        if any(side['episodes'] != expected//2 or side['successes'] < side_minimum for side in sides.values()):
            reasons.append(f'{mode}: unbalanced slots or slot success below {side_minimum}')
    return dict(passed=not reasons, reasons=list(dict.fromkeys(reasons)), modes=metrics,
                metric_version=next(iter(panel_metrics)) if len(panel_metrics) == 1 else None)


def _check_panel(check, *, confirmation=False):
    """Check identity is independent of reusable development scenario identities."""
    if not isinstance(check, dict):
        return dict(passed=False, reasons=['Panel requires check metadata'], modes={})
    report = _panel_gate(check.get('episodes'), expected=100 if confirmation else 20,
                         successes=90 if confirmation else 18, max_falls=5 if confirmation else 1,
                         side_minimum=40 if confirmation else 0)
    if (not _nonempty(check.get('check_id')) or not _sha256(check.get('checkpoint_sha256'))
            or type(check.get('steps')) is not int or check['steps'] < 0):
        report['reasons'].append('Missing valid check identity, checkpoint SHA256 or steps')
    allowed_banks = ('confirmation', 'final') if confirmation else ('development',)
    if check.get('bank') not in allowed_banks:
        report['reasons'].append('Unexpected scenario bank for this gate')
    if isinstance(check.get('episodes'), list) and any(
            isinstance(row, dict) and 'bank' in row and row['bank'] != check.get('bank')
            for row in check['episodes']):
        report['reasons'].append('Episode and panel scenario banks disagree')
    report['passed'] = not report['reasons']
    return report


def fast_gate(checks):
    """Only the last two consecutive checks can support a promotion candidate."""
    if len(checks) < 2:
        return dict(passed=False, status='needs_two_consecutive_checks', reasons=['Two fast checks required'], checks=[])
    reports = [_check_panel(check) for check in checks[-2:]]
    if all(report['passed'] for report in reports):
        previous, current = checks[-2:]
        if current['check_id'] == previous['check_id'] or current['steps'] <= previous['steps']:
            reports[-1]['passed'] = False
            reports[-1]['reasons'].append('Fast checks must be distinct and ordered by increasing steps')
        if reports[0]['metric_version'] != reports[1]['metric_version']:
            reports[-1]['passed'] = False
            reports[-1]['reasons'].append('Consecutive fast checks must use the same stage metric')
    passed = all(report['passed'] for report in reports)
    return dict(passed=passed, status='promotion_candidate' if passed else 'not_qualified', checks=reports,
                reasons=[reason for report in reports for reason in report['reasons']])


def confirmation_gate(check):
    report = _check_panel(check, confirmation=True)
    report['status'] = 'stage_confirmed' if report['passed'] else 'not_qualified'
    return report


def _rates(record, *, expected):
    if not isinstance(record, dict) or any(type(record.get(key)) is not int for key in ('episodes', 'successes', 'falls')):
        raise ValueError('Retention requires integer measured counts')
    total, success, falls = (record[k] for k in ('episodes', 'successes', 'falls'))
    if total != expected or not 0 <= success <= total or not 0 <= falls <= total or success+falls > total:
        raise ValueError('Invalid retention counts')
    return success/total, falls/total


def _retention_provenance(record):
    proof = record.get('provenance')
    if not isinstance(proof, dict) or any(not _nonempty(proof.get(key)) for key in
            ('metric_version', 'measured_at_utc', 'check_id', 'scenario_bank_id')):
        raise ValueError('Retention requires measured baseline/current provenance')
    if any(not _sha256(proof.get(key)) for key in ('checkpoint_sha256', 'source_sha256')):
        raise ValueError('Retention provenance requires checkpoint and source SHA256')
    if proof['metric_version'] != record.get('metric_version'):
        raise ValueError('Retention provenance and reported metric disagree')
    try:
        if not proof['measured_at_utc'].endswith('Z'):
            raise ValueError
        measured = datetime.fromisoformat(proof['measured_at_utc'].replace('Z', '+00:00'))
    except ValueError:
        raise ValueError('Retention measurement time must be an ISO UTC timestamp') from None
    return proof, measured


def retention_gate(current, baseline, *, panel_kind, max_baseline_age_seconds=86400):
    """Fast panels trigger alarms only; promotion retention requires fresh 100s.

    Summaries use the runner's ``modes`` map and a small explicit ``provenance``
    record. Local preservation/source verification remains the caller's duty.
    """
    if panel_kind not in ('fast', 'confirmation'):
        raise ValueError('Choose fast or confirmation retention explicitly')
    if (type(max_baseline_age_seconds) not in (int, float) or not math.isfinite(max_baseline_age_seconds)
            or max_baseline_age_seconds <= 0):
        raise ValueError('Baseline freshness limit must be finite and positive')
    if not baseline.get('metric_version') or current.get('metric_version') != baseline['metric_version']:
        raise ValueError('Retention baseline and current metric must match')
    if baseline.get('verified_unchanged_parent') is not True:
        raise ValueError('Retention requires a freshly verified unchanged-parent baseline')
    old_proof, old_time = _retention_provenance(baseline)
    now_proof, now_time = _retention_provenance(current)
    if (any(old_proof[key] != now_proof[key] for key in ('source_sha256', 'metric_version', 'scenario_bank_id'))
            or old_proof['check_id'] == now_proof['check_id'] or old_time > now_time
            or (now_time-old_time).total_seconds() > max_baseline_age_seconds):
        raise ValueError('Retention requires matched evidence and a preceding distinct parent check')
    expected = 20 if panel_kind == 'fast' else 100
    modes, reasons = {}, []
    for mode in MODES:
        if mode not in current.get('modes', {}) or mode not in baseline.get('modes', {}):
            raise ValueError('Retention requires both action modes')
        now, falls = _rates(current['modes'][mode], expected=expected)
        old, _ = _rates(baseline['modes'][mode], expected=expected)
        drop = (old-now)*100
        passed = (drop < 10-1e-9 and falls <= .10) if panel_kind == 'fast' else (drop <= 5+1e-9 and falls <= .05)
        modes[mode] = dict(passed=passed, baseline_success_rate=old, success_rate=now, fall_rate=falls, drop_pp=drop)
        if not passed:
            reasons.append(f'{mode}: {panel_kind} retention threshold failed')
    passed = not reasons
    status = ('fast_alarm_clear' if passed else 'confirmation_required') if panel_kind == 'fast' else ('retained' if passed else 'retention_failed')
    return dict(passed=passed, status=status, modes=modes, reasons=reasons,
                panel_kind=panel_kind, pause_updates=not passed,
                promotion_qualified=panel_kind == 'confirmation' and passed,
                metric_version=current['metric_version'])


def progression_status(*, fast_passed=False, confirmation_passed=False, retention_passed=False, stop_reason=None, stage='bridge_b1', wider_mastered=False):
    if stage not in ('bridge_b1', 'bridge_b1_contact', 'bridge_b2'):
        raise ValueError('Unsupported progression stage')
    if any(type(value) is not bool for value in (fast_passed, confirmation_passed, retention_passed, wider_mastered)):
        raise ValueError('Progression requires explicit boolean gate results, not gate records')
    if stop_reason:
        return dict(status=stop_reason, mastered=False, stage=stage)
    if not retention_passed:
        status = 'retention_failed'
    elif stage == 'bridge_b2' and not wider_mastered:
        status = 'wider_navigation_required'
    elif not fast_passed:
        status = 'learning'
    elif not confirmation_passed:
        status = 'confirmation_required'
    else:
        status = 'mastered'
    return dict(status=status, mastered=status == 'mastered', stage=stage)


def plateau_gate(checks):
    if len(checks) < 3:
        return dict(review_required=False, status='insufficient_checks', mastered=False)
    recent = checks[-3:]
    keys = ('weaker_successes', 'first_hit_seconds', 'reengagement_seconds')
    if any(any(not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key]) for key in keys) for row in recent):
        return dict(review_required=False, status='insufficient_timing_evidence', mastered=False)
    first = recent[0]
    improved_success = max(row['weaker_successes'] for row in recent[1:]) >= first['weaker_successes']+2
    improved_time = any(row[key] < first[key] for row in recent[1:] for key in keys[1:])
    review = not improved_success and not improved_time
    return dict(review_required=review, status='plateau_review' if review else 'improving', mastered=False,
                interpretation='Coarse diagnostic flag, not evidence of convergence')


def credited_elimination(result):
    damage = result.get('learner_damage')
    return bool(result.get('outcome') == 'win' and type(damage) in (float, int) and math.isfinite(damage) and damage > 0
                and result.get('attribution') == 'learner_hit_within_90_ticks')


def render_markdown_report(report):
    """Keep the detailed evidence next to the observed failure, including exact inputs."""
    lines = ['# Combat curriculum evidence', '', f"Status: {report.get('status', 'not_evaluated')}", '',
             'Diagnosis labels describe observed behavior; they do not establish a causal mechanism.', '']
    for episode in report.get('episodes', []):
        lines.extend([f"## {episode.get('scenario_id', 'unknown')} ({episode.get('mode', 'unknown')})", '',
                      f"Outcome: {episode.get('outcome')}; diagnoses: {', '.join(episode.get('diagnoses', [])) or 'none'}.", ''])
        if 'attack' in episode:
            attack = episode['attack']
            lines.extend([f"Attack requested on {attack['raw_requested_ticks']} ticks; {attack['started_instances']} starts; "
                          f"{attack['unique_hit_instances']} distinct damaging attacks across {attack['damage_frames']} damage frames.", ''])
        for failure in episode.get('failures', []):
            lines.extend([f"Tick {failure['tick']} ({failure['seconds']:.3f} s), distance {failure['distance']:.6f}.",
                          f"Learner: `{json.dumps(failure['learner'], sort_keys=True)}`",
                          f"Target: `{json.dumps(failure['target'], sort_keys=True)}`",
                          f"Raw action: `{json.dumps(failure['raw_action'])}`",
                          f"Executed action: `{json.dumps(failure['executed_action'])}`", ''])
    return '\n'.join(lines)+'\n'
