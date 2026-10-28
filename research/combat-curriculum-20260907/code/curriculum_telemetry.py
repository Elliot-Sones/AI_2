"""Finite, reconstructable B1 evidence. No policy inference or game dependencies."""
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

TRACE_SCHEMA = 'b1-trace-v1'
STAGE_METRICS = {'bridge_b1': 'bridge_b1-v1', 'bridge_b1_contact': 'bridge_b1_contact-v1'}
FPS = 30
LIGHT_MOVES = {'light', '7', 'NLIGHT', 'DLIGHT', 'SLIGHT', 'NAIR', 'DAIR', 'SAIR'}
B1_MASK = [i in (0, 1, 2, 3, 4, 6, 7) for i in range(10)]
TIMEOUT_OUTCOMES = ('timeout', 'response_deadline', 'second_hit_deadline', 'unsafe_relocation')
OUTCOMES = (None, 'ongoing', 'success', 'own_stock_loss', 'opponent_stock_loss', 'unsafe_floor_support', 'insufficient_approach') + TIMEOUT_OUTCOMES


def _json_bytes(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    except (ValueError, TypeError) as error:
        raise ValueError('Evidence must contain only finite JSON values') from error


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_trace(trace):
    """Reject missing geometry/actions and inconsistent execution before serialization."""
    _json_bytes(trace)
    if not isinstance(trace, dict) or trace.get('schema_version') != TRACE_SCHEMA:
        raise ValueError('Unsupported trace schema')
    if trace.get('stage') not in STAGE_METRICS or type(trace.get('tick')) is not int or trace['tick'] < 1:
        raise ValueError('Invalid stage or tick')
    contact_only = trace['stage'] == 'bridge_b1_contact'
    if (('stage_version' in trace or contact_only)
            and trace.get('stage_version') != STAGE_METRICS[trace['stage']]):
        raise ValueError('Stage and metric version disagree')
    if contact_only and trace['tick'] > 300:
        raise ValueError('Contact lesson exceeds its 300-tick deadline')
    if contact_only and (trace.get('outcome') in ('response_deadline', 'second_hit_deadline', 'unsafe_relocation')
                         or trace.get('outcome') == 'timeout' and trace['tick'] != 300):
        raise ValueError('Contact lesson requires its own terminal/deadline semantics')
    if not contact_only and trace.get('outcome') == 'insufficient_approach':
        raise ValueError('Insufficient initial approach is a contact-only outcome')
    if (not (type(trace.get('episode_id')) is int and trace['episode_id'] >= 0
             or isinstance(trace.get('episode_id'), str) and trace['episode_id'].strip())
            or not isinstance(trace.get('scenario_id'), str) or not trace['scenario_id'].strip()
            or type(trace.get('seed')) is not int or trace['seed'] < 0
            or type(trace.get('player_slot')) is not int or trace['player_slot'] not in (0, 1)):
        raise ValueError('Trace requires explicit episode, scenario, seed and learner identity')
    if 'env_index' in trace and (type(trace['env_index']) is not int or trace['env_index'] < 0):
        raise ValueError('Invalid environment identity')
    if trace.get('encounter') not in (1, 2):
        raise ValueError('Invalid encounter')
    if contact_only and trace['encounter'] != 1:
        raise ValueError('Contact lesson has only one encounter')
    for name in ('pre_players', 'post_players'):
        players = trace.get(name)
        if not isinstance(players, list) or len(players) != 2:
            raise ValueError('Each tick needs both player snapshots')
        for slot, player in enumerate(players):
            if not isinstance(player, dict) or player.get('player_slot') != slot:
                raise ValueError('Player snapshots must be ordered by slot')
            if any(not _number(player.get(field)) for field in ('x', 'y', 'vx', 'vy', 'damage')):
                raise ValueError('Missing or invalid player geometry/damage')
            if type(player.get('stocks')) is not int or player['stocks'] < 0:
                raise ValueError('Invalid stocks')
            if player.get('facing') not in (-1, 1) or type(player.get('grounded')) is not bool:
                raise ValueError('Invalid facing/grounded evidence')
            if 'fully_supported' in player and type(player['fully_supported']) is not bool:
                raise ValueError('Invalid full-body floor-support evidence')
            if not isinstance(player.get('state'), str) or 'attack_id' not in player:
                raise ValueError('Missing player state/attack identity')
            if 'stun_time_stored' in player and (not _number(player['stun_time_stored']) or player['stun_time_stored'] < 0):
                raise ValueError('Invalid pending stun evidence')
    for name in ('raw_action', 'executed_action', 'allowed_mask'):
        if not isinstance(trace.get(name), list) or len(trace[name]) != 10:
            raise ValueError('Actions and masks must contain ten dimensions')
    if any(value not in (True, False, 0, 1) for value in trace['allowed_mask']):
        raise ValueError('Mask must be binary')
    if trace['allowed_mask'] != B1_MASK:
        raise ValueError('B1 requires the approved movement and light-attack mask')
    for raw, executed, enabled in zip(trace['raw_action'], trace['executed_action'], trace['allowed_mask']):
        if not _number(raw) or not _number(executed):
            raise ValueError('Actions must be finite numbers')
        expected = min(1., max(0., raw)) if enabled else 0.
        if not math.isclose(executed, expected, abs_tol=1e-7):
            raise ValueError('Executed action does not match clipped/masked raw action')
    if (not isinstance(trace.get('reward_terms'), dict)
            or set(trace['reward_terms']) != {'contact', 'completion', 'stock_loss', 'timeout', 'tick'}
            or any(not _number(v) for v in trace['reward_terms'].values())):
        raise ValueError('Missing finite reward component evidence')
    if trace.get('measured_light_reach') is not None and (not _number(trace['measured_light_reach']) or trace['measured_light_reach'] <= 0):
        raise ValueError('Measured reach must be positive')
    if 'outcome' not in trace or trace['outcome'] not in OUTCOMES:
        raise ValueError('Missing terminal outcome field')
    if not isinstance(trace.get('events'), list):
        raise ValueError('Missing events')
    for event in trace['events']:
        if not isinstance(event, dict) or not isinstance(event.get('type'), str):
            raise ValueError('Each event needs a type')
        kind = event['type']
        if 'tick' in event and event['tick'] != trace['tick']:
            raise ValueError('Event tick differs from containing frame')
        if 'encounter' in event and (type(event['encounter']) is not int or event['encounter'] not in (1, 2)):
            raise ValueError('Event encounter must belong to B1')
        if contact_only and event.get('encounter', 1) != 1:
            raise ValueError('Contact event cannot claim a second encounter')
        if kind in ('attack_started', 'attack_ended', 'attack_active'):
            if event.get('player_slot') not in (0, 1) or not isinstance(event.get('attack_id'), (str, int)):
                raise ValueError('Attack events need player and attack identity')
        if kind in ('damage', 'collision'):
            unknown = kind == 'damage' and event.get('source') == 'unknown' and event.get('source_slot') is None and event.get('attack_id') is None
            if kind == 'damage' and event.get('source') == 'unknown' and not unknown:
                raise ValueError('Unknown damage cannot claim an attacker or attack identity')
            if event.get('target_slot') not in (0, 1) or (not unknown and (event.get('source_slot') not in (0, 1) or event['source_slot'] == event['target_slot'])):
                raise ValueError('Contact/damage needs distinct source and target')
            if not unknown and not isinstance(event.get('attack_id'), (int, str)):
                raise ValueError('Contact/damage requires attack identity')
        if kind == 'damage' and (not _number(event.get('amount')) or event['amount'] <= 0 or not isinstance(event.get('source'), str)):
            raise ValueError('Damage requires positive amount and recorded source')
        if kind == 'relocation':
            if contact_only:
                raise ValueError('Contact lesson forbids all relocations')
            if event.get('player_slot') not in (0, 1) or event.get('reason') != 'encounter_boundary':
                raise ValueError('Relocation must be an explicit encounter boundary')
            for position in ('from', 'to'):
                if not isinstance(event.get(position), dict) or any(not _number(event[position].get(axis)) for axis in ('x', 'y')):
                    raise ValueError('Relocation requires before/after position')
    return trace


def _validate_sequence(traces):
    if not traces:
        raise ValueError('Cannot summarize empty evidence')
    previous = None
    identity_keys = ('episode_id', 'env_index', 'player_slot', 'scenario_id', 'seed', 'stage', 'stage_version')
    identity = tuple(traces[0].get(key) for key in identity_keys)
    for trace in traces:
        validate_trace(trace)
        if tuple(trace.get(key) for key in identity_keys) != identity:
            raise ValueError('aggregate_episode accepts exactly one episode/learner')
        if previous is not None:
            if trace['tick'] != previous['tick'] + 1 or trace['pre_players'] != previous['post_players']:
                raise ValueError('Trace cannot reconstruct consecutive state transitions')
            if trace['allowed_mask'] != previous['allowed_mask']:
                raise ValueError('Allowed controls changed inside an episode')
            if previous['outcome'] not in (None, 'ongoing'):
                raise ValueError('Evidence continues after a terminal outcome')
        elif trace['tick'] != 1:
            raise ValueError('Episode trace must begin at tick 1')
        previous = trace


def _distance(players):
    return math.hypot(players[1]['x'] - players[0]['x'], players[1]['y'] - players[0]['y'])


def _runs(values):
    runs, length = [], 0
    for pressed in values:
        if pressed:
            length += 1
        elif length:
            runs.append(length)
            length = 0
    if length:
        runs.append(length)
    return runs


def _evidence(trace, slot):
    return dict(tick=trace['tick'], seconds=trace['tick']/FPS, distance=_distance(trace['post_players']),
                learner=trace['post_players'][slot], target=trace['post_players'][1-slot],
                raw_action=trace['raw_action'], executed_action=trace['executed_action'],
                outcome=trace['outcome'], encounter=trace['encounter'])


def aggregate_episode(traces, *, learner_slot=0, scenario_id='', mode='deterministic'):
    """Derive metrics from full tick evidence; never infer a hit from AttackState."""
    traces = list(traces)
    _validate_sequence(traces)
    if learner_slot not in (0, 1) or mode not in ('deterministic', 'sampled'):
        raise ValueError('Invalid learner slot or action mode')
    if traces[0].get('player_slot', learner_slot) != learner_slot:
        raise ValueError('Learner identity mismatch')
    if scenario_id and scenario_id != traces[0]['scenario_id']:
        raise ValueError('Scenario identity mismatch')
    if any('mode' in trace and trace['mode'] != mode for trace in traces):
        raise ValueError('Action-mode identity mismatch')
    stage = traces[0]['stage']
    contact_only = stage == 'bridge_b1_contact'
    attacks, damage_events, stock_events, relocations = {}, [], [], []
    toward, reduction, positive_toward, range_ticks = 0., 0., 0., 0
    reward = Counter()
    states = Counter()
    approach_rows, transitions = [], []
    credited_encounters = set()
    first_credited_hit_tick = None
    for trace in traces:
        tick = trace['tick']
        pre, post = trace['pre_players'], trace['post_players']
        adjusted = [dict(p) for p in post]
        own_loss = post[learner_slot]['stocks'] < pre[learner_slot]['stocks']
        opponent_loss = post[1-learner_slot]['stocks'] < pre[1-learner_slot]['stocks']
        if ((trace['outcome'] == 'own_stock_loss') != own_loss
                or (trace['outcome'] == 'opponent_stock_loss') != (opponent_loss and not own_loss)):
            raise ValueError('Terminal stock-loss outcome disagrees with stock transitions')
        contact_credit = 0.
        for event in trace['events']:
            kind = event['type']
            if kind == 'relocation':
                if any(attack['end_tick'] is None for attack in attacks.values()):
                    raise ValueError('Relocation occurred before an observed attack resolved')
                if post[learner_slot].get('fully_supported') is not True:
                    raise ValueError('Relocation requires measured full-body learner support')
                if any('Attack' in p['state'] or 'Stun' in p['state'] or p.get('stun_time_stored') != 0 for p in post):
                    raise ValueError('Relocation requires resolved attack and stun states')
                relocation = dict(event, tick=tick, seconds=tick/FPS)
                relocations.append(relocation)
                for axis in ('x', 'y'):
                    adjusted[event['player_slot']][axis] -= event['to'][axis] - event['from'][axis]
            elif kind == 'attack_started':
                key = (event['player_slot'], event['attack_id'])
                if key in attacks:
                    raise ValueError('Duplicate attack instance identity')
                attacks[key] = dict(attack_id=event['attack_id'], player_slot=event['player_slot'],
                                    attack_type=event.get('attack_type', 'unknown'), encounter=event.get('encounter', trace['encounter']),
                                    start_tick=tick, end_tick=None, hit_ticks=[], damage=0., collision_ticks=[],
                                    contacts=[], damage_evidence=[], end_evidence=None,
                                    start_evidence=_evidence(trace, event['player_slot']))
            elif kind in ('attack_ended', 'attack_active', 'damage', 'collision'):
                if kind == 'damage' and event['source_slot'] is None:
                    damage_events.append(dict(event, tick=tick, seconds=tick/FPS))
                    continue
                key = (event.get('source_slot', event.get('player_slot')), event['attack_id'])
                if key not in attacks:
                    raise ValueError('Attack event refers to an unobserved initiation')
                attack = attacks[key]
                if kind == 'attack_ended':
                    if attack['end_tick'] is not None:
                        raise ValueError('Duplicate attack end')
                    attack['end_tick'] = tick
                    attack['end_evidence'] = _evidence(trace, attack['player_slot'])
                elif kind == 'damage':
                    if attack['end_tick'] is not None and attack['end_tick'] != tick:
                        raise ValueError('Damage recorded after attack ended')
                    attack['hit_ticks'].append(tick)
                    attack['damage'] += event['amount']
                    attack['damage_evidence'].append(dict(event, evidence=_evidence(trace, attack['player_slot'])))
                    damage_events.append(dict(event, tick=tick, seconds=tick/FPS))
                    if (attack['player_slot'] == learner_slot and str(attack['attack_type']) in LIGHT_MOVES
                            and attack['encounter'] not in credited_encounters and not own_loss and not opponent_loss):
                        credited_encounters.add(attack['encounter'])
                        contact_credit = 1.
                        if first_credited_hit_tick is None:
                            first_credited_hit_tick = tick
                elif kind == 'collision':
                    attack['collision_ticks'].append(tick)
                    attack['contacts'].append(dict(event, evidence=_evidence(trace, attack['player_slot'])))
        delta = [adjusted[learner_slot][axis] - pre[learner_slot][axis] for axis in ('x', 'y')]
        direction = [pre[1-learner_slot][axis] - pre[learner_slot][axis] for axis in ('x', 'y')]
        norm = math.hypot(*direction)
        progress = sum(a*b for a,b in zip(delta, direction))/norm if norm else 0.
        distance_change = _distance(pre) - _distance(adjusted)
        horizontal_direction = 1. if pre[1-learner_slot]['x'] >= pre[learner_slot]['x'] else -1.
        horizontal_progress = delta[0]*horizontal_direction
        horizontal_gap_change = abs(pre[1-learner_slot]['x']-pre[learner_slot]['x']) - abs(adjusted[1-learner_slot]['x']-adjusted[learner_slot]['x'])
        toward += progress
        positive_toward += max(0., progress)
        reduction += distance_change
        approach_rows.append(dict(tick=tick, seconds=tick/FPS, distance_before=_distance(pre), distance_after=_distance(post),
                                  toward_target_displacement=progress, distance_reduction_excluding_relocation=distance_change,
                                  horizontal_toward_displacement=horizontal_progress, horizontal_gap_reduction=horizontal_gap_change,
                                  learner_x_displacement=delta[0]))
        reach = trace.get('measured_light_reach')
        range_ticks += int(reach is not None and _distance(pre) <= reach)
        expected_terms = dict(contact=contact_credit, completion=2. if trace['outcome'] == 'success' else 0.,
                              stock_loss=-5. if own_loss else 0.,
                              timeout=-1. if trace['outcome'] in TIMEOUT_OUTCOMES else 0., tick=-.001)
        if any(not math.isclose(trace['reward_terms'][key], value, rel_tol=0., abs_tol=1e-9)
               for key, value in expected_terms.items()):
            raise ValueError('Reward terms disagree with distinct hits, terminal outcome or bounded tick cost')
        reward.update(expected_terms)
        states[post[learner_slot]['state']] += 1
        if pre[learner_slot]['state'] != post[learner_slot]['state']:
            transitions.append(dict(tick=tick, before=pre[learner_slot]['state'], after=post[learner_slot]['state']))
        for slot in (0, 1):
            if post[slot]['stocks'] < pre[slot]['stocks']:
                stock_events.append(dict(tick=tick, seconds=tick/FPS, player_slot=slot,
                                         lost=pre[slot]['stocks']-post[slot]['stocks'], attribution='unknown'))
    learner_attacks = [a for a in attacks.values() if a['player_slot'] == learner_slot]
    hits = [a for a in learner_attacks if a['damage'] > 0]
    for attack in learner_attacks:
        attack['status'] = 'hit' if attack['damage'] else ('ended_without_damage' if attack['end_tick'] is not None else 'incomplete_at_episode_end')
        attack['first_hit_tick'] = min(attack['hit_ticks']) if attack['hit_ticks'] else None
    dealt = [e for e in damage_events if e['source_slot'] == learner_slot]
    taken = [e for e in damage_events if e['target_slot'] == learner_slot]
    raw_pressed = [t['raw_action'][7] > .5 for t in traces]
    executed_pressed = [t['executed_action'][7] > .5 for t in traces]
    patterns = {tuple(a > .5 for a in t['executed_action']) for t in traces}
    fell = any(e['player_slot'] == learner_slot for e in stock_events)
    pursuit = None
    target_relocations = [r for r in relocations if r['player_slot'] != learner_slot]
    if target_relocations:
        relocation = target_relocations[0]
        relocation_players = traces[relocation['tick']-1]['post_players']
        relocation_direction = 1. if relocation_players[1-learner_slot]['x'] >= relocation_players[learner_slot]['x'] else -1.
        response_rows = [r for r in approach_rows if relocation['tick'] < r['tick'] <= relocation['tick'] + 60]
        cumulative_progress, cumulative_gap, response_tick = 0., 0., None
        for row in response_rows:
            cumulative_progress += row['learner_x_displacement']*relocation_direction
            cumulative_gap += row['horizontal_gap_reduction']
            if response_tick is None and cumulative_progress >= 1. and cumulative_gap >= 1.:
                response_tick = row['tick']
        second_hit_ticks = [a['first_hit_tick'] for a in hits if a['encounter'] == 2]
        second_hit = min(second_hit_ticks) if second_hit_ticks else None
        pursuit = dict(metric_version='horizontal_gap_and_learner_progress_v1', relocation_tick=relocation['tick'], response_tick=response_tick,
                       response_seconds=(response_tick-relocation['tick'])/FPS if response_tick else None,
                       toward_target_displacement=cumulative_progress, gap_closed=cumulative_gap,
                       second_hit_tick=second_hit,
                       second_hit_within_deadline=second_hit is not None and 0 < second_hit-relocation['tick'] <= 90,
                       response_within_deadline=response_tick is not None)
    outcome = traces[-1]['outcome']
    light_encounters = {a['encounter'] for a in hits if str(a['attack_type']) in LIGHT_MOVES}
    contact_approach = None
    if contact_only:
        initial = traces[0]['pre_players']
        initial_gap = initial[1-learner_slot]['x'] - initial[learner_slot]['x']
        initial_direction = 1 if initial_gap > 0 else (-1 if initial_gap < 0 else 0)
        first_hit_progress = None
        if first_credited_hit_tick is not None:
            at_hit = traces[first_credited_hit_tick-1]['post_players'][learner_slot]
            first_hit_progress = initial_direction * (at_hit['x']-initial[learner_slot]['x'])
        contact_approach = dict(metric_version='initial_target_horizontal_progress_v1',
                                initial_direction=initial_direction, initial_learner_x=initial[learner_slot]['x'],
                                initial_target_x=initial[1-learner_slot]['x'], first_credited_hit_tick=first_credited_hit_tick,
                                learner_progress_at_first_hit=first_hit_progress, required_progress=.25)
        final_players = traces[-1]['post_players']
        clearance = final_players[learner_slot].get('horizontal_floor_clearance')
        success = bool(outcome == 'success' and not fell and not relocations and credited_encounters == {1}
                       and first_hit_progress is not None and first_hit_progress >= .25
                       and all(a['end_tick'] is not None for a in attacks.values())
                       and all('Attack' not in p['state'] and 'Stun' not in p['state'] and p.get('stun_time_stored') == 0
                               and p['stocks'] > 0 and p['grounded'] and p.get('fully_supported') is True for p in final_players)
                       and _number(clearance) and clearance >= .2)
    else:
        success = bool(outcome == 'success' and not fell and len(hits) >= 2 and light_encounters == {1, 2}
                       and len(target_relocations) == 1 and len(credited_encounters) == 2
                       and pursuit and pursuit['response_within_deadline'] and pursuit['second_hit_within_deadline']
                       and all(a['end_tick'] is not None for a in hits)
                       and all('Attack' not in p['state'] and 'Stun' not in p['state'] and p.get('stun_time_stored') == 0
                               for p in traces[-1]['post_players'])
                       and traces[-1]['post_players'][learner_slot]['grounded']
                       and traces[-1]['post_players'][learner_slot].get('fully_supported') is True)
    if outcome == 'success' and not success:
        raise ValueError('Success outcome lacks reconstructable stage contact/approach/safety evidence')
    diagnoses = []
    if outcome == 'unsafe_floor_support':
        diagnoses.append('unsafe_floor_support')
    if not success:
        if fell:
            diagnoses.append('own_stock_loss')
        if positive_toward < 1e-6:
            diagnoses.append('no_measured_approach')
        if not any(raw_pressed):
            diagnoses.append('no_attack_attempt')
        elif not learner_attacks:
            diagnoses.append('requested_but_no_attack_started')
        elif not hits:
            diagnoses.append('started_but_no_hit')
        if pursuit and not (pursuit['response_within_deadline'] and pursuit['second_hit_within_deadline']):
            diagnoses.append('failed_reengagement')
        if contact_approach and first_hit_progress is not None and first_hit_progress < .25:
            diagnoses.append('insufficient_approach_before_contact')
    total = len(traces)
    lock_windows = []
    # Diagnostic screen, not a causal attribution. Require sustained movement requests
    # and negligible travel as well as attack occupancy; idle attacking alone is not lock.
    for start in range(0, total-59):
        window = traces[start:start+60]
        attack_ticks = sum('Attack' in t['post_players'][learner_slot]['state'] for t in window)
        # Frozen InputHandler order: W/up=0, A/left=1, S/down=2, D/right=3.
        movement_ticks = sum((t['executed_action'][1] > .5) != (t['executed_action'][3] > .5) for t in window)
        travel = sum(math.hypot(t['post_players'][learner_slot]['x']-t['pre_players'][learner_slot]['x'],
                               t['post_players'][learner_slot]['y']-t['pre_players'][learner_slot]['y']) for t in window)
        relocation_in_window = any(e['type'] == 'relocation' and e['player_slot'] == learner_slot for t in window for e in t['events'])
        if attack_ticks >= 54 and movement_ticks >= 54 and travel < .2 and not relocation_in_window:
            lock_windows.append(dict(start_tick=window[0]['tick'], end_tick=window[-1]['tick'],
                                     attack_state_ticks=attack_ticks, movement_requested_ticks=movement_ticks,
                                     learner_travel=travel, start_evidence=_evidence(window[0], learner_slot),
                                     end_evidence=_evidence(window[-1], learner_slot)))
            break
    if lock_windows:
        diagnoses.append('observed_sustained_movement_lock')
    first_hit = min((e['tick'] for e in dealt), default=None)
    raw_means = [sum(t['raw_action'][i] for t in traces)/total for i in range(10)]
    raw_stds = [math.sqrt(sum((t['raw_action'][i]-raw_means[i])**2 for t in traces)/total) for i in range(10)]
    report = dict(schema_version='b1-episode-v1', metric_version=STAGE_METRICS[stage], scenario_id=scenario_id or traces[0].get('scenario_id', ''),
                  episode_id=traces[0].get('episode_id'), seed=traces[0].get('seed'), mode=mode, learner_slot=learner_slot,
                  training_qualified=traces[0].get('training_qualified'),
                  qualification_blocker=traces[0].get('qualification_blocker'),
                  stage=stage, ticks=total, seconds=total/FPS, outcome=outcome, success=success, fell=fell,
                  evidence_valid=True, systemic_movement_lock=bool(lock_windows), diagnoses=diagnoses,
                  diagnosis_limit='Observed missing behavior; causal mechanism is not inferred.',
                  failures=[] if success else [_evidence(traces[-1], learner_slot)],
                  reward_terms=dict(reward), damage_events=damage_events, stock_events=stock_events,
                  attack=dict(raw_requested_ticks=sum(raw_pressed), executed_requested_ticks=sum(executed_pressed),
                              raw_press_run_lengths=_runs(raw_pressed), executed_press_run_lengths=_runs(executed_pressed),
                              started_instances=len(learner_attacks), unique_hit_instances=len(hits),
                              damage_frames=len({e['tick'] for e in dealt}), damage_events=len(dealt),
                              damage_dealt=sum(e['amount'] for e in dealt), damage_taken=sum(e['amount'] for e in taken),
                              unattributed_damage=sum(e['amount'] for e in damage_events if e['source_slot'] is None),
                              first_hit_tick=first_hit, first_hit_seconds=first_hit/FPS if first_hit else None,
                              hit_rate_denominator=len(learner_attacks), hit_rate=len(hits)/len(learner_attacks) if learner_attacks else None,
                              ended_without_damage=sum(a['status'] == 'ended_without_damage' for a in learner_attacks),
                              attack_state_ticks=sum(n for state,n in states.items() if 'Attack' in state),
                              attack_state_occupancy=sum(n for state,n in states.items() if 'Attack' in state)/total,
                              instances=learner_attacks),
                  movement=dict(net_toward_target_displacement=toward, positive_toward_target_displacement=positive_toward,
                                distance_reduction_excluding_relocation=reduction, relocations=relocations, pursuit=pursuit,
                                contact_approach=contact_approach,
                                lock_windows=lock_windows,
                                lock_criterion='Within 60 ticks: at least 54 AttackState ticks and 54 unopposed executed horizontal-movement requests, under 0.2 units actual learner travel, no learner relocation.',
                                time_in_measured_range_seconds=range_ticks/FPS if all(t.get('measured_light_reach') is not None for t in traces) else None,
                                range_measurement='Euclidean center separation versus measured reach; not guaranteed contact.', trajectory=approach_rows),
                  actions=dict(unique_pressed_patterns=len(patterns), pressed_buttons_varied=len(patterns)>1,
                               raw_means=raw_means, raw_sample_stds=raw_stds,
                               raw_press_rates=[sum(t['raw_action'][i]>.5 for t in traces)/total for i in range(10)],
                               action_sources=sorted({t.get('action_source', 'unspecified') for t in traces}),
                               allowed_mask=traces[0]['allowed_mask'],
                               executed_press_rates=[sum(t['executed_action'][i]>.5 for t in traces)/total for i in range(10)],
                               state_occupancy=dict(states), state_transitions=transitions))
    _json_bytes(report)
    return report


def _atomic_write(path, data):
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        # Linking publishes a complete file and refuses to overwrite existing evidence.
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


class AtomicTraceWriter:
    """Append full transitions, including interleaved envs, to immutable gzip chunks."""
    def __init__(self, directory, chunk_size=256):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if any(self.directory.glob('chunk-*.jsonl.gz')) or (self.directory/'inventory.json').exists():
            raise FileExistsError('Trace destination already contains evidence')
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError('chunk_size must be a positive integer')
        self.chunk_size = chunk_size
        self.pending, self.chunks, self.closed = [], [], False

    def append(self, trace):
        if self.closed:
            raise ValueError('Writer is closed')
        validate_trace(trace)
        self.pending.append(_json_bytes(trace))
        if len(self.pending) >= self.chunk_size:
            self._flush()

    def _flush(self):
        if not self.pending:
            return
        name = f'chunk-{len(self.chunks):06d}.jsonl.gz'
        content = gzip.compress(b'\n'.join(self.pending)+b'\n', mtime=0)
        _atomic_write(self.directory/name, content)
        self.chunks.append(dict(path=name, sha256=hashlib.sha256(content).hexdigest(), bytes=len(content), frames=len(self.pending)))
        self.pending.clear()

    def close(self):
        if self.closed:
            return self.inventory
        self._flush()
        self.inventory = dict(schema_version='b1-trace-inventory-v1', trace_schema=TRACE_SCHEMA,
                              frames=sum(c['frames'] for c in self.chunks), chunks=self.chunks)
        _atomic_write(self.directory/'inventory.json', _json_bytes(self.inventory))
        self.closed = True
        return self.inventory


def load_trace_inventory(path):
    """Verify every compressed byte and schema before returning reconstruction input."""
    path = Path(path)
    try:
        if path.is_symlink() or path.parent.is_symlink():
            raise ValueError('Trace inventories must be local regular files')
        inventory = json.loads(path.read_text())
        if inventory['schema_version'] != 'b1-trace-inventory-v1' or inventory['trace_schema'] != TRACE_SCHEMA:
            raise ValueError('Unsupported inventory schema')
        traces, seen = [], set()
        for index, chunk in enumerate(inventory['chunks']):
            name = chunk['path']
            if name != f'chunk-{index:06d}.jsonl.gz' or name in seen:
                raise ValueError('Invalid/duplicate chunk path')
            seen.add(name)
            chunk_path = path.parent/name
            if chunk_path.is_symlink() or not chunk_path.is_file():
                raise ValueError('Trace chunks must be regular files, not links')
            data = chunk_path.read_bytes()
            if len(data) != chunk['bytes'] or hashlib.sha256(data).hexdigest() != chunk['sha256']:
                raise ValueError('Trace checksum/size mismatch')
            rows = [json.loads(line) for line in gzip.decompress(data).splitlines()]
            if len(rows) != chunk['frames']:
                raise ValueError('Chunk frame count mismatch')
            for row in rows:
                validate_trace(row)
            traces.extend(rows)
        if len(traces) != inventory['frames']:
            raise ValueError('Inventory frame count mismatch')
        return traces
    except (KeyError, TypeError, OSError, EOFError, json.JSONDecodeError) as error:
        raise ValueError('Malformed or incomplete trace inventory') from error
