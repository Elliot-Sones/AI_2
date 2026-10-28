"""Replay recorded B1 actions without loading a policy; reject divergent proof.

Observations must match bitwise after float32 conversion even when a nonzero
numeric player-state tolerance is requested. Only each episode's UUID prefix
is normalized in attack identities; attack sequence IDs and all events remain
part of the comparison. One rendered frame represents one 1/30-second tick.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import tempfile

import numpy as np


FPS = 30
FRAME_SIZE = (640, 360)
BUTTONS = ('W', 'A', 'S', 'D', 'jump', 'pickup', 'dodge', 'light', 'heavy', 'taunt')
# Visible Player.render head markers, not Pymunk debug-shape colors.
PLAYER_MARKERS = (('ORANGE', (227, 138, 14)), ('BLUE', (18, 131, 201)))
REPLAY_LESSONS = {
    'bridge_b1-v1': ('bridge_b1', 'make_b1_env'),
    'bridge_b1_contact-v1': ('bridge_b1_contact', 'make_b1_contact_env'),
}
CONTACT_TRACE_FIELDS = (
    'approach_metric', 'initial_learner_x', 'initial_target_x', 'initial_direction',
    'spawn_gap', 'spawn_midpoint', 'approach_progress', 'approach_progress_at_first_hit',
    'required_approach_progress', 'required_spawn_clearance', 'required_final_learner_clearance',
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _write_json(path, value):
    with Path(path).open('xb') as handle:
        handle.write(_json_bytes(value) + b'\n')


def _save_receipt(output, receipt):
    path = output / 'receipt.json'
    _write_json(path, receipt)
    _write_json(output / 'receipt.sha256.json', {'receipt.json': sha256(path)})


def _case(record):
    if isinstance(record, (str, Path)):
        path = Path(record)
        value = json.loads(path.read_text())
        digest = sha256(path)
    else:
        value = dict(record)
        digest = hashlib.sha256(_json_bytes(value)).hexdigest()
    for field in ('id', 'seed', 'player_slot', 'direction', 'task', 'mode', 'ticks', 'outcome', 'terminated', 'truncated'):
        if field not in value:
            raise ValueError(f'Case record is missing {field}')
    metric = value.get('metric_version')
    if value['task'] != 'combat' or not isinstance(metric, str) or metric not in REPLAY_LESSONS:
        raise ValueError('Replay requires a supported B1 combat metric_version')
    if 'stage' in value and value['stage'] != REPLAY_LESSONS[metric][0]:
        raise ValueError('Case stage differs from its metric_version')
    if type(value['seed']) is not int or not 0 <= value['seed'] < 2**32:
        raise ValueError('Case seed must be a uint32 integer')
    if type(value['player_slot']) is not int or value['player_slot'] not in (0, 1):
        raise ValueError('Case player_slot must be zero or one')
    if type(value['direction']) is not int or value['direction'] not in (-1, 1):
        raise ValueError('Case direction must be -1 or 1')
    if type(value['ticks']) is not int or not 1 <= value['ticks'] <= 300:
        raise ValueError('Case ticks must be between 1 and 300')
    if type(value['terminated']) is not bool or type(value['truncated']) is not bool:
        raise ValueError('Case terminal flags must be booleans')
    if value['mode'] not in ('deterministic', 'sampled'):
        raise ValueError('Invalid recorded action mode')
    _json_bytes(value)
    return value, digest


def _normalized(value, episode_id, key=None):
    if key == 'episode_id':
        return '<episode>'
    if key == 'attack_id' and isinstance(value, str) and value.startswith(episode_id + ':'):
        return '<episode>:' + value[len(episode_id) + 1:]
    if isinstance(value, dict):
        return {k: _normalized(v, episode_id, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalized(v, episode_id) for v in value]
    return value


class _Mismatch(Exception):
    pass


class _FFmpegWriter:
    """Stream RGB frames to an existing encoder without an optional plugin."""

    def __init__(self, path):
        executable = shutil.which('ffmpeg')
        if executable is None:
            raise RuntimeError('An existing ffmpeg executable is required for MP4 output')
        self.version = subprocess.run([executable, '-version'], capture_output=True,
                                      text=True, check=True, timeout=10).stdout.splitlines()[0]
        self.errors = tempfile.TemporaryFile()
        self.process = subprocess.Popen([
            executable, '-hide_banner', '-loglevel', 'error', '-nostdin',
            '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size',
            f'{FRAME_SIZE[0]}x{FRAME_SIZE[1]}', '-framerate', str(FPS),
            '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'veryfast',
            '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path),
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.errors)
        self.closed = False

    def append_data(self, frame):
        if self.closed:
            raise RuntimeError('Video writer is closed')
        self.process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.process.stdin.close()
            status = self.process.wait(timeout=30)
            if status:
                self.errors.seek(0)
                raise RuntimeError(f'FFmpeg exited {status}: {self.errors.read().decode(errors="replace")}')
        finally:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait(timeout=10)
            self.errors.close()


class _Checks:
    def __init__(self, receipt, tolerance):
        self.receipt, self.tolerance = receipt, tolerance
        self.tick = 0

    def fail(self, field, expected, actual):
        self.receipt['first_failure_tick'] = self.tick
        self.receipt['failure'] = {'field': field, 'expected': expected, 'actual': actual}
        raise _Mismatch(field)

    def equal(self, field, expected, actual):
        if expected != actual:
            self.fail(field, expected, actual)

    def observation(self, field, expected, actual):
        expected, actual = np.asarray(expected, dtype=np.float32), np.asarray(actual, dtype=np.float32)
        if expected.shape != (64,) or actual.shape != (64,) or not np.isfinite(expected).all() or not np.isfinite(actual).all():
            self.receipt['observation_bitwise_match'] = False
            self.fail(field, '64 finite float32 values', 'invalid observation')
        error = float(np.max(np.abs(expected.astype(np.float64) - actual.astype(np.float64))))
        self.receipt['max_observation_error'] = max(self.receipt['max_observation_error'], error)
        different = expected.view(np.uint32) != actual.view(np.uint32)
        if different.any():
            self.receipt['observation_bitwise_match'] = False
            index = int(np.flatnonzero(different)[0])
            self.fail(f'{field}[{index}]', float(expected[index]), float(actual[index]))

    def players(self, field, expected, actual, expected_episode, actual_episode):
        expected = _normalized(expected, expected_episode)
        actual = _normalized(actual, actual_episode)
        self.equal(field + '.count', len(expected), len(actual))
        for slot, (left, right) in enumerate(zip(expected, actual)):
            self.equal(f'{field}[{slot}].fields', sorted(left), sorted(right))
            for key in left:
                location = f'{field}[{slot}].{key}'
                if type(left[key]) in (int, float) and type(right[key]) in (int, float):
                    error = abs(float(left[key]) - float(right[key]))
                    self.receipt['max_state_error'] = max(self.receipt['max_state_error'], error)
                    # Integer slots, stock counts, facing and state counters
                    # are discrete identities, never approximately equivalent.
                    tolerance = self.tolerance if type(left[key]) is float and type(right[key]) is float else 0.
                    if not math.isfinite(error) or error > tolerance:
                        self.fail(location, left[key], right[key])
                else:
                    self.equal(location, left[key], right[key])

    def transition(self, expected, actual):
        expected_episode, actual_episode = expected['episode_id'], actual['episode_id']
        for field in ('schema_version', 'stage', 'stage_version', 'seed', 'scenario_id',
                      'player_slot', 'tick', 'encounter', 'outcome', 'allowed_mask',
                      'executed_action', 'opponent_action', 'measured_light_reach', 'reengagement_metric', 'reward_terms'):
            self.equal(field, expected.get(field), actual.get(field))
        if expected['stage_version'] == 'bridge_b1_contact-v1':
            for field in CONTACT_TRACE_FIELDS:
                self.equal(field, expected.get(field), actual.get(field))
        for field in ('pre_players', 'post_players'):
            self.players(field, expected[field], actual[field], expected_episode, actual_episode)
        for field in ('observation_before', 'observation_after'):
            self.observation(field, expected.get(field), actual.get(field))
        left, right = _normalized(expected['events'], expected_episode), _normalized(actual['events'], actual_episode)
        if left != right:
            self.receipt['events_match'] = False
            self.fail('events', left, right)


def _frame(env, trace, started, connected):
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    # The frozen renderer returns width,height,channels; the comparison runner
    # uses this same swap to recover an ordinary upright RGB image.
    image = Image.fromarray(np.ascontiguousarray(np.swapaxes(env.render(), 0, 1), dtype=np.uint8))
    image = ImageOps.pad(image, FRAME_SIZE, method=Image.Resampling.LANCZOS, color=(12, 16, 22))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    slot = trace['player_slot']
    color_name, color = PLAYER_MARKERS[slot]
    buttons = ', '.join(name for name, pressed in zip(BUTTONS, trace['executed_action']) if pressed > .5) or 'none'
    draw.rectangle((0, 0, FRAME_SIZE[0], 53), fill=(12, 16, 22))
    draw.text((10, 5), f'Learner: P{slot + 1} / slot {slot} / {color_name} marker    Tick {trace["tick"]}  ({trace["tick"]/FPS:.2f}s)', font=font, fill=color)
    draw.text((10, 21), f'Buttons: {buttons}    Attacks started {len(started)} / connected {len(connected)}', font=font, fill='white')
    draw.text((10, 37), f'Encounter {trace["encounter"]}    Outcome: {trace["outcome"] or "ongoing"}    30 fps / normal speed', font=font, fill='white')
    return np.asarray(image, dtype=np.uint8)


def replay_case(frozen_root, case_record, trace_inventory, output, max_error=0., *, frame_sink=None):
    """Return a durable verification receipt and, by default, a full MP4.

    A mismatch returns ``verified=False`` with the first failing tick and never
    publishes ``replay.mp4``. An optional frame_sink receives decorated uint8
    RGB arrays instead of encoding a movie, useful for an existing video writer.
    Sink frames are provisional until the returned receipt is verified.
    Existing output directories are refused, including empty directories.
    """
    import combat_curriculum
    from curriculum_telemetry import load_trace_inventory

    if not isinstance(max_error, (int, float)) or not math.isfinite(max_error) or max_error < 0:
        raise ValueError('max_error must be finite and nonnegative')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Replay outputs are immutable')
    case, case_digest = _case(case_record)
    stage, factory_name = REPLAY_LESSONS[case['metric_version']]
    inventory_path = Path(trace_inventory)
    traces = load_trace_inventory(inventory_path)
    if not traces or len(traces) != case['ticks']:
        raise ValueError('Trace frame count differs from the complete case')
    episode_id = traces[0].get('episode_id')
    for tick, trace in enumerate(traces, start=1):
        if trace.get('tick') != tick or trace.get('episode_id') != episode_id:
            raise ValueError('Replay requires one complete consecutive episode')
        for field, wanted in (('seed', case['seed']), ('player_slot', case['player_slot']),
                              ('scenario_id', case['id']), ('mode', case['mode']),
                              ('stage_version', case['metric_version']), ('stage', stage)):
            if trace.get(field) != wanted:
                raise ValueError(f'Trace {field} differs from case identity')
    root = Path(frozen_root).resolve()
    receipt = {
        'schema_version': 'b1-replay-receipt-v1', 'verified': False,
        'case_id': case['id'], 'seed': case['seed'], 'learner_slot': case['player_slot'],
        'learner_marker_color': PLAYER_MARKERS[case['player_slot']][0],
        'mode': case['mode'],
        'direction': case['direction'], 'metric_version': case['metric_version'],
        'fps': FPS, 'frames': 0, 'expected_frames': len(traces), 'frame_size': list(FRAME_SIZE),
        'sim_seconds': len(traces) / FPS, 'max_state_error': 0., 'max_error_allowed': float(max_error),
        'max_observation_error': 0., 'observation_bitwise_match': True, 'events_match': True,
        'first_failure_tick': None, 'failure': None, 'movie': None,
        'policy_inference_performed': False, 'training_performed': False,
        'verification_scope': 'Recorded trajectory fidelity; not an independent curriculum safety qualification',
        'identity_normalization': 'episode UUID prefix only; attack sequence suffix retained',
        'frame_output': 'external_frame_sink' if frame_sink is not None else 'mp4',
        'input_sha256': {'case_record': case_digest, 'trace_inventory': sha256(inventory_path)},
        'source_sha256': {
            'original_game': sha256(root / 'UTMIST-AI2-main/environment/environment.py'),
            'combat_curriculum': sha256(Path(__file__).with_name('combat_curriculum.py')),
            'replay': sha256(__file__),
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / 'case.json', case)
    checks = _Checks(receipt, float(max_error))
    previous_random, previous_numpy = random.getstate(), np.random.get_state()
    env = writer = None
    try:
        random.seed(case['seed'])
        np.random.seed(case['seed'])
        factory = getattr(combat_curriculum, factory_name)
        env = factory(root, seed=case['seed'], player_slot=case['player_slot'])
        observation, _ = env.reset(seed=case['seed'], options={'scenario': case})
        with tempfile.TemporaryDirectory(prefix='encoding-', dir=output) as temporary:
            temporary_movie = Path(temporary) / 'replay.mp4'
            if frame_sink is None:
                writer = _FFmpegWriter(temporary_movie)
                receipt['ffmpeg_version'] = writer.version
                sink = writer.append_data
            else:
                sink = frame_sink
            digest = hashlib.sha256()
            history = np.zeros(256, dtype=np.float32)
            started, connected = set(), set()
            reward_total = 0.
            try:
                for expected in traces:
                    checks.tick = expected['tick']
                    checks.observation('observation_before', expected.get('observation_before'), observation)
                    history[:-64] = history[64:]
                    history[-64:] = np.asarray(expected['observation_before'], dtype=np.float32)
                    digest.update(history.tobytes())
                    digest.update(np.asarray(expected['raw_action'], dtype=np.float32).tobytes())
                    observation, reward, terminated, truncated, info = env.step(np.asarray(expected['executed_action'], dtype=np.float32))
                    actual = info['trace']
                    reward_total += reward
                    checks.transition(expected, actual)
                    for event in actual['events']:
                        if event['type'] == 'attack_started' and event['player_slot'] == case['player_slot']:
                            started.add(event['attack_id'])
                        if event['type'] == 'damage' and event.get('source_slot') == case['player_slot']:
                            connected.add(event['attack_id'])
                    sink(_frame(env, actual, started, connected))
                    # Drawing must not alter the physics state, including on the
                    # final tick where no subsequent observation would expose it.
                    checks.observation('observation_after_render', expected['observation_after'], env._observation())
                    checks.players('post_players_after_render', expected['post_players'], env._snapshots(), episode_id, actual['episode_id'])
                    receipt['frames'] += 1
                    final_tick = checks.tick == len(traces)
                    checks.equal('terminal_tick', final_tick, bool(terminated or truncated))
                    if final_tick:
                        checks.equal('case.terminated', case['terminated'], bool(terminated))
                        checks.equal('case.truncated', case['truncated'], bool(truncated))
                        checks.equal('case.outcome', case['outcome'], actual['outcome'])
                        if 'success' in case:
                            checks.equal('case.success', case['success'], info['episode_diagnostics']['success'])
                if 'return' in case:
                    checks.equal('case.return', case['return'], reward_total)
                if 'policy_input_action_sha256' in case:
                    checks.equal('case.policy_input_action_sha256', case['policy_input_action_sha256'], digest.hexdigest())
                receipt['verified'] = True
                receipt['outcome'] = case['outcome']
                receipt['attacks_started'] = len(started)
                receipt['attacks_connected'] = len(connected)
            except _Mismatch:
                pass
            finally:
                if writer is not None:
                    writer.close()
                    writer = None
            if receipt['verified'] and frame_sink is None:
                movie = output / 'replay.mp4'
                temporary_movie.rename(movie)
                receipt['movie'] = {'path': movie.name, 'sha256': sha256(movie), 'bytes': movie.stat().st_size}
    except Exception as error:
        receipt['verified'] = False
        receipt['failure'] = {'field': 'replay_runtime', 'error_type': type(error).__name__, 'message': str(error)}
        receipt['first_failure_tick'] = checks.tick or None
        _save_receipt(output, receipt)
        raise
    finally:
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()
        random.setstate(previous_random)
        np.random.set_state(previous_numpy)
    _save_receipt(output, receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-root', type=Path, required=True)
    parser.add_argument('--case-record', type=Path, required=True)
    parser.add_argument('--trace-inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-error', type=float, default=0.)
    args = parser.parse_args(argv)
    receipt = replay_case(args.frozen_root, args.case_record, args.trace_inventory, args.output, args.max_error)
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    return 0 if receipt['verified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
