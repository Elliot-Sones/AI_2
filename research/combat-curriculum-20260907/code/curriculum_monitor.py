"""Local-only scalar bridge for downloaded B1 pilot evidence; no cloud lifecycle actions."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import tempfile
import threading
import time
import uuid

NOTES = {
    'Goal': 'Teach the preserved Large-145 actor to approach and land one real light hit in bridge_b1_contact-v1 while retaining navigation. This contact-only lesson is a narrow drill, not human-play readiness.',
    'Change': 'Start from 10,027,008 transitions, architecture [1536,1536,1536], 256 inputs and 10 outputs. Selectively calibrate light for 32,768 transitions; clone that result into E1 entropy 0.001 and E2 entropy 0.003. Preserve movement transfer and separate combat/navigation critics.',
    'Expect': 'At least 0.25 world units of learner-directed approach and one real light hit within 10 seconds; success requires resolved safe support for both players and learner horizontal floor clearance at least 0.2, with no own-stock loss. Both deterministic and sampled development panels plus navigation retention qualify progress. A bounded negative result is valid; monitoring never promotes a model.',
    'Setup': 'Corrected Python/Pymunk game on CPU; batched policy/PPO on CUDA. 64 environments x 512 ticks, batch 8192, four epochs, learning rate 0.0001, clip 0.10, target KL 0.01. Three paired rounds of four combat plus one navigation rollout: 491,520 transitions per arm; maintenance entropy 0.001. One active GPU, finite existing budget, scalar monitoring from local downloads only.',
}
ROLLOUT_KEYS = ('rollout', 'transitions', 'cumulative_timesteps', 'collection_seconds', 'update_seconds', 'wall_seconds_including_setup_logging', 'steps_per_second_including_setup_logging', 'episode_count', 'learning_rate')
PPO_KEYS = ('approx_kl', 'clip_fraction', 'entropy_loss', 'explained_variance', 'learning_rate', 'loss', 'n_updates', 'policy_gradient_loss', 'std', 'value_loss', 'active_entropy_sum', 'active_entropy_mean', 'active_entropy_coefficient', 'combat/value_loss', 'combat/explained_variance', 'navigation/value_loss', 'navigation/explained_variance')


def extract_scalars(relative_path, row):
    """Only explicit numeric fields leave the machine; strings/arrays are never forwarded."""
    result = {}
    if not isinstance(row, dict):
        return result
    path = str(relative_path)
    match = re.search(r'(?:^|/)(E[12])(?:[-/]|$)', path)
    arm = match[1] if match else ('calibration' if 'calibration' in Path(path).parts else 'baseline')
    def put(key, value):
        if type(value) in (int, float) and math.isfinite(value):
            result[key] = value
    if re.fullmatch(r'rollout-\d{2}\.json', Path(path).name):
        prefix = f'{arm}/train'
        for key in ROLLOUT_KEYS:
            put(f'{prefix}/{key}', row.get(key))
        ppo = row.get('ppo_metrics', {})
        if isinstance(ppo, dict):
            for key in PPO_KEYS:
                put(f'{prefix}/ppo/{key}', ppo.get('train/'+key))
        stds = row.get('learned_action_std', [])
        if isinstance(stds, list) and len(stds) == 10:
            for index, value in enumerate(stds):
                put(f'{prefix}/action_std/{index}', value)
        episodes = row.get('episodes', [])
        episodes = [episode for episode in episodes if isinstance(episode, dict) and type(episode.get('success')) is bool] if isinstance(episodes, list) else []
        if episodes:
            successes = sum(episode['success'] for episode in episodes)
            falls = sum(episode.get('own_stock_loss', episode.get('fell', episode.get('fall', False))) is True for episode in episodes)
            for key, value in dict(completed_episodes=len(episodes), successes=successes, falls=falls, success_rate=successes/len(episodes), fall_rate=falls/len(episodes)).items():
                put(f'{prefix}/{key}', value)
            hits = [len(episode['distinct_hit_ids']) for episode in episodes if isinstance(episode.get('distinct_hit_ids'), list)]
            times = [episode['first_hit_tick']/30 for episode in episodes if type(episode.get('first_hit_tick')) is int and episode['first_hit_tick'] > 0]
            for key, values in (('mean_hits', hits), ('mean_time_to_first_hit_seconds', times)):
                if values:
                    put(f'{prefix}/{key}', sum(values)/len(values))
                    put(f'{prefix}/{key}_reports', len(values))
    elif Path(path).name == 'summary.json':
        task = row.get('task') if row.get('task') in ('combat', 'near', 'wide') else 'unknown'
        modes = row.get('modes', row)
        for mode in ('deterministic', 'sampled'):
            panel = modes.get(mode, {}) if isinstance(modes, dict) else {}
            if not isinstance(panel, dict):
                continue
            prefix = f'{arm}/eval/{task}/{mode}'
            for key in ('episodes', 'successes', 'falls', 'mean_ticks'):
                put(f'{prefix}/{key}', panel.get(key))
            count = panel.get('episodes')
            if type(count) is int and count > 0:
                for key in ('successes', 'falls'):
                    value = panel.get(key)
                    if type(value) is int and 0 <= value <= count:
                        put(f'{prefix}/'+('success_rate' if key == 'successes' else 'fall_rate'), value/count)
            episodes = row.get('episodes', [])
            for source, target in (('started_instances', 'mean_attack_starts'), ('unique_hit_instances', 'mean_hits'), ('first_hit_seconds', 'mean_time_to_first_hit_seconds')):
                values = [episode['attack'][source] for episode in episodes if isinstance(episode, dict) and episode.get('mode') == mode and isinstance(episode.get('attack'), dict) and type(episode['attack'].get(source)) in (int, float) and math.isfinite(episode['attack'][source])] if isinstance(episodes, list) else []
                if values:
                    put(f'{prefix}/{target}', sum(values)/len(values))
                    put(f'{prefix}/{target}_reports', len(values))
    return result


def _atomic(path, value):
    descriptor, name = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as handle:
            json.dump(value, handle, allow_nan=False, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _verified(path, root, digest):
    for parent in path.parents:
        if parent == root:
            break
        try:
            raw = (parent/'inventory.json').read_bytes()
            inventory, ack = json.loads(raw), json.loads((parent/'preservation_ack.json').read_text())
            relative = path.relative_to(parent).as_posix()
            if ack.get('verified') is True and ack.get('block_id') == inventory.get('block_id') and ack.get('inventory_sha256') == hashlib.sha256(raw).hexdigest():
                return any(item.get('path') == relative and item.get('sha256') == digest for item in inventory.get('files', []) if isinstance(item, dict))
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return False


def run_monitor(root, deadline_unix, sdk, *, interval=20., clock=time.time, wait=None):
    root = Path(root).resolve()
    if not root.is_dir() or type(deadline_unix) not in (int, float) or not math.isfinite(deadline_unix) or deadline_unix <= clock():
        raise ValueError('An existing local session and finite future deadline are required')
    if type(interval) not in (int, float) or not math.isfinite(interval) or not 0 < interval <= 60:
        raise ValueError('Polling interval must be between zero and 60 seconds')
    directory = root/'monitor'
    directory.mkdir(exist_ok=True)
    stop = threading.Event()
    run, outcome, seen = None, 'error', set()
    metadata = {'id': uuid.uuid4().hex[:12], 'pid': os.getpid(), 'root': str(root)}
    history = directory/'scalar-history.jsonl'
    with (directory/'monitor.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('A monitor already owns this local session') from error
        previous_handler = signal.signal(signal.SIGTERM, lambda *_: stop.set())
        try:
            if (directory/'run.json').exists():
                metadata['id'] = json.loads((directory/'run.json').read_text())['id']
            if history.exists():
                for line in history.read_text().splitlines():
                    try:
                        seen.add(json.loads(line)['record_id'])
                    except (ValueError, KeyError, TypeError):
                        continue
            _atomic(directory/'run.json', metadata)
            run = sdk.init(project='utmist-ai2', group='session13', name='session13-b1-paired-pilot', id=metadata['id'], resume='allow', dir=str(directory), config=NOTES, notes='\n\n'.join(f'{key}: {value}' for key,value in NOTES.items()), save_code=False, settings={'disable_code': True, 'disable_git': True, 'console': 'off', 'x_disable_stats': True, 'x_disable_machine_info': True})
            metadata.update(url=run.url, path=run.path)
            _atomic(directory/'run.json', metadata)
            while not stop.is_set() and clock() < deadline_unix:
                candidates = set(root.rglob('rollout-??.json')) | set(root.rglob('summary.json'))
                for path in sorted(candidates):
                    if path.is_symlink() or not path.resolve().is_relative_to(root):
                        continue
                    try:
                        raw = path.read_bytes()
                        digest, relative = hashlib.sha256(raw).hexdigest(), path.relative_to(root).as_posix()
                        verified = int(_verified(path, root, digest))
                        identity = hashlib.sha256(f'{relative}:{digest}:{verified}'.encode()).hexdigest()
                        if identity in seen:
                            continue
                        scalars = extract_scalars(relative, json.loads(raw))
                    except (OSError, ValueError, TypeError, AttributeError):
                        continue
                    if not scalars:
                        continue
                    scalars['evidence/local_verified'] = verified
                    run.log(scalars)
                    with history.open('a') as handle:
                        handle.write(json.dumps({'record_id': identity, 'relative_path': relative, 'sha256': digest, 'unix': clock(), 'scalars': scalars}, allow_nan=False)+'\n')
                        handle.flush()
                        os.fsync(handle.fileno())
                    seen.add(identity)
                _atomic(directory/'status.json', dict(metadata, outcome='running', records=len(seen), updated_unix=clock(), deadline_unix=deadline_unix))
                (wait or stop.wait)(min(interval, max(0., deadline_unix-clock())))
            outcome = 'sigterm' if stop.is_set() else 'deadline'
        except Exception:
            outcome = 'error'  # Never include SDK/file exception contents or credentials.
        finally:
            if run is not None:
                try:
                    run.summary['monitor_outcome'] = outcome
                except Exception:
                    outcome = 'error'
                try:
                    run.finish(exit_code=int(outcome == 'error'))
                except Exception:
                    outcome = 'error'
            signal.signal(signal.SIGTERM, previous_handler)
            _atomic(directory/'status.json', dict(metadata, outcome=outcome, records=len(seen), updated_unix=clock(), deadline_unix=deadline_unix))
    return int(outcome == 'error')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--deadline-unix', type=float, required=True)
    args = parser.parse_args(argv)
    import wandb  # Credentials and the SDK remain in this local process only.
    return run_monitor(args.root, args.deadline_unix, wandb)


if __name__ == '__main__':
    raise SystemExit(main())
