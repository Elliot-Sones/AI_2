#!/usr/bin/env python3
"""Matched raw two-player gameplay/PPO infrastructure benchmark, CUDA policy only.

This is not the contact curriculum and does not establish fighting skill. The
frozen Box(10) Gaussian policy is shared by BOTH players; original input code
thresholds controls. CPU references include transfers and optional spawned
Python/Pymunk workers. Native CPU is a serial C reference; native CUDA runs one
thread per game. Seeds/configurations match, not generator-specific RNG draws
or learned trajectories. Saved-action physics parity is a separate prerequisite.
"""
from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


from ai2.paths import REPO_ROOT as PROJECT_ROOT
from ai2.paths import source_path

import argparse
from contextlib import contextmanager, ExitStack
from dataclasses import asdict, dataclass
import gc
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import time
import traceback
from types import SimpleNamespace

import numpy as np
from tools.validation.training_benchmark_contract import ai2_native_receipt

ROOT = PROJECT_ROOT
BACKENDS = ('python', 'native_cpu', 'native_cuda')
UPDATE_MODES = ('sb3', 'deferred_metrics')
SCHEMA = 'ai2.raw_combat_training_benchmark.v1'


@dataclass(frozen=True)
class BenchmarkConfig:
    games: int = 8
    n_steps: int = 128
    batch_size: int = 256
    n_epochs: int = 4
    net_arch: tuple[int, ...] = (256, 256)
    warmup_rollouts: int = 1
    measured_rollouts: int = 3
    seed: int = 0
    device: str = 'cuda'
    python_workers: int = 1
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    backends: tuple[str, ...] = BACKENDS
    update_mode: str = 'sb3'

    def __post_init__(self):
        for name in ('games', 'n_steps', 'batch_size', 'n_epochs', 'measured_rollouts', 'python_workers'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.warmup_rollouts) is not int or self.warmup_rollouts < 0:
            raise ValueError('warmup_rollouts must be a nonnegative integer')
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 2*self.games:
            raise ValueError('seed and all player seeds must fit uint32')
        if not self.net_arch or any(type(v) is not int or v <= 0 for v in self.net_arch):
            raise ValueError('net_arch must contain positive hidden layer widths')
        if self.batch_size <= 1 or self.player_transitions_per_rollout % self.batch_size:
            raise ValueError('batch_size must divide both-player rollout transitions and exceed 1')
        if self.python_workers > self.games:
            raise ValueError('python_workers cannot exceed games')
        if not self.backends or len(set(self.backends)) != len(self.backends) or set(self.backends) - set(BACKENDS):
            raise ValueError('backends must be distinct supported names')
        if self.update_mode not in UPDATE_MODES:
            raise ValueError('update_mode must be sb3 or deferred_metrics')
        if self.device != 'cuda' and not (self.device.startswith('cuda:') and self.device[5:].isdigit()):
            raise ValueError('policy/PPO require an explicit CUDA device; no CPU fallback')
        for name in ('learning_rate', 'gamma', 'gae_lambda', 'clip_range', 'ent_coef', 'vf_coef', 'max_grad_norm'):
            value = getattr(self, name)
            if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'invalid {name}')
        if not 0 < self.gamma <= 1 or not 0 < self.gae_lambda <= 1 or self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError('invalid discount, learning rate, or gradient bound')

    @property
    def game_ticks_per_rollout(self):
        return self.games * self.n_steps

    @property
    def player_transitions_per_rollout(self):
        return 2 * self.game_ticks_per_rollout


def require_cuda(device: str, *, torch_module=None):
    if device != 'cuda' and not (device.startswith('cuda:') and device[5:].isdigit()):
        raise ValueError('policy/PPO require CUDA; no CPU fallback')
    if torch_module is None:
        import torch as torch_module
    if not torch_module.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no policy construction, inference, or training allowed')
    value = torch_module.device(device)
    index = value.index if value.index is not None else torch_module.cuda.current_device()
    if index >= torch_module.cuda.device_count():
        raise ValueError('requested CUDA device does not exist')
    return torch_module.device('cuda', index)


def _linux_cpu_model(cpuinfo: str | None = None) -> str:
    if cpuinfo is None:
        try:
            cpuinfo = Path('/proc/cpuinfo').read_text(encoding='utf-8', errors='replace')
        except OSError:
            return 'unknown-cpu-model'
    for line in cpuinfo.splitlines():
        key, separator, value = line.partition(':')
        if separator and key.strip().lower() == 'model name' and value.strip():
            return value.strip()
    return 'unknown-cpu-model'


class OriginalGameBatch:
    """Uninstrumented original CPU games; NumPy-only, safe in spawn workers.

    Global RNG swapping is required because the source uses Python globals even
    after reset. Autoreset continues each game's stream, as the native lanes do.
    No wrappers, action masks, reward shaping, relocations, or gameplay patches.
    """
    def __init__(self, frozen_root: Path, game_ids: list[int]):
        from ai2.evaluation.combat_episode_capture import original_game
        self.game_ids = game_ids
        self.games = []
        self._contexts = ExitStack()
        self._closed = False
        self._initialized = False
        self.valid_steps = np.zeros(len(game_ids), dtype=np.uint64)
        self.stack = np.zeros((2*len(game_ids), 256), dtype=np.float32)
        py_state, np_state = random.getstate(), np.random.get_state()
        try:
            for game_id in game_ids:
                game, _ = self._contexts.enter_context(original_game(Path(frozen_root), game_id))
                self.games.append(game)
        except BaseException:
            self._contexts.close()
            raise
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @contextmanager
    def _rng(self, index):
        py_state, np_state = random.getstate(), np.random.get_state()
        try:
            random.setstate(self.py_states[index])
            np.random.set_state(self.np_states[index])
            yield
        finally:
            self.py_states[index], self.np_states[index] = random.getstate(), np.random.get_state()
            random.setstate(py_state)
            np.random.set_state(np_state)

    @staticmethod
    def _observations(observations):
        result = np.asarray([observations[0], observations[1]], dtype=np.float32)
        if result.shape != (2, 64) or not np.isfinite(result).all():
            raise RuntimeError('invalid original-game observations')
        return result

    def reset(self, seeds):
        if self._closed or len(seeds) != len(self.games):
            raise ValueError('closed batch or wrong game seed count')
        self.py_states = [random.Random(int(seed)).getstate() for seed in seeds]
        self.np_states = [np.random.RandomState(int(seed)).get_state() for seed in seeds]
        self.returns = np.zeros((len(self.games), 2), dtype=np.float64)
        self.lengths = np.zeros(len(self.games), dtype=np.int64)
        self.episodes = np.zeros(len(self.games), dtype=np.int64)
        self.stack.fill(0)
        for index, (game, seed) in enumerate(zip(self.games, seeds)):
            with self._rng(index):
                obs, _ = game.reset(seed=int(seed))
            self.stack[2*index:2*index+2, -64:] = self._observations(obs)
        self._initialized = True
        return self.stack.copy()

    def step(self, actions):
        if self._closed or not self._initialized:
            raise RuntimeError('reset original games before stepping')
        actions = np.asarray(actions)
        if actions.shape != (len(self.games)*2, 10) or not np.isfinite(actions).all():
            raise ValueError('invalid both-player original-game action batch')
        n = len(self.games)*2
        terminal = np.empty_like(self.stack)
        rewards = np.empty(n, dtype=np.float32)
        terminated, truncated = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
        infos = []
        for index, game in enumerate(self.games):
            pair = slice(2*index, 2*index+2)
            with self._rng(index):
                obs, reward, term, trunc, _ = game.step({p: actions[2*index+p] for p in (0, 1)})
                frame = self._observations(obs)
                terminal[pair, :192] = self.stack[pair, 64:]
                terminal[pair, 192:] = frame
                raw_reward = np.asarray([reward[0], reward[1]], dtype=np.float64)
                if not np.isfinite(raw_reward).all():
                    raise RuntimeError('nonfinite raw game reward')
                rewards[pair] = raw_reward
                self.returns[index] += raw_reward
                self.lengths[index] += 1
                self.valid_steps[index] += 1
                terminated[pair], truncated[pair] = bool(term), bool(trunc)
                if term or trunc:
                    for p in (0, 1):
                        player = game.players[p]
                        infos.append({'episode': {'r': round(float(self.returns[index, p]), 6),
                                                  'l': int(self.lengths[index])},
                                      'combat': {'game': self.game_ids[index], 'player': p,
                                                 'episode_index': int(self.episodes[index]),
                                                 'damage_done': float(player.damage_done),
                                                 'damage_taken': float(player.damage_taken_total),
                                                 'stocks': int(player.stocks)}})
                    obs, _ = game.reset()
                    frame = self._observations(obs)
                    self.stack[pair, :192] = 0
                    self.returns[index] = 0
                    self.lengths[index] = 0
                    self.episodes[index] += 1
                else:
                    self.stack[pair, :192] = terminal[pair, :192]
                self.stack[pair, 192:] = frame
        return dict(observations=self.stack.copy(), terminal_observations=terminal,
                    rewards=rewards, terminated=terminated, truncated=truncated,
                    infos=infos, valid_steps=self.valid_steps.copy())

    def close(self):
        if self._closed:
            return
        self._closed = True
        py_state, np_state = random.getstate(), np.random.get_state()
        try:
            self._contexts.close()
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)


def _python_worker(connection, frozen_root, game_ids):
    """No policy construction, inference or updates; original game + arrays only."""
    try:
        with OriginalGameBatch(Path(frozen_root), game_ids) as batch:
            connection.send(('ok', None))
            while True:
                operation, payload = connection.recv()
                if operation == 'close':
                    break
                if operation not in ('reset', 'step'):
                    raise ValueError('unknown original-game worker operation')
                connection.send(('ok', getattr(batch, operation)(payload)))
    except BaseException:
        try:
            connection.send(('error', traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class PythonGameCPUEnv:
    """CPU tensor transport compatible with NativeLaneCPUAdapter, including spawn."""
    def __init__(self, games, frozen_root, *, seed=0, workers=1):
        import torch
        from ai2.simulation.native_lane_vec_env import frozen_spaces
        if type(games) is not int or games <= 0 or type(workers) is not int or not 1 <= workers <= games:
            raise ValueError('require positive games and workers <= games')
        self.device, self._torch = torch.device('cpu'), torch
        self.num_envs = games*2
        self.observation_space, self.action_space = frozen_spaces()
        self.backend = SimpleNamespace(kind='cpu', valid_steps=np.zeros(games, dtype=np.uint64),
                                       errors=np.zeros(games, dtype=np.int32))
        self._infos, self._processes, self._connections = [], [], []
        self._batch = None
        self._closed = False
        self._initialized = False
        self._groups = [group.tolist() for group in np.array_split(np.arange(games), workers)]
        self.seed([seed+i for i in range(self.num_envs)])
        try:
            if workers == 1:
                self._batch = OriginalGameBatch(Path(frozen_root), list(range(games)))
            else:
                context = mp.get_context('spawn')
                for group in self._groups:
                    parent, child = context.Pipe()
                    process = context.Process(target=_python_worker, args=(child, str(Path(frozen_root).resolve()), group))
                    process.start()
                    child.close()
                    self._processes.append(process)
                    self._connections.append(parent)
                for connection in self._connections:
                    self._receive(connection)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _receive(connection):
        if not connection.poll(180):
            raise RuntimeError('original-game worker timed out')
        try:
            status, result = connection.recv()
        except (EOFError, OSError) as exc:
            raise RuntimeError('original-game worker exited without a result') from exc
        if status != 'ok':
            raise RuntimeError(f'original-game worker failed:\n{result}')
        return result

    def seed(self, seeds):
        if len(seeds) != self.num_envs or any(type(v) is not int or not 0 <= v < 2**32 for v in seeds):
            raise ValueError('one uint32 seed per player slot is required')
        self._seeds = list(seeds)
        return list(seeds)

    def reset_tensor(self, seeds=None):
        if self._closed:
            raise RuntimeError('closed original-game vector')
        if seeds is not None:
            self.seed(seeds)
        game_seeds = self._seeds[::2]
        if self._batch is not None:
            obs = self._batch.reset(game_seeds)
        else:
            for connection, group in zip(self._connections, self._groups):
                connection.send(('reset', [game_seeds[i] for i in group]))
            obs = np.concatenate([self._receive(connection) for connection in self._connections])
        self._initialized = True
        return self._torch.from_numpy(obs)

    def step_tensor(self, actions):
        from ai2.training.tensor_ppo import TensorStep
        if self._closed or not self._initialized:
            raise RuntimeError('reset original-game vector before step')
        torch = self._torch
        if (not torch.is_tensor(actions) or actions.device.type != 'cpu' or actions.dtype != torch.float32
                or tuple(actions.shape) != (self.num_envs, 10) or actions.requires_grad):
            raise ValueError('require CPU float32 both-player actions without gradients')
        array = actions.detach().contiguous().numpy()
        if self._batch is not None:
            rows = [self._batch.step(array)]
        else:
            for connection, group in zip(self._connections, self._groups):
                connection.send(('step', array[2*group[0]:2*(group[-1]+1)]))
            rows = [self._receive(connection) for connection in self._connections]
        self.backend.valid_steps[:] = np.concatenate([row['valid_steps'] for row in rows])
        self._infos.extend(info for row in rows for info in row['infos'])
        return TensorStep(**{key: torch.from_numpy(np.concatenate([row[key] for row in rows]))
                             for key in ('observations', 'rewards', 'terminated', 'truncated', 'terminal_observations')})

    def drain_episode_infos(self):
        infos, self._infos = self._infos, []
        return infos

    def close(self):
        if self._closed:
            return
        self._closed = True
        for connection in self._connections:
            try:
                connection.send(('close', None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self._processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        for connection in self._connections:
            connection.close()
        if self._batch is not None:
            self._batch.close()


def build_backend(name, config, frozen_root, device):
    from ai2.simulation.native_game_lane_vec_env import build_game_env
    from ai2.simulation.native_lane_cpu_adapter import NativeLaneCPUAdapter
    from ai2.training.tensor_ppo import TensorVecEnv
    if name not in BACKENDS:
        raise ValueError('unknown benchmark backend')
    if name == 'python':
        raw = PythonGameCPUEnv(config.games, frozen_root, seed=config.seed, workers=config.python_workers)
    else:
        # Worst case: each game ends on every collected tick; no record loss.
        raw = build_game_env(config.games, device=device if name == 'native_cuda' else 'cpu',
                             seed=config.seed, record_capacity=config.game_ticks_per_rollout)
    try:
        wrapped = raw if name == 'native_cuda' else NativeLaneCPUAdapter(raw, device=device)
        return TensorVecEnv(wrapped, device=device), raw.backend
    except BaseException:
        raw.close()
        raise


LOSS_KEYS = ('train/loss', 'train/policy_gradient_loss', 'train/value_loss', 'train/entropy_loss')


def loss_metrics(values):
    if any(key not in values or not math.isfinite(float(values[key])) for key in LOSS_KEYS):
        raise RuntimeError('PPO did not report all required finite losses')
    return {key: float(value) if math.isfinite(float(value)) else None
            for key, value in values.items() if key.startswith('train/') and isinstance(value, (int, float, np.number))}


def _rates(row):
    for basis in ('collection', 'total'):
        seconds = row[f'{basis}_seconds']
        if not math.isfinite(seconds) or seconds <= 0:
            raise RuntimeError('invalid benchmark elapsed time')
        row[f'{basis}_game_ticks_per_second'] = row['game_ticks']/seconds
        row[f'{basis}_player_transitions_per_second'] = row['player_transitions']/seconds
    return row


def measure_rollout(model, callback, backend, config, *, synchronize):
    before = np.array(backend.valid_steps, copy=True)
    synchronize()
    started_unix = time.time()
    start = time.perf_counter()
    collected = model.collect_rollouts(model.env, callback, model.rollout_buffer, config.n_steps)
    synchronize()
    collection_end = time.perf_counter()
    if not collected:
        raise RuntimeError('rollout cancelled; incomplete work is not a timing result')
    # Drain occurs inside collect_rollouts and refreshes CUDA valid_steps/errors.
    errors = np.asarray(backend.errors).astype(int).tolist()
    delta = np.asarray(backend.valid_steps) - before
    if any(errors) or delta.shape != (config.games,) or not np.all(delta == config.n_steps):
        raise RuntimeError(f'invalid game ticks: delta={delta.tolist()}, errors={errors}')
    model._update_current_progress_remaining(model.num_timesteps, model._total_timesteps)
    model.train()
    synchronize()
    end = time.perf_counter()
    finished_unix = time.time()
    ticks = sum(int(value) for value in delta)
    return _rates(dict(game_ticks=ticks, player_transitions=2*ticks,
                       started_unix=started_unix, finished_unix=finished_unix,
                       collection_seconds=collection_end-start, update_seconds=end-collection_end,
                       total_seconds=end-start, losses_finite=True,
                       metrics=loss_metrics(model.logger.name_to_value), errors=errors,
                       completed_player_episodes=len(callback.rollout_infos),
                       completed_game_episodes=len({(v['combat']['game'], v['combat']['episode_index'])
                                                    for v in callback.rollout_infos})))


def summarize_rollouts(rows):
    measured = [row for row in rows if row['measured']]
    if not measured:
        raise RuntimeError('no measured rollouts')
    return _rates({key: sum(row[key] for row in measured) for key in
                   ('game_ticks', 'player_transitions', 'collection_seconds', 'update_seconds', 'total_seconds')})


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    os.replace(temporary, path)


def _append_json(path, value):
    with Path(path).open('a') as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False)+'\n')


def policy_receipt(policy, device):
    """Hash bytes outside timed windows; every parameter remains CUDA-resident."""
    import torch
    digest, count = hashlib.sha256(), 0
    for name, parameter in policy.named_parameters():
        if parameter.device != device or not parameter.is_cuda:
            raise RuntimeError('policy parameter escaped required CUDA device')
        if not bool(torch.isfinite(parameter).all().item()):
            raise RuntimeError('nonfinite policy parameter')
        count += parameter.numel()
        digest.update(name.encode()+str(tuple(parameter.shape)).encode())
        digest.update(parameter.detach().cpu().numpy().tobytes())
    if count == 0:
        raise RuntimeError('empty policy')
    return dict(sha256=digest.hexdigest(), parameter_count=count)


def _make_callback():
    from stable_baselines3.common.callbacks import BaseCallback
    class Records(BaseCallback):
        rollout_infos = None
        def _on_rollout_start(self):
            self.rollout_infos = []
        def _on_step(self):
            return True
        def _on_rollout_end(self):
            self.rollout_infos = list(self.locals.get('drained_infos', []))
    return Records()


def make_benchmark_ppo_class(config):
    """Keep the baseline class exact; opt in only to the source-pinned update."""
    from ai2.training.tensor_ppo import TensorPPO
    if config.update_mode == 'sb3':
        return TensorPPO
    from ai2.training.ppo_update_optimizations import SB3_VERSION, sb3_train_source_metadata
    metadata = sb3_train_source_metadata()
    if not metadata['supported']:
        raise RuntimeError(f'deferred PPO update requires the validated SB3 {SB3_VERSION} train source; got {metadata}')
    from ai2.training.experimental_native_lane_training import OptimizationOptions, make_ppo_class
    return make_ppo_class(OptimizationOptions(deferred_metrics=True))


def run_backend(name, config, output, frozen_root, device):
    import torch
    from stable_baselines3.common.logger import configure
    ppo_class = make_benchmark_ppo_class(config)
    env, backend = build_backend(name, config, frozen_root, device)
    logger = None
    callback = None
    try:
        if env.num_envs != config.games*2 or env.obs_shape != (256,) or type(env.action_space).__name__ != 'Box':
            raise RuntimeError('backend does not implement the matched two-player stacked Box workload')
        model = ppo_class('MlpPolicy', env, device=device, seed=config.seed,
                          n_steps=config.n_steps, batch_size=config.batch_size, n_epochs=config.n_epochs,
                          learning_rate=config.learning_rate, gamma=config.gamma, gae_lambda=config.gae_lambda,
                          clip_range=config.clip_range, ent_coef=config.ent_coef, vf_coef=config.vf_coef,
                          max_grad_norm=config.max_grad_norm, normalize_advantage=True, target_kl=None,
                          policy_kwargs={'net_arch': {'pi': list(config.net_arch), 'vf': list(config.net_arch)},
                                         'activation_fn': torch.nn.Tanh}, verbose=0)
        initial = policy_receipt(model.policy, device)
        logger = configure(str(output), ['json'])
        model.set_logger(logger)
        total = config.player_transitions_per_rollout*(config.warmup_rollouts+config.measured_rollouts)
        _, callback = model._setup_learn(total, callback=_make_callback())
        callback.on_training_start(locals(), globals())
        rows = []
        for index in range(config.warmup_rollouts+config.measured_rollouts):
            try:
                row = measure_rollout(model, callback, backend, config,
                                      synchronize=lambda: torch.cuda.synchronize(device))
            except BaseException as exc:
                _append_json(output/'rollouts.jsonl', dict(rollout=index, status='failed', error=str(exc),
                             measured=index >= config.warmup_rollouts,
                             valid_game_ticks=np.asarray(backend.valid_steps).tolist(),
                             errors=np.asarray(backend.errors).tolist()))
                raise
            row.update(rollout=index, measured=index >= config.warmup_rollouts,
                       policy_player_timesteps=model.num_timesteps)
            rows.append(row)
            _append_json(output/'rollouts.jsonl', row)
            for info in callback.rollout_infos:
                _append_json(output/'episodes.jsonl', {'rollout': index, **info})
            model.logger.dump(step=model.num_timesteps)
            message = json.dumps({'backend': name, **row}, sort_keys=True)
            with (output/'benchmark.log').open('a') as handle:
                handle.write(message+'\n')
            print(message, flush=True)
        callback.on_training_end()
        final = policy_receipt(model.policy, device)
        if final['sha256'] == initial['sha256']:
            raise RuntimeError('policy parameters did not change; training not established')
        checkpoint = output/'policy_final.zip'
        model.save(checkpoint)
        return dict(backend=name, status='passed', summary=summarize_rollouts(rows), rollouts=rows,
                    initial_policy=initial, final_policy=final, parameters_changed=True,
                    policy_architecture=str(model.policy), checkpoint=str(checkpoint), checkpoint_sha256=_sha(checkpoint),
                    python_workers=config.python_workers if name == 'python' else 0,
                    baseline_scope=('parallel Python/Pymunk' if config.python_workers > 1 else 'serial diagnostic Python/Pymunk')
                                   if name == 'python' else name,
                    measured_work='collection includes game, policy, transfers, GAE, episode drain; update includes PPO and sync')
    finally:
        if callback is not None:
            # BaseCallback retains collect_rollouts' locals, including itself,
            # the model and CUDA buffers. Detach, never clear aliased globals().
            callback.locals, callback.globals, callback.model = {}, {}, None
        try:
            if logger is not None:
                logger.close()
        finally:
            env.close()


def provenance(config, frozen_root, device):
    import torch
    from ai2.evaluation.combat_episode_capture import FROZEN_SHA
    from ai2.simulation.native_attack_tables import build_tables
    from ai2.training.ppo_update_optimizations import SB3_VERSION, SB3_TRAIN_SHA256, sb3_train_source_metadata
    source = Path(frozen_root)/'UTMIST-AI2-main/environment/environment.py'
    if _sha(source) != FROZEN_SHA:
        raise ValueError('frozen environment source hash mismatch')
    tables = build_tables(source.parent)
    candidates = {name: source_path(ROOT, name) for name in ('benchmark_native_combat_training.py', 'native_game_lane_vec_env.py',
                  'native_lane_vec_env.py', 'native_lane_cpu_adapter.py', 'tensor_ppo.py', 'tensor_rollout_buffer.py',
                  'native_game_episode.py', 'combat_episode_capture.py', 'native_attack_tables.py',
                  'experimental_native_lane_training.py', 'ppo_update_optimizations.py',
                  'native/navigation_spaces.json')}
    candidates.update({str(path.relative_to(ROOT)): path for path in
                       sorted((ROOT/'native').glob('*.h'))+sorted((ROOT/'native').glob('game_*.c*'))})
    return dict(config=asdict(config), environment_sha256=FROZEN_SHA, attack_assets_sha256=tables['asset_sha256'],
                sources={name: _sha(path) for name, path in candidates.items()},
                ppo_update=dict(mode=config.update_mode,
                    training_algorithm=('installed stable_baselines3.PPO.train' if config.update_mode == 'sb3'
                                        else 'SB3-compatible deferred-metrics PPO; eager loss'),
                    source_guard=dict(required=config.update_mode == 'deferred_metrics',
                                      expected_version=SB3_VERSION, expected_train_sha256=SB3_TRAIN_SHA256,
                                      **sb3_train_source_metadata())),
                system=platform.platform(), python=platform.python_version(), cpu_count=os.cpu_count(),
                cpu_model=_linux_cpu_model(),
                cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                packages={name: importlib.metadata.version(name) for name in ('torch', 'stable_baselines3', 'numpy', 'pymunk', 'pygame')},
                cuda_version=torch.version.cuda, gpu=torch.cuda.get_device_name(device), policy_device=str(device),
                torch_cpu_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                workload='raw game rewards, both players jointly train shared Gaussian MLP; Box10, four-frame float32[256]',
                player_order='game0/player0, game0/player1, game1/player0, game1/player1, ...',
                game_seed_rule='seed+2*game_index; Python and native generators differ; no trajectory-equivalence claim',
                exclusions='startup/build/initial reset/model initialization, logging, parameter hashing, checkpoint serialization excluded from timing; episode autoresets are timed',
                limitation='infrastructure only; not curriculum training or evidence of combat mastery')


def run_benchmark(config: BenchmarkConfig, output: Path, frozen_root: Path):
    output, frozen_root = Path(output).resolve(), Path(frozen_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('output must be new or empty; existing receipts are not overwritten')
    output.mkdir(parents=True, exist_ok=True)
    result = dict(schema=SCHEMA, status='running', config=asdict(config), backends=[])
    _write_json(output/'result.json', result)
    try:
        device = require_cuda(config.device)
        result['provenance'] = provenance(config, frozen_root, device)
        for name in config.backends:
            directory = output/name
            directory.mkdir()
            try:
                row = run_backend(name, config, directory, frozen_root, device)
            except BaseException as exc:
                _write_json(directory/'failure.json', {'status': 'failed', 'error': str(exc)})
                raise
            finally:
                # Release learner/env cycles before constructing the next lane;
                # collection timing must not inherit another backend's cleanup.
                gc.collect()
            if result['backends'] and row['initial_policy'] != result['backends'][0]['initial_policy']:
                raise RuntimeError('backend initial policy parameters differ: unmatched seeded workload')
            result['backends'].append(row)
            _write_json(output/'result.json', result)
        result['status'] = 'passed'
        _write_json(output/'result.json', result)
        for backend in result['backends']:
            _write_json(output/backend['backend']/'training-benchmark-contract.json',
                        ai2_native_receipt(result, backend['backend']))
        return result
    except BaseException as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        _write_json(output/'result.json', result)
        (output/'failure.log').write_text(traceback.format_exc())
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backends', nargs='+', choices=BACKENDS, default=list(BACKENDS))
    for name, default in (('games', 8), ('n-steps', 128), ('batch-size', 256), ('n-epochs', 4),
                          ('warmup-rollouts', 1), ('measured-rollouts', 3), ('seed', 0), ('python-workers', 1)):
        parser.add_argument('--'+name, type=int, default=default)
    parser.add_argument('--net-arch', type=int, nargs='+', default=[256, 256])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--update-mode', choices=UPDATE_MODES, default='sb3')
    args = vars(parser.parse_args(argv))
    output, frozen_root = args.pop('output'), args.pop('frozen_root')
    args['net_arch'], args['backends'] = tuple(args['net_arch']), tuple(args['backends'])
    run_benchmark(BenchmarkConfig(**args), output, frozen_root)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
