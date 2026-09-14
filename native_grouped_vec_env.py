"""Opt-in grouped CPU game workers using the native game hotpath patch.

This is deliberately compatible with benchmark_campaign --grouped-env-path.
Native handles are constructed only inside workers, never pickled or inherited
from a parent CUDA context. Collision solving remains original Pymunk.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any

try:
    from grouped_vec_env import GroupedSubprocVecEnv as _GroupedSubprocVecEnv
except ModuleNotFoundError:
    _GroupedSubprocVecEnv = object


@dataclass(frozen=True)
class NativeFactory:
    factory: Callable[[], Any]
    source_path: Path

    def __call__(self):
        from native_game_patch import install_native_game_patch
        env = self.factory()
        try:
            active = install_native_game_patch(env, source_path=self.source_path)
        except BaseException:
            env.close()
            raise
        original_close = env.close

        def close(*args, **kwargs):
            try:
                return original_close(*args, **kwargs)
            finally:
                if active is not None:
                    active.restore()

        env.close = close
        return env


def native_factory(factory, *, source_path):
    return NativeFactory(factory, Path(source_path))


class GroupedSubprocVecEnv(_GroupedSubprocVecEnv):
    def __init__(self, env_fns, *, group_size=4, start_method=None, native_source_path=None):
        if _GroupedSubprocVecEnv is object:
            raise ImportError("grouped native game workers require the installed RL runtime")
        source = Path(native_source_path) if native_source_path is not None else Path.cwd() / "UTMIST-AI2-main/environment/environment.py"
        from native_game_patch import verify_source_hash
        verify_source_hash(source)
        super().__init__(
            [native_factory(fn, source_path=source.resolve()) for fn in env_fns],
            group_size=group_size,
            start_method=start_method,
        )
