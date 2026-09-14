"""Native game patch plus isolated per-lane RNG grouped worker.

This module is a small composition helper for ``benchmark_campaign
--grouped-env-path``.  It wraps each original env factory with
``native_grouped_vec_env.NativeFactory`` first, then constructs the opt-in
isolated RNG grouped worker.  It intentionally avoids cooperative inheritance:
``isolated_rng_vec_env.GroupedSubprocVecEnv`` returns the base grouped object
from ``__new__``, so an MRO composition would not reliably apply native
factory wrapping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class GroupedSubprocVecEnv:
    def __new__(
        cls,
        env_fns: Any,
        *,
        group_size: int = 4,
        start_method: str | None = None,
        native_source_path: str | Path | None = None,
    ) -> Any:
        import isolated_rng_vec_env
        import native_grouped_vec_env
        from native_game_patch import verify_source_hash

        source = (
            Path(native_source_path)
            if native_source_path is not None
            else Path.cwd() / "UTMIST-AI2-main" / "environment" / "environment.py"
        )
        verify_source_hash(source)
        resolved = source.resolve()
        wrapped_fns = [native_grouped_vec_env.native_factory(fn, source_path=resolved) for fn in env_fns]
        return isolated_rng_vec_env.GroupedSubprocVecEnv(
            wrapped_fns,
            group_size=group_size,
            start_method=start_method,
        )

