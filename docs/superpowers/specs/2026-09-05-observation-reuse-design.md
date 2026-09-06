# Reuse player observations within a game transition

The user approved removing duplicate observation calculations and testing the
change locally before considering a C rewrite. Preserve existing unrelated
edits and keep all work local; this task does not launch cloud instances.

`WarehouseBrawl.step` and `reset` currently call `observe` once per agent.
Each call builds both players' 32-value lists, so returning two observations
builds four player lists. `Player.get_obs` reads positions, velocities, floor
bounds, combat state, weapons, spawners, and platform state. The repeated
floor bounding-box cache refresh is idempotent at this boundary.

Introduce a private `_observe_all` helper that computes the two player lists
once, then returns independently owned NumPy arrays ordered player/opponent
for each agent. Use it only at the existing step/reset observation boundary.
Keep `observe(agent)` fresh on every call, with its existing signature and
values. Reuse lasts only within one helper call; no persistent state cache or
invalidation mechanism is needed. There are no local subclasses overriding
`WarehouseBrawl.observe`; step/reset will use the new batch helper directly.

Preserve dtype, shape, ordering, action thresholds, rewards, physics, resets,
episode rules, and training settings. Do not change opponent generation,
observation contents, the inactive alternate WarehouseBrawl file, or PPO.

Success requires two real `Player.get_obs` calls per step/reset instead of
four, exact observations and independent arrays, fresh reads after direct
state changes, unchanged termination/truncation behavior, matching old/new
fixed-action replay hashes and natural episode outcomes, and isolated local
before/after throughput measurements. Report measured speed and limitations;
neither a CPU speedup nor identical short traces establish learning quality.
