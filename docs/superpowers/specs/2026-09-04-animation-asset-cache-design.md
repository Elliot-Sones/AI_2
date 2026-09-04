# Animation asset reuse

User-approved scope: load animation assets once and reuse them, preserving
gameplay timing, then repeat the performance baseline. Work in the current
workspace; preserve existing benchmark artifacts and unrelated edits.

Chosen design: a process-local bounded LRU of decoded animation frames and GIF
durations, keyed by canonical file path, file modification nanoseconds, and
file size. Capacity 128 exceeds the current animation corpus. Spawned workers
have independent caches. Entries persist across episode resets but are bounded.
Changed file fingerprints cause a reload; failed decodes must not be cached.

Each load returns a distinct Animation object with separate lists, reusing only
the source Pygame surfaces. Existing renderers transform copies, not originals.
Playback counters stay on AnimationSprite2D. Compute frames_per_step with the
existing max(1, round(duration / 1000 * ENV_FPS)) formula for each instance;
retain all VFX objects and weapon pickup gates. Do not cache mutable sprite
instances or remove graphics-related gameplay metadata.

Rejected: removing VFX in headless mode, because animation duration controls
pickup eligibility. Rejected: a folder cache or global shared-memory service,
because file-level decode reuse fixes the measured cost with less state.

Success: fewer real image decodes on repeated loads/resets; identical pixels,
durations, frame advance and pickup timing; no policy/reward/config changes;
repeat benchmarks with commands, hashes, tests, and caveats recorded.

Cloud: reuse instance 49902188 only, preserve the original remote run, upload
optimized code into a new run directory. Maximum 30 minutes of additional
compute at the currently quoted $0.3167/hour, approximately $0.16. Stop earlier
when results are retrieved and verified. Never destroy/delete without the
user's explicit confirmation. Stopped storage continues charging.
