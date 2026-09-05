# Interactive training-flow explainer

Open `index.html` directly in a browser; the page is self-contained and needs
no build, server, dependencies, network calls, cloud instance or API keys.

- Drag the timeline or use Previous / Next / Play.
- **One round** reconstructs the sequence using measured stage averages.
- **Episode → learning** shows the lifecycle on an event axis, not fabricated
  wall-clock timestamps. An episode ending does not trigger a PPO update.
- Expand **Inside a CPU worker** to inspect calculations and worker timings.
- Expand the evidence section for provenance, denominators and uncertainty.

Timing source: `baselines/pipeline-profile-20260904/artifacts/natural_16env/pipeline_profile.json`.
The round axis omits startup, rollout-end work, learning updates and between-rollout
setup. Its 8.218938 ms average is not the earlier 9.206 ms all-cost-amortized round.
GPU utilization, kernel-only timing, PCIe bandwidth and a full episode timestamp
trace were not measured. Data sizes are logical float32 payloads calculated from
shapes, not observed physical traffic. Source links work when opened directly
or when the repository root is served locally.

Pure timing/content checks, without installing anything:

```sh
node --test tests/test_training_flow.cjs
```

UI verification also exercises timeline clicks, keyboard/pointer scrubbing,
play/pause, all 25 stage selections, expandable calculations and 360/736/1440 px
layouts in Chromium. Screenshots are under `output/playwright/` when captured.
The illustrative hardware highlights are not GPU-utilization measurements.
