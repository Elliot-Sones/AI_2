"""Canonical source locations, with explicit support for frozen flat checkouts.

Logical legacy filenames remain receipt keys. Resolve them within the supplied
checkout; never substitute live code for a missing frozen experiment source.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_LAYOUT_VERSION = 2
SOURCE_LOCATIONS = {'audit_navigation_targets.py': 'tools/validation/audit_navigation_targets.py',
    'batched_rng_vec_env.py': 'ai2/simulation/batched_rng_vec_env.py',
    'benchmark_campaign.py': 'ai2/runtime/benchmark_campaign.py',
    'benchmark_contact_fixtures.py': 'tools/benchmarks/benchmark_contact_fixtures.py',
    'benchmark_curriculum_throughput.py': 'ai2/training/benchmark_curriculum_throughput.py',
    'benchmark_diagnostics.py': 'tools/benchmarks/benchmark_diagnostics.py',
    'benchmark_graph_rollout_parity.py': 'ai2/evaluation/benchmark_graph_rollout_parity.py',
    'benchmark_native_combat_training.py': 'ai2/training/benchmark_native_combat_training.py',
    'benchmark_native_contact_training.py': 'ai2/training/benchmark_native_contact_training.py',
    'benchmark_native_lane_training.py': 'ai2/training/benchmark_native_lane_training.py',
    'benchmark_policy_capacity.py': 'tools/benchmarks/benchmark_policy_capacity.py',
    'benchmark_policy_compile.py': 'ai2/training/benchmark_policy_compile.py',
    'benchmark_policy_compile_components.py': 'tools/benchmarks/benchmark_policy_compile_components.py',
    'benchmark_policy_cuda_graph.py': 'ai2/training/benchmark_policy_cuda_graph.py',
    'benchmark_policy_rollout_parity.py': 'tools/benchmarks/benchmark_policy_rollout_parity.py',
    'benchmark_replay.py': 'ai2/evaluation/benchmark_replay.py',
    'benchmark_tensor_rollout_parity.py': 'ai2/evaluation/benchmark_tensor_rollout_parity.py',
    'benchmark_tensor_training.py': 'tools/benchmarks/benchmark_tensor_training.py',
    'benchmark_training.py': 'ai2/evaluation/benchmark_training.py',
    'benchmark_transport.py': 'tools/benchmarks/benchmark_transport.py',
    'benchmark_worker_profile.py': 'tools/benchmarks/benchmark_worker_profile.py',
    'campaign_candidate_pairs.py': 'baselines/throughput-scaling-20260906/scripts/campaign_candidate_pairs.py',
    'campaign_candidate_scaling.py': 'baselines/throughput-scaling-20260906/scripts/campaign_candidate_scaling.py',
    'campaign_cloud.py': 'baselines/native-ppo-session10/scripts/campaign_cloud.py',
    'campaign_corrected_scaling.py': 'baselines/throughput-scaling-20260906/scripts/campaign_corrected_scaling.py',
    'campaign_graph_pairs.py': 'baselines/throughput-scaling-20260906/scripts/campaign_graph_pairs.py',
    'campaign_grouped_pairs.py': 'baselines/throughput-scaling-20260906/scripts/campaign_grouped_pairs.py',
    'campaign_initial_eval.py': 'baselines/throughput-scaling-20260906/scripts/campaign_initial_eval.py',
    'campaign_learning.py': 'ai2/runtime/campaign_learning.py',
    'campaign_native_lanes.py': 'ai2/runtime/campaign_native_lanes.py',
    'campaign_native_ppo_ablations.py': 'ai2/runtime/campaign_native_ppo_ablations.py',
    'campaign_suite.py': 'ai2/runtime/campaign_suite.py',
    'causal_retention_campaign.py': 'ai2/runtime/causal_retention_campaign.py',
    'checkpoint_hardlinks.py': 'tools/analysis/checkpoint_hardlinks.py',
    'classic_real_semantic_smoke.py': 'tools/validation/classic_real_semantic_smoke.py',
    'combat_episode_capture.py': 'ai2/evaluation/combat_episode_capture.py',
    'combat_episode_parity.py': 'ai2/evaluation/combat_episode_parity.py',
    'compare_training_checkpoints.py': 'ai2/evaluation/compare_training_checkpoints.py',
    'continue_contact_campaign.py': 'ai2/runtime/continue_contact_campaign.py',
    'continue_reengagement_campaign.py': 'ai2/runtime/continue_reengagement_campaign.py',
    'continue_retention_campaign.py': 'ai2/runtime/continue_retention_campaign.py',
    'cuda_mlp_graph.py': 'ai2/training/cuda_mlp_graph.py',
    'curriculum_ppo.py': 'ai2/training/curriculum_ppo.py',
    'evaluate_fullgame_bank.py': 'ai2/evaluation/evaluate_fullgame_bank.py',
    'evaluate_native_contact_training.py': 'ai2/evaluation/evaluate_native_contact_training.py',
    'evaluate_reengagement_curriculum.py': 'ai2/evaluation/evaluate_reengagement_curriculum.py',
    'evaluate_retention_campaign.py': 'ai2/evaluation/evaluate_retention_campaign.py',
    'experimental_native_lane_training.py': 'ai2/training/experimental_native_lane_training.py',
    'explore_positions.py': 'tools/visualization/explore_positions.py',
    'fixed_action_trace_parity.py': 'tools/validation/fixed_action_trace_parity.py',
    'grouped_vec_env.py': 'ai2/simulation/grouped_vec_env.py',
    'isolated_rng_vec_env.py': 'ai2/simulation/isolated_rng_vec_env.py',
    'kl_diagnostics_runner.py': 'tools/analysis/kl_diagnostics_runner.py',
    'learning_runner.py': 'ai2/training/learning_runner.py',
    'monitor.py': 'tools/analysis/monitor.py',
    'native_attack_tables.py': 'ai2/simulation/native_attack_tables.py',
    'native_batched_vec_env.py': 'ai2/simulation/native_batched_vec_env.py',
    'native_capacity_campaign.py': 'baselines/native-capacity-session11/scripts/native_capacity_campaign.py',
    'native_chipmunk_bridge.py': 'ai2/simulation/native_chipmunk_bridge.py',
    'native_combat_reward_gate.py': 'ai2/simulation/native_combat_reward_gate.py',
    'native_contact_episode.py': 'ai2/simulation/native_contact_episode.py',
    'native_contact_lane_vec_env.py': 'ai2/simulation/native_contact_lane_vec_env.py',
    'native_curriculum_ppo.py': 'ai2/training/native_curriculum_ppo.py',
    'native_curriculum_runner.py': 'ai2/training/native_curriculum_runner.py',
    'native_fullgame_action_table.py': 'ai2/evaluation/native_fullgame_action_table.py',
    'native_game_episode.py': 'ai2/simulation/native_game_episode.py',
    'native_game_lane_vec_env.py': 'ai2/simulation/native_game_lane_vec_env.py',
    'native_game_patch.py': 'ai2/simulation/native_game_patch.py',
    'native_game_qualification.py': 'ai2/evaluation/native_game_qualification.py',
    'native_game_snapshot.py': 'ai2/evaluation/native_game_snapshot.py',
    'native_grouped_vec_env.py': 'ai2/simulation/native_grouped_vec_env.py',
    'native_isolated_vec_env.py': 'ai2/simulation/native_isolated_vec_env.py',
    'native_lane_cpu_adapter.py': 'ai2/simulation/native_lane_cpu_adapter.py',
    'native_lane_vec_env.py': 'ai2/simulation/native_lane_vec_env.py',
    'native_learning_runner.py': 'ai2/training/native_learning_runner.py',
    'native_live_input.py': 'ai2/simulation/native_live_input.py',
    'native_live_observation.py': 'ai2/simulation/native_live_observation.py',
    'native_navigation_contact.py': 'ai2/simulation/native_navigation_contact.py',
    'native_navigation_contract.py': 'ai2/simulation/native_navigation_contract.py',
    'native_navigation_curriculum_env.py': 'ai2/simulation/native_navigation_curriculum_env.py',
    'native_navigation_episode.py': 'ai2/simulation/native_navigation_episode.py',
    'native_navigation_geometry.py': 'ai2/simulation/native_navigation_geometry.py',
    'native_navigation_movement.py': 'ai2/simulation/native_navigation_movement.py',
    'native_navigation_observation.py': 'ai2/simulation/native_navigation_observation.py',
    'native_navigation_physics.py': 'ai2/simulation/native_navigation_physics.py',
    'native_navigation_platform.py': 'ai2/simulation/native_navigation_platform.py',
    'native_reengagement_episode.py': 'ai2/simulation/native_reengagement_episode.py',
    'native_reengagement_lane_vec_env.py': 'ai2/simulation/native_reengagement_lane_vec_env.py',
    'native_shared_batched_vec_env.py': 'ai2/simulation/native_shared_batched_vec_env.py',
    'navigation_contact_capture.py': 'ai2/evaluation/navigation_contact_capture.py',
    'navigation_contract_capture.py': 'ai2/evaluation/navigation_contract_capture.py',
    'navigation_episode_capture.py': 'ai2/evaluation/navigation_episode_capture.py',
    'navigation_movement_capture.py': 'ai2/evaluation/navigation_movement_capture.py',
    'navigation_physics_capture.py': 'ai2/evaluation/navigation_physics_capture.py',
    'navigation_positions_header.py': 'tools/validation/navigation_positions_header.py',
    'play_vs_agent.py': 'tools/visualization/play_vs_agent.py',
    'ppo_optimizer_counters.py': 'ai2/training/ppo_optimizer_counters.py',
    'ppo_update_optimizations.py': 'ai2/training/ppo_update_optimizations.py',
    'profile_native_combat_training.py': 'tools/profiling/profile_native_combat_training.py',
    'profile_native_lane_training.py': 'tools/profiling/profile_native_lane_training.py',
    'profile_native_ppo_optimizations.py': 'tools/profiling/profile_native_ppo_optimizations.py',
    'qualify_native_contact_training.py': 'tools/validation/qualify_native_contact_training.py',
    'qualify_native_lanes.py': 'tools/validation/qualify_native_lanes.py',
    'qualify_native_ppo_optimizations.py': 'tools/validation/qualify_native_ppo_optimizations.py',
    'qualify_native_reengagement.py': 'tools/validation/qualify_native_reengagement.py',
    'qualify_navigation_curriculum.py': 'tools/validation/qualify_navigation_curriculum.py',
    'record_campaign_navigation.py': 'tools/visualization/record_campaign_navigation.py',
    'reengagement_curriculum.py': 'ai2/evaluation/reengagement_curriculum.py',
    'rollout_cache.py': 'ai2/training/rollout_cache.py',
    'rollout_policy_graph.py': 'ai2/training/rollout_policy_graph.py',
    'run_campaign_job.py': 'ai2/runtime/run_campaign_job.py',
    'run_retention_cuda_qualification.py': 'tools/validation/run_retention_cuda_qualification.py',
    'session10_capture_external_logs.py': 'ai2/runtime/session10_capture_external_logs.py',
    'session10_cloud_start.py': 'baselines/native-ppo-session10/scripts/session10_cloud_start.py',
    'session10_evaluate_optimizations.py': 'baselines/native-ppo-session10/scripts/session10_evaluate_optimizations.py',
    'session10_monitor.py': 'baselines/native-ppo-session10/scripts/session10_monitor.py',
    'session10_profile_analysis.py': 'tools/analysis/session10_profile_analysis.py',
    'session10_retrieval_handoff.py': 'baselines/native-ppo-session10/scripts/session10_retrieval_handoff.py',
    'session10_summarize_ablations.py': 'tools/analysis/session10_summarize_ablations.py',
    'session11_capture_external_logs.py': 'baselines/native-capacity-session11/scripts/session11_capture_external_logs.py',
    'session11_monitor.py': 'baselines/native-capacity-session11/scripts/session11_monitor.py',
    'shared_action_vec_env.py': 'ai2/simulation/shared_action_vec_env.py',
    'summarize_campaign.py': 'tools/analysis/summarize_campaign.py',
    'summarize_learning.py': 'tools/analysis/summarize_learning.py',
    'summarize_native_capacity.py': 'tools/analysis/summarize_native_capacity.py',
    'tensor_cpu_vec_env.py': 'ai2/simulation/tensor_cpu_vec_env.py',
    'tensor_frame_stack.py': 'ai2/training/tensor_frame_stack.py',
    'tensor_ppo.py': 'ai2/training/tensor_ppo.py',
    'tensor_rollout_buffer.py': 'ai2/training/tensor_rollout_buffer.py',
    'tracking.py': 'ai2/runtime/tracking.py',
    'train.py': 'ai2/training/train.py',
    'train_contact_campaign_block.py': 'ai2/training/train_contact_campaign_block.py',
    'train_reengagement_campaign_block.py': 'ai2/training/train_reengagement_campaign_block.py',
    'train_retention_campaign_block.py': 'ai2/training/train_retention_campaign_block.py',
    'training_benchmark_contract.py': 'tools/validation/training_benchmark_contract.py',
    'validate_grouped_candidate.py': 'tools/validation/validate_grouped_candidate.py',
    'validate_native_combat_checkpoint.py': 'tools/validation/validate_native_combat_checkpoint.py',
    'validate_native_game_patch.py': 'tools/validation/validate_native_game_patch.py',
    'validate_native_navigation.py': 'tools/validation/validate_native_navigation.py',
    'validate_production_action_cuda_gate.py': 'tools/validation/validate_production_action_cuda_gate.py',
    'visualize_positions.py': 'tools/visualization/visualize_positions.py',
    'watch_games.py': 'tools/visualization/watch_games.py',
    'watch_navigation.py': 'tools/visualization/watch_navigation.py'}


def source_path(root: Path | str, filename: Path | str) -> Path:
    """Resolve a legacy logical filename inside its own source checkout."""
    root = Path(root)
    relative = str(filename)
    if (root / "ai2/paths.py").is_file():
        relative = SOURCE_LOCATIONS.get(relative, relative)
    return root / relative


def source_inventory(root: Path | str) -> dict[str, Path]:
    """Return every required Python input, failing closed for incomplete trees.

    Old flat snapshots retain their old inventory. New layouts also hash package
    initializers, path resolution, and checkpoint compatibility entry points.
    """
    root = Path(root)
    if not (root / "ai2/paths.py").is_file():
        return {p.name: p for p in sorted(root.glob("*.py"))}
    relative_paths = set(SOURCE_LOCATIONS.values())
    relative_paths.update({"train.py", "curriculum_ppo.py", "ai2/paths.py"})
    for filename in SOURCE_LOCATIONS.values():
        path = Path(filename)
        if path.parts[0] in {"ai2", "tools"}:
            for parent in path.parents:
                if parent != Path("."):
                    relative_paths.add(str(parent / "__init__.py"))
    for directory in ("ai2", "tools"):
        relative_paths.add(f"{directory}/__init__.py")
        relative_paths.update(str(p.relative_to(root)) for p in (root / directory).rglob("*.py"))
    inventory = {name: root / name for name in sorted(relative_paths)}
    missing = [name for name, path in inventory.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete source layout {SOURCE_LAYOUT_VERSION}: {missing}")
    return inventory
