import importlib.util
import json
import sys
import textwrap
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "benchmark_campaign.py"


def load_campaign():
    spec = importlib.util.spec_from_file_location("benchmark_campaign", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_campaign"] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def preserved_import_state(*module_names):
    original_modules = {name: sys.modules.get(name) for name in module_names}
    missing = {name for name in module_names if name not in sys.modules}
    original_path = list(sys.path)
    try:
        yield
    finally:
        sys.path[:] = original_path
        for name in module_names:
            if name in missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original_modules[name]


class BenchmarkCampaignTests(unittest.TestCase):
    def test_profile_summary_labels_actual_timing_mode(self):
        campaign = load_campaign()
        plan = campaign.plan_run(n_envs=1, n_steps=2, batch_size=2,
                                 epochs=1, warmup=0, cycles=1)
        for timing, expected in [('natural', 'natural host spans'),
                                 ('sync', 'synchronized update stages')]:
            with self.subTest(timing=timing):
                state = campaign.CampaignState(output=Path('/tmp'), plan=plan,
                    sync=lambda: None, mode='profile', profile_timing=timing)
                self.assertIn(expected, state.summary(status='completed')['profile_timing'])

    def test_instrumentation_state_is_not_serialized_into_policy_checkpoints(self):
        campaign = load_campaign()
        class Base:
            def _excluded_save_params(self): return ['policy']
        state = campaign.CampaignState(output=Path('/tmp'),
            plan=campaign.plan_run(n_envs=1, n_steps=2, batch_size=2, epochs=1, warmup=0, cycles=1), sync=lambda: None)
        model = campaign.make_measured_ppo_class(Base, state)()
        self.assertTrue({'policy', 'campaign_state', 'pipeline_timer', 'update_timer'} <= set(model._excluded_save_params()))

    def test_cycle_clock_includes_between_rollout_and_update_work(self):
        campaign = load_campaign()
        with TemporaryDirectory() as tmp:
            state = campaign.CampaignState(output=Path(tmp),
                plan=campaign.plan_run(n_envs=2, n_steps=4, batch_size=4,
                                       epochs=1, warmup=1, cycles=1), sync=lambda: None)
            model = types.SimpleNamespace(num_timesteps=8)
            with patch.object(campaign.time, 'perf_counter', return_value=10):
                state.mark_learn_started()
            state.record_rollout(2, True)
            with patch.object(campaign.time, 'perf_counter', return_value=15):
                state.record_update(1, model)
            state.record_rollout(2, True)
            with patch.object(campaign.time, 'perf_counter', return_value=23):
                state.record_update(1, model)
            self.assertEqual(state.measured_records()[0]['cycle_seconds'], 8)
            self.assertEqual(state.measured_records()[0]['boundary_seconds'], 5)

    def test_failed_update_not_completed_and_cache_is_cleared(self):
        campaign = load_campaign()
        with TemporaryDirectory() as tmp:
            state = campaign.CampaignState(output=Path(tmp),
                plan=campaign.plan_run(n_envs=1, n_steps=2, batch_size=2,
                                       epochs=1, warmup=0, cycles=1), sync=lambda: None)
            cleared = []
            class Base:
                def __init__(self):
                    self.rollout_buffer = types.SimpleNamespace(clear_cache=lambda: cleared.append(True))
                def collect_rollouts(self): return True
                def train(self): raise RuntimeError('partial optimizer failure')
            model = campaign.make_measured_ppo_class(Base, state)()
            model.collect_rollouts()
            with self.assertRaisesRegex(RuntimeError, 'partial optimizer'):
                model.train()
            self.assertEqual(state.measured_cycle_count, 0)
            self.assertEqual(cleared, [True])

    def test_plan_steps_are_warmup_plus_measured_cycles(self):
        campaign = load_campaign()

        plan = campaign.plan_run(
            n_envs=64,
            n_steps=512,
            batch_size=8192,
            epochs=8,
            warmup=2,
            cycles=100,
        )

        self.assertEqual(plan.transitions_per_cycle, 32768)
        self.assertEqual(plan.steps, 3342336)
        self.assertEqual(plan.minibatches_per_cycle, 32)

    def test_output_must_be_absolute(self):
        campaign = load_campaign()

        with self.assertRaisesRegex(ValueError, "--output must be absolute"):
            campaign.validate_output_path(Path("relative-output"))

    def test_fake_train_job_records_only_completed_measured_cycles(self):
        campaign = load_campaign()
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_job_path = tmp_path / "fake_train_job.py"
            train_job_path.write_text(
                textwrap.dedent(
                    """
                    from pathlib import Path

                    RUN_ARGS = None
                    PPO = None

                    class PPOBase:
                        def __init__(self, *args, **kwargs):
                            self.num_timesteps = 16001056

                        @classmethod
                        def load(cls, *args, **kwargs):
                            return cls()

                        def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
                            self.num_timesteps += 64 * n_rollout_steps
                            return True

                        def train(self):
                            return None

                    PPO = PPOBase

                    def run_job(args):
                        global RUN_ARGS
                        RUN_ARGS = args
                        model = PPO.load(args.checkpoint, env=None, device=args.device)
                        total_cycles = args.steps // (args.n_envs * args.n_steps)
                        for _ in range(total_cycles):
                            model.collect_rollouts(None, None, None, args.n_steps)
                            model.train()
                        Path(args.output).mkdir(parents=True, exist_ok=True)
                        return None
                    """
                )
            )
            output = tmp_path / "out"
            args = campaign.build_parser().parse_args(
                [
                    "--config",
                    str(tmp_path / "config.yaml"),
                    "--checkpoint",
                    str(tmp_path / "checkpoint.zip"),
                    "--source",
                    str(tmp_path),
                    "--train-job",
                    str(train_job_path),
                    "--output",
                    str(output),
                    "--cycles",
                    "3",
                    "--warmup",
                    "2",
                    "--device",
                    "cuda",
                ]
            )

            with preserved_import_state("fake_train_job", "campaign_target_train_job"):
                code = campaign.run_campaign(args)

                self.assertEqual(code, 0)
                metrics = json.loads((output / "campaign_metrics.jsonl").read_text().strip().splitlines()[-1])
                summary = json.loads((output / "campaign_summary.json").read_text())
                self.assertEqual(metrics["cycle_index"], 5)
                self.assertEqual(metrics["measured_cycle"], 3)
                self.assertEqual(summary["completed_measured_cycles"], 3)
                self.assertEqual(summary["status"], "completed")
                self.assertEqual(summary["plan"]["steps"], 163840)
                module = sys.modules["campaign_target_train_job"]
                self.assertEqual(module.RUN_ARGS.steps, 163840)

    def test_incomplete_measured_cycles_fail_closed(self):
        campaign = load_campaign()
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_job_path = tmp_path / "short_train_job.py"
            train_job_path.write_text(
                textwrap.dedent(
                    """
                    class PPOBase:
                        def __init__(self):
                            self.num_timesteps = 0
                        @classmethod
                        def load(cls, *args, **kwargs):
                            return cls()
                        def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
                            self.num_timesteps += 64 * n_rollout_steps
                            return True
                        def train(self):
                            return None
                    PPO = PPOBase
                    def run_job(args):
                        model = PPO.load(args.checkpoint)
                        for _ in range(2):
                            model.collect_rollouts(None, None, None, args.n_steps)
                            model.train()
                    """
                )
            )
            output = tmp_path / "out"
            args = campaign.build_parser().parse_args(
                [
                    "--config",
                    str(tmp_path / "config.yaml"),
                    "--checkpoint",
                    str(tmp_path / "checkpoint.zip"),
                    "--source",
                    str(tmp_path),
                    "--train-job",
                    str(train_job_path),
                    "--output",
                    str(output),
                    "--cycles",
                    "3",
                    "--warmup",
                    "1",
                ]
            )

            with preserved_import_state("short_train_job", "campaign_target_train_job"):
                code = campaign.run_campaign(args)

                self.assertEqual(code, 2)
                summary = json.loads((output / "campaign_summary.json").read_text())
                self.assertEqual(summary["status"], "incomplete")
                self.assertEqual(summary["completed_measured_cycles"], 1)

    def test_resident_buffer_hook_is_passed_to_constructor_and_load_custom_objects(self):
        campaign = load_campaign()

        class BasePPO:
            constructor_kwargs = None
            load_kwargs = None

            def __init__(self, *args, **kwargs):
                type(self).constructor_kwargs = kwargs

            @classmethod
            def load(cls, *args, **kwargs):
                cls.load_kwargs = kwargs
                return cls()

            def collect_rollouts(self, *args, **kwargs):
                return True

            def train(self):
                return None

        class CachedRolloutBuffer:
            pass

        state = campaign.CampaignState(
            output=Path("/tmp/campaign-test-output"),
            plan=campaign.plan_run(n_envs=64, n_steps=512, batch_size=8192, epochs=8, warmup=0, cycles=1),
            sync=lambda: None,
            mode="baseline",
            resident_buffer_class=CachedRolloutBuffer,
        )
        measured = campaign.make_measured_ppo_class(BasePPO, state)

        measured("MlpPolicy", None)
        measured.load("checkpoint.zip", custom_objects={})

        self.assertIs(measured.constructor_kwargs["rollout_buffer_class"], CachedRolloutBuffer)
        self.assertIs(measured.load_kwargs["custom_objects"]["rollout_buffer_class"], CachedRolloutBuffer)

    def test_profile_hooks_are_loaded_from_explicit_paths(self):
        campaign = load_campaign()
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pipeline_path = tmp_path / "probe_pipeline.py"
            update_path = tmp_path / "update_hooks.py"
            pipeline_path.write_text(
                "class PipelineTimer:\n"
                "    def __init__(self, enabled=False, sync=None): self.enabled=enabled; self.sync=sync\n"
                "def instrument_collect_rollouts_source(source): return source\n"
            )
            update_path.write_text(
                "class UpdateTimer:\n"
                "    def __init__(self, device='cpu', synchronize_stages=False): self.device=device\n"
                "    def report(self): return {'setup': {'count': 1}}\n"
                "def instrument_train(function): return function\n"
                "def instrument_buffer(buffer, timer): return lambda: None\n"
            )

            hooks = campaign.load_profile_hooks(pipeline_path, update_path, device="cuda", sync=lambda: None)

            self.assertEqual(Path(hooks.pipeline_module.__file__).resolve(), pipeline_path.resolve())
            self.assertEqual(Path(hooks.update_module.__file__).resolve(), update_path.resolve())

    def test_profile_timing_controls_update_timer_synchronization(self):
        campaign = load_campaign()
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pipeline_path = tmp_path / "probe_pipeline.py"
            update_path = tmp_path / "update_hooks.py"
            pipeline_path.write_text(
                "class PipelineTimer:\n"
                "    def __init__(self, enabled=False, sync=None): self.enabled=enabled; self.sync=sync\n"
                "def instrument_collect_rollouts_source(source): return source\n"
            )
            update_path.write_text(
                "class UpdateTimer:\n"
                "    def __init__(self, device='cpu', synchronize_stages=False):\n"
                "        self.device=device; self.synchronize_stages=synchronize_stages\n"
                "    def report(self): return {}\n"
                "def instrument_train(function): return function\n"
            )

            natural = campaign.load_profile_hooks(
                pipeline_path, update_path, device="cuda", sync=lambda: None, profile_timing="natural"
            )
            synced = campaign.load_profile_hooks(
                pipeline_path, update_path, device="cuda", sync=lambda: None, profile_timing="sync"
            )

            self.assertFalse(natural.update_timer.synchronize_stages)
            self.assertTrue(synced.update_timer.synchronize_stages)

    def test_group_size_default_leaves_train_job_build_env_unchanged(self):
        campaign = load_campaign()
        original = object()
        fake_train_job = types.SimpleNamespace(SubprocVecEnv=original)
        args = types.SimpleNamespace(group_size=1, grouped_env_path=Path("/unused/grouped.py"))

        cleanup = campaign.patch_grouped_vec_env(fake_train_job, args)

        self.assertIsNone(cleanup)
        self.assertIs(fake_train_job.SubprocVecEnv, original)

    def test_train_job_import_uses_real_importable_module_name_for_spawn(self):
        campaign = load_campaign()
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_job_path = tmp_path / "train_job.py"
            train_job_path.write_text("VALUE = 7\n")

            with preserved_import_state("train_job", "campaign_target_train_job"):
                module = campaign.import_train_job_module(train_job_path)

                self.assertIs(sys.modules["train_job"], module)
                self.assertIs(sys.modules["campaign_target_train_job"], module)
                self.assertIn(str(tmp_path.resolve()), sys.path)
                self.assertEqual(module.__name__, "train_job")

    def test_runtime_helper_imports_use_real_stems_for_spawn_and_checkpointing(self):
        campaign = load_campaign()
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_path = tmp_path / "rollout_cache.py"
            grouped_path = tmp_path / "grouped_vec_env.py"
            cache_path.write_text("class CachedRolloutBuffer: pass\n")
            grouped_path.write_text("class GroupedSubprocVecEnv: pass\n")

            with preserved_import_state("rollout_cache", "campaign_rollout_cache", "grouped_vec_env", "campaign_grouped_vec_env"):
                cache_cls = campaign.load_resident_buffer(cache_path)
                grouped_module = campaign.load_grouped_env_module(grouped_path)

                self.assertEqual(cache_cls.__module__, "rollout_cache")
                self.assertIs(sys.modules["rollout_cache"].CachedRolloutBuffer, cache_cls)
                self.assertIs(sys.modules["campaign_rollout_cache"], sys.modules["rollout_cache"])
                self.assertEqual(grouped_module.__name__, "grouped_vec_env")
                self.assertIs(sys.modules["campaign_grouped_vec_env"], grouped_module)
                self.assertIn(str(tmp_path.resolve()), sys.path)

    def test_group_size_over_one_wraps_subproc_factory_and_records_provenance(self):
        campaign = load_campaign()
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            grouped_path = tmp_path / "grouped_vec_env.py"
            grouped_path.write_text(
                "CALLS=[]\n"
                "class GroupedSubprocVecEnv:\n"
                "    def __init__(self, env_fns, group_size, start_method=None):\n"
                "        CALLS.append((list(env_fns), group_size, start_method))\n"
            )
            original_calls = []

            class OriginalSubprocVecEnv:
                def __init__(self, env_fns, start_method=None):
                    original_calls.append((env_fns, start_method))

            fake_train_job = types.SimpleNamespace(SubprocVecEnv=OriginalSubprocVecEnv)
            args = types.SimpleNamespace(group_size=4, grouped_env_path=grouped_path)

            with preserved_import_state("grouped_vec_env", "campaign_grouped_vec_env"):
                cleanup = campaign.patch_grouped_vec_env(fake_train_job, args)
                try:
                    created = fake_train_job.SubprocVecEnv(["a", "b"], start_method="spawn")
                finally:
                    cleanup()

                grouped_module = sys.modules["campaign_grouped_vec_env"]
                self.assertIsInstance(created, grouped_module.GroupedSubprocVecEnv)
                self.assertEqual(grouped_module.CALLS, [(["a", "b"], 4, "spawn")])
                self.assertEqual(original_calls, [])
                self.assertIs(fake_train_job.SubprocVecEnv, OriginalSubprocVecEnv)

    def test_grouped_args_are_written_to_provenance(self):
        campaign = load_campaign()
        with TemporaryDirectory() as tmp:
            output = Path(tmp)
            args = types.SimpleNamespace(
                config=output / "config.yaml",
                checkpoint=output / "checkpoint.zip",
                source=output,
                train_job=output / "train_job.py",
                pipeline_hooks=output / "probe_pipeline.py",
                update_hooks=output / "update_hooks.py",
                resident_buffer=False,
                rollout_cache_path=output / "rollout_cache.py",
                group_size=8,
                grouped_env_path=output / "grouped_vec_env.py",
                profile_timing="natural",
                rollout_policy_graph=False,
                rollout_policy_graph_path=output / "rollout_policy_graph.py",
                rollout_policy_graph_capture_warmup=3,
                rollout_policy_graph_tolerance=1e-5,
            )
            train_job = types.SimpleNamespace(PPO=None)
            state = campaign.CampaignState(
                output=output,
                plan=campaign.plan_run(n_envs=64, n_steps=512, batch_size=8192, epochs=8, warmup=2, cycles=3),
                sync=lambda: None,
            )

            campaign.write_provenance(output, args=args, train_job_module=train_job, state=state)

            provenance = json.loads((output / "campaign_provenance.json").read_text())
            self.assertEqual(provenance["grouping"], {
                "group_size": 8,
                "grouped_env_path": str((output / "grouped_vec_env.py").resolve()),
            })
            self.assertEqual(provenance["profile_timing"], "natural")

    def test_rollout_policy_graph_args_are_written_to_provenance(self):
        campaign = load_campaign()
        with TemporaryDirectory() as tmp:
            output = Path(tmp)
            graph_path = output / "rollout_policy_graph.py"
            graph_path.write_text("# helper\n")
            args = types.SimpleNamespace(
                config=output / "config.yaml",
                checkpoint=output / "checkpoint.zip",
                source=output,
                train_job=output / "train_job.py",
                pipeline_hooks=output / "probe_pipeline.py",
                update_hooks=output / "update_hooks.py",
                resident_buffer=False,
                rollout_cache_path=output / "rollout_cache.py",
                group_size=1,
                grouped_env_path=output / "grouped_vec_env.py",
                profile_timing="natural",
                rollout_policy_graph=True,
                rollout_policy_graph_path=graph_path,
                rollout_policy_graph_capture_warmup=5,
                rollout_policy_graph_tolerance=2e-6,
            )
            train_job = types.SimpleNamespace(PPO=None)
            state = campaign.CampaignState(
                output=output,
                plan=campaign.plan_run(n_envs=64, n_steps=512, batch_size=8192, epochs=8, warmup=2, cycles=3),
                sync=lambda: None,
                rollout_policy_graph_module=types.SimpleNamespace(__file__=str(graph_path)),
                rollout_policy_graph_capture_warmup=5,
                rollout_policy_graph_tolerance=2e-6,
            )

            campaign.write_provenance(output, args=args, train_job_module=train_job, state=state)

            provenance = json.loads((output / "campaign_provenance.json").read_text())
            self.assertEqual(provenance["rollout_policy_graph"]["enabled"], True)
            self.assertEqual(provenance["rollout_policy_graph"]["capture_warmup"], 5)
            self.assertEqual(provenance["rollout_policy_graph"]["tolerance"], 2e-6)
            self.assertEqual(provenance["files"]["rollout_policy_graph"]["sha256"], campaign.sha256_file(graph_path))

    def test_rollout_policy_graph_wraps_collect_rollouts_temporarily(self):
        campaign = load_campaign()
        events = []

        class FakeGraphModule:
            @staticmethod
            def policy_forward_graph_context(policy, **kwargs):
                events.append(("context_kwargs", kwargs))

                @contextmanager
                def manager():
                    original = policy.forward
                    adapter = types.SimpleNamespace(
                        preflight_receipt={"status": "passed"},
                        summary=lambda: {"captures": 1, "replays": 1, "preflight": {"status": "passed"}},
                    )
                    policy.forward = lambda obs, deterministic=False: ("graph", deterministic)
                    try:
                        events.append(("entered", policy.forward("obs", deterministic=False)))
                        yield adapter
                    finally:
                        policy.forward = original
                        events.append(("restored", policy.forward("obs", deterministic=True)))

                return manager()

        class Base:
            def __init__(self):
                self.policy = types.SimpleNamespace(
                    forward=lambda obs, deterministic=False: ("eager", deterministic),
                )

            def collect_rollouts(self):
                events.append(("inside", self.policy.forward("obs", deterministic=False)))
                return True

            def train(self):
                return None

        with TemporaryDirectory() as tmp:
            state = campaign.CampaignState(
                output=Path(tmp),
                plan=campaign.plan_run(n_envs=64, n_steps=512, batch_size=8192, epochs=8, warmup=0, cycles=1),
                sync=lambda: None,
                rollout_policy_graph_module=FakeGraphModule,
                rollout_policy_graph_capture_warmup=7,
                rollout_policy_graph_tolerance=3e-6,
            )
            model = campaign.make_measured_ppo_class(Base, state)()

            self.assertTrue(model.collect_rollouts())

            self.assertEqual(events[0][0], "context_kwargs")
            self.assertEqual(events[0][1]["expected_batch_size"], 64)
            self.assertEqual(events[0][1]["capture_warmup"], 7)
            self.assertEqual(events[1], ("entered", ("graph", False)))
            self.assertEqual(events[2], ("inside", ("graph", False)))
            self.assertEqual(events[3], ("restored", ("eager", True)))
            self.assertEqual(state.records, [])
            self.assertEqual(
                state.rollout_policy_graph_records,
                [{"captures": 1, "preflight": {"status": "passed"}, "replays": 1}],
            )

    def test_rollout_policy_graph_collect_requires_passed_preflight_receipt(self):
        campaign = load_campaign()

        class FakeGraphModule:
            @staticmethod
            def policy_forward_graph_context(policy, **kwargs):
                @contextmanager
                def manager():
                    yield types.SimpleNamespace(summary=lambda: {"captures": 1}, preflight_receipt=None)
                return manager()

        class Base:
            def __init__(self):
                self.policy = types.SimpleNamespace(forward=lambda obs, deterministic=False: ("eager", deterministic))

            def collect_rollouts(self):
                return True

            def train(self):
                return None

        with TemporaryDirectory() as tmp:
            state = campaign.CampaignState(
                output=Path(tmp),
                plan=campaign.plan_run(n_envs=64, n_steps=512, batch_size=8192, epochs=8, warmup=0, cycles=1),
                sync=lambda: None,
                rollout_policy_graph_module=FakeGraphModule,
            )
            model = campaign.make_measured_ppo_class(Base, state)()

            with self.assertRaisesRegex(RuntimeError, "preflight"):
                model.collect_rollouts()

    def test_rollout_policy_graph_is_opt_in_and_default_parser_disables_it(self):
        campaign = load_campaign()

        args = campaign.build_parser().parse_args(["--output", "/tmp/out"])

        self.assertFalse(args.rollout_policy_graph)


if __name__ == "__main__":
    unittest.main()
