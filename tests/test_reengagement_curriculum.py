from __future__ import annotations

import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import reengagement_curriculum as rc


def resolve_frozen_source_root() -> Path:
    override = os.environ.get("AI2_FROZEN_SOURCE_ROOT")
    root = Path(override).expanduser() if override else Path(__file__).resolve().parents[1] / "baselines/fullgame-20260908/oracle"
    root = root.resolve()
    source = root / "UTMIST-AI2-main/environment/environment.py"
    if not source.is_file():
        raise FileNotFoundError(
            f"AI2 frozen original-game fixture not found at {root}. "
            "Set AI2_FROZEN_SOURCE_ROOT to the frozen oracle checkout root."
        )
    return root


@pytest.fixture(scope="module")
def frozen_root() -> Path:
    return resolve_frozen_source_root()


def zero_action() -> np.ndarray:
    return np.zeros(10, dtype=np.float32)


def test_frozen_fixture_defaults_to_reviewed_worktree_when_no_override(monkeypatch):
    monkeypatch.delenv("AI2_FROZEN_SOURCE_ROOT", raising=False)
    expected = Path(__file__).resolve().parents[1] / "baselines/fullgame-20260908/oracle"

    if (expected / "UTMIST-AI2-main/environment/environment.py").is_file():
        assert resolve_frozen_source_root() == expected
    else:
        with pytest.raises(FileNotFoundError, match="AI2 frozen original-game fixture not found") as exc:
            resolve_frozen_source_root()
        assert str(expected) in str(exc.value)


def test_frozen_fixture_uses_explicit_override_and_fails_clearly_for_missing_path(monkeypatch, tmp_path):
    explicit = tmp_path / "oracle"
    source = explicit / "UTMIST-AI2-main/environment/environment.py"
    source.parent.mkdir(parents=True)
    source.write_text("# fixture shape only; source hash is validated by real env construction\n")

    monkeypatch.setenv("AI2_FROZEN_SOURCE_ROOT", str(explicit))
    assert resolve_frozen_source_root() == explicit.resolve()

    missing = tmp_path / "missing-oracle"
    monkeypatch.setenv("AI2_FROZEN_SOURCE_ROOT", str(missing))
    with pytest.raises(FileNotFoundError, match="AI2 frozen original-game fixture not found"):
        resolve_frozen_source_root()


def player_snapshot(slot: int, *, x=0.0, y=0.35, stocks=3, damage=0.0, damage_done=0.0,
                    damage_taken_total=0.0, grounded=True, fully_supported=True,
                    clearance=1.0, attack_id=None, state="StandingState"):
    return {
        "player_slot": slot,
        "x": float(x),
        "y": float(y),
        "vx": 0.0,
        "vy": 0.0,
        "facing": 1,
        "stocks": int(stocks),
        "damage": float(damage),
        "damage_done": float(damage_done),
        "damage_taken_total": float(damage_taken_total),
        "state": state,
        "grounded": grounded,
        "fully_supported": fully_supported,
        "horizontal_floor_clearance": float(clearance),
        "stun_time_stored": 0,
        "attack_id": attack_id,
    }


def state_machine_boundary_harness(*, tick=1, events=None, pre=None, post=None):
    """Build a minimal ReengagementEnv shell for state-machine boundary tests.

    This bypasses the frozen original game and must not be used for integration
    claims about physics, collision, controller timing, or fixture provenance.
    """

    env = object.__new__(rc.ReengagementEnv)
    env.player_slot = 0
    env.direction = 1
    env.seed_value = 1
    env.scenario = {"id": "controlled", "seed": 1, "player_slot": 0, "direction": 1, "gap": 3.2, "midpoint": 4.5}
    env.episode_id = "controlled-episode"
    env.tick = tick - 1
    env.encounter = 1
    env.events = []
    env.hit_ids = []
    env.first_hit_tick = None
    env.second_hit_tick = None
    env.hit_attack_id = None
    env.relocation_tick = None
    env.relocation_gap = None
    env.relocation_learner_x = None
    env.relocation_target_x = None
    env.pursuit_direction = None
    env.pursuit_tick = None
    env.outcome = None
    env.previous_light = False
    env.initial_stocks = [3, 3]
    env.reward_totals = {name: 0.0 for name in rc.REWARD_NAMES}
    env._trace_handle = None
    env._observation_before = np.zeros(64, dtype=np.float32)
    env._test_events = list(events or [])
    env._test_pre = pre or [player_snapshot(0), player_snapshot(1, x=3.2)]
    env._test_post = post or [player_snapshot(0), player_snapshot(1, x=3.2)]

    def fake_game_step(action):
        env.tick = tick
        env.events = list(env._test_events)
        return np.asarray(action, dtype=np.float32), np.asarray(action, dtype=np.float32), env._test_pre, env._test_post

    env._game_step = fake_game_step
    env._resolved = lambda: True
    env._snapshots = lambda: env._test_post
    env._observation = lambda: np.zeros(64, dtype=np.float32)
    env._relocate = lambda post_value: False
    return env


def damage_event(*, attack_id="attack-1", attack_type="DLIGHT", source_slot=0, target_slot=1, amount=1.0):
    return {
        "type": "damage",
        "source_slot": source_slot,
        "target_slot": target_slot,
        "attack_id": attack_id,
        "attack_type": attack_type,
        "amount": amount,
        "damage_before": 0.0,
        "damage_after": amount,
        "source": "test",
    }


def finish_episode(env: rc.ReengagementEnv, controller: str, *, limit: int = 300):
    info = None
    for _ in range(limit):
        if controller == "scripted":
            action = rc.scripted_reengagement_action(env)
        elif controller == "held_light":
            action = zero_action()
            action[7] = 1
        elif controller == "one_hit_then_idle":
            action = rc.scripted_reengagement_action(env) if env.first_hit_tick is None else zero_action()
        elif controller == "move_without_second_hit":
            action = zero_action()
            if env.relocation_tick is None:
                action = rc.scripted_reengagement_action(env)
            else:
                target = env.raw_env.players[1 - env.player_slot]
                learner = env.raw_env.players[env.player_slot]
                direction = 1 if target.body.position.x >= learner.body.position.x else -1
                action[3 if direction == 1 else 1] = 1
        elif controller == "walkoff":
            action = zero_action()
            action[3 if env.direction == 1 else 1] = 1
        else:
            action = zero_action()
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return info
    raise AssertionError("episode did not terminate")


def test_identity_schema_is_frozen():
    assert rc.CURRICULUM_VERSION == "bridge_b2_reengagement-v1"
    assert rc.BANK_VERSION == "b2-reengagement-bank-v1"
    assert rc.TRACE_VERSION == "b2-reengagement-trace-v1"
    assert rc.REQUIRED_PYMUNK_VERSION == "6.2.1"
    assert rc.REWARD_NAMES == (
        "first_hit",
        "pursuit",
        "second_hit",
        "completion",
        "stock_loss",
        "unsafe_support",
        "deadline",
        "tick",
    )
    assert rc.OUTCOMES == (
        None,
        "success",
        "own_stock_loss",
        "opponent_stock_loss",
        "unsafe_support",
        "response_deadline",
        "second_hit_deadline",
        "total_deadline",
    )
    assert rc.OBSERVATION_RESIDUAL_VELOCITY_EPSILON == 1e-15
    assert rc.OBSERVATION_VELOCITY_INDICES == (2, 3, 34, 35)


def test_public_observation_canonicalizes_only_sub_epsilon_velocity_residue():
    eps = rc.OBSERVATION_RESIDUAL_VELOCITY_EPSILON
    obs = np.arange(64, dtype=np.float64)
    below = np.nextafter(eps, 0.0)
    above = np.nextafter(eps, np.inf)
    obs[2] = below
    obs[3] = -below
    obs[34] = eps
    obs[35] = above
    obs[4] = below

    got = rc.canonicalize_public_observation(obs)

    assert got.dtype == np.float32
    assert got[2].view(np.uint32) == np.float32(0.0).view(np.uint32)
    assert got[3].view(np.uint32) == np.float32(0.0).view(np.uint32)
    assert got[34].view(np.uint32) == np.float32(obs[34]).view(np.uint32)
    assert got[35].view(np.uint32) == np.float32(obs[35]).view(np.uint32)
    assert got[4].view(np.uint32) == np.float32(obs[4]).view(np.uint32)


def test_original_game_load_fails_closed_without_pinned_pymunk(monkeypatch, tmp_path):
    source = tmp_path / "UTMIST-AI2-main/environment/environment.py"
    source.parent.mkdir(parents=True)
    source.write_text("raise AssertionError('environment import should not execute')\n")
    monkeypatch.setattr(rc, "ORIGINAL_GAME_SHA256", __import__("hashlib").sha256(source.read_bytes()).hexdigest())
    monkeypatch.setitem(__import__("sys").modules, "pymunk", SimpleNamespace(version="6.9.0"))

    with pytest.raises(RuntimeError, match="requires pymunk==6.2.1, got '6.9.0'"):
        rc.load_original_game(tmp_path)


def test_choose_target_anchor_uses_floor_bounds_half_width_gap_and_tie_direction():
    half_width = 0.435
    left_anchor = -1.0 + half_width + 0.2
    right_anchor = 10.0 - half_width - 0.2

    assert rc.choose_target_anchor(-1.0, 10.0, half_width, 4.5, 1) == pytest.approx(right_anchor)
    assert rc.choose_target_anchor(-1.0, 10.0, half_width, 4.5, -1) == pytest.approx(left_anchor)
    assert rc.choose_target_anchor(-1.0, 4.0, half_width, 1.8, 1) == pytest.approx(left_anchor)
    assert rc.choose_target_anchor(5.0, 10.0, half_width, 7.2, -1) == pytest.approx(right_anchor)
    assert rc.choose_target_anchor(-1.0, 2.0, half_width, 0.5, 1) is None


@pytest.mark.parametrize("slot", [0, 1])
@pytest.mark.parametrize("direction", [-1, 1])
def test_reset_is_deterministic_private_rng_and_safe_for_both_slots_directions(frozen_root, slot, direction):
    random.seed(987)
    np.random.seed(654)
    py_before = random.getstate()
    np_before = np.random.get_state()

    env = rc.make_reengagement_env(frozen_root, seed=17, player_slot=slot)
    try:
        first_obs, first_info = env.reset(
            seed=100 + slot,
            options={"scenario": {"player_slot": slot, "direction": direction, "id": "deterministic"}},
        )
        random.setstate(py_before)
        np.random.set_state(np_before)
        second_obs, second_info = env.reset(
            seed=100 + slot,
            options={"scenario": {"player_slot": slot, "direction": direction, "id": "deterministic"}},
        )
    finally:
        env.close()

    assert random.getstate() == py_before
    assert np.array_equal(np.random.get_state()[1], np_before[1])
    np.testing.assert_array_equal(first_obs.view(np.uint32), second_obs.view(np.uint32))
    first = first_info["trace"]
    second = second_info["trace"]
    assert first["scenario"] == second["scenario"]
    assert 3.1 <= first["spawn_gap"] <= 3.4
    assert 4.35 <= first["spawn_midpoint"] <= 4.65
    assert min(p["horizontal_floor_clearance"] for p in first["post_players"]) >= 0.2
    target = first["post_players"][1 - slot]
    assert target["vx"] == 0.0
    assert target["vy"] == 0.0
    assert target["state"] == "StandingState"
    assert first["opponent_action"] == [0.0] * 10


def test_constructor_honors_player_slot_for_initial_episode(frozen_root):
    env = rc.make_reengagement_env(frozen_root, seed=18, player_slot=1)
    try:
        assert env.player_slot == 1
        assert env.scenario["player_slot"] == 1
        assert env._snapshots()[1]["x"] == pytest.approx(env.scenario["midpoint"] - env.direction * env.scenario["gap"] / 2)
    finally:
        env.close()


@pytest.mark.parametrize(
    "scenario",
    [
        {"player_slot": 2, "direction": 1},
        {"player_slot": 0, "direction": 0},
        {"player_slot": 0, "direction": 1, "gap": 3.0},
        {"player_slot": 0, "direction": 1, "gap": 3.2, "midpoint": float("nan")},
    ],
)
def test_invalid_scenarios_are_rejected_before_reset_mutates(frozen_root, scenario):
    env = rc.make_reengagement_env(frozen_root, seed=17)
    try:
        _, info = env.reset(seed=17, options={"scenario": {"player_slot": 0, "direction": 1}})
        before = info["trace"]["post_players"]
        with pytest.raises(ValueError):
            env.reset(seed=17, options={"scenario": scenario})
        after = env._snapshots()
    finally:
        env.close()
    assert after == before


@pytest.mark.parametrize(
    "seed_value, scenario_override",
    [
        (True, {}),
        (1.9, {}),
        ("7", {}),
        (17, {"player_slot": True}),
        (17, {"direction": True}),
        (17, {"gap": "3.2"}),
        (17, {"midpoint": "4.5"}),
    ],
)
def test_scenario_reset_rejects_raw_invalid_types_without_mutating_episode(frozen_root, seed_value, scenario_override):
    env = rc.make_reengagement_env(frozen_root, seed=17, player_slot=0)
    try:
        _, info = env.reset(seed=17, options={"scenario": {"player_slot": 0, "direction": 1}})
        before = {
            "seed": env.seed_value,
            "scenario": dict(env.scenario),
            "episode_number": env._episode_number,
            "tick": env.tick,
            "post_players": info["trace"]["post_players"],
        }
        with pytest.raises(ValueError):
            env.reset(seed=seed_value, options={"scenario": scenario_override})
        after = {
            "seed": env.seed_value,
            "scenario": dict(env.scenario),
            "episode_number": env._episode_number,
            "tick": env.tick,
            "post_players": env._snapshots(),
        }
    finally:
        env.close()
    assert after == before


def test_relocation_recreates_only_target_and_preserves_required_state(frozen_root, monkeypatch):
    env = rc.make_reengagement_env(frozen_root, seed=17, player_slot=0)
    try:
        env.reset(seed=134040, options={"scenario": {"player_slot": 0, "direction": 1}})
        learner = env.raw_env.players[env.player_slot]
        target = env.raw_env.players[1 - env.player_slot]
        learner.body.position = (4.5, env.floor_y)
        target.stocks = 7
        target.damage = 12.5
        target.damage_done = 4.25
        target.damage_taken_total = 9.75
        learner_before_object = learner
        target_before_object = target
        learner_before = {
            "body": learner.body,
            "shape": learner.shape,
            "state": learner.state,
            "position": tuple(learner.body.position),
            "velocity": tuple(learner.body.velocity),
            "stocks": learner.stocks,
            "damage": learner.damage,
            "damage_done": learner.damage_done,
            "damage_taken_total": learner.damage_taken_total,
            "facing": int(learner.facing),
        }
        target_before_snapshot = env._snapshot(target)
        monkeypatch.setattr(rc, "choose_target_anchor", lambda *args: 8.0)

        assert env._relocate(env._snapshots()) is True

        learner_after = env.raw_env.players[env.player_slot]
        target_after = env.raw_env.players[1 - env.player_slot]
        assert learner_after is learner_before_object
        assert learner_after.body is learner_before["body"]
        assert learner_after.shape is learner_before["shape"]
        assert learner_after.state is learner_before["state"]
        assert tuple(learner_after.body.position) == learner_before["position"]
        assert tuple(learner_after.body.velocity) == learner_before["velocity"]
        assert learner_after.stocks == learner_before["stocks"]
        assert learner_after.damage == learner_before["damage"]
        assert learner_after.damage_done == learner_before["damage_done"]
        assert learner_after.damage_taken_total == learner_before["damage_taken_total"]
        assert int(learner_after.facing) == learner_before["facing"]
        assert target_after is not target_before_object
        assert target_after.body is not target_before_object.body
        assert target_after.shape is not target_before_object.shape
        assert target_after.stocks == 7
        assert target_after.damage == pytest.approx(12.5)
        assert target_after.damage_done == pytest.approx(4.25)
        assert target_after.damage_taken_total == pytest.approx(9.75)
        assert target_after.body.position.x == pytest.approx(8.0)
        assert target_after.body.position.y == pytest.approx(env.floor_y)
        assert tuple(target_after.body.velocity) == (0.0, 0.0)
        assert type(target_after.state).__name__ == "StandingState"
        assert int(target_after.facing) == -1
        relocation = next(event for event in env.events if event["type"] == "relocation")
        assert relocation["pre_target"] == target_before_snapshot
        assert relocation["post_target"]["x"] == pytest.approx(8.0)
        assert relocation["relocation_gap"] == pytest.approx(3.5)
        assert env.relocation_gap == pytest.approx(3.5)
        assert env.relocation_tick == env.tick
    finally:
        env.close()


def test_relocation_pending_when_no_safe_anchor_does_not_recreate_target(frozen_root, monkeypatch):
    env = rc.make_reengagement_env(frozen_root, seed=17, player_slot=0)
    try:
        env.reset(seed=134041, options={"scenario": {"player_slot": 0, "direction": 1}})
        learner = env.raw_env.players[env.player_slot]
        target = env.raw_env.players[1 - env.player_slot]
        learner.body.position = (4.5, env.floor_y)
        target_before = target
        monkeypatch.setattr(rc, "choose_target_anchor", lambda *args: None)

        assert env._relocate(env._snapshots()) is False

        assert env.raw_env.players[1 - env.player_slot] is target_before
        assert env.relocation_tick is None
        assert env.encounter == 1
        assert any(event["type"] == "relocation_pending" for event in env.events)
    finally:
        env.close()


@pytest.mark.parametrize("slot", [0, 1])
@pytest.mark.parametrize("direction", [-1, 1])
def test_scripted_hit_resolve_pursue_hit_resolve_succeeds(frozen_root, slot, direction):
    env = rc.make_reengagement_env(frozen_root, seed=17, player_slot=slot)
    try:
        env.reset(seed=134000 + slot, options={"scenario": {"player_slot": slot, "direction": direction}})
        info = finish_episode(env, "scripted")
    finally:
        env.close()
    diag = info["episode_diagnostics"]
    assert diag["outcome"] == "success"
    assert diag["sequence_completed"] is True
    assert diag["first_hit_tick"] is not None
    assert diag["pursuit_tick"] is not None
    assert diag["second_hit_tick"] is not None
    assert diag["distinct_hit_ids"][0] != diag["distinct_hit_ids"][1]
    assert diag["first_hit_tick"] < diag["relocation_tick"] < diag["pursuit_tick"] < diag["second_hit_tick"]
    assert diag["second_hit_tick"] <= diag["ticks"]
    assert diag["min_clearance"] >= 0.2
    assert diag["failure_category"] == "success"


@pytest.mark.parametrize(
    "controller, expected",
    [
        ("idle", "total_deadline"),
        ("held_light", "total_deadline"),
        ("one_hit_then_idle", "response_deadline"),
        ("move_without_second_hit", "second_hit_deadline"),
        ("walkoff", "own_stock_loss"),
    ],
)
def test_shortcut_controllers_fail(frozen_root, controller, expected):
    env = rc.make_reengagement_env(frozen_root, seed=17)
    try:
        env.reset(seed=134010, options={"scenario": {"player_slot": 0, "direction": 1}})
        info = finish_episode(env, controller)
    finally:
        env.close()
    assert info["episode_diagnostics"]["outcome"] == expected
    assert info["episode_diagnostics"]["success"] is False


def test_repeated_first_attack_id_and_opponent_self_loss_fail_without_success_transition(frozen_root):
    env = rc.make_reengagement_env(frozen_root, seed=17)
    try:
        env.reset(seed=134020, options={"scenario": {"player_slot": 0, "direction": 1}})
        while env.relocation_tick is None:
            env.step(rc.scripted_reengagement_action(env))
        first_id = env.hit_ids[0]
        original = env.raw_env.step

        def repeated_first_id_step(actions):
            result = original(actions)
            env.events.append(
                {
                    "type": "damage",
                    "source_slot": env.player_slot,
                    "target_slot": 1 - env.player_slot,
                    "attack_id": first_id,
                    "attack_type": "DLIGHT",
                    "amount": 1.0,
                    "damage_before": 0.0,
                    "damage_after": 1.0,
                    "source": "test-corruption",
                }
            )
            return result

        env.raw_env.step = repeated_first_id_step
        with pytest.raises(RuntimeError, match="replayed first encounter attack id after relocation"):
            env.step(zero_action())
    finally:
        env.close()

    env = rc.make_reengagement_env(frozen_root, seed=17)
    try:
        env.reset(seed=134021, options={"scenario": {"player_slot": 0, "direction": 1}})
        target = env.raw_env.players[1]
        target.stocks -= 1
        _, reward, terminated, truncated, info = env.step(zero_action())
    finally:
        env.close()
    assert terminated is True
    assert truncated is False
    assert info["episode_diagnostics"]["outcome"] == "opponent_stock_loss"
    assert reward == pytest.approx(-1.001)


def test_invalid_damage_evidence_requires_nonzero_id_valid_type_and_correct_attribution():
    invalid_events = [
        damage_event(attack_id=None),
        damage_event(attack_id=""),
        damage_event(attack_type="HEAVY"),
        damage_event(source_slot=1),
        damage_event(target_slot=0),
    ]
    for event in invalid_events:
        env = state_machine_boundary_harness(events=[event])
        with pytest.raises(RuntimeError, match="invalid damage evidence"):
            rc.ReengagementEnv._step(env, zero_action())


def test_simultaneous_stock_loss_prioritizes_own_loss_and_adds_hit_penalty_and_tick():
    pre = [player_snapshot(0, stocks=3), player_snapshot(1, stocks=3)]
    post = [player_snapshot(0, stocks=2), player_snapshot(1, stocks=2)]
    env = state_machine_boundary_harness(pre=pre, post=post, events=[damage_event(attack_id="terminal-hit")])

    _, reward, terminated, truncated, info = rc.ReengagementEnv._step(env, zero_action())

    assert terminated is True
    assert truncated is False
    assert env.outcome == "own_stock_loss"
    assert reward == pytest.approx(0.5 - 5.0 - 0.001)
    assert info["trace"]["reward_terms"]["first_hit"] == pytest.approx(0.5)
    assert info["trace"]["reward_terms"]["stock_loss"] == pytest.approx(-5.0)
    assert info["trace"]["reward_terms"]["tick"] == pytest.approx(-0.001)


def test_step_restores_global_rng_state_for_python_and_numpy(frozen_root):
    env = rc.make_reengagement_env(frozen_root, seed=17, player_slot=0)
    try:
        env.reset(seed=134050, options={"scenario": {"player_slot": 0, "direction": 1}})
        random.seed(444)
        np.random.seed(555)
        py_before = random.getstate()
        np_before = np.random.get_state()
        for _ in range(3):
            env.step(zero_action())
            assert random.getstate() == py_before
            assert np.array_equal(np.random.get_state()[1], np_before[1])
    finally:
        env.close()


def test_pursuit_on_elapsed_tick_60_credits_before_deadline():
    post = [player_snapshot(0, x=1.0), player_snapshot(1, x=3.0)]
    env = state_machine_boundary_harness(tick=160, post=post)
    env.encounter = 2
    env.relocation_tick = 100
    env.relocation_gap = 4.0
    env.relocation_learner_x = 0.0
    env.relocation_target_x = 4.0
    env.pursuit_direction = 1

    _, reward, terminated, _, info = rc.ReengagementEnv._step(env, zero_action())

    assert env.pursuit_tick == 160
    assert terminated is False
    assert env.outcome is None
    assert reward == pytest.approx(0.5 - 0.001)
    assert info["trace"]["reward_terms"]["pursuit"] == pytest.approx(0.5)


def test_absent_pursuit_ends_exactly_on_elapsed_tick_60():
    env = state_machine_boundary_harness(tick=160)
    env.encounter = 2
    env.relocation_tick = 100
    env.relocation_gap = 4.0
    env.relocation_learner_x = 0.0
    env.relocation_target_x = 4.0
    env.pursuit_direction = 1

    _, reward, terminated, truncated, info = rc.ReengagementEnv._step(env, zero_action())

    assert terminated is True
    assert truncated is False
    assert env.tick == 160
    assert env.outcome == "response_deadline"
    assert reward == pytest.approx(-1.001)
    assert info["episode_diagnostics"]["outcome"] == "response_deadline"


def test_distinct_second_hit_on_elapsed_tick_90_credits_before_deadline_and_can_succeed():
    post = [player_snapshot(0, x=2.0), player_snapshot(1, x=3.0)]
    env = state_machine_boundary_harness(tick=190, post=post, events=[damage_event(attack_id="second-hit")])
    env.encounter = 2
    env.first_hit_tick = 20
    env.hit_ids = ["first-hit"]
    env.hit_attack_id = "first-hit"
    env.relocation_tick = 100
    env.relocation_gap = 4.0
    env.relocation_learner_x = 0.0
    env.relocation_target_x = 4.0
    env.pursuit_direction = 1
    env.pursuit_tick = 160

    _, reward, terminated, truncated, info = rc.ReengagementEnv._step(env, zero_action())

    assert terminated is True
    assert truncated is False
    assert env.outcome == "success"
    assert env.second_hit_tick == 190
    assert reward == pytest.approx(1.0 + 2.0 - 0.001)
    assert info["trace"]["reward_terms"]["second_hit"] == pytest.approx(1.0)
    assert info["trace"]["reward_terms"]["completion"] == pytest.approx(2.0)


def test_replayed_second_hit_damage_frame_is_ignored_while_resolution_completes_safely():
    post = [player_snapshot(0, x=2.0), player_snapshot(1, x=3.0)]
    env = state_machine_boundary_harness(tick=191, post=post, events=[damage_event(attack_id="second-hit")])
    env.encounter = 2
    env.first_hit_tick = 20
    env.second_hit_tick = 190
    env.hit_ids = ["first-hit", "second-hit"]
    env.hit_attack_id = "second-hit"
    env.relocation_tick = 100
    env.relocation_gap = 4.0
    env.relocation_learner_x = 0.0
    env.relocation_target_x = 4.0
    env.pursuit_direction = 1
    env.pursuit_tick = 160
    env.attributed_damage_events = [damage_event(attack_id="first-hit"), damage_event(attack_id="second-hit")]

    _, reward, terminated, truncated, info = rc.ReengagementEnv._step(env, zero_action())

    assert terminated is True
    assert truncated is False
    assert env.outcome == "success"
    assert reward == pytest.approx(2.0 - 0.001)
    assert info["trace"]["reward_terms"]["second_hit"] == 0.0
    assert info["trace"]["reward_terms"]["completion"] == pytest.approx(2.0)
    assert env.hit_ids == ["first-hit", "second-hit"]
    assert len(env.attributed_damage_events) == 2


def test_absent_second_hit_ends_on_elapsed_tick_90_and_rejects_tick_91():
    env = state_machine_boundary_harness(tick=190)
    env.encounter = 2
    env.first_hit_tick = 20
    env.hit_ids = ["first-hit"]
    env.hit_attack_id = None
    env.relocation_tick = 100
    env.relocation_gap = 4.0
    env.relocation_learner_x = 0.0
    env.relocation_target_x = 4.0
    env.pursuit_direction = 1
    env.pursuit_tick = 160

    _, reward, terminated, truncated, info = rc.ReengagementEnv._step(env, zero_action())

    assert terminated is True
    assert truncated is False
    assert env.tick == 190
    assert env.outcome == "second_hit_deadline"
    assert reward == pytest.approx(-1.001)
    assert info["episode_diagnostics"]["outcome"] == "second_hit_deadline"
    with pytest.raises(RuntimeError):
        rc.ReengagementEnv._step(env, zero_action())


def test_total_tick_300_processes_eligible_second_hit_success_before_total_deadline():
    post = [player_snapshot(0, x=2.0), player_snapshot(1, x=3.0)]
    env = state_machine_boundary_harness(tick=300, post=post, events=[damage_event(attack_id="second-hit")])
    env.encounter = 2
    env.first_hit_tick = 20
    env.hit_ids = ["first-hit"]
    env.hit_attack_id = "first-hit"
    env.relocation_tick = 250
    env.relocation_gap = 4.0
    env.relocation_learner_x = 0.0
    env.relocation_target_x = 4.0
    env.pursuit_direction = 1
    env.pursuit_tick = 280

    _, reward, terminated, truncated, info = rc.ReengagementEnv._step(env, zero_action())

    assert terminated is True
    assert truncated is False
    assert env.tick == 300
    assert env.outcome == "success"
    assert info["episode_diagnostics"]["outcome"] == "success"
    assert reward == pytest.approx(1.0 + 2.0 - 0.001)
    with pytest.raises(RuntimeError):
        rc.ReengagementEnv._step(env, zero_action())


def test_deadline_boundaries_process_tick_300_but_never_301_in_real_env(frozen_root):
    env = rc.make_reengagement_env(frozen_root, seed=17)
    try:
        env.reset(seed=134030, options={"scenario": {"player_slot": 0, "direction": 1}})
        while env.relocation_tick is None:
            env.step(rc.scripted_reengagement_action(env))
        relocation_tick = env.relocation_tick
        while env.tick - relocation_tick < 60:
            _, _, terminated, _, info = env.step(zero_action())
        assert terminated is True
        assert info["episode_diagnostics"]["outcome"] == "response_deadline"
        assert env.tick - relocation_tick == 60
    finally:
        env.close()


def test_completed_diagnostics_include_reconstructable_relocation_damage_clearance_and_stock_fields(frozen_root):
    env = rc.make_reengagement_env(frozen_root, seed=17, player_slot=0)
    try:
        env.reset(seed=134060, options={"scenario": {"player_slot": 0, "direction": 1}})
        info = finish_episode(env, "scripted")
    finally:
        env.close()
    diag = info["episode_diagnostics"]
    for field in (
        "relocation_pre_target",
        "relocation_post_target",
        "relocation_gap",
        "gap_closed",
        "learner_progress",
        "attributed_damage_events",
        "min_clearance",
        "unsupported_ticks",
        "falls",
        "stock_changes",
        "failure_category",
    ):
        assert field in diag
    assert diag["relocation_pre_target"]["player_slot"] == 1
    assert diag["relocation_post_target"]["player_slot"] == 1
    assert diag["relocation_gap"] >= 2.0
    assert diag["gap_closed"] >= 1.0
    assert diag["learner_progress"] >= 1.0
    assert len(diag["attributed_damage_events"]) >= 2
    assert diag["min_clearance"] >= 0.2
    assert diag["unsupported_ticks"] == {0: 0, 1: 0}
    assert diag["falls"] == {0: 0, 1: 0}
    assert diag["stock_changes"] == {0: 0, 1: 0}
    assert diag["failure_category"] == "success"


@pytest.mark.parametrize(
    "controller, expected",
    [
        ("idle", "no_attack_request"),
        ("held_light", "started_missed"),
        ("one_hit_then_idle", "first_hit_only"),
        ("move_without_second_hit", "pursued_missed_second"),
        ("walkoff", "own_stock_loss"),
    ],
)
def test_representative_failure_categories(frozen_root, controller, expected):
    env = rc.make_reengagement_env(frozen_root, seed=17, player_slot=0)
    try:
        env.reset(seed=134061, options={"scenario": {"player_slot": 0, "direction": 1}})
        info = finish_episode(env, controller)
    finally:
        env.close()
    assert info["episode_diagnostics"]["failure_category"] == expected


def test_started_missed_request_without_contact_takes_precedence_over_request_only():
    attack_started = {"type": "attack_started", "player_slot": 0, "attack_id": "miss", "attack_type": "DLIGHT"}
    env = state_machine_boundary_harness(tick=300, events=[attack_started])
    env.light_request_ticks = 1
    action = zero_action()
    action[7] = 1
    _, _, terminated, _, info = rc.ReengagementEnv._step(env, action)
    assert terminated is True
    assert info["episode_diagnostics"]["failure_category"] == "started_missed"

    env = state_machine_boundary_harness(tick=300)
    env.light_request_ticks = 1
    action = zero_action()
    action[7] = 1
    _, _, terminated, _, info = rc.ReengagementEnv._step(env, action)
    assert terminated is True
    assert info["episode_diagnostics"]["failure_category"] == "requested_no_start"


def test_bad_support_category_is_reported_from_terminal_diagnostics():
    post = [
        player_snapshot(0, fully_supported=False, clearance=-0.1),
        player_snapshot(1, fully_supported=True, clearance=1.0),
    ]
    env = state_machine_boundary_harness(tick=10, post=post)
    env.hit_attack_id = "first-hit"
    _, _, terminated, _, info = rc.ReengagementEnv._step(env, zero_action())
    assert terminated is True
    assert info["episode_diagnostics"]["failure_category"] == "unsafe_support"


def test_opponent_stock_loss_and_deadline_categories_are_reported(frozen_root):
    pre = [player_snapshot(0, stocks=3), player_snapshot(1, stocks=3)]
    post = [player_snapshot(0, stocks=3), player_snapshot(1, stocks=2)]
    env = state_machine_boundary_harness(pre=pre, post=post)
    _, _, terminated, _, info = rc.ReengagementEnv._step(env, zero_action())
    assert terminated is True
    assert info["episode_diagnostics"]["failure_category"] == "opponent_stock_loss"

    env = state_machine_boundary_harness(tick=160)
    env.encounter = 2
    env.relocation_tick = 100
    env.relocation_gap = 4.0
    env.relocation_learner_x = 0.0
    env.relocation_target_x = 4.0
    env.pursuit_direction = 1
    _, _, terminated, _, info = rc.ReengagementEnv._step(env, zero_action())
    assert terminated is True
    assert info["episode_diagnostics"]["failure_category"] == "failed_renewed_pursuit"
    assert info["episode_diagnostics"]["terminal_reason"] == "response_deadline"

    env = rc.make_reengagement_env(frozen_root, seed=17)
    try:
        env.reset(seed=134031, options={"scenario": {"player_slot": 0, "direction": 1}})
        while env.tick < 299:
            _, _, terminated, _, _ = env.step(zero_action())
            assert not terminated
        _, _, terminated, truncated, info = env.step(zero_action())
        assert terminated is True
        assert truncated is False
        assert env.tick == 300
        assert info["episode_diagnostics"]["outcome"] == "total_deadline"
        with pytest.raises(RuntimeError):
            env.step(zero_action())
    finally:
        env.close()
