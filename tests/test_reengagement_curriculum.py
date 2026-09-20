from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pytest

import reengagement_curriculum as rc


FROZEN = Path("/Users/elliot18/Projects/ml-research/AI_2/baselines/fullgame-20260908/oracle")


def zero_action() -> np.ndarray:
    return np.zeros(10, dtype=np.float32)


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
def test_reset_is_deterministic_private_rng_and_safe_for_both_slots_directions(slot, direction):
    random.seed(987)
    np.random.seed(654)
    py_before = random.getstate()
    np_before = np.random.get_state()

    env = rc.make_reengagement_env(FROZEN, seed=17, player_slot=slot)
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


@pytest.mark.parametrize(
    "scenario",
    [
        {"player_slot": 2, "direction": 1},
        {"player_slot": 0, "direction": 0},
        {"player_slot": 0, "direction": 1, "gap": 3.0},
        {"player_slot": 0, "direction": 1, "gap": 3.2, "midpoint": float("nan")},
    ],
)
def test_invalid_scenarios_are_rejected_before_reset_mutates(scenario):
    env = rc.make_reengagement_env(FROZEN, seed=17)
    try:
        _, info = env.reset(seed=17, options={"scenario": {"player_slot": 0, "direction": 1}})
        before = info["trace"]["post_players"]
        with pytest.raises(ValueError):
            env.reset(seed=17, options={"scenario": scenario})
        after = env._snapshots()
    finally:
        env.close()
    assert after == before


@pytest.mark.parametrize("slot", [0, 1])
@pytest.mark.parametrize("direction", [-1, 1])
def test_scripted_hit_resolve_pursue_hit_resolve_succeeds(slot, direction):
    env = rc.make_reengagement_env(FROZEN, seed=17, player_slot=slot)
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
def test_shortcut_controllers_fail(controller, expected):
    env = rc.make_reengagement_env(FROZEN, seed=17)
    try:
        env.reset(seed=134010, options={"scenario": {"player_slot": 0, "direction": 1}})
        info = finish_episode(env, controller)
    finally:
        env.close()
    assert info["episode_diagnostics"]["outcome"] == expected
    assert info["episode_diagnostics"]["success"] is False


def test_repeated_first_attack_id_and_opponent_self_loss_fail_without_success_transition():
    env = rc.make_reengagement_env(FROZEN, seed=17)
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
        with pytest.raises(RuntimeError, match="invalid damage evidence"):
            env.step(zero_action())
    finally:
        env.close()

    env = rc.make_reengagement_env(FROZEN, seed=17)
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


def test_deadline_boundaries_process_tick_60_90_and_300_but_never_301():
    env = rc.make_reengagement_env(FROZEN, seed=17)
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

    env = rc.make_reengagement_env(FROZEN, seed=17)
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
