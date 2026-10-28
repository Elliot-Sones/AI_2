"""Frozen-source parity for the complete host/device player state reducer."""
from __future__ import annotations

import ctypes
import ast
from dataclasses import dataclass
from enum import Enum, auto
import hashlib
import json
import itertools
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FROZEN = Path(os.environ.get("AI2_FROZEN_SOURCE_ROOT") or
    "/Users/elliot18/Projects/ml-research/AI_2-learning-validation-20260905/"
    "baselines/navigation-repair-20260905/payload/variants/corrected")
SOURCE = FROZEN / "UTMIST-AI2-main/environment/environment.py"
STATE_NAMES = ("WalkingState", "StandingState", "TurnaroundState", "AirTurnaroundState",
               "SprintingState", "StunState", "InAirState", "DodgeState", "AttackState",
               "DashState", "BackDashState", "KOState", "TauntState")
STATE_KEYS = ("walking", "standing", "turnaround", "air_turnaround", "sprinting", "stun",
              "in_air", "dodge", "attack", "dash", "backdash", "KO", "taunt")
MOVE_NAMES = ("NLight", "DLight", "SLight", "NSig", "DSig", "SSig", "NAir", "DAir", "SAir", "Recovery", "Groundpound")


def frozen_namespace():
    """Compile unchanged source AST with real Pymunk/pygame, excluding UI/import startup."""
    import abc
    import pygame
    import pymunk
    data = SOURCE.read_bytes()
    assert hashlib.sha256(data).hexdigest() == "c0116fc7b4a46d45c37367d9aa4d7f44041296ad96ba01e8b1f3f1f158e24230"
    tree = ast.parse(data, filename=str(SOURCE))
    names = set(STATE_NAMES) | {"PlayerObjectState", "GroundState", "Facing", "MoveType", "CompactMoveState",
        "KeyStatus", "HorizontalState", "PlayerInputHandler", "MoveManager", "HurtboxPositionChange",
        "CasterPositionChange", "DealtPositionTarget", "CasterVelocitySet", "CasterVelocityAddXY",
        "CasterVelocitySetXY", "HitVelocitySetXY", "HitVelocityAddXY", "HitPosSetXY", "CasterVelocityDampXY",
        "CastFrameChangeHolder", "Cast", "Power", "Player", "Capsule", "CapsuleCollider"}
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in names:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "m_state_to_move" for t in node.targets):
            nodes.append(node)
    ns = dict(__name__=__name__, ABC=abc.ABC, dataclass=dataclass, Enum=Enum, auto=auto,
              np=np, pygame=pygame, pymunk=pymunk, random=random.Random(91), math=math,
              GameObject=type("GameObject", (), {}), WarehouseBrawl=types.SimpleNamespace(BRAWL_TO_UNITS=1.024/320))
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(SOURCE), "exec"), ns)
    return ns


def oracle_players(ns, positions=((3.0, .35), (4.1, .35))):
    """Minimal real-body scene; player methods/state classes are the frozen source itself."""
    pymunk = ns["pymunk"]
    signal = types.SimpleNamespace(emit=lambda **kw: None)
    env = types.SimpleNamespace(stage_width_tiles=29.8, stage_height_tiles=16.8, fps=30, dt=1/30,
                                logger={}, hit_during_stun=signal, knockout_signal=signal, objects={})
    for key, pos, vertices in (
        ("ground1", (4.5, 1.0), [(-2.5,-.15),(2.5,-.15),(2.5,4),(-2.5,4)]),
        ("ground2", (-4.5, 3.0), [(-2.5,-.15),(2.5,-.15),(2.5,4),(-2.5,4)]),
        ("platform1", (0, 1), [(-1,-.1),(1,-.1),(1,.1),(-1,.1)]),
    ):
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = pos
        env.objects[key] = types.SimpleNamespace(body=body, shape=pymunk.Poly(body, vertices))
    for directory, family, attr in (("unarmed_attacks", "Unarmed", "attacks"), ("spear_attacks", "Spear", "spear_attacks"), ("hammer_attacks", "Hammer", "hammer_attacks")):
        setattr(env, attr, {ns["MoveType"](i+2): json.loads((SOURCE.parent / directory / f"{family} {name}.json").read_text()) for i,name in enumerate(MOVE_NAMES)})
    players = []
    for i, pos in enumerate(positions):
        p = ns["Player"].__new__(ns["Player"])
        p.env, p.agent_id = env, i
        p.body = pymunk.Body(1, 1e9)
        p.body.position = pos
        p.shape = pymunk.Poly.create_box(p.body, size=(.87, 1.0))
        p.start_position = pos
        p.input = ns["PlayerInputHandler"]()
        p.facing = ns["Facing"].RIGHT if i == 0 else ns["Facing"].LEFT
        p.states = {key: ns[name](p) for key,name in zip(STATE_KEYS, STATE_NAMES)}
        p.state = p.states["standing"]
        p.state_str = "StandingState"
        p.weapon = "Punch"
        p.damage = p.damage_taken_this_stock = p.damage_taken_total = p.damage_taken_this_frame = p.damage_done = 0.0
        p.stocks = 3
        p.damage_velocity = p.target_vel = (0.0, 0.0)
        p.prev_x, p.prev_y = pos
        p.just_got_hit = False
        p.hitboxes_to_draw, p.points_to_draw = [], []
        p.hurtbox_collider = ns["CapsuleCollider"](pos, .928, 1.024)
        for attr,value in dict(move_speed=6.75,jump_speed=8.9,max_fall_speed=12,fast_fall_ease=14.25/30,
            in_air_ease=10.75/30,bounce_coef=.7,run_speed=8,dash_speed=10,backdash_speed=4,
            turnaround_time=1,taunt_time=30,backdash_time=7,dodge_time=10,grounded_dodge_cooldown=30,
            air_dodge_cooldown=82,invincible_time=90,jump_cooldown=6,dash_time=9).items():
            setattr(p,attr,value)
        players.append(p)
    players[0].opponent, players[1].opponent = players[1],players[0]
    env.players = players
    return players


def player_snapshot(p):
    atk = p.states["attack"]
    mm = getattr(atk, "move_manager", None)
    pw = mm.current_power if mm is not None else None
    cast = pw.casts[pw.cast_idx] if pw is not None else None
    return [*p.body.position, *p.body.velocity, p.damage, p.stocks, int(p.facing), STATE_NAMES.index(type(p.state).__name__),
        p.damage_taken_this_frame,p.damage_taken_this_stock,p.damage_taken_total,p.damage_done,
        *p.hurtbox_collider.center,p.hurtbox_collider.width,p.hurtbox_collider.height,
        p.state.stun_time_stored,getattr(p.states["stun"],"stun_frames",0),
        pw.cast_idx if pw else 0,mm.frame if mm else 0,pw.total_frame_count if pw else 0,
        pw.frames_into_recovery if pw else 0,cast.frame_idx if cast else 0,int(pw.hit_anyone) if pw else 0,
        sum(1 << q.agent_id for q in mm.all_hit_agents) if mm else 0,
        sum(1 << q.agent_id for q in pw.agents_hit_this_power) if pw else 0,
        *p.damage_velocity,*p.target_vel]


def run_native_command(command, *, input=None):
    """Launch once; keep native failures intact and avoid macOS OpenMP atfork hooks.

    An absolute executable and close_fds=False select CPython's posix_spawn path
    on macOS. Python-created descriptors remain non-inheritable by default; this
    test harness does not create inheritable descriptors or pass extra FDs.
    """
    executable = shutil.which(command[0])
    if executable is None:
        raise FileNotFoundError(command[0])
    result = subprocess.run([executable, *command[1:]], input=input, text=True,
                            capture_output=True, close_fds=sys.platform != "darwin")
    if result.returncode:
        raise AssertionError(repr(result))
    return result


def compiled_pair(directory):
    import ai2.simulation.native_navigation_episode as m
    output = Path(directory) / "player.so"
    run_native_command(["cc","-std=c99","-O2","-shared","-fPIC","-ffp-contract=off","-I",str(ROOT/"native"),
        str(ROOT/"native/game_player.c"),str(ROOT/"native/navigation_episode.c"),"-lm","-o",str(output)])
    return m, m.EpisodeReference(output)


class NativePlayerHarnessTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "darwin", "macOS OpenMP fork regression")
    def test_compiler_uses_posix_spawn_without_running_fork_child_handlers(self):
        with tempfile.TemporaryDirectory(prefix="ai2-game-player-launch-") as directory:
            with patch.object(subprocess, "_fork_exec", side_effect=AssertionError(
                    "compiler must not fork after OpenMP initialization")):
                with patch.object(os, "posix_spawn", wraps=os.posix_spawn) as spawn:
                    compiled_pair(directory)
            self.assertEqual(spawn.call_count, 1)

    def test_compiler_abort_keeps_command_status_and_both_streams_without_retry(self):
        failure = subprocess.CompletedProcess(["cc"], -6, "stdout marker", "")
        with patch.object(subprocess, "run", return_value=failure) as run:
            with self.assertRaises(AssertionError) as raised:
                compiled_pair("unused-after-compiler-failure")
        run.assert_called_once()
        message = str(raised.exception)
        self.assertIn("returncode=-6", message)
        self.assertIn("args=['cc']", message)
        self.assertIn("stdout='stdout marker'", message)
        self.assertIn("stderr=''", message)


class NativeGamePlayerTests(unittest.TestCase):
    def test_malformed_indices_are_rejected_before_access_under_ubsan(self):
        """Run hostile state in subprocesses so sanitizer failures cannot kill unittest."""
        probe = r'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "game_player.c"
int main(int argc, char **argv) {
    AI2NEpisode g, old_g; AI2GCombatPlayer e[2], old_e[2]; AI2NRng rng, old_rng;
    int field, value, expected, rc, done=99, source=0;
    if (argc!=4) return 91;
    field=atoi(argv[1]); value=atoi(argv[2]); expected=atoi(argv[3]);
    if (ai2g_probe_scene(&g,e,&rng)) return 92;
    if (field>=13 && field<=16) {
        if (ai2g_attack_begin(&g,e,0,2)) return 93;
        g.players[0].ms.state_id=AI2G_STATE_ATTACK;
    }
    switch (field) {
        case 0: g.players[0].body=value; break;
        case 1: g.players[1].body=value; break;
        case 2: g.platform_body=value; break;
        case 3: g.players[0].shape=value; break;
        case 4: g.players[1].shape=value; break;
        case 5: g.platform_shape=value; break;
        case 6: g.world.shapes[0].body=value; break;
        case 7: g.world.shapes[1].body=value; break;
        case 8: g.world.body_count=value; break;
        case 9: g.world.shape_count=value; break;
        case 10: g.world.shapes[0].count=value; break;
        case 11: g.world.shapes[g.players[0].shape].count=value; break;
        case 12: g.world.shapes[g.platform_shape].count=value; break;
        case 13: e[0].attack.move_idx=value; break;
        case 14: e[0].attack.power_idx=value; break;
        case 15: e[0].attack.cast_idx=value; break;
        case 16: e[0].attack.first_hit_agent=value; break;
        case 17: e[0].event_count=value; break;
        case 18: source=value; break;
        case 19: e[0].total_tick_events=value; break;
        case 20:
            if (ai2g_attack_begin(&g,e,0,2)) return 94;
            e[0].attack.move_idx=value; break;
        default: return 95;
    }
    memcpy(&old_g,&g,sizeof(g)); memcpy(old_e,e,sizeof(e)); memcpy(&old_rng,&rng,sizeof(rng));
    if (field>=17 && field<=19) rc=ai2g_emit(e,source,AI2G_EVENT_ATTACK_STARTED,-1,-1,0);
    else if (field==20) rc=ai2g_attack_step(&g,e,0,&done);
    else rc=ai2g_player_physics(&g,e,0,&rng);
    if (rc!=expected) { fprintf(stderr,"field=%d value=%d expected=%d got=%d\n",field,value,expected,rc); return 96; }
    /* An error flag may change, but no scene, RNG, event payload or gameplay state may. */
    e[0].event_overflow=old_e[0].event_overflow;
    e[1].event_overflow=old_e[1].event_overflow;
    if (memcmp(&g,&old_g,sizeof(g)) || memcmp(e,old_e,sizeof(e)) || memcmp(&rng,&old_rng,sizeof(rng))) {
        fprintf(stderr,"field=%d value=%d changed non-error state\n",field,value); return 97;
    }
    return 0;
}
'''
        cases = []
        for field in (0, 1, 2, 6, 7):
            cases.extend((field, value, 101) for value in (-1, 5, 12))
        for field in (3, 4, 5):
            cases.extend((field, value, 101) for value in (-1, 5, 12))
        cases.extend((8, value, 101) for value in (-1, 0, 13))
        cases.extend((9, value, 101) for value in (-1, 1, 13))
        for field in (10, 11, 12):
            cases.extend((field, value, 101) for value in (-1, 0, 5))
        cases.extend((13, value, 102) for value in (-1, 33))
        cases.extend((14, value, 102) for value in (-1, 107))
        cases.extend((15, value, 102) for value in (-1, 9))
        cases.extend((16, value, 102) for value in (-2, 2, 32))
        cases.extend((17, value, 104) for value in (-1, 32))
        cases.extend((18, value, 101) for value in (-1, 2))
        cases.extend((19, value, 104) for value in (-1, 2147483647))
        cases.extend((20, value, 102) for value in (-1, 33))
        with tempfile.TemporaryDirectory(prefix="ai2-game-player-invalid-") as directory:
            executable = Path(directory)/"guard-probe"
            run_native_command(["cc", "-std=c99", "-O1", "-g", "-fsanitize=undefined",
                "-fno-sanitize-recover=undefined", "-ffp-contract=off", "-I", str(ROOT/"native"),
                "-x", "c", "-", "-lm", "-o", str(executable)], input=probe)
            for field, value, expected in cases:
                with self.subTest(field=field, value=value):
                    run_native_command([str(executable), str(field), str(value), str(expected)])

    def test_combat_extra_initialization_preserves_legacy_scene(self):
        with tempfile.TemporaryDirectory(prefix="ai2-game-player-") as directory:
            run_native_command(
                ["cc", "-std=c99", "-shared", "-fPIC", "-ffp-contract=off",
                 "-I", str(ROOT / "native"), str(ROOT / "native/game_player.c"),
                 "-lm", "-o", str(Path(directory) / "player.so")],
            )
            lib = ctypes.CDLL(str(Path(directory) / "player.so"))
            lib.ai2g_player_initialization_probe.restype = ctypes.c_int
            self.assertEqual(lib.ai2g_player_initialization_probe(), 0)

    def test_attack_state_is_not_rejected_as_navigation_input(self):
        with tempfile.TemporaryDirectory(prefix="ai2-game-player-") as directory:
            run_native_command(
                ["cc", "-std=c99", "-shared", "-fPIC", "-ffp-contract=off", "-I", str(ROOT / "native"),
                 str(ROOT / "native/game_player.c"), "-lm", "-o", str(Path(directory) / "player.so")],
            )
            lib = ctypes.CDLL(str(Path(directory) / "player.so"))
            self.assertTrue(hasattr(lib, "ai2g_player_physics_export"), "full gameplay player entry point is missing")

    def test_dynamic_python_attribute_presence_is_explicit(self):
        with tempfile.TemporaryDirectory(prefix="ai2-game-player-presence-") as directory:
            _, ref = compiled_pair(directory)
            self.assertTrue(hasattr(ref.lib,"ai2g_player_presence_probe"),"inactive-state attribute presence is missing")
            self.assertEqual(ref.lib.ai2g_player_presence_probe(),0)

    def test_events_keep_global_cross_player_execution_order(self):
        with tempfile.TemporaryDirectory(prefix="ai2-game-player-events-") as directory:
            _, ref = compiled_pair(directory)
            self.assertTrue(hasattr(ref.lib,"ai2g_player_event_order_probe"),"cross-player event order is missing")
            self.assertEqual(ref.lib.ai2g_player_event_order_probe(),0)

    def test_fullgame_transition_priority_and_source_quirks(self):
        with tempfile.TemporaryDirectory(prefix="ai2-game-player-priority-") as directory:
            _, ref = compiled_pair(directory)
            self.assertTrue(hasattr(ref.lib,"ai2g_player_priority_probe"),"transition priority proof is missing")
            self.assertEqual(ref.lib.ai2g_player_priority_probe(),0)

    def test_every_loaded_attack_matches_frozen_player_timeline_both_slots(self):
        ns = frozen_namespace()
        with tempfile.TemporaryDirectory(prefix="ai2-game-player-oracle-") as directory:
            m, ref = compiled_pair(directory)
            self.assertTrue(hasattr(ref.lib,"ai2g_player_force_attack_export"),"attack execution is missing")
            lib = ref.lib
            lib.ai2g_combat_player_size.restype = ctypes.c_int
            for name in ("ai2g_player_init_export","ai2g_player_pre_process_export"):
                getattr(lib,name).argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]
            lib.ai2g_player_set_pose_export.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]+[ctypes.c_double]*4+[ctypes.c_int,ctypes.c_int]
            lib.ai2g_player_force_attack_export.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
            lib.ai2g_player_physics_export.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p]
            lib.ai2g_player_decode_export.argtypes = [ctypes.c_void_p,ctypes.c_int,ctypes.POINTER(ctypes.c_double)]
            lib.ai2g_player_snapshot_export.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.POINTER(ctypes.c_double)]
            cfg = ref.make_config(dict(min_dist=2,max_dist=6,fall_penalty=-3,completion_mode="sequence",control_mode="movement",
                settle_speed=1,max_targets=3,time_limit=600,settle_steps=9,post_goal_required_steps=15))
            cases=0
            for weapon,direction,gap in itertools.product(range(3),(-1,1),(1.1,8.0)):
                for actor in (0,1):
                    for move in range(2,13):
                        with self.subTest(weapon=weapon,actor=actor,move=move,direction=direction,gap=gap):
                            cases+=1
                            rng = m.Rng()
                            ref.lib.ai2n_rng_seed_export(ctypes.byref(rng),91,0)
                            ep = m.Episode()
                            self.assertEqual(ref.reset(ep,cfg,rng),0)
                            extra = ctypes.create_string_buffer(lib.ai2g_combat_player_size()*2)
                            y=.35 if move<8 else -2.0
                            positions=[None,None]
                            positions[actor]=(3.5,y)
                            positions[1-actor]=(3.5+direction*gap,y)
                            if gap<2:
                                family=("Unarmed","Spear","Hammer")[weapon]
                                directory=("unarmed_attacks","spear_attacks","hammer_attacks")[weapon]
                                data=json.loads((SOURCE.parent/directory/f"{family} {MOVE_NAMES[move-2]}.json").read_text())
                                hitbox=next(h for power in data["powers"] for cast in power["casts"]
                                    for h in (cast.get("hitboxes") or [])
                                    if cast.get("baseDamage",0)>0 and cast.get("attackFrames",0)//2>0)
                                hx=3.5+hitbox["xOffset"]*2*(1.024/320)*direction
                                hy=y+hitbox["yOffset"]*2*(1.024/320)
                                width=hitbox["width"]*2*(1.024/320)
                                height=hitbox["height"]*2*(1.024/320)
                                radius=min(width,height)/2
                                # Aim at an actual cap, not a quantized-away pygame.Rect middle.
                                if width<height:
                                    hy-=height/2-radius
                                elif width>height:
                                    hx-=width/2-radius
                                positions[1-actor]=(hx,hy)
                            players = oracle_players(ns,positions)
                            for i,p in enumerate(players):
                                lib.ai2g_player_init_export(ctypes.byref(ep),extra,i)
                                facing=direction if i==actor else -direction
                                p.facing=ns["Facing"](facing)
                                lib.ai2g_player_set_pose_export(ctypes.byref(ep),extra,i,*positions[i],0.,0.,1,facing)
                                # Native scene initialization creates InAirState fields; mirror them.
                                p.states["in_air"].set_jumps(0,0,0)
                                p.states["in_air"].is_base=True
                            players[actor].weapon = ("Punch","Spear","Hammer")[weapon]
                            ep.players[actor].weapon = weapon
                            attack = players[actor].states["attack"]
                            attack.refresh()
                            attack.give_move(ns["MoveType"](move))
                            attack.reset(players[actor].state)
                            players[actor].state = attack
                            attack.enter()
                            self.assertEqual(lib.ai2g_player_force_attack_export(ctypes.byref(ep),extra,actor,move,ctypes.byref(rng)),0)
                            for tick in range(180):
                                for i,p in enumerate(players):
                                    action = np.zeros(10)
                                    if i == actor and tick < 12:
                                        action[8 if move in (5,6,7,11,12) else 7] = 1
                                    p.pre_process()
                                    p.input.update(action)
                                    lib.ai2g_player_pre_process_export(ctypes.byref(ep),extra,i)
                                    self.assertEqual(lib.ai2g_player_decode_export(ctypes.byref(ep),i,(ctypes.c_double*10)(*action)),0)
                                for i,p in enumerate(players):
                                    p.physics_process(1/30)
                                    self.assertEqual(lib.ai2g_player_physics_export(ctypes.byref(ep),extra,i,ctypes.byref(rng)),0)
                                for i,p in enumerate(players):
                                    got = (ctypes.c_double*30)()
                                    lib.ai2g_player_snapshot_export(ctypes.byref(ep),extra,i,got)
                                    self.assertEqual(list(got),player_snapshot(p),f"tick={tick} player={i}")
                                    for sid,key in enumerate(STATE_KEYS):
                                        state=p.states[key]
                                        for field,attr in m.SLOT_NAMES.items():
                                            expected_present=attr in state.__dict__
                                            self.assertEqual(bool(ep.players[i].ms.bank[sid].present[field]),expected_present,
                                                f"tick={tick} player={i} state={key} field={attr} presence")
                                            if expected_present:
                                                self.assertEqual(ep.players[i].ms.bank[sid].value[field],int(state.__dict__[attr]),
                                                    f"tick={tick} player={i} state={key} field={attr}")
                                if players[actor].state is not attack:
                                    break
                            else:
                                self.fail("loaded acyclic attack did not complete within 180 player ticks")
                            if gap<2:
                                self.assertGreater(players[actor].damage_done,0,"positive fixture must actually damage its target")
                            else:
                                self.assertEqual(players[actor].damage_done,0,"whiff fixture must not damage its target")
            self.assertEqual(cases,264)


if __name__ == "__main__":
    unittest.main()
