"""Policy-free weapon lifecycle tests; compile the same host/device header."""
from __future__ import annotations

import ctypes as C
import ast
import hashlib
import importlib
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import types
from typing import List
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
nav = importlib.import_module('native_navigation_episode')
INT32 = C.c_int32
D = C.c_double


class Aux(C.Structure):
    _fields_ = [(name, D) for name in ('hurtbox_x', 'hurtbox_y', 'hurtbox_width', 'hurtbox_height')] + [
        ('pickup_lock_until', INT32), ('pad', INT32)]


class Weapon(C.Structure):
    _fields_ = [('world_pos', D * 2)] + [(name, INT32) for name in (
        'weapon', 'active', 'body', 'shape', 'spawn_frame', 'physics_on')]


class Spawner(C.Structure):
    _fields_ = [('world_pos', D * 2)] + [(name, INT32) for name in (
        'id', 'kind', 'og_cooldown_frames', 'cooldown_frames', 'last_spawn_frame',
        'despawn_frames', 'weapon_slot', 'flag', 'spawned_once', 'done', 'flipped', 'weapon')]


class Event(C.Structure):
    _fields_ = [(name, INT32) for name in ('kind', 'agent', 'weapon', 'spawner_id')]


class Weapons(C.Structure):
    _fields_ = [('px_per_world', D)] + [(name, INT32) for name in (
        'initialized', 'next_spawner_id', 'spawner_count', 'weapon_count', 'pool_count',
        'event_count', 'last_error', 'pad')] + [
        ('spawners', Spawner * 8), ('weapons', Weapon * 8), ('pool', INT32 * 8), ('events', Event * 8)]


FROZEN = Path(os.environ.get('AI2_FROZEN_SOURCE_ROOT') or
              '/Users/elliot18/Projects/ml-research/AI_2-learning-validation-20260905/baselines/'
              'navigation-repair-20260905/payload/variants/corrected') / 'UTMIST-AI2-main/environment'
FROZEN_SHA = 'c0116fc7b4a46d45c37367d9aa4d7f44041296ad96ba01e8b1f3f1f158e24230'


def frozen_weapon_classes():
    """Real frozen class bodies; substitute only rendering and event observers.

    Pymunk bodies/shapes and pygame capsule math remain real. This is lifecycle
    differential testing, not a claim of complete match/physics parity.
    """
    import pygame
    import pymunk
    from enum import Enum, auto

    source = (FROZEN/'environment.py').read_bytes()
    if hashlib.sha256(source).hexdigest() != FROZEN_SHA:
        raise AssertionError('frozen weapon oracle hash mismatch')

    class Visual:
        def __init__(self, **kwargs):
            self.world_pos = kwargs['world_pos']

        def _steps(self, name):
            return 36

        def show_spawn(self):
            pass

        show_despawn = show_pickup = show_spawn

    names = {'MoveType', 'Facing', 'WeaponGO', 'WeaponPool', 'WeaponSpawner',
             'WeaponSpawnController', 'DroppedWeaponSpawner', 'CapsuleCollider'}
    selected = [node for node in ast.parse(source).body
                if isinstance(node, ast.ClassDef) and node.name in names]
    namespace = {'pygame': pygame, 'pymunk': pymunk, 'Enum': Enum, 'auto': auto, 'GameObject': object,
                 'Player': object, 'SpawnerVFX': Visual, 'List': List,
                 'random': random.Random(31), '_get_asset_path': lambda name: name,
                 'WEAPON_CAT': 4, 'GROUND_CAT': 2, 'AttackState': type('AttackState', (), {})}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(FROZEN/'environment.py'), 'exec'), namespace)
    return namespace


class WeaponTests(unittest.TestCase):
    library = None

    def setUp(self):
        self.assertTrue((ROOT/'native/game_weapons.h').is_file(), 'full-game weapon header missing')
        if self.__class__.library is None:
            temporary = tempfile.TemporaryDirectory(prefix='ai2-game-weapons-')
            self.__class__.addClassCleanup(temporary.cleanup)
            library = Path(temporary.name)/'weapons.so'
            subprocess.run(['cc', '-std=c99', '-O2', '-shared', '-fPIC', '-ffp-contract=off',
                            str(ROOT/'native/game_weapons.c'), str(ROOT/'native/navigation_episode.c'),
                            '-lm', '-o', str(library)], check=True, capture_output=True, text=True)
            self.__class__.library = C.CDLL(str(library))
            lib = self.library
            for name, expected in (('state', Weapons), ('aux', Aux), ('weapon', Weapon), ('spawner', Spawner)):
                self.assertEqual(getattr(lib, 'ai2g_weapons_'+name+'_size')(), C.sizeof(expected))
            self.__class__.reference = nav.EpisodeReference(library)
            lib.ai2g_weapons_init_export.argtypes = [C.POINTER(nav.Episode), C.POINTER(Weapons), C.POINTER(Aux), D]
            lib.ai2g_weapons_mirror_export.argtypes = [C.POINTER(nav.Episode), C.POINTER(Weapons)]
            for name in ('drop', 'pickup'):
                getattr(lib, 'ai2g_weapons_'+name+'_export').argtypes = [
                    C.POINTER(nav.Episode), C.POINTER(Weapons), C.POINTER(Aux), C.POINTER(nav.Rng)]
            lib.ai2g_weapons_update_export.argtypes = [C.POINTER(nav.Episode), C.POINTER(Weapons), C.POINTER(nav.Rng)]
        self.scene = nav.Episode()
        self.system = Weapons()
        self.aux = (Aux * 2)()
        self.rng = nav.Rng()
        self.library.ai2n_rng_seed_export(C.byref(self.rng), 7, 0)
        config = self.reference.make_config({
            'completion_mode': 'sequence', 'control_mode': 'movement', 'max_targets': 3,
            'time_limit': 600, 'settle_steps': 9, 'post_goal_required_steps': 15,
            'settle_speed': 1.0, 'fall_penalty': -3.0, 'min_dist': 2.0, 'max_dist': 6.0})
        self.assertEqual(self.reference.reset(self.scene, config, self.rng), 0)
        self.assertEqual(self.init(), 0)
        for spawner in self.system.spawners[:2]:
            spawner.cooldown_frames = 100000
            spawner.last_spawn_frame = 0

    def init(self, scale=48.0):
        return self.library.ai2g_weapons_init_export(C.byref(self.scene), C.byref(self.system), self.aux, scale)

    def call(self, name):
        args = [C.byref(self.scene), C.byref(self.system)]
        if name in ('drop', 'pickup'):
            args.append(self.aux)
        if name != 'mirror':
            args.append(C.byref(self.rng))
        return getattr(self.library, 'ai2g_weapons_'+name+'_export')(*args)

    def tape(self, *draws):
        self.rng, self.keep = self.reference.replay_rng([
            {'kind': kind, 'a': a, 'b': b, 'value': value} for kind, a, b, value in draws])

    def spawn(self, weapon=1):
        self.scene.steps = 10
        self.system.spawners[0].cooldown_frames = 10
        self.tape((2, 2.6, 6.5, 4.0), (3, 0, 1, weapon-1))
        self.assertEqual(self.call('update'), 0)
        self.assertEqual(self.rng.cursor, 2)
        return self.system.weapons[self.system.spawners[0].weapon_slot]

    def aim_pickup(self, player=0, edge=True):
        weapon = self.system.weapons[self.system.spawners[0].weapon_slot]
        self.aux[player].hurtbox_x, self.aux[player].hurtbox_y = weapon.world_pos
        self.scene.players[player].input_state[5] = 1
        self.scene.players[player].input_state[15] = int(edge)

    def drop(self, player=0, weapon=1):
        self.scene.players[player].weapon = weapon
        self.scene.players[player].input_state[15] = 1
        self.tape((3, 0, 350, 123))
        self.assertEqual(self.call('drop'), 0)
        self.assertEqual(self.rng.cursor, 1)
        return self.system.spawners[self.system.spawner_count-1]

    def test_init_copies_clocks_without_rng_and_keeps_next_id_on_reset(self):
        self.system.next_spawner_id = 71
        self.assertEqual(self.init(), 0)
        self.assertEqual(self.system.next_spawner_id, 71)
        self.assertEqual(self.system.spawner_count, 2)
        self.assertEqual(self.system.spawners[0].cooldown_frames, self.scene.spawners[0].cooldown_frames)
        self.assertEqual(self.aux[0].hurtbox_width, 145*2*(1.024/320))
        self.assertEqual(self.aux[0].hurtbox_height, 160*2*(1.024/320))
        self.assertEqual(self.aux[0].pickup_lock_until, -1)

    def test_regular_spawn_body_and_eligibility_boundary(self):
        weapon = self.spawn(2)
        self.assertEqual(weapon.weapon, 2)
        self.assertEqual(tuple(weapon.world_pos), (4.0, 0.7))
        self.assertEqual(self.scene.world.shapes[weapon.shape].categories, 4)
        self.assertEqual(self.scene.world.shapes[weapon.shape].mask, 2)
        self.aim_pickup()
        self.scene.steps = 46
        self.assertEqual(self.call('pickup'), 0)
        self.assertEqual(self.scene.players[0].weapon, 0)
        self.assertEqual(self.call('update'), 0)
        self.assertEqual(self.system.spawners[0].flag, 1)
        self.scene.steps = 47
        original = self.system.spawners[0].og_cooldown_frames
        self.tape((3, int(original*.5), int(original*1.5), original))
        self.assertEqual(self.call('pickup'), 0)
        self.assertEqual(self.scene.players[0].weapon, 2)
        self.assertEqual(self.system.spawners[0].last_spawn_frame, 47)
        self.assertEqual(self.system.events[0].kind, 2)

    def test_regular_held_pickup_slot_zero_wins_and_attack_eight_is_excluded(self):
        self.spawn()
        self.system.spawners[0].flag = 1
        self.aim_pickup(0, edge=False)
        self.aim_pickup(1, edge=False)
        self.scene.players[0].ms.state_id = 8
        original = self.system.spawners[0].og_cooldown_frames
        self.tape((3, int(original*.5), int(original*1.5), original))
        self.assertEqual(self.call('pickup'), 0)
        self.assertEqual([p.weapon for p in self.scene.players], [0, 1])

    def test_regular_expiry_is_strict_and_keeps_spawn_clock(self):
        self.spawn()
        self.scene.steps = 360
        self.assertEqual(self.call('update'), 0)
        self.assertGreaterEqual(self.system.spawners[0].weapon_slot, 0)
        self.scene.steps = 361
        original = self.system.spawners[0].og_cooldown_frames
        self.tape((3, int(original*.5), int(original*1.5), original))
        self.assertEqual(self.call('update'), 0)
        self.assertEqual(self.system.spawners[0].weapon_slot, -1)
        self.assertEqual(self.system.spawners[0].last_spawn_frame, 10)
        self.assertEqual(self.system.pool_count, 1)

    def test_drop_position_lock_spawn_phase_and_inclusive_expiry(self):
        self.scene.steps = 20
        self.scene.players[0].ms.facing = -1
        body = self.scene.world.bodies[self.scene.players[0].body]
        dropped = self.drop()
        self.assertEqual(tuple(dropped.world_pos), (body.p.x-.1, body.p.y+.27))
        self.assertEqual(dropped.weapon_slot, -1)
        self.assertEqual(self.aux[0].pickup_lock_until, 35)
        self.assertEqual(self.scene.players[0].weapon, 0)
        self.assertEqual(self.call('pickup'), 0)
        self.assertEqual(self.call('update'), 0)
        self.assertEqual(self.system.spawners[2].spawned_once, 1)
        self.scene.steps = 269
        self.assertEqual(self.call('update'), 0)
        self.assertEqual(self.system.spawner_count, 3)
        self.scene.steps = 270
        self.assertEqual(self.call('update'), 0)
        self.assertEqual(self.system.spawner_count, 2)

    def test_dropped_pickup_requires_edge_and_lock_expiry(self):
        self.scene.steps = 20
        self.drop(weapon=2)
        self.assertEqual(self.call('update'), 0)
        dropped = self.system.spawners[2]
        self.aux[0].hurtbox_x, self.aux[0].hurtbox_y = dropped.world_pos
        self.scene.steps = 34
        self.assertEqual(self.call('pickup'), 0)
        self.assertEqual(self.scene.players[0].weapon, 0)
        self.scene.steps = 35
        self.scene.players[0].input_state[15] = 0
        self.scene.players[0].input_state[5] = 1
        self.assertEqual(self.call('pickup'), 0)
        self.assertEqual(self.scene.players[0].weapon, 0)
        self.scene.players[0].input_state[15] = 1
        self.assertEqual(self.call('pickup'), 0)
        self.assertEqual(self.scene.players[0].weapon, 2)
        self.assertEqual(self.call('update'), 0)
        self.assertEqual(self.system.spawner_count, 2)

    def test_controller_length_blocks_regular_spawning_until_next_update_after_prune(self):
        self.drop()
        self.system.spawners[0].cooldown_frames = 1
        self.scene.steps = 10
        self.assertEqual(self.call('update'), 0)
        self.assertEqual(self.system.spawners[0].weapon_slot, -1)
        self.scene.steps = 260
        self.assertEqual(self.call('update'), 0)
        self.assertEqual(self.system.spawner_count, 2)
        self.assertEqual(self.system.spawners[0].weapon_slot, -1)
        self.tape((2, 2.6, 6.5, 3.0), (3, 0, 1, 0))
        self.assertEqual(self.call('update'), 0)
        self.assertGreaterEqual(self.system.spawners[0].weapon_slot, 0)

    def test_mirror_changes_weapon_position_but_not_spawner_anchor(self):
        weapon = self.spawn()
        self.scene.world.bodies[weapon.body].p.x = 5.25
        self.scene.world.bodies[weapon.body].p.y = 2.0
        self.assertEqual(self.call('mirror'), 0)
        self.assertEqual(tuple(weapon.world_pos), (5.25, 2.0))
        self.assertEqual(tuple(self.system.spawners[0].world_pos), (4.0, .7))

    def test_pool_reuses_matching_weapon_and_removes_body_shape(self):
        weapon = self.spawn()
        body, shape = weapon.body, weapon.shape
        self.system.spawners[0].flag = 1
        self.aim_pickup()
        original = self.system.spawners[0].og_cooldown_frames
        self.tape((3, int(original*.5), int(original*1.5), original))
        self.assertEqual(self.call('pickup'), 0)
        self.assertEqual(self.scene.world.body_active[body], 0)
        self.assertEqual(self.scene.world.shape_active[shape], 0)
        self.scene.steps = 11
        self.system.spawners[0].cooldown_frames = 1
        self.tape((2, 2.6, 6.5, 3.0), (3, 0, 1, 0))
        self.assertEqual(self.call('update'), 0)
        self.assertEqual(self.system.weapon_count, 1)
        self.assertEqual(self.system.pool_count, 0)
        self.assertEqual(self.system.spawners[0].weapon_slot, 0)

    def test_bad_geometry_and_replay_exhaustion_fail_closed(self):
        self.assertNotEqual(self.init(float('nan')), 0)
        self.assertEqual(self.init(), 0)
        self.system.spawners[0].cooldown_frames = 0
        self.tape()
        self.assertNotEqual(self.call('update'), 0)
        self.assertNotEqual(self.system.last_error, 0)

    def test_capacity_exhaustion_does_not_overwrite(self):
        self.system.spawner_count = 8
        self.scene.players[0].weapon = 1
        self.scene.players[0].input_state[15] = 1
        self.tape((3, 0, 350, 100))
        self.assertNotEqual(self.call('drop'), 0)
        self.assertEqual(self.scene.players[0].weapon, 1)
        self.assertEqual(self.system.spawner_count, 8)

    def test_regular_claim_order_stun_allowed_and_lock_does_not_apply(self):
        self.spawn()
        self.system.spawners[0].flag = 1
        self.aim_pickup(0)
        self.aim_pickup(1)
        self.scene.players[0].ms.state_id = 5
        self.aux[0].pickup_lock_until = 99999
        original = self.system.spawners[0].og_cooldown_frames
        self.tape((3, int(original*.5), int(original*1.5), original))
        self.assertEqual(self.call('pickup'), 0)
        self.assertEqual([p.weapon for p in self.scene.players], [1, 0])

    def test_bad_replay_value_and_hurtbox_geometry_are_errors_not_misses(self):
        self.system.spawners[0].cooldown_frames = 0
        self.tape((2, 2.6, 6.5, float('nan')))
        self.assertEqual(self.call('update'), 105)
        self.assertEqual(self.init(), 0)
        for sp in self.system.spawners[:2]:
            sp.cooldown_frames = 100000
            sp.last_spawn_frame = 0
        self.spawn()
        self.system.spawners[0].flag = 1
        self.aim_pickup()
        self.aux[0].hurtbox_width = float('nan')
        self.assertEqual(self.call('pickup'), 103)
        self.assertEqual(self.scene.players[0].weapon, 0)

    def test_generated_mode_runs_all_weapon_phases_and_recycles(self):
        self.system.spawners[0].cooldown_frames = 1
        self.system.spawners[1].cooldown_frames = 1
        equipped = set()
        for tick in range(1, 1600):
            self.scene.steps = tick
            self.assertEqual(self.call('mirror'), 0)
            active = [sp for sp in self.system.spawners[:self.system.spawner_count] if sp.weapon_slot >= 0]
            for i in range(2):
                if active:
                    weapon = self.system.weapons[active[i % len(active)].weapon_slot]
                    self.aux[i].hurtbox_x, self.aux[i].hurtbox_y = weapon.world_pos
                    body = self.scene.world.bodies[self.scene.players[i].body]
                    body.p.x, body.p.y = weapon.world_pos
                self.scene.players[i].input_state[5] = int(tick % 9 in (0, 1, 2))
                self.scene.players[i].input_state[15] = int(tick % 9 == 0)
            for name in ('drop', 'pickup', 'update'):
                self.assertEqual(self.call(name), 0, (tick, name))
            equipped.update(e.weapon for e in self.system.events[:self.system.event_count] if e.kind == 2)
        self.assertEqual(equipped, {1, 2})
        self.assertGreater(self.system.next_spawner_id, 10)
        self.assertEqual(self.rng.mode, 1)
        self.assertEqual(self.rng.error, 0)

    def test_body_capacity_exhaustion_returns_sticky_error(self):
        for i in range(12):
            self.scene.world.body_active[i] = 1
        self.scene.world.body_count = 12
        self.system.spawners[0].cooldown_frames = 0
        self.tape((2, 2.6, 6.5, 4.0), (3, 0, 1, 0))
        self.assertEqual(self.call('update'), 104)
        self.assertEqual(self.system.weapon_count, 0)
        self.assertEqual(self.call('mirror'), 104)

    def test_corrupt_world_counts_fail_before_indexing_in_every_phase(self):
        for field, capacity in (('body_count', 12), ('shape_count', 12),
                                ('arbiter_count', 24), ('solve_count', 24), ('handler_count', 4)):
            for value in (-1, capacity + 1):
                for phase in ('mirror', 'drop', 'pickup', 'update', 'init'):
                    with self.subTest(field=field, value=value, phase=phase):
                        self.system.last_error = 0
                        previous = getattr(self.scene.world, field)
                        setattr(self.scene.world, field, value)
                        before = bytes(self.scene)
                        result = self.init() if phase == 'init' else self.call(phase)
                        self.assertEqual(result, 102)
                        self.assertEqual(bytes(self.scene), before)
                        setattr(self.scene.world, field, previous)

    def test_frozen_spawn_asset_has_exact_36_tick_readiness(self):
        from PIL import Image, ImageSequence
        path = FROZEN/'spawnervfx/spawn.gif'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         'b76d02d162556865d56366d37a9648591a461deab40f989e20671c3263f77de2')
        with Image.open(path) as image:
            steps = sum(max(1, round(frame.info.get('duration', 100)/1000*30))
                        for frame in ImageSequence.Iterator(image))
        self.assertEqual(steps, 36)

    def test_lifecycle_matches_frozen_classes_rng_pool_order_and_body_geometry(self):
        ns = frozen_weapon_classes()
        pygame, pymunk = ns['pygame'], ns['pymunk']

        class RecordedRandom:
            def __init__(self):
                self.generator = random.Random(1100)  # First two spawns cover Spear and Hammer.
                self.draws = []

            def uniform(self, a, b):
                value = self.generator.uniform(a, b)
                self.draws.append((2, a, b, value))
                return value

            def randint(self, a, b):
                value = self.generator.randint(a, b)
                self.draws.append((3, a, b, value))
                return value

        rng = RecordedRandom()
        ns['random'] = rng
        camera = types.SimpleNamespace(scale_gtp=lambda: 48.0)
        env = types.SimpleNamespace(camera=camera, space=pymunk.Space(), objects={}, steps=0)
        env.weapon_images = {name: pygame.Surface((40, 16)) for name in ('Spear', 'Hammer')}
        env.weapon_pool = ns['WeaponPool'](env.weapon_images)
        signals = []
        env.weapon_equip_signal = types.SimpleNamespace(emit=lambda **kw: signals.append(('equip', kw['agent'])))
        env.weapon_drop_signal = types.SimpleNamespace(emit=lambda **kw: signals.append(('drop', kw['agent'])))
        env.players = []
        for i in range(2):
            player = types.SimpleNamespace(
                agent_id=i, env=env, weapon='Punch', state=object(), facing=ns['Facing'].RIGHT,
                body=types.SimpleNamespace(position=pymunk.Vec2d(0, 0)),
                input=types.SimpleNamespace(key_status={'h': types.SimpleNamespace(held=False, just_pressed=False)}),
                hurtbox_collider=ns['CapsuleCollider']((0, 0), self.aux[i].hurtbox_width, self.aux[i].hurtbox_height))
            env.players.append(player)
        spawners = []
        for i in range(2):
            native = self.system.spawners[i]
            sp = ns['WeaponSpawner'](camera, i, env, env.weapon_pool, list(native.world_pos),
                                     native.og_cooldown_frames, native.despawn_frames)
            sp.cooldown_frames, sp.last_spawn_frame = native.cooldown_frames, native.last_spawn_frame
            spawners.append(sp)
        env.weapon_controller = ns['WeaponSpawnController'](spawners)
        identities = {}
        seen_weapons = set()
        names = {'Punch': 0, 'Spear': 1, 'Hammer': 2}

        def identity(weapon):
            if id(weapon) not in identities:
                identities[id(weapon)] = len(identities)
            return identities[id(weapon)]

        def compare():
            self.assertEqual(self.system.spawner_count, len(env.weapon_controller.spawners))
            for index, original in enumerate(env.weapon_controller.spawners):
                native = self.system.spawners[index]
                self.assertEqual(tuple(native.world_pos), tuple(original.world_pos))
                for field in ('id', 'og_cooldown_frames', 'cooldown_frames', 'last_spawn_frame', 'despawn_frames'):
                    self.assertEqual(getattr(native, field), getattr(original, field), (env.steps, field))
                if isinstance(original, ns['DroppedWeaponSpawner']):
                    self.assertEqual(native.spawned_once, original._spawned_once)
                    self.assertEqual(native.done, original.done)
                    self.assertEqual(native.weapon, names[original.weapon_name])
                else:
                    self.assertEqual(native.flag, original.flag)
                if original.active_weapon is None:
                    self.assertEqual(native.weapon_slot, -1)
                else:
                    original_weapon = original.active_weapon
                    handle = identity(original_weapon)
                    self.assertEqual(native.weapon_slot, handle)
                    weapon = self.system.weapons[handle]
                    self.assertEqual(weapon.weapon, names[original_weapon.name])
                    seen_weapons.add(weapon.weapon)
                    self.assertEqual(tuple(weapon.world_pos), tuple(original_weapon.world_pos))
                    self.assertEqual(weapon.spawn_frame, original_weapon.spawn_frame)
                    body = self.scene.world.bodies[weapon.body]
                    self.assertEqual((body.p.x, body.p.y), tuple(original_weapon.body.position))
                    native_shape = self.scene.world.shapes[weapon.shape]
                    self.assertEqual(native_shape.categories, original_weapon.shape.filter.categories)
                    self.assertEqual(native_shape.mask, original_weapon.shape.filter.mask)
                    self.assertEqual(native_shape.u, original_weapon.shape.friction)
                    self.assertEqual(native_shape.e, original_weapon.shape.elasticity)
                    self.assertEqual([(p.x, p.y) for p in native_shape.v0[:native_shape.count]],
                                     [(p.x, p.y) for p in original_weapon.shape.get_vertices()])
            self.assertEqual(list(self.system.pool[:self.system.pool_count]),
                             [identity(w) for w in env.weapon_pool.pool])
            for i, player in enumerate(env.players):
                self.assertEqual(self.scene.players[i].weapon, names[player.weapon])
                self.assertEqual(self.aux[i].pickup_lock_until, getattr(player, 'pickup_lock_until', -1))
            self.assertEqual(self.system.next_spawner_id, getattr(env, '_next_spawner_id', 0))

        def phase(name):
            rng.draws.clear()
            if name == 'drop':
                ns['DroppedWeaponSpawner'].try_drop(env)
            elif name == 'pickup':
                env.weapon_controller.try_pick_up_all(env.players, env.steps)
            else:
                env.weapon_controller.update(env.steps)
            self.tape(*rng.draws)
            self.assertEqual(self.call(name), 0, (env.steps, name))
            self.assertEqual(self.rng.cursor, len(rng.draws), (env.steps, name))
            compare()

        # Continuous state and real oracle methods, with deterministic scripted
        # positions (this test deliberately does not integrate either world).
        for tick in range(1, 650):
            env.steps = self.scene.steps = tick
            if tick == 1:
                for i in range(2):
                    env.weapon_controller.spawners[i].cooldown_frames = 1
                    self.system.spawners[i].cooldown_frames = 1
            for i, player in enumerate(env.players):
                loose = [sp for sp in env.weapon_controller.spawners if sp.active_weapon]
                location = loose[i % len(loose)].active_weapon.world_pos if loose else [4.0 if i == 0 else -4.0, .7]
                player.body.position = pymunk.Vec2d(*location)
                player.hurtbox_collider.center = pygame.Vector2(location)
                self.aux[i].hurtbox_x, self.aux[i].hurtbox_y = location
                body = self.scene.world.bodies[self.scene.players[i].body]
                body.p.x, body.p.y = location
                held = tick % 7 in (0, 1)
                pressed = tick % 7 == 0
                player.input.key_status['h'].held = held
                player.input.key_status['h'].just_pressed = pressed
                self.scene.players[i].input_state[5] = int(held)
                self.scene.players[i].input_state[15] = int(pressed)
                player.facing = ns['Facing'].LEFT if i else ns['Facing'].RIGHT
                self.scene.players[i].ms.facing = -1 if i else 1
            signals.clear()
            self.assertEqual(self.call('mirror'), 0)
            for weapon in [sp.active_weapon for sp in env.weapon_controller.spawners if sp.active_weapon]:
                weapon.physics_process(1/30)
            for name in ('drop', 'pickup', 'update'):
                phase(name)
            expected = [('drop' if e.kind == 1 else 'equip', 'player' if e.agent == 0 else 'opponent')
                        for e in self.system.events[:self.system.event_count]]
            self.assertEqual(expected, signals)
        self.assertGreater(self.system.next_spawner_id, 6)
        self.assertGreater(len(identities), 1)
        self.assertEqual(seen_weapons, {1, 2})


if __name__ == '__main__':
    unittest.main()
