"""Frozen constructor and compiled-table parity; no game/model/cloud execution."""
import ast
import copy
import ctypes
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = Path(os.environ.get('AI2_FROZEN_SOURCE_ROOT',
                   '/Users/elliot18/Projects/ml-research/AI_2-learning-validation-20260905/'
                   'baselines/navigation-repair-20260905/payload/variants/corrected')) / 'UTMIST-AI2-main/environment'
COUNTS = dict(moves=33, powers=107, casts=206, hitboxes=207, points=4, changes=194)
MOVE_NAMES = ('NLight', 'DLight', 'SLight', 'NSig', 'DSig', 'SSig',
              'NAir', 'DAir', 'SAir', 'Recovery', 'Groundpound')
WEAPONS = (('unarmed_attacks', 'Unarmed'), ('spear_attacks', 'Spear'), ('hammer_attacks', 'Hammer'))
FLAGS = (('is_charge', 1), ('disable_caster_gravity', 2), ('disable_hit_gravity', 4),
         ('target_all_hit_agents', 8), ('transition_on_instant_hit', 16),
         ('on_hit_velocity_set_active', 32), ('allow_left_right_mobility', 64),
         ('hit_all_hit_agents', 128), ('enable_floor_drag', 256), ('damage_over_life_of_hitbox', 512))


def constructors():
    names = {'HurtboxPositionChange', 'CasterPositionChange', 'DealtPositionTarget',
             'CasterVelocitySet', 'CasterVelocityAddXY', 'CasterVelocitySetXY',
             'HitVelocitySetXY', 'HitVelocityAddXY', 'HitPosSetXY', 'CasterVelocityDampXY',
             'CastFrameChangeHolder', 'Cast', 'Power'}
    source = ENVIRONMENT / 'environment.py'
    tree = ast.parse(source.read_text(), filename=str(source))
    selected = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    if {node.name for node in selected} != names:
        raise AssertionError('Missing frozen constructor oracle')
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    scope = {'math': math}
    exec(compile(module, str(source), 'exec'), scope)
    return scope


def oracle_change(holder):
    hurt, position, dealt = holder.hurtbox_position_change, holder.caster_position_change, holder.dealt_position_target
    velocity = holder.caster_velocity_set
    row = dict(frame=holder.frame // 2, hurtbox_active=int(hurt.active),
               caster_position_active=int(position.active), dealt_active=int(dealt.active),
               caster_velocity_active=int(velocity.active) if velocity else 0,
               hurtbox_x=float(hurt.xOffset), hurtbox_y=float(hurt.yOffset),
               hurtbox_width=float(hurt.width), hurtbox_height=float(hurt.height),
               caster_position_x=float(position.x), caster_position_y=float(position.y),
               dealt_x=float(dealt.xOffset), dealt_y=float(dealt.yOffset), dealt_mult=float(dealt.mult),
               caster_speed=float(velocity.magnitude) if velocity else 0.,
               caster_angle=float(velocity.directionDeg) if velocity else 0.)
    groups = [('caster_damp', holder.caster_velocity_damp_xy, 'dampX', 'dampY', 1.),
              ('caster_set', holder.caster_velocity_set_xy, 'magnitudeX', 'magnitudeY', 0.),
              ('caster_add', holder.caster_velocity_add_xy, 'magnitudeX', 'magnitudeY', 0.),
              ('hit_set', holder.hit_velocity_set_xy, 'magnitudeX', 'magnitudeY', 0.),
              ('hit_add', holder.hit_velocity_add_xy, 'magnitudeX', 'magnitudeY', 0.),
              ('hit_position', holder.hit_pos_set_xy, 'positionX', 'positionY', 0.)]
    for name, value, x, y, default in groups:
        row[name] = dict(x=float(getattr(value, x)) if value else default,
                         y=float(getattr(value, y)) if value else default,
                         active_x=int(value.activeX) if value else 0,
                         active_y=int(value.activeY) if value else 0)
    return row


class AttackTableTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('native_attack_tables'), 'Attack table generator is missing')
        self.generator = importlib.import_module('native_attack_tables')
        self.assets = {path.relative_to(ENVIRONMENT).as_posix(): json.loads(path.read_text())
                       for directory, _ in WEAPONS for path in (ENVIRONMENT / directory).glob('*.json')}

    def exact(self, actual, expected):
        if isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.exact(actual[key], expected[key])
        elif isinstance(expected, float):
            self.assertEqual(struct.pack('=d', actual), struct.pack('=d', expected))
        else:
            self.assertEqual(actual, expected)

    def test_frozen_hash_counts_and_move_order(self):
        tables = self.generator.build_tables(ENVIRONMENT)
        self.assertEqual(tables['asset_sha256'], 'b0ca5343c4a1b4ad0ba17c238b70afeff15ee0c9ce822f5e9c8933d35632b644')
        self.assertEqual(tables['source_sha256'], 'c0116fc7b4a46d45c37367d9aa4d7f44041296ad96ba01e8b1f3f1f158e24230')
        self.assertEqual({name: len(tables[name]) for name in COUNTS}, COUNTS)
        self.assertEqual(tables['move_paths'], [f'{directory}/{weapon} {move}.json'
                                               for directory, weapon in WEAPONS for move in MOVE_NAMES])

    def test_every_frozen_constructor_field_and_global_offset_is_exact(self):
        tables = self.generator.build_tables(ENVIRONMENT)
        oracle = constructors()
        power_offset = cast_offset = box_offset = point_offset = change_offset = 0
        for index, path in enumerate(tables['move_paths']):
            asset = self.assets[path]
            self.exact(tables['moves'][index], dict(initial_power=power_offset + asset['move']['initialPowerIndex'],
                                                   action_key={'LIGHT': 7, 'HEAVY': 8}[asset['move']['actionKey']]))
            for local_power, data in enumerate(asset['powers']):
                power = oracle['Power'].get_power(data)
                expected = dict(first_cast=cast_offset, cast_count=len(power.casts),
                                fixed_recovery=power.fixed_recovery, recovery=power.recovery,
                                cooldown=power.cooldown, min_charge=power.min_charge, stun_time=power.stun_time,
                                flags=sum(bit for name, bit in FLAGS if getattr(power, name)),
                                hit_angle=float(power.hit_angle_deg), on_hit_speed=float(power.on_hit_velocity_set_magnitude),
                                on_hit_angle=float(power.on_hit_velocity_set_direction_deg))
                for name in ('hit', 'miss', 'ground'):
                    reference = getattr(power, f'on_{name}_next_power_index')
                    expected['on_' + name] = -1 if reference == -1 else power_offset + reference
                self.exact(tables['powers'][power_offset + local_power], expected)
                for cast in power.casts:
                    changes = cast.cast_data.get('frameChanges', [])
                    self.exact(tables['casts'][cast_offset], dict(startup=cast.startup_frames, attack=cast.attack_frames,
                        first_hitbox=box_offset, hitbox_count=len(cast.hitboxes), first_point=point_offset,
                        point_count=len(cast.collision_check_points), first_change=change_offset,
                        change_count=len(changes), must_be_held=int(cast.must_be_held), damage=float(cast.base_damage),
                        variable_force=float(cast.variable_force), fixed_force=float(cast.fixed_force), hit_angle=float(cast.hit_angle_deg)))
                    for box in cast.hitboxes:
                        self.exact(tables['hitboxes'][box_offset], dict(x=float(box['xOffset']), y=float(box['yOffset']),
                                                                      width=float(box['width']), height=float(box['height'])))
                        box_offset += 1
                    for point in cast.collision_check_points:
                        self.exact(tables['points'][point_offset], dict(x=float(point['xOffset']), y=float(point['yOffset'])))
                        point_offset += 1
                    for change in changes:
                        self.exact(tables['changes'][change_offset], oracle_change(oracle['CastFrameChangeHolder'](change)))
                        change_offset += 1
                    cast_offset += 1
            power_offset += len(asset['powers'])

    def test_frame_defaults_y_signs_ignored_damp_fields_and_duplicate_order(self):
        oracle = constructors()['CastFrameChangeHolder']
        changes = [dict(frame=3), dict(frame=2, casterVelocitySetXY=dict(magnitudeY=3, activeY=True),
                       hitPosSetXY=dict(positionY=3, activeY=True), casterVelocityDampXY=dict(magnitudeX=99, magnitudeY=-8)),
                   dict(frame=3, casterVelocityAddXY={}, hitVelocitySetXY={}, hitVelocityAddXY={})]
        for asset in self.assets.values():
            for power in asset['powers']:
                for cast in power['casts']:
                    cast.pop('frameChanges', None)
        first = next(iter(self.assets.values()))['powers'][0]['casts'][0]
        first['frameChanges'] = changes
        tables = self.generator.compile_assets(self.assets)
        path = next(iter(self.assets))
        move = tables['moves'][tables['move_paths'].index(path)]
        power = tables['powers'][move['initial_power']]
        cast = tables['casts'][power['first_cast']]
        rows = tables['changes'][cast['first_change']:cast['first_change'] + cast['change_count']]
        self.assertEqual([row['frame'] for row in rows], [1, 1, 1])
        for actual, data in zip(rows, changes):
            self.exact(actual, oracle_change(oracle(data)))

    def test_unknown_keys_fail_closed_at_each_level(self):
        for level in ('root', 'move', 'power', 'cast', 'hitbox', 'point', 'change', 'nested'):
            with self.subTest(level=level):
                assets = copy.deepcopy(self.assets)
                root = next(iter(assets.values()))
                power = root['powers'][0]
                cast = power['casts'][0]
                cast['hitboxes'] = [dict(xOffset=0, yOffset=0, width=1, height=1)]
                cast['collisionCheckPoints'] = [dict(xOffset=0, yOffset=0)]
                cast['frameChanges'] = [dict(frame=0, casterVelocitySetXY={})]
                levels = dict(root=root, move=root['move'], power=power, cast=cast, hitbox=cast['hitboxes'][0],
                              point=cast['collisionCheckPoints'][0], change=cast['frameChanges'][0],
                              nested=cast['frameChanges'][0]['casterVelocitySetXY'])
                levels[level]['unknown_behavior'] = 1
                with self.assertRaisesRegex(ValueError, 'Unsupported'):
                    self.generator.compile_assets(assets)

    def test_capacity_invalid_references_and_numeric_types_fail_closed(self):
        for name, maximum in COUNTS.items():
            with self.subTest(capacity=name), self.assertRaisesRegex(ValueError, 'capacity'):
                self.generator.compile_assets(self.assets, capacities={name: maximum - 1})
        for field, value in [('onHitNextPowerIndex', 999), ('onMissNextPowerIndex', -2),
                             ('cooldown', True), ('stunTime', 1.5), ('recovery', -1),
                             ('hitAngleDeg', float('nan')), ('hitAngleDeg', float('inf')),
                             ('fixedRecovery', 2**40), ('isCharge', 1)]:
            with self.subTest(field=field, value=value):
                assets = copy.deepcopy(self.assets)
                next(iter(assets.values()))['powers'][0][field] = value
                with self.assertRaises(ValueError):
                    self.generator.compile_assets(assets)
        with self.assertRaisesRegex(ValueError, 'capacity'):
            self.generator.compile_assets(self.assets, capacities={'powers': COUNTS['powers'] + 1})

    def test_missing_extra_assets_and_wrong_action_fail_closed(self):
        assets = copy.deepcopy(self.assets)
        assets.pop(next(iter(assets)))
        with self.assertRaises(ValueError):
            self.generator.compile_assets(assets)

        assets = copy.deepcopy(self.assets)
        assets['extra.json'] = copy.deepcopy(next(iter(assets.values())))
        with self.assertRaises(ValueError):
            self.generator.compile_assets(assets)
        assets = copy.deepcopy(self.assets)
        next(iter(assets.values()))['move']['actionKey'] = 'THROW'
        with self.assertRaises(ValueError):
            self.generator.compile_assets(assets)

    def test_optional_constructor_flags_are_not_silently_dropped(self):
        assets = copy.deepcopy(self.assets)
        path = self.generator.MOVE_PATHS[0]
        power = assets[path]['powers'][0]
        for key in self.generator.FLAGS:
            power[key] = True
        power['casts'][0]['mustBeHeld'] = True
        tables = self.generator.compile_assets(assets)
        self.assertEqual(tables['powers'][0]['flags'], 1023)
        self.assertEqual(tables['casts'][0]['must_be_held'], 1)

    def test_regeneration_is_byte_identical_and_check_never_rewrites(self):
        tables = self.generator.build_tables(ENVIRONMENT)
        rendered = self.generator.render_header(tables)
        self.assertEqual(rendered, self.generator.render_header(tables))
        self.assertEqual(rendered.encode(), (ROOT / 'native/game_attack_data.h').read_bytes())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'table.h'
            self.generator.main(['--source-root', str(ENVIRONMENT), '--output', str(output)])
            self.assertEqual(output.read_bytes(), rendered.encode())
            self.generator.main(['--source-root', str(ENVIRONMENT), '--output', str(output), '--check'])
            output.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                self.generator.main(['--source-root', str(ENVIRONMENT), '--output', str(output), '--check'])
            self.assertEqual(output.read_bytes(), b'changed')

    def test_source_and_asset_drift_fail_before_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            shutil.copyfile(ENVIRONMENT / 'environment.py', root / 'environment.py')
            for folder, _ in WEAPONS:
                shutil.copytree(ENVIRONMENT / folder, root / folder)
            self.generator.build_tables(root)
            asset = next((root / 'unarmed_attacks').glob('*.json'))
            asset.write_bytes(asset.read_bytes() + b' ')
            with self.assertRaisesRegex(ValueError, 'hash'):
                self.generator.build_tables(root)
            shutil.copyfile(ENVIRONMENT / asset.relative_to(root), asset)
            (root / 'environment.py').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'hash'):
                self.generator.build_tables(root)

    @unittest.skipUnless(shutil.which('cc'), 'C compiler required')
    def test_compiled_header_fields_match_all_python_rows(self):
        tables = self.generator.build_tables(ENVIRONMENT)
        header = self.generator.render_header(tables)
        self.assertIn('__device__ const', header)
        self.assertNotIn('__constant__', header)
        self.assertIn('#if defined(__CUDA_ARCH__)', header)
        types = {'int32_t': ctypes.c_int32, 'uint32_t': ctypes.c_uint32, 'double': ctypes.c_double}
        structures = {}
        for name, fields in self.generator.STRUCTS.items():
            structures[name] = type(name, (ctypes.Structure,), {'_fields_': [(field, types.get(kind, structures.get(kind)))
                                                                                  for kind, field in fields]})
        source = '#include "native/game_attack_data.h"\n'
        for name, kind in self.generator.TABLE_TYPES.items():
            source += f'const {kind} *get_{name}(void) {{ return ai2g_{name}; }}\n'
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / 'tables.so'
            command = ['cc', '-std=c11', '-Wall', '-Wextra', '-Werror', '-shared', '-fPIC', '-I', str(ROOT),
                       '-x', 'c', '-', '-o', str(library)]
            compiled = subprocess.run(command, input=source, text=True, capture_output=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            loaded = ctypes.CDLL(str(library))
            def unpack(value):
                return {name: unpack(getattr(value, name)) if isinstance(getattr(value, name), ctypes.Structure)
                        else getattr(value, name) for name, _ in value._fields_}
            for name, kind in self.generator.TABLE_TYPES.items():
                getter = getattr(loaded, 'get_' + name)
                getter.restype = ctypes.POINTER(structures[kind])
                for index, expected in enumerate(tables[name]):
                    self.exact(unpack(getter()[index]), expected)

    @unittest.skipUnless(shutil.which('nvcc'), 'CUDA compiler unavailable; no device compilation claimed')
    def test_cuda_header_compiles_in_one_host_device_translation_unit(self):
        source = '#include "native/game_attack_data.h"\n'
        source += '__host__ __device__ double attack_table_probe(int i) { return '
        source += ' + '.join(f'ai2g_{name}[i % AI2G_{self.generator.COUNT_MACROS[name]}_COUNT].{field}'
                             for name, field in [('moves', 'initial_power'), ('powers', 'hit_angle'),
                                                 ('casts', 'damage'), ('hitboxes', 'width'),
                                                 ('points', 'x'), ('changes', 'frame')])
        source += '; }\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'probe.cu'
            path.write_text(source)
            result = subprocess.run(['nvcc', '-std=c++17', '-dc', '-I', str(ROOT), str(path),
                                     '-o', str(path.with_suffix('.o'))], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
