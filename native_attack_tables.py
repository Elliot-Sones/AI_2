#!/usr/bin/env python3
"""Exact frozen attack constructors as bounded C/CUDA tables, never game execution.

Source-root is the frozen ``environment`` directory. Power transitions become
global indices; frame timing uses Python floor division, without sorting changes.
Known ignored source fields: move.gadgetThrow (false only), power.powerID
(animation metadata), and casterVelocityDampXY.magnitudeX/magnitudeY. The latter
do NOT substitute for dampX/dampY, matching CastFrameChangeHolder exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = Path('/Users/elliot18/Projects/ml-research/AI_2-learning-validation-20260905/'
                      'baselines/navigation-repair-20260905/payload/variants/corrected/'
                      'UTMIST-AI2-main/environment')
if os.environ.get('AI2_FROZEN_SOURCE_ROOT'):
    DEFAULT_SOURCE = Path(os.environ['AI2_FROZEN_SOURCE_ROOT']) / 'UTMIST-AI2-main/environment'
SOURCE_SHA256 = 'c0116fc7b4a46d45c37367d9aa4d7f44041296ad96ba01e8b1f3f1f158e24230'
ASSET_SHA256 = 'b0ca5343c4a1b4ad0ba17c238b70afeff15ee0c9ce822f5e9c8933d35632b644'
CAPACITIES = dict(moves=33, powers=107, casts=206, hitboxes=207, points=4, changes=194)
MOVE_NAMES = ('NLight', 'DLight', 'SLight', 'NSig', 'DSig', 'SSig',
              'NAir', 'DAir', 'SAir', 'Recovery', 'Groundpound')
WEAPONS = (('unarmed_attacks', 'Unarmed'), ('spear_attacks', 'Spear'), ('hammer_attacks', 'Hammer'))
MOVE_PATHS = tuple(f'{directory}/{weapon} {move}.json'
                   for directory, weapon in WEAPONS for move in MOVE_NAMES)
FLAGS = dict(isCharge=1, disableCasterGravity=2, disableHitGravity=4,
             targetAllHitAgents=8, transitionOnInstantHit=16, onHitVelocitySetActive=32,
             allowLeftRightMobility=64, hitAllHitAgents=128, enableFloorDrag=256,
             damageOverLifeOfHitbox=512)
POWER_TIMES = dict(fixed_recovery='fixedRecovery', recovery='recovery', cooldown='cooldown',
                   min_charge='minCharge', stun_time='stunTime')
TRANSITIONS = dict(on_hit='onHitNextPowerIndex', on_miss='onMissNextPowerIndex', on_ground='onGroundNextPowerIndex')
POWER_FLOATS = dict(hit_angle='hitAngleDeg', on_hit_speed='onHitVelocitySetMagnitude',
                    on_hit_angle='onHitVelocitySetDirectionDeg')
CAST_FLOATS = dict(damage='baseDamage', variable_force='variableForce', fixed_force='fixedForce', hit_angle='hitAngleDeg')
XY_GROUPS = {
    'caster_damp': ('casterVelocityDampXY', 'dampX', 'dampY', 1., False),
    'caster_set': ('casterVelocitySetXY', 'magnitudeX', 'magnitudeY', 0., True),
    'caster_add': ('casterVelocityAddXY', 'magnitudeX', 'magnitudeY', 0., True),
    'hit_set': ('hitVelocitySetXY', 'magnitudeX', 'magnitudeY', 0., True),
    'hit_add': ('hitVelocityAddXY', 'magnitudeX', 'magnitudeY', 0., True),
    'hit_position': ('hitPosSetXY', 'positionX', 'positionY', 0., False),
}


def fields(kind, names):
    return [(kind, name) for name in names.split()]


# Ordered schema is shared only by the generator/renderer, not the frozen oracle.
STRUCTS = {
    'AI2GMoveData': fields('int32_t', 'initial_power action_key'),
    'AI2GPowerData': fields('int32_t', 'first_cast cast_count fixed_recovery recovery cooldown min_charge stun_time on_hit on_miss on_ground')
        + fields('uint32_t', 'flags') + fields('double', 'hit_angle on_hit_speed on_hit_angle'),
    'AI2GCastData': fields('int32_t', 'startup attack first_hitbox hitbox_count first_point point_count first_change change_count must_be_held')
        + fields('double', 'damage variable_force fixed_force hit_angle'),
    'AI2GHitboxData': fields('double', 'x y width height'),
    'AI2GPointData': fields('double', 'x y'),
    'AI2GXYChange': fields('double', 'x y') + fields('int32_t', 'active_x active_y'),
    'AI2GFrameChange': fields('int32_t', 'frame hurtbox_active caster_position_active dealt_active caster_velocity_active')
        + fields('double', 'hurtbox_x hurtbox_y hurtbox_width hurtbox_height caster_position_x caster_position_y dealt_x dealt_y dealt_mult caster_speed caster_angle')
        + fields('AI2GXYChange', 'caster_damp caster_set caster_add hit_set hit_add hit_position'),
}
TABLE_TYPES = dict(moves='AI2GMoveData', powers='AI2GPowerData', casts='AI2GCastData',
                   hitboxes='AI2GHitboxData', points='AI2GPointData', changes='AI2GFrameChange')
COUNT_MACROS = dict(moves='MOVE', powers='POWER', casts='CAST', hitboxes='HITBOX', points='POINT', changes='CHANGE')


def _keys(value, allowed, context, required=()):
    if not isinstance(value, dict):
        raise ValueError(f'Expected object: {context}')
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f'Unsupported keys in {context}: {sorted(unknown)}')
    if set(required) - set(value):
        raise ValueError(f'Missing required keys: {context}')


def _integer(value, *, minimum=0):
    if type(value) is not int or not minimum <= value <= 2147483647:
        raise ValueError('Expected bounded int32 timing/index')
    return value


def _number(value):
    if type(value) not in (int, float):
        raise ValueError('Expected finite exact double')
    try:
        result = float(value)
    except OverflowError:
        raise ValueError('Expected finite exact double') from None
    if not math.isfinite(result) or result != value:
        raise ValueError('Expected finite exact double')
    return result


def _boolean(value):
    if type(value) is not bool:
        raise ValueError('Expected boolean flag')
    return int(value)


def _list(value):
    if not isinstance(value, list):
        raise ValueError('Expected an ordered list')
    return value


def _reference(value, first_power, count):
    value = _integer(value, minimum=-1)
    if value >= count:
        raise ValueError('Power reference outside owning move')
    return -1 if value == -1 else first_power + value


def _change(data):
    _keys(data, {'frame', 'hurtboxPositionChange', 'casterPositionChange', 'dealtPositionTarget',
                 'casterVelocitySet', *(group[0] for group in XY_GROUPS.values())}, 'frame change', ('frame',))
    result = dict(frame=_integer(data['frame']) // 2)
    groups = (
        ('hurtboxPositionChange', 'hurtbox_active', dict(hurtbox_x=('xOffset', 0.), hurtbox_y=('yOffset', 0.),
             hurtbox_width=('width', 0.), hurtbox_height=('height', 0.))),
        ('casterPositionChange', 'caster_position_active', dict(caster_position_x=('x', 0.), caster_position_y=('y', 0.))),
        ('dealtPositionTarget', 'dealt_active', dict(dealt_x=('xOffset', 0.), dealt_y=('yOffset', 0.), dealt_mult=('mult', 2.))),
        ('casterVelocitySet', 'caster_velocity_active', dict(caster_speed=('magnitude', 0.), caster_angle=('directionDeg', 0.))),
    )
    for key, active, coordinates in groups:
        value = data.get(key, {})
        _keys(value, {'active', *(name for name, _ in coordinates.values())}, key)
        result[active] = _boolean(value.get('active', False))
        result.update({target: _number(value.get(source, default)) for target, (source, default) in coordinates.items()})
    for target, (key, x, y, default, negate_y) in XY_GROUPS.items():
        value = data.get(key, {})
        ignored = {'magnitudeX', 'magnitudeY'} if target == 'caster_damp' else set()
        _keys(value, {x, y, 'activeX', 'activeY'} | ignored, key)
        for name in ignored & value.keys():
            _number(value[name])
        # Absent holder members are None; present XY constructors negate their
        # input before conversion. Preserve integer-zero versus float -0.0 too.
        raw_y = value.get(y, default)
        _number(raw_y)
        final_y = -raw_y if negate_y and key in data else raw_y
        result[target] = dict(x=_number(value.get(x, default)), y=_number(final_y),
                              active_x=_boolean(value.get('activeX', False)), active_y=_boolean(value.get('activeY', False)))
    return result


def compile_assets(assets, *, capacities=None):
    """Compile validated in-memory assets; production must use hash-bound build_tables."""
    if not isinstance(assets, dict) or set(assets) != set(MOVE_PATHS):
        raise ValueError('Expected exactly the 33 frozen weapon/move asset paths')
    limits = dict(CAPACITIES)
    for name, maximum in (capacities or {}).items():
        if name not in limits or type(maximum) is not int or not 0 <= maximum <= limits[name]:
            raise ValueError('Invalid or expanded table capacity')
        limits[name] = maximum
    tables = {name: [] for name in TABLE_TYPES}
    tables['move_paths'] = list(MOVE_PATHS)
    for path in MOVE_PATHS:
        asset = assets[path]
        _keys(asset, {'move', 'powers'}, path, ('move', 'powers'))
        move, powers = asset['move'], _list(asset['powers'])
        _keys(move, {'initialPowerIndex', 'actionKey', 'gadgetThrow'}, 'move', ('initialPowerIndex', 'actionKey'))
        if not powers or move['actionKey'] not in ('LIGHT', 'HEAVY') or _boolean(move.get('gadgetThrow', False)):
            raise ValueError('Unsupported move action or empty powers')
        first_power = len(tables['powers'])
        initial = _reference(move['initialPowerIndex'], first_power, len(powers))
        if initial < 0:
            raise ValueError('Move initial power must exist')
        tables['moves'].append(dict(initial_power=initial, action_key={'LIGHT': 7, 'HEAVY': 8}[move['actionKey']]))
        for power in powers:
            _keys(power, {'powerID', 'casts', *POWER_TIMES.values(), *TRANSITIONS.values(), *POWER_FLOATS.values(), *FLAGS},
                  'power', ('casts',))
            _integer(power.get('powerID', -1), minimum=-1)
            casts = _list(power['casts'])
            if not casts:
                raise ValueError('A power must contain casts')
            row = dict(first_cast=len(tables['casts']), cast_count=len(casts),
                       flags=sum(bit * _boolean(power.get(name, False)) for name, bit in FLAGS.items()))
            row.update({name: _integer(power.get(key, 0)) // 2 for name, key in POWER_TIMES.items()})
            row.update({name: _reference(power.get(key, -1), first_power, len(powers)) for name, key in TRANSITIONS.items()})
            row.update({name: _number(power.get(key, 0.)) for name, key in POWER_FLOATS.items()})
            tables['powers'].append(row)
            for cast in casts:
                _keys(cast, {'startupFrames', 'attackFrames', 'mustBeHeld', 'hitboxes', 'collisionCheckPoints', 'frameChanges',
                             *CAST_FLOATS.values()}, 'cast')
                row = dict(startup=_integer(cast.get('startupFrames', 0)) // 2,
                           attack=_integer(cast.get('attackFrames', 0)) // 2,
                           must_be_held=_boolean(cast.get('mustBeHeld', False)))
                row.update({name: _number(cast.get(key, 0.)) for name, key in CAST_FLOATS.items()})
                for table, key, prefix in [('hitboxes', 'hitboxes', 'hitbox'), ('points', 'collisionCheckPoints', 'point'),
                                           ('changes', 'frameChanges', 'change')]:
                    values = _list(cast.get(key, []))
                    row['first_' + prefix], row[prefix + '_count'] = len(tables[table]), len(values)
                    for value in values:
                        if table == 'changes':
                            tables[table].append(_change(value))
                        else:
                            names = dict(x='xOffset', y='yOffset')
                            if table == 'hitboxes':
                                names.update(width='width', height='height')
                            _keys(value, names.values(), table, names.values())
                            tables[table].append({name: _number(value[key]) for name, key in names.items()})
                tables['casts'].append(row)
    for name, maximum in limits.items():
        if len(tables[name]) > maximum:
            raise ValueError(f'{name} table capacity exceeded: {len(tables[name])} > {maximum}')
    return tables


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Unsupported duplicate JSON key')
        result[key] = value
    return result


def build_tables(environment_dir=DEFAULT_SOURCE, *, capacities=None):
    """Read-only production entrypoint: pinned constructor and complete asset hashes."""
    root = Path(environment_dir)
    source = root / 'environment.py'
    if source.is_symlink() or hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError('Frozen constructor source hash mismatch')
    paths = sorted(path for directory, _ in WEAPONS for path in (root / directory).glob('*.json'))
    if {path.relative_to(root).as_posix() for path in paths} != set(MOVE_PATHS):
        raise ValueError('Frozen asset path set mismatch')
    digest, assets = hashlib.sha256(), {}
    for path in paths:
        if path.is_symlink() or path.resolve() != path.absolute():
            raise ValueError('Frozen assets must not be symlinks')
        relative, raw = path.relative_to(root).as_posix(), path.read_bytes()
        digest.update(relative.encode('utf8') + b'\0' + raw)
        assets[relative] = json.loads(raw, object_pairs_hook=_unique_object)
    if digest.hexdigest() != ASSET_SHA256:
        raise ValueError('Frozen attack asset hash mismatch')
    tables = compile_assets(assets, capacities=capacities)
    if {name: len(tables[name]) for name in CAPACITIES} != CAPACITIES:
        raise ValueError('Frozen attack table counts mismatch')
    tables.update(asset_sha256=ASSET_SHA256, source_sha256=SOURCE_SHA256)
    return tables


def _initializer(kind, row):
    if set(row) != {name for _, name in STRUCTS[kind]}:
        raise ValueError('Generated row does not match the native schema')
    values = []
    for field_kind, name in STRUCTS[kind]:
        value = row[name]
        values.append(_initializer(field_kind, value) if field_kind in STRUCTS else
                      repr(_number(value)) if field_kind == 'double' else str(_integer(value, minimum=-1)))
    return '{' + ', '.join(values) + '}'


def render_header(tables):
    """Pure deterministic renderer; dual immutable host/device global tables."""
    if tables.get('asset_sha256') != ASSET_SHA256 or tables.get('source_sha256') != SOURCE_SHA256:
        raise ValueError('Only hash-bound frozen tables can be rendered')
    lines = ['#ifndef AI2_GAME_ATTACK_DATA_H', '#define AI2_GAME_ATTACK_DATA_H', '#include <stdint.h>',
             '/* Generated by native_attack_tables.py; do not edit. Raw asset units.',
             ' * Move index = weapon * 11 + (MoveType - 2); Punch=0, Spear=1, Hammer=2.',
             ' * Powers use global references; duplicate halved frames retain source order.',
             ' * CUDA global readonly storage intentionally avoids the 64 KiB constant limit. */',
             f'#define AI2G_ATTACK_ASSET_SHA256 "{ASSET_SHA256}"', f'#define AI2G_ATTACK_SOURCE_SHA256 "{SOURCE_SHA256}"', '']
    for kind, members in STRUCTS.items():
        lines.append(f'typedef struct {kind} {{')
        lines.extend(f'    {field_kind} {name};' for field_kind, name in members)
        lines.extend([f'}} {kind};', ''])
    for name, kind in TABLE_TYPES.items():
        if len(tables[name]) != CAPACITIES[name]:
            raise ValueError('Frozen table capacity/count mismatch')
        macro = 'AI2G_' + COUNT_MACROS[name] + '_COUNT'
        initializer = 'AI2G_' + name.upper() + '_INITIALIZER'
        lines.extend([f'#define {macro} {len(tables[name])}', f'#define {initializer} {{ \\'])
        lines.extend('    ' + _initializer(kind, row) + ', \\' for row in tables[name])
        lines.extend(['}', f'static const {kind} ai2g_{name}_host[{macro}] = {initializer};', '#ifdef __CUDACC__',
                      f'static __device__ const {kind} ai2g_{name}_device[{macro}] = {initializer};', '#endif',
                      '#if defined(__CUDA_ARCH__)', f'#define ai2g_{name} ai2g_{name}_device', '#else',
                      f'#define ai2g_{name} ai2g_{name}_host', '#endif', f'#undef {initializer}', ''])
    return '\n'.join([*lines, '#endif', ''])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=ROOT / 'native/game_attack_data.h')
    parser.add_argument('--check', action='store_true', help='Compare exact bytes; never write')
    args = parser.parse_args(argv)
    rendered = render_header(build_tables(args.source_root)).encode('utf8')
    if args.check:
        if args.output.read_bytes() != rendered:
            raise ValueError('Generated attack header differs; refusing stale tables')
    else:
        args.output.write_bytes(rendered)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
