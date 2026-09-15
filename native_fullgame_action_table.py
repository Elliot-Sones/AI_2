"""Canonical runtime action semantics for the native full-game MVP."""

from copy import deepcopy


ACTION_TABLE = {
    0: {"key": "W", "meaning": "up_aim_axis", "trace_required": True},
    1: {"key": "A", "meaning": "left", "trace_required": True},
    2: {"key": "S", "meaning": "down_aim_fastfall_axis", "trace_required": True},
    3: {"key": "D", "meaning": "right", "trace_required": True},
    4: {"key": "Space", "meaning": "jump", "trace_required": True},
    5: {"key": "H", "meaning": "pickup_drop", "trace_required": True},
    6: {"key": "L", "meaning": "dash_dodge", "trace_required": True},
    7: {"key": "J", "meaning": "light_attack", "trace_required": True},
    8: {"key": "K", "meaning": "heavy_attack", "trace_required": True},
    9: {"key": "G", "meaning": "taunt", "trace_required": True},
}

_MOVEMENT_MASK = (0, 1, 2, 3, 4, 6)
_CONTACT_MASK = (0, 1, 2, 3, 4, 6, 7)
_STALE_LABEL_SCAN_PATTERNS = {
    "5=dodge": r"5=dodge",
    "6=pickup": r"6=pickup",
    "9=throw": r"9=throw",
    "indices_5_8_9_grouped_as_tools": r"\[5,\s*8,\s*9\].*tool",
    "throw_mentions_index_9": r"throw.*index 9",
    "index_9_mentions_throw": r"index 9.*throw",
}


def canonical_action_table():
    """Return the runtime-backed Box10 action table."""
    return deepcopy(ACTION_TABLE)


def movement_mask():
    """Return the current movement curriculum action indices."""
    return list(_MOVEMENT_MASK)


def contact_mask():
    """Return the current contact curriculum action indices."""
    return list(_CONTACT_MASK)


def stale_label_scan_patterns():
    """Return named patterns for stale action-label audits."""
    return dict(_STALE_LABEL_SCAN_PATTERNS)
