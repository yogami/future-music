"""Slice 5 (REQ-10): 18-Action MacBook Keyboard Empath Deck Domain Specification."""

from dataclasses import dataclass
from enum import StrEnum

from future_music.domain.empath_memory import ControlGear, ScaleMode


class DeckKeyCode(StrEnum):
    """Canonical key codes for the 18 MacBook Keyboard Empath Deck controls (REQ-10)."""

    KEY_1 = "1"
    KEY_2 = "2"
    KEY_3 = "3"
    KEY_Q = "Q"
    KEY_W = "W"
    KEY_E = "E"
    HOLD_V = "HOLD_V"
    RELEASE_V = "RELEASE_V"
    HOLD_H = "HOLD_H"
    RELEASE_H = "RELEASE_H"
    RETURN = "RETURN"
    BACKSPACE = "BACKSPACE"
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    HOLD_SPACE = "HOLD_SPACE"
    RELEASE_SPACE = "RELEASE_SPACE"


class DeckActionCategory(StrEnum):
    """Functional categories for MacBook Empath Deck actions."""

    ORB_SELECT = "ORB_SELECT"
    SCALE_SHIFT = "SCALE_SHIFT"
    INPUT_GATE = "INPUT_GATE"
    GHOST_DECISION = "GHOST_DECISION"
    EXPRESSION_STEP = "EXPRESSION_STEP"
    TENSION_OVERRIDE = "TENSION_OVERRIDE"


@dataclass(frozen=True)
class DeckKeySpec:
    """Immutable specification of a MacBook Empath Deck key binding (REQ-10)."""

    key_code: DeckKeyCode
    control_name: str
    category: DeckActionCategory
    target_gear: ControlGear | None
    orb_slot: int | None = None
    scale_mode: ScaleMode | None = None


DECK_KEY_REGISTRY: dict[DeckKeyCode, DeckKeySpec] = {
    DeckKeyCode.KEY_1: DeckKeySpec(
        DeckKeyCode.KEY_1,
        "Memory Orb 1",
        DeckActionCategory.ORB_SELECT,
        ControlGear.GEAR_2_WHIM_STEERING,
        orb_slot=1,
    ),
    DeckKeyCode.KEY_2: DeckKeySpec(
        DeckKeyCode.KEY_2,
        "Memory Orb 2",
        DeckActionCategory.ORB_SELECT,
        ControlGear.GEAR_2_WHIM_STEERING,
        orb_slot=2,
    ),
    DeckKeyCode.KEY_3: DeckKeySpec(
        DeckKeyCode.KEY_3,
        "Memory Orb 3",
        DeckActionCategory.ORB_SELECT,
        ControlGear.GEAR_2_WHIM_STEERING,
        orb_slot=3,
    ),
    DeckKeyCode.KEY_Q: DeckKeySpec(
        DeckKeyCode.KEY_Q,
        "Scale: Natural Minor",
        DeckActionCategory.SCALE_SHIFT,
        None,
        scale_mode=ScaleMode.NATURAL_MINOR,
    ),
    DeckKeyCode.KEY_W: DeckKeySpec(
        DeckKeyCode.KEY_W,
        "Scale: Phrygian",
        DeckActionCategory.SCALE_SHIFT,
        None,
        scale_mode=ScaleMode.PHRYGIAN,
    ),
    DeckKeyCode.KEY_E: DeckKeySpec(
        DeckKeyCode.KEY_E,
        "Scale: Dorian",
        DeckActionCategory.SCALE_SHIFT,
        None,
        scale_mode=ScaleMode.DORIAN,
    ),
    DeckKeyCode.HOLD_V: DeckKeySpec(
        DeckKeyCode.HOLD_V,
        "Gate Voice Mic",
        DeckActionCategory.INPUT_GATE,
        ControlGear.GEAR_2_WHIM_STEERING,
    ),
    DeckKeyCode.RELEASE_V: DeckKeySpec(
        DeckKeyCode.RELEASE_V,
        "Close Voice Mic",
        DeckActionCategory.INPUT_GATE,
        None,
    ),
    DeckKeyCode.HOLD_H: DeckKeySpec(
        DeckKeyCode.HOLD_H,
        "Gate Hum Mic",
        DeckActionCategory.INPUT_GATE,
        ControlGear.GEAR_2_WHIM_STEERING,
    ),
    DeckKeyCode.RELEASE_H: DeckKeySpec(
        DeckKeyCode.RELEASE_H,
        "Close Hum Mic",
        DeckActionCategory.INPUT_GATE,
        None,
    ),
    DeckKeyCode.RETURN: DeckKeySpec(
        DeckKeyCode.RETURN,
        "Commit Ghost Preview",
        DeckActionCategory.GHOST_DECISION,
        ControlGear.GEAR_1_AUTOPILOT,
    ),
    DeckKeyCode.BACKSPACE: DeckKeySpec(
        DeckKeyCode.BACKSPACE,
        "Discard Ghost Preview",
        DeckActionCategory.GHOST_DECISION,
        ControlGear.GEAR_1_AUTOPILOT,
    ),
    DeckKeyCode.UP: DeckKeySpec(
        DeckKeyCode.UP,
        "Filter Morph Up",
        DeckActionCategory.EXPRESSION_STEP,
        None,
    ),
    DeckKeyCode.DOWN: DeckKeySpec(
        DeckKeyCode.DOWN,
        "Filter Morph Down",
        DeckActionCategory.EXPRESSION_STEP,
        None,
    ),
    DeckKeyCode.LEFT: DeckKeySpec(
        DeckKeyCode.LEFT,
        "Note Density Sparse",
        DeckActionCategory.EXPRESSION_STEP,
        None,
    ),
    DeckKeyCode.RIGHT: DeckKeySpec(
        DeckKeyCode.RIGHT,
        "Note Density Busy",
        DeckActionCategory.EXPRESSION_STEP,
        None,
    ),
    DeckKeyCode.HOLD_SPACE: DeckKeySpec(
        DeckKeyCode.HOLD_SPACE,
        "Tension Hold Override",
        DeckActionCategory.TENSION_OVERRIDE,
        ControlGear.GEAR_3_TACTILE_OVERRIDE,
    ),
    DeckKeyCode.RELEASE_SPACE: DeckKeySpec(
        DeckKeyCode.RELEASE_SPACE,
        "Drop Release",
        DeckActionCategory.TENSION_OVERRIDE,
        ControlGear.GEAR_1_AUTOPILOT,
    ),
}


def resolve_deck_key(raw_key: str) -> DeckKeySpec:
    """Normalize raw_key and return its DeckKeySpec or raise ValueError."""
    normalized = raw_key.strip().upper()
    try:
        code = DeckKeyCode(normalized)
    except ValueError as err:
        raise ValueError(f"Unsupported deck key '{raw_key}'") from err
    return DECK_KEY_REGISTRY[code]
