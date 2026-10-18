"""Unit tests for Slice 5 (REQ-10): 18-Action MacBook Keyboard Empath Deck Domain."""

import pytest
from future_music.domain.empath_deck import (
    DECK_KEY_REGISTRY,
    DeckActionCategory,
    DeckKeyCode,
    resolve_deck_key,
)

from future_music.domain.empath_memory import ControlGear, ScaleMode


def test_deck_key_registry_contains_all_18_macbook_empath_deck_actions() -> None:
    """REQ-10: Registry maps all 18 deterministic MacBook Keyboard + Mic gate actions."""
    assert len(DECK_KEY_REGISTRY) == 18
    assert set(DECK_KEY_REGISTRY.keys()) == set(DeckKeyCode)


def test_orb_keys_1_2_3_map_to_orb_slots_and_gear2_whim_steering() -> None:
    """REQ-10: Keys '1', '2', '3' map to orb_slot 1, 2, 3 and GEAR_2_WHIM_STEERING."""
    for slot_num in (1, 2, 3):
        spec = resolve_deck_key(str(slot_num))
        assert spec.category == DeckActionCategory.ORB_SELECT
        assert spec.orb_slot == slot_num
        assert spec.target_gear == ControlGear.GEAR_2_WHIM_STEERING


def test_scale_keys_q_w_e_map_to_natural_minor_phrygian_and_dorian() -> None:
    """REQ-10: Keys 'Q', 'W', 'E' map to F Natural Minor, F Phrygian, and F Dorian."""
    spec_q = resolve_deck_key("q")
    spec_w = resolve_deck_key("W")
    spec_e = resolve_deck_key("e")
    assert spec_q.scale_mode == ScaleMode.NATURAL_MINOR
    assert spec_w.scale_mode == ScaleMode.PHRYGIAN
    assert spec_e.scale_mode == ScaleMode.DORIAN
    assert spec_q.category == DeckActionCategory.SCALE_SHIFT


def test_voice_gate_keys_hold_v_and_release_v_map_to_input_gate_category() -> None:
    """REQ-10: HOLD_V engages GEAR_2_WHIM_STEERING; RELEASE_V closes voice mic gate."""
    hold_v = resolve_deck_key("HOLD_V")
    rel_v = resolve_deck_key("RELEASE_V")
    assert hold_v.category == DeckActionCategory.INPUT_GATE
    assert hold_v.target_gear == ControlGear.GEAR_2_WHIM_STEERING
    assert rel_v.category == DeckActionCategory.INPUT_GATE
    assert rel_v.target_gear is None


def test_hum_gate_keys_hold_h_and_release_h_map_to_input_gate_category() -> None:
    """REQ-10: HOLD_H engages GEAR_2_WHIM_STEERING; RELEASE_H closes hum mic gate."""
    hold_h = resolve_deck_key("HOLD_H")
    rel_h = resolve_deck_key("RELEASE_H")
    assert hold_h.category == DeckActionCategory.INPUT_GATE
    assert hold_h.target_gear == ControlGear.GEAR_2_WHIM_STEERING
    assert rel_h.category == DeckActionCategory.INPUT_GATE
    assert rel_h.target_gear is None


def test_ghost_decision_keys_return_and_backspace_return_to_gear1_autopilot() -> None:
    """REQ-10: RETURN commits and BACKSPACE discards Ghost Preview into GEAR_1_AUTOPILOT."""
    ret_spec = resolve_deck_key("RETURN")
    bs_spec = resolve_deck_key("BACKSPACE")
    assert ret_spec.category == DeckActionCategory.GHOST_DECISION
    assert ret_spec.target_gear == ControlGear.GEAR_1_AUTOPILOT
    assert bs_spec.category == DeckActionCategory.GHOST_DECISION
    assert bs_spec.target_gear == ControlGear.GEAR_1_AUTOPILOT


def test_arrow_keys_up_down_left_right_map_to_expression_step_category() -> None:
    """REQ-10: UP, DOWN, LEFT, RIGHT map to EXPRESSION_STEP while preserving current gear."""
    for arrow in ("UP", "DOWN", "LEFT", "RIGHT"):
        spec = resolve_deck_key(arrow)
        assert spec.category == DeckActionCategory.EXPRESSION_STEP
        assert spec.target_gear is None


def test_spacebar_hold_and_release_map_to_gear3_override_and_gear1_drop() -> None:
    """REQ-10: HOLD_SPACE enters GEAR_3_TACTILE_OVERRIDE; RELEASE_SPACE returns to GEAR_1."""
    hold_sp = resolve_deck_key("HOLD_SPACE")
    rel_sp = resolve_deck_key("RELEASE_SPACE")
    assert hold_sp.category == DeckActionCategory.TENSION_OVERRIDE
    assert hold_sp.target_gear == ControlGear.GEAR_3_TACTILE_OVERRIDE
    assert rel_sp.category == DeckActionCategory.TENSION_OVERRIDE
    assert rel_sp.target_gear == ControlGear.GEAR_1_AUTOPILOT


def test_resolve_deck_key_normalizes_whitespace_and_lowercase() -> None:
    """REQ-10: resolve_deck_key trims whitespace and normalizes lowercase key strings."""
    spec = resolve_deck_key("  hold_space  ")
    assert spec.key_code == DeckKeyCode.HOLD_SPACE


def test_resolve_deck_key_raises_value_error_for_unmapped_key() -> None:
    """REQ-10: Unknown key codes raise ValueError rather than silently failing."""
    with pytest.raises(ValueError, match="Unsupported deck key"):
        resolve_deck_key("F12")
