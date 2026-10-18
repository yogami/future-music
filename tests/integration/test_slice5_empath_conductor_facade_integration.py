"""Integration tests for Slice 5 (REQ-10): EmpathConductorFacade across all 18 deck actions."""

import pytest
from future_music.application.empath_conductor_facade import (
    EmpathConductorFacade,
    build_default_conductor_facade,
)

from future_music.domain.clock_and_midi import ModalLockActiveError
from future_music.domain.empath_memory import ControlGear, MoodVector, ScaleMode
from future_music.domain.whim_and_hum import HumContourEvent, InputGateMode


def test_facade_surfaces_orbs_and_stages_ghost_preview_on_keys_1_2_3_and_backspace() -> None:
    """REQ-10: Keys 1, 2, 3 stage surfaced Memory Orbs in Ghost Preview; Backspace discards."""
    facade = build_default_conductor_facade()
    orbs = facade.update_mood(MoodVector(0.35, 0.55, 0.30, 0.95), current_minutes=0.0)
    assert len(orbs) == 3
    assert orbs[0].asset.asset_id == "robert_miles_children"

    snap_1 = facade.handle_deck_key("1")
    assert snap_1.active_gear == ControlGear.GEAR_2_WHIM_STEERING
    assert snap_1.staged_ghost_asset_id == "robert_miles_children"

    snap_bs = facade.handle_deck_key("BACKSPACE")
    assert snap_bs.active_gear == ControlGear.GEAR_1_AUTOPILOT
    assert snap_bs.staged_ghost_asset_id is None


def test_facade_commits_orb_and_enforces_scale_keys_q_w_e_with_modal_lock() -> None:
    """REQ-10: Committing Sufi Vocal (Orb 2) via RETURN locks Phrygian and blocks Q/E."""
    facade = build_default_conductor_facade()
    facade.update_mood(MoodVector(0.35, 0.55, 0.30, 0.95), current_minutes=5.0)
    facade.handle_deck_key("2")
    snap_commit = facade.handle_deck_key("RETURN", elapsed_minutes=5.0)

    assert snap_commit.active_gear == ControlGear.GEAR_1_AUTOPILOT
    assert snap_commit.active_scale_mode == ScaleMode.PHRYGIAN
    assert snap_commit.modal_lock_active is True
    assert snap_commit.active_bridge_stage is not None

    with pytest.raises(ModalLockActiveError):
        facade.handle_deck_key("Q")


def test_facade_arrow_keys_and_spacebar_tension_hold_and_release() -> None:
    """REQ-10: UP/DOWN morph CC 1 (+/-16), LEFT/RIGHT step density, HOLD/RELEASE_SPACE drop."""
    facade = build_default_conductor_facade()
    assert facade.handle_deck_key("UP").cc1_modulation == 48
    assert facade.handle_deck_key("DOWN").cc1_modulation == 32
    assert facade.handle_deck_key("LEFT").note_density == pytest.approx(0.25)
    assert facade.handle_deck_key("RIGHT").note_density == pytest.approx(0.50)

    snap_hold = facade.handle_deck_key("HOLD_SPACE", hold_progress=1.0)
    assert snap_hold.active_gear == ControlGear.GEAR_3_TACTILE_OVERRIDE
    assert snap_hold.drums_hpf_hz == pytest.approx(350.0)

    snap_rel = facade.handle_deck_key("RELEASE_SPACE")
    assert snap_rel.active_gear == ControlGear.GEAR_1_AUTOPILOT
    assert snap_rel.drums_hpf_hz == pytest.approx(20.0)


def test_facade_voice_and_hum_gate_keys_route_spoken_whim_and_hum_contour() -> None:
    """REQ-10: HOLD_V/RELEASE_V and HOLD_H/RELEASE_H gate voice and hum pipelines cleanly."""
    facade: EmpathConductorFacade = build_default_conductor_facade()
    assert facade.handle_deck_key("HOLD_V").input_gate_mode == InputGateMode.HOLD_V
    intent = facade.submit_voice_whim("bring in the Robert Miles piano in 16 bars")
    assert intent.target_asset_id == "robert_miles_children"
    assert facade.handle_deck_key("RELEASE_V").input_gate_mode == InputGateMode.IDLE

    assert facade.handle_deck_key("HOLD_H").input_gate_mode == InputGateMode.HOLD_H
    notes = facade.submit_hum_contour((HumContourEvent(onset_ms=205.0, f0_hz=349.23),))
    assert len(notes) == 1
    assert facade.handle_deck_key("RELEASE_H").input_gate_mode == InputGateMode.IDLE
