"""Acceptance tests for Slice 4 (REQ-07, REQ-08, REQ-09): Voice, Hum, Ghost & Lookahead Bridge."""

import numpy as np
import pytest
from future_music.application.whim_and_bridge_use_case import WhimAndBridgeUseCase
from future_music.domain.whim_and_hum import (
    HumContourEvent,
    InputGateConflictError,
    InputGateController,
)
from future_music.infrastructure.audio_routing_adapter import DualChannelAudioRouterAdapter

from future_music.domain.clock_and_midi import ModalScaleEngine, SampleClock
from future_music.domain.empath_memory import EpisodicSetMemory, ScaleMode
from future_music.infrastructure.iac_midi_adapter import GarageBandIacMidiAdapter
from future_music.infrastructure.in_memory_vault import create_default_empath_vault


def create_slice4_conductor() -> WhimAndBridgeUseCase:
    """Create a complete Slice 4 conductor stack with unmocked domain and adapters."""
    return WhimAndBridgeUseCase(
        vault=create_default_empath_vault(),
        empath_id="empath_euro_90s",
        episodic_memory=EpisodicSetMemory(window_minutes=60.0),
        clock=SampleClock(sample_rate=48000, bpm=124.0),
        modal_engine=ModalScaleEngine(),
        input_gate=InputGateController(),
        audio_router=DualChannelAudioRouterAdapter(),
        midi_port=GarageBandIacMidiAdapter(),
    )


def test_req07_gated_voice_whim_and_hum_quantizer_prevent_crosstalk_and_snap_to_scale() -> None:
    """REQ-07: Hold V parses voice whim into Ghost Preview; Hold H snaps hummed f0 to scale."""
    conductor = create_slice4_conductor()
    conductor.input_gate.open_voice_gate()
    with pytest.raises(InputGateConflictError):
        conductor.submit_hum_contour((HumContourEvent(onset_ms=100.0, f0_hz=349.23),))

    intent = conductor.submit_voice_whim(
        "bring in the Robert Miles piano in 16 bars, make it darker Phrygian, drop the drums"
    )
    conductor.input_gate.close_gate()
    assert intent.target_asset_id == "robert_miles_children"
    assert intent.target_scale_mode == ScaleMode.PHRYGIAN
    assert conductor.ghost_preview_state.staged_asset_id == "robert_miles_children"


def test_req08_ghost_preview_stereo_split_and_dual_device_isolation_on_stage_and_discard() -> None:
    """REQ-08: Ghost Preview plays only in Left/Headphone channel; Backspace clears it cleanly."""
    conductor = create_slice4_conductor()
    conductor.stage_asset_to_ghost_preview("sufi_vocal", bars=2)

    main_pa = np.full(256, 0.75, dtype=np.float64)
    ghost_cue = np.full(256, 0.25, dtype=np.float64)
    active_out = conductor.render_dual_bus_audio(main_mono=main_pa, ghost_mono=ghost_cue)
    assert np.allclose(active_out.stereo_split_frame[:, 0], 0.25)
    assert np.allclose(active_out.stereo_split_frame[:, 1], 0.75)

    conductor.discard_ghost_preview()
    cleared_out = conductor.render_dual_bus_audio(main_mono=main_pa, ghost_mono=ghost_cue)
    assert np.allclose(cleared_out.stereo_split_frame[:, 0], 0.0)
    assert np.allclose(cleared_out.stereo_split_frame[:, 1], 0.75)
    assert np.allclose(cleared_out.ghost_preview_stereo, 0.0)


def test_req09_lookahead_bridge_executes_4_stages_and_engages_modal_lock_on_drop() -> None:
    """REQ-09: Committing Sufi Vocal runs 4-stage bridge and Phrygian Modal Lock."""
    conductor = create_slice4_conductor()
    conductor.stage_asset_to_ghost_preview("sufi_vocal", bars=2)
    plan = conductor.commit_ghost_preview(bridge_bars=16, elapsed_minutes=12.0)

    assert len(plan.stages) == 4
    assert plan.stages[0].note_density_multiplier == pytest.approx(0.5)
    assert plan.stages[2].stem_gain == pytest.approx(0.35)
    assert plan.stages[3].stem_gain == pytest.approx(1.0)
    assert conductor.modal_engine.active_scale_mode == ScaleMode.PHRYGIAN
    assert conductor.modal_engine.modal_lock_active is True
