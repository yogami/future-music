"""Integration tests for Slice 4 (REQ-07, REQ-08, REQ-09): Whim, Ghost Preview & Bridge UseCase."""

import numpy as np
import pytest
from future_music.application.whim_and_bridge_use_case import WhimAndBridgeUseCase
from future_music.domain.whim_and_hum import (
    HumContourEvent,
    InputGateController,
    InputGateInactiveError,
)
from future_music.infrastructure.audio_routing_adapter import DualChannelAudioRouterAdapter

from future_music.domain.clock_and_midi import ModalScaleEngine, SampleClock, ScheduledMidiNote
from future_music.domain.empath_memory import ControlGear, EpisodicSetMemory
from future_music.infrastructure.iac_midi_adapter import GarageBandIacMidiAdapter
from future_music.infrastructure.in_memory_vault import (
    InMemoryPersonalVault,
    create_default_empath_vault,
)


def build_slice4_use_case() -> tuple[
    WhimAndBridgeUseCase,
    InMemoryPersonalVault,
    DualChannelAudioRouterAdapter,
    GarageBandIacMidiAdapter,
]:
    """Assemble WhimAndBridgeUseCase with real infrastructure adapters."""
    vault = create_default_empath_vault()
    router = DualChannelAudioRouterAdapter()
    midi = GarageBandIacMidiAdapter()
    use_case = WhimAndBridgeUseCase(
        vault=vault,
        empath_id="empath_euro_90s",
        episodic_memory=EpisodicSetMemory(window_minutes=60.0),
        clock=SampleClock(sample_rate=48000, bpm=124.0),
        modal_engine=ModalScaleEngine(),
        input_gate=InputGateController(),
        audio_router=router,
        midi_port=midi,
    )
    return use_case, vault, router, midi


def test_stage_and_discard_ghost_preview_keeps_main_pa_isolated_in_stereo_split() -> None:
    """REQ-08: Staging Ghost Preview routes strictly to Left Channel; Backspace clears it."""
    use_case, _, router, _ = build_slice4_use_case()
    ghost_state = use_case.stage_asset_to_ghost_preview("robert_miles_children", bars=2)
    assert ghost_state.active is True
    assert ghost_state.gear == ControlGear.GEAR_2_WHIM_STEERING
    assert ghost_state.emitted_tool_call is not None
    assert ghost_state.emitted_tool_call.tool_name == "route_to_ghost_cue"

    main_signal = np.ones(512, dtype=np.float64) * 0.6
    ghost_signal = np.ones(512, dtype=np.float64) * 0.4
    out = router.route_signals(
        main_mono=main_signal, ghost_mono=ghost_signal, ghost_active=ghost_state.active
    )
    assert np.allclose(out.stereo_split_frame[:, 0], 0.4)
    assert np.allclose(out.stereo_split_frame[:, 1], 0.6)

    discarded = use_case.discard_ghost_preview()
    assert discarded.active is False
    assert discarded.gear == ControlGear.GEAR_1_AUTOPILOT


def test_commit_ghost_preview_builds_4_stage_bridge_and_records_episodic_memory() -> None:
    """REQ-08 & REQ-09: Pressing Return commits Ghost Preview into 4-Stage Bridge Planner."""
    use_case, _, _, _ = build_slice4_use_case()
    use_case.stage_asset_to_ghost_preview("sufi_vocal", bars=2)
    plan = use_case.commit_ghost_preview(bridge_bars=16, elapsed_minutes=20.0)

    assert plan.target_asset_id == "sufi_vocal"
    assert plan.total_bars == 16
    assert len(plan.stages) == 4
    assert use_case.modal_engine.modal_lock_active is True
    assert use_case.episodic_memory.compute_recency_penalty("sufi_vocal", 30.0) > 0.0


def test_gated_hum_contour_dispatches_lag_compensated_pentatonic_midi_notes() -> None:
    """REQ-07: Holding H and humming dispatches -80ms lag-compensated pentatonic MIDI notes."""
    use_case, _, _, midi = build_slice4_use_case()
    with pytest.raises(InputGateInactiveError):
        use_case.submit_hum_contour((HumContourEvent(onset_ms=205.0, f0_hz=349.23),))

    use_case.input_gate.open_hum_gate()
    quantized = use_case.submit_hum_contour(
        (HumContourEvent(onset_ms=205.0, f0_hz=380.0, velocity=105),)
    )
    use_case.input_gate.close_gate()

    assert len(quantized) == 1
    assert midi.sent_messages[-1].message_type == "note_on"
    assert midi.sent_messages[-1].note == quantized[0].midi_note


def test_pre_play_spectral_critic_carves_midrange_midi_before_iac_dispatch() -> None:
    """REQ-09: Scheduling a bar with vocal presence (1-3kHz) shifts MIDI C4-C6 down 1 octave."""
    use_case, _, _, midi = build_slice4_use_case()
    raw_notes = (
        ScheduledMidiNote(note=72, velocity=100, trigger_sample=500),
        ScheduledMidiNote(note=48, velocity=100, trigger_sample=1000),
    )
    critic_result = use_case.schedule_bar_with_spectral_critic(
        stem_freq_band_hz=(1100.0, 2900.0), scheduled_notes=raw_notes
    )
    assert critic_result.spectral_carve_applied is True
    assert midi.sent_messages[0].note == 60
    assert midi.sent_messages[0].velocity == 85
    assert midi.sent_messages[1].note == 48
    assert midi.sent_messages[1].velocity == 100
