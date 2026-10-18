"""Integration tests for Slice 2 (REQ-02, REQ-03, REQ-04): HarmonicClockUseCase and IAC Adapter."""

import pytest
from future_music.application.harmonic_clock_use_case import HarmonicClockUseCase
from future_music.domain.clock_and_midi import (
    GarageBandExpressionState,
    ModalScaleEngine,
    SampleClock,
    UnsupportedGarageBandMidiCcError,
)
from future_music.infrastructure.iac_midi_adapter import GarageBandIacMidiAdapter

from future_music.domain.empath_memory import MemoryAsset, ScaleMode


def build_harmonic_use_case() -> tuple[HarmonicClockUseCase, GarageBandIacMidiAdapter]:
    """Assemble HarmonicClockUseCase wired to a real GarageBandIacMidiAdapter."""
    adapter = GarageBandIacMidiAdapter(port_name="IAC Driver Empath Bus 1")
    use_case = HarmonicClockUseCase(
        clock=SampleClock(sample_rate=48000, bpm=124.0, midi_advance_ms=18.0),
        expression=GarageBandExpressionState(),
        modal_engine=ModalScaleEngine(),
        midi_port=adapter,
    )
    return use_case, adapter


def test_harmonic_clock_schedules_midi_motif_864_samples_ahead_of_beat_over_iac() -> None:
    """REQ-02 & REQ-04: Motif notes dispatch to IAC with -18ms (864 sample) pre-fire offset."""
    use_case, adapter = build_harmonic_use_case()
    asset = MemoryAsset(
        asset_id="robert_miles_children",
        title="Robert Miles - Children",
        emotional_embedding=(0.35, 0.55, 0.30, 0.95),
        native_bpm=124.0,
        root_key="F",
        native_mode=ScaleMode.NATURAL_MINOR,
        audio_stem_id="robert_miles_piano.wav",
        symbolic_motif=(65, 68, 72, 70),
        requires_modal_lock=False,
        dominant_freq_hz=(800.0, 2800.0),
    )
    use_case.load_active_asset(asset)
    use_case.shift_scale_mode(ScaleMode.PHRYGIAN)
    dispatched = use_case.schedule_bar_motif(bar_start_beat=4, velocity=100)

    assert len(dispatched) == 2
    expected_beat4_sample = use_case.clock.beat_sample_index(4) - 864
    assert dispatched[0].trigger_sample == expected_beat4_sample
    assert len(adapter.sent_messages) == 2


def test_harmonic_clock_filter_and_density_controls_emit_cc1_and_scale_midi_notes() -> None:
    """REQ-03: Morphing filter up/down emits CC 1 over IAC, and density scales note count."""
    use_case, adapter = build_harmonic_use_case()
    asset = MemoryAsset(
        asset_id="robert_miles_children",
        title="Robert Miles - Children",
        emotional_embedding=(0.35, 0.55, 0.30, 0.95),
        native_bpm=124.0,
        root_key="F",
        native_mode=ScaleMode.NATURAL_MINOR,
        audio_stem_id="robert_miles_piano.wav",
        symbolic_motif=(65, 68, 72, 70),
        requires_modal_lock=False,
        dominant_freq_hz=(800.0, 2800.0),
    )
    use_case.load_active_asset(asset)
    cc_val = use_case.step_filter_up(sample_pos=1000)
    assert cc_val == 48
    assert adapter.sent_messages[-1].message_type == "control_change"
    assert adapter.sent_messages[-1].control == 1
    assert adapter.sent_messages[-1].value == 48

    use_case.step_density_right()
    use_case.step_density_right()
    dispatched = use_case.schedule_bar_motif(bar_start_beat=1, velocity=95)
    assert len(dispatched) == 4


def test_harmonic_clock_iac_adapter_rejects_unsupported_cc74_and_allows_sustain_bend() -> None:
    """REQ-03: GarageBandIacMidiAdapter accepts CC 64 and Pitch Bend but rejects CC 74."""
    _, adapter = build_harmonic_use_case()
    adapter.send_sustain_pedal(sustain_on=True, trigger_sample=500)
    adapter.send_pitch_bend(pitch_bend=-2048, trigger_sample=600)
    assert adapter.sent_messages[-2].control == 64
    assert adapter.sent_messages[-2].value == 127
    assert adapter.sent_messages[-1].message_type == "pitchwheel"
    assert adapter.sent_messages[-1].pitch == -2048
    with pytest.raises(UnsupportedGarageBandMidiCcError):
        adapter.send_cc(cc_number=74, value=64, trigger_sample=700)
