"""Acceptance tests for Slice 2 (REQ-02, REQ-03, REQ-04): Clock Sync, IAC MIDI & Modal Safety."""

import pytest
from future_music.application.harmonic_clock_use_case import HarmonicClockUseCase
from future_music.domain.clock_and_midi import (
    GarageBandExpressionState,
    ModalLockActiveError,
    ModalScaleEngine,
    SampleClock,
)
from future_music.infrastructure.iac_midi_adapter import GarageBandIacMidiAdapter

from future_music.domain.empath_memory import ScaleMode
from future_music.infrastructure.in_memory_vault import create_default_empath_vault


def test_req02_req03_sample_counted_clock_fires_iac_midi_18ms_early_with_alchemy_controls() -> None:
    """REQ-02 & REQ-03: Master clock fires IAC MIDI 864 samples early and sweeps CC 1."""
    vault = create_default_empath_vault()
    robert_miles = vault.get_assets("empath_euro_90s")[0]
    adapter = GarageBandIacMidiAdapter(port_name="IAC Driver Empath Bus 1")
    use_case = HarmonicClockUseCase(
        clock=SampleClock(sample_rate=48000, bpm=124.0, midi_advance_ms=18.0),
        expression=GarageBandExpressionState(cc1_modulation=32, note_density=1.0),
        modal_engine=ModalScaleEngine(),
        midi_port=adapter,
    )
    use_case.load_active_asset(robert_miles)
    use_case.step_filter_up(sample_pos=0)
    use_case.step_filter_up(sample_pos=1024)
    events = use_case.schedule_bar_motif(bar_start_beat=4, velocity=110)

    beat_4_acoustic = use_case.clock.beat_sample_index(4)
    assert events[0].trigger_sample == beat_4_acoustic - 864
    assert use_case.expression.cc1_modulation == 64
    assert [e.note for e in events] == [65, 68, 72, 70, 68, 67, 65, 63]


def test_req04_robert_miles_mutates_to_phrygian_without_clash_while_sufi_locks_mode() -> None:
    """REQ-04: Robert Miles mutates G4(67)->Gb4(66) in Phrygian; Sufi vocal locks to Phrygian."""
    vault = create_default_empath_vault()
    euro_assets = {a.asset_id: a for a in vault.get_assets("empath_euro_90s")}
    adapter = GarageBandIacMidiAdapter()
    use_case = HarmonicClockUseCase(
        clock=SampleClock(),
        expression=GarageBandExpressionState(note_density=1.0),
        modal_engine=ModalScaleEngine(),
        midi_port=adapter,
    )

    use_case.load_active_asset(euro_assets["robert_miles_children"])
    state_phrygian = use_case.shift_scale_mode(ScaleMode.PHRYGIAN)
    assert state_phrygian.audio_stem_gain == 0.0
    events = use_case.schedule_bar_motif(bar_start_beat=0, velocity=100)
    assert [e.note for e in events] == [65, 68, 72, 70, 68, 66, 65, 63]

    state_sufi = use_case.load_active_asset(euro_assets["sufi_vocal"])
    assert state_sufi.modal_lock_active is True
    assert state_sufi.active_scale_mode == ScaleMode.PHRYGIAN
    with pytest.raises(ModalLockActiveError):
        use_case.shift_scale_mode(ScaleMode.NATURAL_MINOR)
