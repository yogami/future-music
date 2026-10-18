"""Unit tests for Slice 2 (REQ-02, REQ-03, REQ-04): SampleClock, GarageBand MIDI, and Modal Lock."""

import pytest
from future_music.domain.clock_and_midi import (
    CC1_STEP_DELTA,
    DEFAULT_CC1_MODULATION,
    DEFAULT_MIDI_ADVANCE_MS,
    NOTE_DENSITY_STEP,
    GarageBandExpressionState,
    ModalLockActiveError,
    ModalScaleEngine,
    SampleClock,
    UnsupportedGarageBandMidiCcError,
    validate_garageband_cc,
    validate_midi_note_and_velocity,
    validate_pitch_bend,
)

from future_music.domain.empath_memory import MemoryAsset, ScaleMode


def make_robert_miles_asset() -> MemoryAsset:
    """Build Robert Miles Children asset with F Natural Minor audio stem and symbolic MIDI motif."""
    return MemoryAsset(
        asset_id="robert_miles_children",
        title="Robert Miles - Children",
        emotional_embedding=(0.35, 0.55, 0.30, 0.95),
        native_bpm=124.0,
        root_key="F",
        native_mode=ScaleMode.NATURAL_MINOR,
        audio_stem_id="robert_miles_piano.wav",
        symbolic_motif=(65, 68, 72, 70, 68, 67, 65, 63),
        requires_modal_lock=False,
        dominant_freq_hz=(800.0, 2800.0),
    )


def make_sufi_vocal_asset() -> MemoryAsset:
    """Build Sufi vocal asset in F Phrygian requiring Modal Lock."""
    return MemoryAsset(
        asset_id="sufi_vocal",
        title="Sufi Vocal in F Phrygian",
        emotional_embedding=(0.35, 0.55, 0.30, 0.95),
        native_bpm=124.0,
        root_key="F",
        native_mode=ScaleMode.PHRYGIAN,
        audio_stem_id="sufi_vocal.wav",
        symbolic_motif=(),
        requires_modal_lock=True,
        dominant_freq_hz=(1000.0, 3000.0),
    )


def test_sample_clock_computes_exact_samples_per_beat_and_bar_at_124_bpm_48khz() -> None:
    """REQ-02: At 124 BPM and 48000 Hz, samples_per_beat is 23225.806... and bar is 4x."""
    clock = SampleClock(sample_rate=48000, bpm=124.0, midi_advance_ms=DEFAULT_MIDI_ADVANCE_MS)
    expected_beat = 48000.0 * 60.0 / 124.0
    assert clock.samples_per_beat == pytest.approx(expected_beat)
    assert clock.samples_per_bar == pytest.approx(expected_beat * 4.0)


def test_sample_clock_computes_18ms_midi_advance_offset_of_864_samples_and_clamps_at_zero() -> None:
    """REQ-02: 18.0 ms advance at 48 kHz equals 864 samples; beat 0 clamps at sample 0."""
    clock = SampleClock(sample_rate=48000, bpm=124.0, midi_advance_ms=18.0)
    assert clock.midi_advance_samples == 864
    assert clock.midi_trigger_sample(beat_number=0) == 0
    beat_1_sample = clock.beat_sample_index(beat_number=1)
    assert clock.midi_trigger_sample(beat_number=1) == beat_1_sample - 864


def test_sample_clock_advances_frame_counter_and_resolves_1_indexed_bar_and_beat() -> None:
    """REQ-02: Advancing audio callback frames tracks sample position and (bar, beat_in_bar)."""
    clock = SampleClock(sample_rate=48000, bpm=120.0)
    prev, curr = clock.advance_samples(24000)
    assert (prev, curr) == (0, 24000)
    assert clock.current_bar_and_beat(0) == (1, 1)
    assert clock.current_bar_and_beat(24000) == (1, 2)
    assert clock.current_bar_and_beat(96000) == (2, 1)


def test_garageband_expression_allows_cc1_and_cc64_and_rejects_arbitrary_cc74() -> None:
    """REQ-03: GarageBand only obeys CC 1 (Mod Wheel) and CC 64 (Sustain); CC 74 is rejected."""
    validate_garageband_cc(cc_number=1, value=64)
    validate_garageband_cc(cc_number=64, value=127)
    with pytest.raises(UnsupportedGarageBandMidiCcError, match="CC 74"):
        validate_garageband_cc(cc_number=74, value=100)
    with pytest.raises(ValueError, match="CC value"):
        validate_garageband_cc(cc_number=1, value=128)


def test_garageband_expression_validates_note_velocity_and_pitch_bend_ranges() -> None:
    """REQ-03: MIDI notes/velocities and pitch bend (-8192..8191) validate bounds."""
    validate_midi_note_and_velocity(note=65, velocity=100)
    validate_pitch_bend(pitch_bend=4096)
    with pytest.raises(ValueError, match="velocity"):
        validate_midi_note_and_velocity(note=65, velocity=0)
    with pytest.raises(ValueError, match="pitch_bend"):
        validate_pitch_bend(pitch_bend=9000)


def test_garageband_filter_morph_up_and_down_step_by_16_and_clamp_to_0_127() -> None:
    """REQ-03: [UP] and [DOWN] step CC 1 by +/-16 and clamp within [0, 127]."""
    expr = GarageBandExpressionState(cc1_modulation=DEFAULT_CC1_MODULATION)
    up_event = expr.morph_filter_up()
    assert up_event.cc_number == 1
    assert up_event.value == DEFAULT_CC1_MODULATION + CC1_STEP_DELTA

    for _ in range(10):
        up_event = expr.morph_filter_up()
    assert up_event.value == 127

    for _ in range(10):
        down_event = expr.morph_filter_down()
    assert down_event.value == 0


def test_garageband_note_density_left_and_right_step_by_025_and_filter_notes() -> None:
    """REQ-03: [LEFT] and [RIGHT] step note_density by 0.25 and scale active MIDI notes."""
    expr = GarageBandExpressionState()
    motif = (65, 68, 72, 70, 68, 67, 65, 63)
    assert expr.note_density == pytest.approx(0.5)
    assert expr.filter_notes_by_density(motif) == (65, 68, 72, 70)

    expr.adjust_note_density_left()
    assert expr.note_density == pytest.approx(0.5 - NOTE_DENSITY_STEP)
    assert expr.filter_notes_by_density(motif) == (65, 68)

    expr.adjust_note_density_left()
    assert expr.note_density == 0.0
    assert expr.filter_notes_by_density(motif) == ()

    for _ in range(6):
        expr.adjust_note_density_right()
    assert expr.note_density == 1.0
    assert expr.filter_notes_by_density(motif) == motif


def test_modal_scale_engine_plays_robert_miles_audio_stem_in_native_f_natural_minor() -> None:
    """REQ-04: In native F Natural Minor, Robert Miles plays audio stem at gain 1.0."""
    engine = ModalScaleEngine()
    state = engine.activate_asset(make_robert_miles_asset())
    assert state.active_scale_mode == ScaleMode.NATURAL_MINOR
    assert state.modal_lock_active is False
    assert state.audio_stem_gain == 1.0
    assert state.mutated_midi_motif == (65, 68, 72, 70, 68, 67, 65, 63)


def test_modal_scale_engine_mutes_audio_stem_and_flattens_g_to_gb_in_phrygian() -> None:
    """REQ-04: Shifting Robert Miles to F Phrygian mutes audio stem and mutates G4(67)->Gb4(66)."""
    engine = ModalScaleEngine()
    engine.activate_asset(make_robert_miles_asset())
    state = engine.shift_scale(ScaleMode.PHRYGIAN)
    assert state.active_scale_mode == ScaleMode.PHRYGIAN
    assert state.audio_stem_gain == 0.0
    assert state.mutated_midi_motif == (65, 68, 72, 70, 68, 66, 65, 63)


def test_modal_scale_engine_mutes_audio_stem_and_raises_6th_degree_in_dorian() -> None:
    """REQ-04: Shifting to F Dorian mutes audio stem and raises 6th degree Db(61)->D(62)."""
    engine = ModalScaleEngine()
    asset_with_6th = MemoryAsset(
        asset_id="minor_sixth_motif",
        title="Motif with 6th Degree",
        emotional_embedding=(0.2, 0.5, 0.5, 0.5),
        native_bpm=124.0,
        root_key="F",
        native_mode=ScaleMode.NATURAL_MINOR,
        audio_stem_id="motif.wav",
        symbolic_motif=(53, 61, 65),
        requires_modal_lock=False,
        dominant_freq_hz=(800.0, 2000.0),
    )
    engine.activate_asset(asset_with_6th)
    state = engine.shift_scale(ScaleMode.DORIAN)
    assert state.audio_stem_gain == 0.0
    assert state.mutated_midi_motif == (53, 62, 65)


def test_modal_scale_engine_enforces_modal_lock_on_sufi_vocal_and_rejects_clash() -> None:
    """REQ-04: Activating sufi_vocal locks mode to Phrygian and blocks Natural Minor or Dorian."""
    engine = ModalScaleEngine()
    state = engine.activate_asset(make_sufi_vocal_asset())
    assert state.active_scale_mode == ScaleMode.PHRYGIAN
    assert state.modal_lock_active is True
    assert state.audio_stem_gain == 1.0
    with pytest.raises(ModalLockActiveError, match="sufi_vocal"):
        engine.shift_scale(ScaleMode.NATURAL_MINOR)
    with pytest.raises(ModalLockActiveError, match="sufi_vocal"):
        engine.shift_scale(ScaleMode.DORIAN)


def test_modal_scale_engine_releases_modal_lock_when_switching_to_symbolic_asset() -> None:
    """REQ-04: Transitioning from sufi_vocal back to Robert Miles releases Modal Lock."""
    engine = ModalScaleEngine()
    engine.activate_asset(make_sufi_vocal_asset())
    assert engine.modal_lock_active is True
    engine.activate_asset(make_robert_miles_asset())
    assert engine.modal_lock_active is False
    shifted = engine.shift_scale(ScaleMode.DORIAN)
    assert shifted.active_scale_mode == ScaleMode.DORIAN
