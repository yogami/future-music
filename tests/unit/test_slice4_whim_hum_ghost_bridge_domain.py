"""Unit tests for Slice 4 (REQ-07, REQ-08, REQ-09): Input Gating, Hum Quantizer, Ghost & Bridge."""

import pytest
from future_music.domain.ghost_and_bridge import (
    BridgeStageName,
    build_lookahead_bridge_plan,
    evaluate_spectral_collision,
)
from future_music.domain.whim_and_hum import (
    HUMAN_REACTION_LAG_MS,
    HumContourEvent,
    InputGateConflictError,
    InputGateController,
    InputGateInactiveError,
    InputGateMode,
    parse_voice_whim,
    quantize_hum_contour,
)

from future_music.domain.clock_and_midi import SampleClock, ScheduledMidiNote
from future_music.domain.empath_memory import MemoryAsset, ScaleMode


def make_test_asset(
    asset_id: str = "sufi_vocal",
    mode: ScaleMode = ScaleMode.PHRYGIAN,
    modal_lock: bool = True,
) -> MemoryAsset:
    """Create a deterministic MemoryAsset fixture for Slice 4 unit tests."""
    return MemoryAsset(
        asset_id=asset_id,
        title="Sufi Vocal Phrase",
        emotional_embedding=(0.10, 0.70, 0.65, 0.60),
        native_bpm=124.0,
        root_key="F",
        native_mode=mode,
        audio_stem_id=f"{asset_id}.wav",
        symbolic_motif=(65, 66, 68, 72),
        requires_modal_lock=modal_lock,
        dominant_freq_hz=(1000.0, 3000.0),
    )


def test_input_gate_controller_rejects_un_gated_mic_input_in_idle_mode() -> None:
    """REQ-07: Microphone capture while IDLE raises InputGateInactiveError."""
    gate = InputGateController()
    assert gate.active_mode == InputGateMode.IDLE
    with pytest.raises(InputGateInactiveError):
        gate.require_voice_gate()
    with pytest.raises(InputGateInactiveError):
        gate.require_hum_gate()


def test_input_gate_controller_enforces_mutual_exclusion_between_hold_v_and_hold_h() -> None:
    """REQ-07: Opening HOLD_H while HOLD_V is active raises InputGateConflictError."""
    gate = InputGateController()
    assert gate.open_voice_gate() == InputGateMode.HOLD_V
    with pytest.raises(InputGateConflictError):
        gate.open_hum_gate()
    with pytest.raises(InputGateConflictError):
        gate.require_hum_gate()

    assert gate.close_gate() == InputGateMode.IDLE
    assert gate.open_hum_gate() == InputGateMode.HOLD_H
    with pytest.raises(InputGateConflictError):
        gate.open_voice_gate()


def test_voice_whim_parser_resolves_robert_miles_phrygian_strip_drums_and_16_bars() -> None:
    """REQ-07: Deterministic voice whim parser extracts asset, Phrygian mode, drums, and 16 bars."""
    intent = parse_voice_whim(
        "bring in the Robert Miles piano in 16 bars, make it darker Phrygian, drop the drums"
    )
    assert intent.is_recognized is True
    assert intent.target_asset_id == "robert_miles_children"
    assert intent.target_scale_mode == ScaleMode.PHRYGIAN
    assert intent.strip_drums is True
    assert intent.bridge_bars == 16


def test_voice_whim_parser_resolves_sufi_vocal_and_dorian_brighter_cues() -> None:
    """REQ-07: Parser maps 'sufi vocal' to sufi_vocal and 'brighter' to ScaleMode.DORIAN."""
    intent = parse_voice_whim("tease the sufi vocal and shift brighter into Dorian")
    assert intent.is_recognized is True
    assert intent.target_asset_id == "sufi_vocal"
    assert intent.target_scale_mode == ScaleMode.DORIAN
    assert intent.bridge_bars == 4


def test_voice_whim_parser_returns_unrecognized_noop_for_unrelated_speech() -> None:
    """REQ-07: Unrecognized utterances return is_recognized=False with zero state directives."""
    intent = parse_voice_whim("testing one two three check microphone levels")
    assert intent.is_recognized is False
    assert intent.target_asset_id is None
    assert intent.target_scale_mode is None
    assert intent.strip_drums is False


def test_hum_to_midi_quantizer_subtracts_80ms_lag_and_snaps_to_sixteenth_grid() -> None:
    """REQ-07: Subtracts 80ms human reaction lag from onset and snaps to nearest 1/16th grid."""
    clock = SampleClock(sample_rate=48000, bpm=120.0)
    # At 120 BPM, 1 beat = 500ms, 1/16th = 125ms (6000 samples).
    # Onset at 205ms - 80ms lag = 125ms -> exact 1st 16th note (sample 6000).
    event = HumContourEvent(onset_ms=205.0, f0_hz=349.23, velocity=110)
    quantized = quantize_hum_contour(
        events=(event,), clock=clock, root_key="F", mode=ScaleMode.NATURAL_MINOR
    )
    assert len(quantized) == 1
    assert quantized[0].compensated_onset_ms == pytest.approx(205.0 - HUMAN_REACTION_LAG_MS)
    assert quantized[0].sixteenth_index == 1
    assert quantized[0].quantized_sample == 6000


def test_hum_to_midi_quantizer_clamps_early_onset_lag_compensation_at_zero_ms() -> None:
    """REQ-07: Onsets < 80ms clamp compensated_onset_ms at 0.0ms (sample 0)."""
    clock = SampleClock(sample_rate=48000, bpm=124.0)
    event = HumContourEvent(onset_ms=35.0, f0_hz=349.23, velocity=95)
    quantized = quantize_hum_contour(events=(event,), clock=clock)
    assert quantized[0].compensated_onset_ms == 0.0
    assert quantized[0].quantized_sample == 0
    assert quantized[0].sixteenth_index == 0


def test_hum_to_midi_quantizer_snaps_out_of_tune_hum_to_f_phrygian_pentatonic_subset() -> None:
    """REQ-07: Out-of-tune 380 Hz hum snaps to Gb4 (MIDI 66) in F Phrygian pentatonic."""
    clock = SampleClock(sample_rate=48000, bpm=124.0)
    # 380 Hz is ~MIDI 66.46; G4 (67) is outside F Phrygian, so it snaps to Gb4 (66).
    event = HumContourEvent(onset_ms=80.0, f0_hz=380.0, velocity=100)
    quantized = quantize_hum_contour(
        events=(event,), clock=clock, root_key="F", mode=ScaleMode.PHRYGIAN
    )
    assert quantized[0].midi_note == 66


def test_four_stage_lookahead_bridge_planner_generates_16_bar_four_stage_progression() -> None:
    """REQ-09: 16-bar bridge plan divides evenly into 4 stages of 4 bars each with tool calls."""
    asset = make_test_asset()
    plan = build_lookahead_bridge_plan(
        target_asset=asset,
        source_mode=ScaleMode.NATURAL_MINOR,
        target_mode=ScaleMode.PHRYGIAN,
        total_bars=16,
    )
    assert plan.total_bars == 16
    assert len(plan.stages) == 4
    assert plan.stages[0].stage_name == BridgeStageName.STAGE_1_THIN_MIDRANGE
    assert (plan.stages[0].start_bar, plan.stages[0].end_bar) == (1, 4)
    assert plan.stages[1].stage_name == BridgeStageName.STAGE_2_PIVOT_CHORDS
    assert (plan.stages[1].start_bar, plan.stages[1].end_bar) == (5, 8)
    assert plan.stages[2].stage_name == BridgeStageName.STAGE_3_TEASE_AND_RISER
    assert plan.stages[2].stem_gain == pytest.approx(0.35)
    assert plan.stages[3].stage_name == BridgeStageName.STAGE_4_SUBBASS_RESOLVE_AND_DROP
    assert plan.stages[3].stem_gain == pytest.approx(1.0)


def test_bridge_planner_stages_emit_expected_structured_tool_calls() -> None:
    """REQ-09: Bridge stages emit write_midi_clip, warp_stem, and modulate_filter_cutoff calls."""
    asset = make_test_asset(asset_id="robert_miles_children", mode=ScaleMode.NATURAL_MINOR)
    plan = build_lookahead_bridge_plan(
        target_asset=asset,
        source_mode=ScaleMode.NATURAL_MINOR,
        target_mode=ScaleMode.PHRYGIAN,
        total_bars=4,
    )
    stage3_tools = [tc.tool_name for tc in plan.stages[2].tool_calls]
    stage4_tools = [tc.tool_name for tc in plan.stages[3].tool_calls]
    assert "warp_stem" in stage3_tools
    assert "modulate_filter_cutoff" in stage3_tools
    assert "warp_stem" in stage4_tools
    assert "write_midi_clip" in stage4_tools


def test_spectral_critic_shifts_colliding_midrange_synth_notes_down_one_octave() -> None:
    """REQ-09: Vocal stem in 1-3 kHz shifts MIDI notes in [60, 84] down 12 semitones."""
    notes = (
        ScheduledMidiNote(note=65, velocity=100, trigger_sample=0),
        ScheduledMidiNote(note=41, velocity=100, trigger_sample=1000),
    )
    result = evaluate_spectral_collision(
        stem_freq_band_hz=(1200.0, 2800.0), scheduled_notes=notes
    )
    assert result.spectral_carve_applied is True
    assert result.carved_note_count == 1
    assert result.adjusted_notes[0].note == 53
    assert result.adjusted_notes[0].velocity == 85
    assert result.adjusted_notes[1].note == 41
    assert result.adjusted_notes[1].velocity == 100


def test_spectral_critic_leaves_midi_notes_untouched_when_stem_has_no_1_to_3khz_overlap() -> None:
    """REQ-09: Sub-bass stem in (40, 250) Hz does not trigger spectral carving on midrange MIDI."""
    notes = (ScheduledMidiNote(note=68, velocity=95, trigger_sample=0),)
    result = evaluate_spectral_collision(
        stem_freq_band_hz=(40.0, 250.0), scheduled_notes=notes
    )
    assert result.spectral_carve_applied is False
    assert result.carved_note_count == 0
    assert result.adjusted_notes[0].note == 68
    assert result.adjusted_notes[0].velocity == 95
