"""Unit tests for Slice 6 (REQ-12): 124 BPM Stem Synthesis, WAV Encoding, & Bridge Transport."""

import struct
import wave
from io import BytesIO

import numpy as np
from future_music.domain.stem_synthesis import (
    compute_bridge_stage_for_bar,
    encode_stereo_pcm16_wav,
    synthesize_default_124bpm_stems,
)

from future_music.domain.dsp_tension_drop import apply_biquad_hpf, compute_biquad_hpf_coeffs
from future_music.domain.empath_memory import MemoryAsset, ScaleMode
from future_music.domain.ghost_and_bridge import BridgeStageName, build_lookahead_bridge_plan


def test_synthesize_default_124bpm_stems_generates_exact_1_bar_sample_length() -> None:
    """REQ-12: At 48000 Hz and 124.0 BPM, 1 bar (4 beats) is exactly 92,903 samples."""
    stems = synthesize_default_124bpm_stems(sample_rate=48000, bpm=124.0)
    expected_keys = {
        "drums",
        "bass",
        "robert_miles_children",
        "sufi_vocal",
        "berlin_acid_synth",
    }
    assert set(stems.keys()) == expected_keys
    for buf in stems.values():
        assert buf.shape == (92903,)
        assert buf.dtype == np.float64


def test_synthesized_drums_stem_contains_both_low_kick_and_high_snare_energy() -> None:
    """REQ-12: Drums stem has strong sub-100Hz kick energy plus >1kHz snare/hat sizzle."""
    stems = synthesize_default_124bpm_stems(sample_rate=48000, bpm=124.0)
    drums = stems["drums"]
    full_rms = float(np.sqrt(np.mean(np.square(drums))))
    hpf_coeffs = compute_biquad_hpf_coeffs(cutoff_hz=350.0, sample_rate=48000)
    high_only = apply_biquad_hpf(drums, hpf_coeffs)
    high_rms = float(np.sqrt(np.mean(np.square(high_only))))
    assert full_rms > 0.20
    assert 0.04 < high_rms < full_rms * 0.75


def test_synthesized_bass_stem_is_dominated_by_subbass_energy_below_350hz() -> None:
    """REQ-12: Bass stem is concentrated in F1/C2 sub-bass (<350Hz)."""
    stems = synthesize_default_124bpm_stems(sample_rate=48000, bpm=124.0)
    bass = stems["bass"]
    full_rms = float(np.sqrt(np.mean(np.square(bass))))
    hpf_coeffs = compute_biquad_hpf_coeffs(cutoff_hz=350.0, sample_rate=48000)
    filtered_rms = float(np.sqrt(np.mean(np.square(apply_biquad_hpf(bass, hpf_coeffs)))))
    assert full_rms > 0.25
    assert filtered_rms < full_rms * 0.15


def test_synthesized_melodic_stems_have_distinct_harmonic_signatures_and_nonzero_rms() -> None:
    """REQ-12: Robert Miles, Sufi Vocal, and Acid Synth stems are audible and distinct."""
    stems = synthesize_default_124bpm_stems(sample_rate=48000, bpm=124.0)
    rm = stems["robert_miles_children"]
    sufi = stems["sufi_vocal"]
    acid = stems["berlin_acid_synth"]
    assert float(np.sqrt(np.mean(np.square(rm)))) > 0.10
    assert float(np.sqrt(np.mean(np.square(sufi)))) > 0.10
    assert float(np.sqrt(np.mean(np.square(acid)))) > 0.10
    assert not np.allclose(rm, sufi)


def test_encode_stereo_pcm16_wav_produces_valid_48khz_2ch_riff_wave_bytes() -> None:
    """REQ-12: encode_stereo_pcm16_wav encodes (N, 2) float64 array into 48kHz stereo WAV."""
    stereo = np.column_stack((np.full(480, 0.25), np.full(480, -0.50)))
    wav_bytes = encode_stereo_pcm16_wav(stereo, sample_rate=48000)
    with wave.open(BytesIO(wav_bytes), "rb") as wf:
        assert wf.getnchannels() == 2
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 48000
        assert wf.getnframes() == 480


def test_encode_stereo_pcm16_wav_clamps_out_of_range_samples_without_integer_wrap() -> None:
    """REQ-12: Samples exceeding [-1.0, 1.0] clamp cleanly to [-32767, 32767]."""
    stereo = np.array([[2.5, -3.0]], dtype=np.float64)
    wav_bytes = encode_stereo_pcm16_wav(stereo, sample_rate=48000)
    with wave.open(BytesIO(wav_bytes), "rb") as wf:
        raw_frames = wf.readframes(1)
    left_val, right_val = struct.unpack("<hh", raw_frames)
    assert left_val == 32767
    assert right_val == -32767


def test_compute_active_bridge_stage_for_bar_progresses_across_4_stages() -> None:
    """REQ-12: compute_bridge_stage_for_bar maps 1-indexed bridge bar to stages 1..4."""
    asset = MemoryAsset(
        "sufi_vocal",
        "Sufi Vocal",
        (0.1, 0.7, 0.6, 0.6),
        124.0,
        "F",
        ScaleMode.PHRYGIAN,
        "sufi_vocal.wav",
        (65, 66, 68, 72),
        True,
        (1000.0, 3000.0),
    )
    plan = build_lookahead_bridge_plan(asset, ScaleMode.NATURAL_MINOR, ScaleMode.PHRYGIAN, 4)
    assert compute_bridge_stage_for_bar(plan, 1) == BridgeStageName.STAGE_1_THIN_MIDRANGE
    assert compute_bridge_stage_for_bar(plan, 2) == BridgeStageName.STAGE_2_PIVOT_CHORDS
    assert compute_bridge_stage_for_bar(plan, 3) == BridgeStageName.STAGE_3_TEASE_AND_RISER
    assert compute_bridge_stage_for_bar(plan, 4) == BridgeStageName.STAGE_4_SUBBASS_RESOLVE_AND_DROP


def test_compute_active_bridge_stage_returns_none_when_bridge_completes() -> None:
    """REQ-12: Once bar_in_bridge exceeds plan.total_bars, bridge stage returns None."""
    asset = MemoryAsset(
        "sufi_vocal",
        "Sufi Vocal",
        (0.1, 0.7, 0.6, 0.6),
        124.0,
        "F",
        ScaleMode.PHRYGIAN,
        "sufi_vocal.wav",
        (65, 66, 68, 72),
        True,
        (1000.0, 3000.0),
    )
    plan = build_lookahead_bridge_plan(asset, ScaleMode.NATURAL_MINOR, ScaleMode.PHRYGIAN, 4)
    assert compute_bridge_stage_for_bar(plan, 5) is None
