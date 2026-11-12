"""Unit tests for Slice 7 (REQ-14): Click-Free Envelope Attack, Bar Loop Continuity, & Melody."""

import numpy as np

from future_music.domain.stem_synthesis import synthesize_default_124bpm_stems


def test_req14_synthesized_stems_start_near_zero_without_sample_zero_pop() -> None:
    """REQ-14: All synthesized 124 BPM stems ramp in smoothly from near zero at sample 0."""
    stems = synthesize_default_124bpm_stems(sample_rate=48000, bpm=124.0)
    for stem_name, buf in stems.items():
        assert abs(float(buf[0])) < 0.05, f"{stem_name} starts with click at sample 0: {buf[0]}"
        assert float(np.max(np.abs(buf))) < 0.65, f"{stem_name} individual peak too high"


def test_req14_synthesized_stems_seamlessly_loop_without_bar_boundary_click() -> None:
    """REQ-14: End-to-start transition across the 1-bar loop boundary is click-free."""
    stems = synthesize_default_124bpm_stems(sample_rate=48000, bpm=124.0)
    for stem_name, buf in stems.items():
        jump = abs(float(buf[-1]) - float(buf[0]))
        assert jump < 0.05, f"{stem_name} has bar-loop seam discontinuity: {jump}"


def _dominant_hz(chunk: np.ndarray, sample_rate: int = 48000) -> float:
    """Return dominant frequency in Hz of a mono signal window."""
    spectrum = np.abs(np.fft.rfft(chunk))
    peak_bin = int(np.argmax(spectrum))
    return float(peak_bin * sample_rate / len(chunk))


def test_req14_robert_miles_stem_varies_musically_across_eighth_note_steps() -> None:
    """REQ-14: Melodic dream piano stem articulates F4 -> C5 across eighth-note steps."""
    stems = synthesize_default_124bpm_stems(sample_rate=48000, bpm=124.0)
    piano = stems["robert_miles_children"]
    eighth_len = len(piano) // 8
    hz_step_0 = _dominant_hz(piano[:eighth_len])
    hz_step_2 = _dominant_hz(piano[2 * eighth_len : 3 * eighth_len])
    assert abs(hz_step_0 - 349.23) < 10.0
    assert abs(hz_step_2 - 523.25) < 10.0
    assert hz_step_2 > hz_step_0 + 120.0
