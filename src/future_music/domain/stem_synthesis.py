"""Slice 6 (REQ-12): Deterministic 124 BPM Stem Synthesis, WAV Encoding, & Bridge Bar Step."""

import math
import wave
from io import BytesIO

import numpy as np

from future_music.domain.clock_and_midi import (
    BEATS_PER_BAR,
    DEFAULT_BPM,
    DEFAULT_SAMPLE_RATE,
    SampleClock,
)
from future_music.domain.ghost_and_bridge import (
    BridgeStageName,
    BridgeStageSpec,
    LookaheadBridgePlan,
)

PCM16_MAX: float = 32767.0
STEREO_CHANNELS: int = 2
PCM16_SAMPWIDTH: int = 2

KICK_FREQ_HZ: float = 55.0
SNARE_FREQ_HZ: float = 2400.0
BASS_F1_HZ: float = 43.65
BASS_C2_HZ: float = 65.41
F4_HZ: float = 349.23
GB4_HZ: float = 369.99
AB4_HZ: float = 415.30
C5_HZ: float = 523.25
F3_HZ: float = 174.61
F6_FORMANT_HZ: float = 1746.14


def _synth_drums_and_bass(t_sec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize 1-bar 55Hz kick + 2.4kHz snare drums stem and F1/C2 sub-bass stem."""
    two_pi = 2.0 * math.pi
    drums = 0.45 * np.sin(two_pi * KICK_FREQ_HZ * t_sec) + 0.12 * np.sin(
        two_pi * SNARE_FREQ_HZ * t_sec
    )
    bass = 0.50 * np.sin(two_pi * BASS_F1_HZ * t_sec) + 0.25 * np.sin(
        two_pi * BASS_C2_HZ * t_sec
    )
    drums[0] = 1.0
    bass[0] = 1.0
    return (drums, bass)


def _harmonic_triad(t_sec: np.ndarray, f1: float, f2: float, f3: float) -> np.ndarray:
    """Synthesize a 3-partial harmonic waveform from frequencies (f1, f2, f3)."""
    two_pi = 2.0 * math.pi
    return (
        0.25 * np.sin(two_pi * f1 * t_sec)
        + 0.20 * np.sin(two_pi * f2 * t_sec)
        + 0.15 * np.sin(two_pi * f3 * t_sec)
    )


def _synth_melodic_stems(t_sec: np.ndarray) -> dict[str, np.ndarray]:
    """Synthesize F Natural Minor piano, F Phrygian Sufi vocal, and 303 acid lead stems."""
    return {
        "robert_miles_children": _harmonic_triad(t_sec, F4_HZ, AB4_HZ, C5_HZ),
        "sufi_vocal": _harmonic_triad(t_sec, F4_HZ, GB4_HZ, F6_FORMANT_HZ),
        "berlin_acid_synth": _harmonic_triad(t_sec, F3_HZ, GB4_HZ, F4_HZ * 2.0),
    }


def synthesize_default_124bpm_stems(
    sample_rate: int = DEFAULT_SAMPLE_RATE, bpm: float = DEFAULT_BPM
) -> dict[str, np.ndarray]:
    """Generate deterministic 1-bar sample-locked stems for drums, bass, and 3 motifs."""
    clock = SampleClock(sample_rate=sample_rate, bpm=bpm)
    bar_samples = clock.beat_sample_index(BEATS_PER_BAR)
    t_sec = np.arange(bar_samples, dtype=np.float64) / float(sample_rate)
    drums, bass = _synth_drums_and_bass(t_sec)
    stems = {"drums": drums, "bass": bass}
    stems.update(_synth_melodic_stems(t_sec))
    return stems


def encode_stereo_pcm16_wav(
    stereo_frame: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE
) -> bytes:
    """Encode a (num_samples, 2) float64 stereo buffer into 16-bit RIFF WAVE bytes."""
    clamped = np.clip(np.asarray(stereo_frame, dtype=np.float64), -1.0, 1.0)
    pcm16 = np.round(clamped * PCM16_MAX).astype("<i2")
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(STEREO_CHANNELS)
        wf.setsampwidth(PCM16_SAMPWIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


def _find_matching_stage(
    stages: tuple[BridgeStageSpec, ...], bar_in_bridge: int
) -> BridgeStageName | None:
    """Return stage_name whose [start_bar, end_bar] contains bar_in_bridge."""
    for stage in stages:
        if stage.start_bar <= bar_in_bridge <= stage.end_bar:
            return stage.stage_name
    return None


def compute_bridge_stage_for_bar(
    plan: LookaheadBridgePlan | None, bar_in_bridge: int
) -> BridgeStageName | None:
    """Return the active BridgeStageName for 1-indexed bar_in_bridge, or None if finished."""
    if plan is None:
        return None
    return _find_matching_stage(plan.stages, bar_in_bridge)
