"""Slice 6 & 7 (REQ-12, REQ-14): Clean Non-Clipping 124 BPM Stem Synthesis & WAV Encoding."""

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
EDGE_RAMP_SEC: float = 0.006

KICK_FREQ_HZ: float = 55.0
SNARE_FREQ_HZ: float = 2400.0
BASS_F1_HZ: float = 43.65
BASS_C2_HZ: float = 65.41
F3_HZ: float = 174.61
EB4_HZ: float = 311.13
F4_HZ: float = 349.23
GB4_HZ: float = 369.99
G4_HZ: float = 392.00
AB4_HZ: float = 415.30
BB4_HZ: float = 466.16
C5_HZ: float = 523.25
F5_HZ: float = 698.46
F6_FORMANT_HZ: float = 1746.14

DREAM_PIANO_PHRASE_HZ: tuple[float, ...] = (
    F4_HZ, AB4_HZ, C5_HZ, BB4_HZ, AB4_HZ, G4_HZ, F4_HZ, EB4_HZ
)
SUFI_VOCAL_PHRASE_HZ: tuple[float, ...] = (
    F4_HZ, GB4_HZ, AB4_HZ, C5_HZ, C5_HZ, AB4_HZ, GB4_HZ, F4_HZ
)
ACID_LEAD_PHRASE_HZ: tuple[float, ...] = (
    F3_HZ, F3_HZ, F4_HZ, GB4_HZ, AB4_HZ, GB4_HZ, F4_HZ, F3_HZ
)


def _bar_edge_envelope(t_sec: np.ndarray, total_sec: float) -> np.ndarray:
    """Apply smooth 6ms fade-in/out so looping bar boundaries stay click-free."""
    fade_in = np.minimum(t_sec / EDGE_RAMP_SEC, 1.0)
    fade_out = np.minimum(np.maximum(total_sec - t_sec, 0.0) / EDGE_RAMP_SEC, 1.0)
    return fade_in * fade_out


def _synth_drums_and_bass(
    t_sec: np.ndarray, beat_dur_sec: float, edge_env: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize smooth-attack 55Hz kick + 2.4kHz hat drums stem and F1/C2 bass stem."""
    two_pi = 2.0 * math.pi
    beat_t = np.mod(t_sec, beat_dur_sec)
    kick_env = np.minimum(beat_t / 0.005, 1.0) * (0.60 + 0.40 * np.exp(-beat_t / 0.22))
    off_t = np.mod(t_sec + beat_dur_sec * 0.5, beat_dur_sec)
    hat_env = np.minimum(off_t / 0.004, 1.0) * (0.22 + 0.78 * np.exp(-off_t / 0.16))
    drums_raw = 0.37 * kick_env * np.sin(two_pi * KICK_FREQ_HZ * t_sec) + (
        0.135 * hat_env * np.sin(two_pi * SNARE_FREQ_HZ * t_sec)
    )
    bass_raw = 0.35 * np.sin(two_pi * BASS_F1_HZ * t_sec) + 0.12 * np.sin(
        two_pi * BASS_C2_HZ * t_sec
    )
    return (drums_raw * edge_env, bass_raw * edge_env)


def _synth_stepped_phrase(
    t_sec: np.ndarray, total_sec: float, phrase_hz: tuple[float, ...],
    overtone_ratio: float, overtone_mix: float,
) -> np.ndarray:
    """Synthesize a click-free 8-step musical motif across one 4/4 bar."""
    step_dur = total_sec / float(len(phrase_hz))
    step_idx = np.minimum((t_sec // step_dur).astype(int), len(phrase_hz) - 1)
    step_t = np.mod(t_sec, step_dur)
    freqs = np.asarray(phrase_hz, dtype=np.float64)[step_idx]
    attack = np.minimum(step_t / 0.010, 1.0)
    release = np.minimum(np.maximum(step_dur - step_t, 0.0) / 0.010, 1.0)
    note_env = attack * release * (0.72 + 0.28 * np.exp(-step_t / 0.16))
    phase = 2.0 * math.pi * freqs * t_sec
    wave = (1.0 - overtone_mix) * np.sin(phase) + overtone_mix * np.sin(overtone_ratio * phase)
    return 0.24 * note_env * wave


def _synth_melodic_stems(t_sec: np.ndarray, total_sec: float) -> dict[str, np.ndarray]:
    """Synthesize F Natural Minor piano, F Phrygian Sufi vocal, and 303 acid lead stems."""
    return {
        "robert_miles_children": _synth_stepped_phrase(
            t_sec, total_sec, DREAM_PIANO_PHRASE_HZ, overtone_ratio=2.0, overtone_mix=0.18
        ),
        "sufi_vocal": _synth_stepped_phrase(
            t_sec, total_sec, SUFI_VOCAL_PHRASE_HZ, overtone_ratio=3.0, overtone_mix=0.22
        ),
        "berlin_acid_synth": _synth_stepped_phrase(
            t_sec, total_sec, ACID_LEAD_PHRASE_HZ, overtone_ratio=2.0, overtone_mix=0.25
        ),
    }


def synthesize_default_124bpm_stems(
    sample_rate: int = DEFAULT_SAMPLE_RATE, bpm: float = DEFAULT_BPM
) -> dict[str, np.ndarray]:
    """Generate deterministic 1-bar click-free stems for drums, bass, and 3 motifs."""
    clock = SampleClock(sample_rate=sample_rate, bpm=bpm)
    bar_samples = clock.beat_sample_index(BEATS_PER_BAR)
    t_sec = np.arange(bar_samples, dtype=np.float64) / float(sample_rate)
    total_sec = float(bar_samples) / float(sample_rate)
    edge_env = _bar_edge_envelope(t_sec, total_sec)
    drums, bass = _synth_drums_and_bass(t_sec, total_sec / float(BEATS_PER_BAR), edge_env)
    stems = {"drums": drums, "bass": bass}
    stems.update(_synth_melodic_stems(t_sec, total_sec))
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
