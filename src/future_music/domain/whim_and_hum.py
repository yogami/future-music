"""Slice 4 (REQ-07): Input Gating, Voice Whim Intent Parser, and -80ms Hum Quantizer."""

import math
from dataclasses import dataclass
from enum import StrEnum

from future_music.domain.clock_and_midi import (
    MAX_MIDI_VAL,
    MIN_MIDI_VAL,
    MS_PER_SECOND,
    OCTAVE_SEMITONES,
    ROOT_PITCH_CLASSES,
    SampleClock,
)
from future_music.domain.empath_memory import ScaleMode

HUMAN_REACTION_LAG_MS: float = 80.0
A4_FREQ_HZ: float = 440.0
A4_MIDI_NOTE: float = 69.0
SIXTEENTH_PER_BEAT: float = 4.0
DEFAULT_BRIDGE_BARS: int = 4
EXTENDED_BRIDGE_BARS: int = 16
DEFAULT_HUM_VELOCITY: int = 100

PENTATONIC_SUBSETS: dict[ScaleMode, tuple[int, ...]] = {
    ScaleMode.NATURAL_MINOR: (0, 3, 5, 7, 10),
    ScaleMode.PHRYGIAN: (0, 1, 3, 7, 8),
    ScaleMode.DORIAN: (0, 3, 5, 7, 9),
}

ASSET_KEYWORD_MAP: tuple[tuple[str, str], ...] = (
    ("robert miles", "robert_miles_children"),
    ("children", "robert_miles_children"),
    ("piano", "robert_miles_children"),
    ("sufi", "sufi_vocal"),
    ("qawwali", "sufi_vocal"),
    ("vocal", "sufi_vocal"),
    ("synth", "berlin_acid_synth"),
    ("acid", "berlin_acid_synth"),
)

SCALE_KEYWORD_MAP: tuple[tuple[str, ScaleMode], ...] = (
    ("phrygian", ScaleMode.PHRYGIAN),
    ("darker", ScaleMode.PHRYGIAN),
    ("hijaz", ScaleMode.PHRYGIAN),
    ("dorian", ScaleMode.DORIAN),
    ("brighter", ScaleMode.DORIAN),
    ("natural minor", ScaleMode.NATURAL_MINOR),
    ("aeolian", ScaleMode.NATURAL_MINOR),
)

DRUM_STRIP_KEYWORDS: tuple[str, ...] = (
    "drop the drums",
    "strip the drums",
    "strip the hi-hats",
)


class InputGateMode(StrEnum):
    """Push-to-input microphone gate states (REQ-07)."""

    IDLE = "IDLE"
    HOLD_V = "HOLD_V"
    HOLD_H = "HOLD_H"


class InputGateInactiveError(RuntimeError):
    """Raised when microphone input is submitted while the gate is IDLE."""


class InputGateConflictError(RuntimeError):
    """Raised when attempting simultaneous or mismatched HOLD_V and HOLD_H gates."""


@dataclass
class InputGateController:
    """Enforces mutually exclusive push-to-input gating for Hold V and Hold H (REQ-07)."""

    active_mode: InputGateMode = InputGateMode.IDLE

    def open_voice_gate(self) -> InputGateMode:
        """Open HOLD_V voice whim gate or raise InputGateConflictError if HOLD_H is open."""
        if self.active_mode == InputGateMode.HOLD_H:
            raise InputGateConflictError("Cannot open HOLD_V while HOLD_H is active")
        self.active_mode = InputGateMode.HOLD_V
        return self.active_mode

    def open_hum_gate(self) -> InputGateMode:
        """Open HOLD_H hum melody gate or raise InputGateConflictError if HOLD_V is open."""
        if self.active_mode == InputGateMode.HOLD_V:
            raise InputGateConflictError("Cannot open HOLD_H while HOLD_V is active")
        self.active_mode = InputGateMode.HOLD_H
        return self.active_mode

    def close_gate(self) -> InputGateMode:
        """Return microphone gate to IDLE."""
        self.active_mode = InputGateMode.IDLE
        return self.active_mode

    def require_voice_gate(self) -> None:
        """Verify HOLD_V is currently active before processing voice transcripts."""
        if self.active_mode == InputGateMode.IDLE:
            raise InputGateInactiveError("Voice whim requires HOLD_V gate")
        if self.active_mode == InputGateMode.HOLD_H:
            raise InputGateConflictError("Voice whim blocked while HOLD_H is active")

    def require_hum_gate(self) -> None:
        """Verify HOLD_H is currently active before processing hum pitch contours."""
        if self.active_mode == InputGateMode.IDLE:
            raise InputGateInactiveError("Hum contour requires HOLD_H gate")
        if self.active_mode == InputGateMode.HOLD_V:
            raise InputGateConflictError("Hum contour blocked while HOLD_V is active")


@dataclass(frozen=True)
class VoiceWhimIntent:
    """Structured intent parsed deterministically from a spoken Empath whim (REQ-07)."""

    raw_transcript: str
    is_recognized: bool
    target_asset_id: str | None = None
    target_scale_mode: ScaleMode | None = None
    strip_drums: bool = False
    bridge_bars: int = DEFAULT_BRIDGE_BARS


def _match_asset_keyword(lowered_text: str) -> str | None:
    """Return the first matching asset_id from ASSET_KEYWORD_MAP or None."""
    for phrase, asset_id in ASSET_KEYWORD_MAP:
        if phrase in lowered_text:
            return asset_id
    return None


def _match_scale_keyword(lowered_text: str) -> ScaleMode | None:
    """Return the first matching ScaleMode from SCALE_KEYWORD_MAP or None."""
    for phrase, mode in SCALE_KEYWORD_MAP:
        if phrase in lowered_text:
            return mode
    return None


def _has_any_phrase(lowered_text: str, phrases: tuple[str, ...]) -> bool:
    """Return True if any phrase in phrases occurs in lowered_text."""
    return any(phrase in lowered_text for phrase in phrases)


def parse_voice_whim(transcript: str) -> VoiceWhimIntent:
    """Parse spoken Empath cue deterministically into VoiceWhimIntent (REQ-07)."""
    lowered = transcript.lower()
    asset_id = _match_asset_keyword(lowered)
    scale_mode = _match_scale_keyword(lowered)
    strip_drums = _has_any_phrase(lowered, DRUM_STRIP_KEYWORDS)
    bars = EXTENDED_BRIDGE_BARS if "16 bars" in lowered else DEFAULT_BRIDGE_BARS
    recognized = any((asset_id is not None, scale_mode is not None, strip_drums))
    return VoiceWhimIntent(
        raw_transcript=transcript,
        is_recognized=recognized,
        target_asset_id=asset_id,
        target_scale_mode=scale_mode,
        strip_drums=strip_drums,
        bridge_bars=bars,
    )


@dataclass(frozen=True)
class HumContourEvent:
    """Detected vocal hum onset and fundamental frequency f0 in Hz."""

    onset_ms: float
    f0_hz: float
    velocity: int = DEFAULT_HUM_VELOCITY


@dataclass(frozen=True)
class QuantizedMidiNote:
    """Lag-compensated, 1/16th-grid-snapped, pentatonic-quantized MIDI note (REQ-07)."""

    midi_note: int
    velocity: int
    compensated_onset_ms: float
    quantized_sample: int
    sixteenth_index: int


def freq_to_raw_midi(f0_hz: float) -> float:
    """Convert fundamental frequency in Hz to fractional MIDI pitch."""
    return A4_MIDI_NOTE + float(OCTAVE_SEMITONES) * math.log2(f0_hz / A4_FREQ_HZ)


def _candidate_pentatonic_notes(
    center_note: int, root_pc: int, allowed_pcs: tuple[int, ...]
) -> list[int]:
    """Collect candidate MIDI notes within +/-1 octave belonging to allowed_pcs."""
    candidates: list[int] = []
    for offset in range(-OCTAVE_SEMITONES, OCTAVE_SEMITONES + 1):
        note = center_note + offset
        if (note - root_pc) % OCTAVE_SEMITONES in allowed_pcs:
            candidates.append(note)
    return candidates


def snap_pitch_to_scale_pentatonic(
    raw_midi: float, root_key: str, mode: ScaleMode
) -> int:
    """Snap fractional MIDI pitch to nearest chord-tone/pentatonic degree of mode."""
    root_pc = ROOT_PITCH_CLASSES.get(root_key, 5)
    allowed_pcs = PENTATONIC_SUBSETS[mode]
    candidates = _candidate_pentatonic_notes(round(raw_midi), root_pc, allowed_pcs)
    best = min(candidates, key=lambda n: (abs(n - raw_midi), n))
    return max(MIN_MIDI_VAL, min(MAX_MIDI_VAL, best))


def _quantize_single_hum(
    event: HumContourEvent, clock: SampleClock, root_key: str, mode: ScaleMode, lag_ms: float
) -> QuantizedMidiNote:
    """Apply -80ms lag compensation, 1/16th grid snap, and pentatonic pitch snap."""
    compensated_ms = max(0.0, event.onset_ms - lag_ms)
    raw_sample = (compensated_ms * clock.sample_rate) / MS_PER_SECOND
    samples_per_sixteenth = clock.samples_per_beat / SIXTEENTH_PER_BEAT
    sixteenth_idx = round(raw_sample / samples_per_sixteenth)
    quantized_sample = round(sixteenth_idx * samples_per_sixteenth)
    snapped = snap_pitch_to_scale_pentatonic(freq_to_raw_midi(event.f0_hz), root_key, mode)
    return QuantizedMidiNote(
        midi_note=snapped,
        velocity=event.velocity,
        compensated_onset_ms=compensated_ms,
        quantized_sample=quantized_sample,
        sixteenth_index=sixteenth_idx,
    )


def quantize_hum_contour(
    events: tuple[HumContourEvent, ...],
    clock: SampleClock,
    root_key: str = "F",
    mode: ScaleMode = ScaleMode.NATURAL_MINOR,
    lag_ms: float = HUMAN_REACTION_LAG_MS,
) -> tuple[QuantizedMidiNote, ...]:
    """Quantize a sequence of hummed pitch events into scale-locked MIDI notes (REQ-07)."""
    return tuple(
        _quantize_single_hum(ev, clock, root_key, mode, lag_ms) for ev in events
    )
