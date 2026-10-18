"""Slice 2 (REQ-02, REQ-03, REQ-04): SampleClock, GarageBand MIDI Expression, and Modal Lock."""

from dataclasses import dataclass

from future_music.domain.empath_memory import MAX_UNIT, MIN_UNIT, MemoryAsset, ScaleMode

DEFAULT_SAMPLE_RATE: int = 48000
DEFAULT_BPM: float = 124.0
DEFAULT_MIDI_ADVANCE_MS: float = 18.0
SECONDS_PER_MINUTE: float = 60.0
MS_PER_SECOND: float = 1000.0
BEATS_PER_BAR: int = 4

DEFAULT_CC1_MODULATION: int = 32
CC1_STEP_DELTA: int = 16
CC1_MOD_WHEEL: int = 1
CC64_SUSTAIN_PEDAL: int = 64
ALLOWED_GARAGEBAND_CCS: frozenset[int] = frozenset({CC1_MOD_WHEEL, CC64_SUSTAIN_PEDAL})

MIN_MIDI_VAL: int = 0
MAX_MIDI_VAL: int = 127
MIN_VELOCITY: int = 1
MIN_PITCH_BEND: int = -8192
MAX_PITCH_BEND: int = 8191

DEFAULT_NOTE_DENSITY: float = 0.5
NOTE_DENSITY_STEP: float = 0.25
OCTAVE_SEMITONES: int = 12
ROOT_PITCH_CLASSES: dict[str, int] = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}


class UnsupportedGarageBandMidiCcError(ValueError):
    """Raised when an unsupported MIDI CC number is sent to GarageBand."""


class ModalLockActiveError(RuntimeError):
    """Raised when attempting a conflicting scale shift while Modal Lock is active."""


def validate_garageband_cc(cc_number: int, value: int) -> None:
    """Validate that cc_number is natively supported by GarageBand and value is 0..127."""
    if cc_number not in ALLOWED_GARAGEBAND_CCS:
        allowed = sorted(ALLOWED_GARAGEBAND_CCS)
        raise UnsupportedGarageBandMidiCcError(
            f"GarageBand rejects CC {cc_number}; allowed: {allowed}"
        )
    if not (MIN_MIDI_VAL <= value <= MAX_MIDI_VAL):
        raise ValueError(f"CC value must be within [0, 127], got {value}")


def validate_midi_note_and_velocity(note: int, velocity: int) -> None:
    """Validate MIDI note (0..127) and Note-On velocity (1..127)."""
    if not (MIN_MIDI_VAL <= note <= MAX_MIDI_VAL):
        raise ValueError(f"note must be within [0, 127], got {note}")
    if not (MIN_VELOCITY <= velocity <= MAX_MIDI_VAL):
        raise ValueError(f"velocity must be within [1, 127], got {velocity}")


def validate_pitch_bend(pitch_bend: int) -> None:
    """Validate 14-bit signed MIDI pitch bend (-8192..8191)."""
    if not (MIN_PITCH_BEND <= pitch_bend <= MAX_PITCH_BEND):
        raise ValueError(f"pitch_bend must be within [-8192, 8191], got {pitch_bend}")


@dataclass
class SampleClock:
    """Sample-counted master beat clock with pre-fire MIDI advance offset (REQ-02)."""

    sample_rate: int = DEFAULT_SAMPLE_RATE
    bpm: float = DEFAULT_BPM
    midi_advance_ms: float = DEFAULT_MIDI_ADVANCE_MS
    current_sample: int = 0

    @property
    def samples_per_beat(self) -> float:
        """Exact fractional audio samples per quarter-note beat."""
        return (self.sample_rate * SECONDS_PER_MINUTE) / self.bpm

    @property
    def samples_per_bar(self) -> float:
        """Exact fractional audio samples per 4/4 bar."""
        return self.samples_per_beat * BEATS_PER_BAR

    @property
    def midi_advance_samples(self) -> int:
        """Number of audio samples to pre-fire IAC MIDI ahead of the acoustic beat."""
        return round((self.midi_advance_ms * self.sample_rate) / MS_PER_SECOND)

    def beat_sample_index(self, beat_number: int) -> int:
        """Return rounded audio sample index for a 0-indexed global beat_number."""
        return round(beat_number * self.samples_per_beat)

    def midi_trigger_sample(self, beat_number: int) -> int:
        """Return pre-fired sample index for MIDI note dispatch (clamped at 0)."""
        return max(0, self.beat_sample_index(beat_number) - self.midi_advance_samples)

    def advance_samples(self, frame_count: int) -> tuple[int, int]:
        """Advance sample counter by frame_count and return (start_sample, end_sample)."""
        start = self.current_sample
        self.current_sample += frame_count
        return (start, self.current_sample)

    def current_bar_and_beat(self, sample_pos: int) -> tuple[int, int]:
        """Return 1-indexed (bar_number, beat_in_bar) for sample_pos."""
        total_beats = int(sample_pos // self.samples_per_beat)
        bar = (total_beats // BEATS_PER_BAR) + 1
        beat_in_bar = (total_beats % BEATS_PER_BAR) + 1
        return (bar, beat_in_bar)


@dataclass(frozen=True)
class MidiControlEvent:
    """Validated GarageBand MIDI CC event."""

    cc_number: int
    value: int


@dataclass(frozen=True)
class ScheduledMidiNote:
    """MIDI note scheduled at a specific pre-fired hardware sample index."""

    note: int
    velocity: int
    trigger_sample: int


@dataclass
class GarageBandExpressionState:
    """Tracks GarageBand Alchemy CC 1 filter morph and MIDI note density (REQ-03)."""

    cc1_modulation: int = DEFAULT_CC1_MODULATION
    note_density: float = DEFAULT_NOTE_DENSITY

    def morph_filter_up(self) -> MidiControlEvent:
        """Increment CC 1 Modulation Wheel by +16 clamped to 127 ([UP] key)."""
        self.cc1_modulation = min(MAX_MIDI_VAL, self.cc1_modulation + CC1_STEP_DELTA)
        return MidiControlEvent(cc_number=CC1_MOD_WHEEL, value=self.cc1_modulation)

    def morph_filter_down(self) -> MidiControlEvent:
        """Decrement CC 1 Modulation Wheel by -16 clamped to 0 ([DOWN] key)."""
        self.cc1_modulation = max(MIN_MIDI_VAL, self.cc1_modulation - CC1_STEP_DELTA)
        return MidiControlEvent(cc_number=CC1_MOD_WHEEL, value=self.cc1_modulation)

    def adjust_note_density_left(self) -> float:
        """Decrease MIDI note density by 0.25 clamped at 0.0 ([LEFT] key)."""
        self.note_density = max(MIN_UNIT, round(self.note_density - NOTE_DENSITY_STEP, 4))
        return self.note_density

    def adjust_note_density_right(self) -> float:
        """Increase MIDI note density by 0.25 clamped at 1.0 ([RIGHT] key)."""
        self.note_density = min(MAX_UNIT, round(self.note_density + NOTE_DENSITY_STEP, 4))
        return self.note_density

    def filter_notes_by_density(self, notes: tuple[int, ...]) -> tuple[int, ...]:
        """Select active notes deterministically according to note_density."""
        if self.note_density <= MIN_UNIT or not notes:
            return ()
        keep_count = max(1, round(len(notes) * self.note_density))
        return notes[:keep_count]


@dataclass(frozen=True)
class ModalRenderState:
    """Snapshot of active scale mode, modal lock status, stem gain, and mutated MIDI motif."""

    active_scale_mode: ScaleMode
    modal_lock_active: bool
    audio_stem_gain: float
    mutated_midi_motif: tuple[int, ...]


def mutate_note_between_modes(
    note: int, root_pc: int, source_mode: ScaleMode, target_mode: ScaleMode
) -> int:
    """Transpose a single MIDI note from source_mode degree to matching target_mode degree."""
    rel_semitone = (note - root_pc) % OCTAVE_SEMITONES
    if rel_semitone not in source_mode.intervals:
        return note
    degree_index = source_mode.intervals.index(rel_semitone)
    target_semitone = target_mode.intervals[degree_index]
    return note + (target_semitone - rel_semitone)


@dataclass
class ModalScaleEngine:
    """Manages symbolic motif scale mutation and enforces Modal Lock on vocal stems (REQ-04)."""

    active_asset: MemoryAsset | None = None
    active_scale_mode: ScaleMode = ScaleMode.NATURAL_MINOR
    modal_lock_active: bool = False

    def activate_asset(self, asset: MemoryAsset) -> ModalRenderState:
        """Load a MemoryAsset onto the Main Mix and engage Modal Lock if required."""
        self.active_asset = asset
        self.active_scale_mode = asset.native_mode
        self.modal_lock_active = asset.requires_modal_lock
        return self.current_state()

    def _is_locked_conflict(self, target_mode: ScaleMode) -> bool:
        """Return True if Modal Lock blocks switching to target_mode."""
        if not self.modal_lock_active or self.active_asset is None:
            return False
        return target_mode != self.active_asset.native_mode

    def shift_scale(self, target_mode: ScaleMode) -> ModalRenderState:
        """Shift active scale mode or raise ModalLockActiveError if locked vocal clashes."""
        if self._is_locked_conflict(target_mode) and self.active_asset is not None:
            raise ModalLockActiveError(f"Modal Lock active for '{self.active_asset.asset_id}'")
        self.active_scale_mode = target_mode
        return self.current_state()

    def _compute_mutated_motif(self, asset: MemoryAsset) -> tuple[int, ...]:
        """Mutate asset's symbolic_motif into active_scale_mode."""
        root_pc = ROOT_PITCH_CLASSES.get(asset.root_key, 5)
        return tuple(
            mutate_note_between_modes(n, root_pc, asset.native_mode, self.active_scale_mode)
            for n in asset.symbolic_motif
        )

    def current_state(self) -> ModalRenderState:
        """Compute current ModalRenderState for the active asset and scale mode."""
        if self.active_asset is None:
            return ModalRenderState(self.active_scale_mode, False, MIN_UNIT, ())
        is_native = self.active_scale_mode == self.active_asset.native_mode
        stem_gain = MAX_UNIT if is_native else MIN_UNIT
        motif = self._compute_mutated_motif(self.active_asset)
        return ModalRenderState(
            active_scale_mode=self.active_scale_mode,
            modal_lock_active=self.modal_lock_active,
            audio_stem_gain=stem_gain,
            mutated_midi_motif=motif,
        )
