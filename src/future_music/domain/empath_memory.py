"""Slice 1 (REQ-01): Autobiographical Emotion-to-Sound Graph, Episodic Memory & Co-Pilot domain."""

import math
from dataclasses import dataclass, field
from enum import StrEnum

LEADING_PROJECTION_WEIGHT: float = 0.25
DEFAULT_EPISODIC_WINDOW_MINUTES: float = 60.0
RECENCY_PENALTY_WEIGHT: float = 0.35
STAGNATION_BARS_THRESHOLD: int = 16
STATIC_TENSION_EPSILON: float = 0.05
MIN_VALENCE: float = -1.0
MAX_UNIT: float = 1.0
MIN_UNIT: float = 0.0
DEFAULT_TOP_K_ORBS: int = 3
Vector4D = tuple[float, float, float, float]


class IntentMode(StrEnum):
    """Whether the Empath is absorbing room mood (PACING) or projecting ahead (LEADING)."""

    PACING = "PACING"
    LEADING = "LEADING"


class ScaleMode(StrEnum):
    """Supported musical modes and their semitone interval structures."""

    NATURAL_MINOR = "NATURAL_MINOR"
    PHRYGIAN = "PHRYGIAN"
    DORIAN = "DORIAN"

    @property
    def intervals(self) -> tuple[int, ...]:
        """Return semitone offsets relative to root pitch class."""
        mapping = {
            ScaleMode.NATURAL_MINOR: (0, 2, 3, 5, 7, 8, 10),
            ScaleMode.PHRYGIAN: (0, 1, 3, 5, 7, 8, 10),
            ScaleMode.DORIAN: (0, 2, 3, 5, 7, 9, 10),
        }
        return mapping[self]


class ControlGear(StrEnum):
    """Three gears of Variable-Grip Conductor autonomy."""

    GEAR_1_AUTOPILOT = "GEAR_1_AUTOPILOT"
    GEAR_2_WHIM_STEERING = "GEAR_2_WHIM_STEERING"
    GEAR_3_TACTILE_OVERRIDE = "GEAR_3_TACTILE_OVERRIDE"


def _validate_range(name: str, value: float, low: float, high: float) -> None:
    """Raise ValueError if a value falls outside [low, high]."""
    if value < low or value > high:
        raise ValueError(f"{name} must be within [{low}, {high}], got {value}")


@dataclass(frozen=True)
class MoodVector:
    """Affective state vector of the Empath performer."""

    valence: float
    arousal: float
    tension: float
    nostalgia: float
    intent_mode: IntentMode = IntentMode.PACING

    def __post_init__(self) -> None:
        """Validate affective coordinate bounds."""
        _validate_range("valence", self.valence, MIN_VALENCE, MAX_UNIT)
        _validate_range("arousal", self.arousal, MIN_UNIT, MAX_UNIT)
        _validate_range("tension", self.tension, MIN_UNIT, MAX_UNIT)
        _validate_range("nostalgia", self.nostalgia, MIN_UNIT, MAX_UNIT)

    def effective_embedding(self) -> Vector4D:
        """Compute query embedding accounting for PACING vs LEADING projection."""
        if self.intent_mode == IntentMode.LEADING:
            projected_arousal = min(MAX_UNIT, self.arousal + LEADING_PROJECTION_WEIGHT)
            projected_tension = min(MAX_UNIT, self.tension + LEADING_PROJECTION_WEIGHT)
            return (self.valence, projected_arousal, projected_tension, self.nostalgia)
        return (self.valence, self.arousal, self.tension, self.nostalgia)


@dataclass(frozen=True)
class MemoryAsset:
    """Autobiographical stem and symbolic motif entry in the Empath's personal vault."""

    asset_id: str
    title: str
    emotional_embedding: Vector4D
    native_bpm: float
    root_key: str
    native_mode: ScaleMode
    audio_stem_id: str
    symbolic_motif: tuple[int, ...]
    requires_modal_lock: bool
    dominant_freq_hz: tuple[float, float]


@dataclass(frozen=True)
class MemoryOrb:
    """Surfaced candidate memory projected onto one of the Empath's 3 HUD orb slots."""

    orb_slot: int
    asset: MemoryAsset
    score: float


@dataclass(frozen=True)
class EpisodicPlayRecord:
    """Record of a played motif or transition in the active DJ set."""

    asset_id: str
    scale_mode: ScaleMode
    transition_type: str
    timestamp_minutes: float


def cosine_similarity_4d(vec_a: Vector4D, vec_b: Vector4D) -> float:
    """Compute normalized cosine similarity between two 4D vectors."""
    denom = math.hypot(*vec_a) * math.hypot(*vec_b)
    if denom == 0.0:
        return 0.0
    dot = vec_a[0] * vec_b[0] + vec_a[1] * vec_b[1] + vec_a[2] * vec_b[2] + vec_a[3] * vec_b[3]
    return dot / denom


@dataclass
class EpisodicSetMemory:
    """Rolling episodic memory that penalizes recently played motifs within a time window."""

    window_minutes: float = DEFAULT_EPISODIC_WINDOW_MINUTES
    records: list[EpisodicPlayRecord] = field(default_factory=list)

    def record_play(
        self,
        asset_id: str,
        scale_mode: ScaleMode,
        transition_type: str,
        timestamp_minutes: float,
    ) -> None:
        """Append a newly executed transition to episodic set history."""
        self.records.append(
            EpisodicPlayRecord(asset_id, scale_mode, transition_type, timestamp_minutes)
        )

    def _single_record_penalty(self, rec: EpisodicPlayRecord, current_minutes: float) -> float:
        """Compute linear decay penalty for one play record."""
        elapsed = current_minutes - rec.timestamp_minutes
        if elapsed < MIN_UNIT or elapsed >= self.window_minutes:
            return MIN_UNIT
        return RECENCY_PENALTY_WEIGHT * (MAX_UNIT - (elapsed / self.window_minutes))

    def compute_recency_penalty(self, asset_id: str, current_minutes: float) -> float:
        """Sum linear recency decay penalties for all plays of asset_id in window."""
        matching = [r for r in self.records if r.asset_id == asset_id]
        return sum(self._single_record_penalty(r, current_minutes) for r in matching)


class AutobiographicalEmotionPrism:
    """Maps Empath MoodVectors to ranked Personal Memory Orbs with episodic penalties."""

    def score_asset(
        self,
        mood: MoodVector,
        asset: MemoryAsset,
        episodic_memory: EpisodicSetMemory,
        current_minutes: float,
    ) -> float:
        """Compute net relevance score for an asset at current_minutes."""
        raw_sim = cosine_similarity_4d(mood.effective_embedding(), asset.emotional_embedding)
        penalty = episodic_memory.compute_recency_penalty(asset.asset_id, current_minutes)
        return raw_sim - penalty

    def surface_memory_orbs(
        self,
        mood: MoodVector,
        assets: tuple[MemoryAsset, ...],
        episodic_memory: EpisodicSetMemory,
        current_minutes: float,
        top_k: int = DEFAULT_TOP_K_ORBS,
    ) -> tuple[MemoryOrb, ...]:
        """Rank assets descending by score and assign 1-indexed orb slots."""
        scored = [
            (self.score_asset(mood, asset, episodic_memory, current_minutes), asset)
            for asset in assets
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return tuple(
            MemoryOrb(orb_slot=idx + 1, asset=pair[1], score=pair[0])
            for idx, pair in enumerate(scored[:top_k])
        )


@dataclass
class ProactiveCoPilotMonitor:
    """Monitors static tension duration in GEAR_1_AUTOPILOT to surface fresh orbs."""

    stagnation_bars_threshold: int = STAGNATION_BARS_THRESHOLD
    bars_in_static_tension: int = 0
    anchor_tension: float | None = None

    def _update_static_streak(self, tension: float) -> None:
        """Increment or reset static tension streak against anchor_tension."""
        shifted = (
            self.anchor_tension is None
            or abs(tension - self.anchor_tension) >= STATIC_TENSION_EPSILON
        )
        if shifted:
            self.anchor_tension = tension
            self.bars_in_static_tension = 1
        else:
            self.bars_in_static_tension += 1

    def on_bar_advanced(self, tension: float, gear: ControlGear) -> bool:
        """Advance one bar and return True when proactive orb resurfacing should fire."""
        if gear != ControlGear.GEAR_1_AUTOPILOT:
            self.bars_in_static_tension = 0
            self.anchor_tension = None
            return False
        self._update_static_streak(tension)
        return self.bars_in_static_tension >= self.stagnation_bars_threshold
