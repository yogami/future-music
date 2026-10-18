"""Slice 4 (REQ-08, REQ-09): Ghost Preview State, 4-Stage Lookahead Bridge, & Spectral Critic."""

from dataclasses import dataclass
from enum import StrEnum

from future_music.domain.clock_and_midi import (
    DEFAULT_CC1_MODULATION,
    MAX_MIDI_VAL,
    MIN_MIDI_VAL,
    MIN_VELOCITY,
    OCTAVE_SEMITONES,
    ScheduledMidiNote,
)
from future_music.domain.empath_memory import (
    MAX_UNIT,
    MIN_UNIT,
    ControlGear,
    MemoryAsset,
    ScaleMode,
)

DEFAULT_PREVIEW_BARS: int = 2
NUM_BRIDGE_STAGES: int = 4
STAGE1_DENSITY_MULTIPLIER: float = 0.5
STAGE2_DENSITY_MULTIPLIER: float = 0.75
STAGE3_TEASE_GAIN: float = 0.35
STAGE3_RISER_CC1: int = 96

PRESENCE_BAND_HZ: tuple[float, float] = (1000.0, 3000.0)
MIDI_PRESENCE_NOTE_RANGE: tuple[int, int] = (60, 84)
SPECTRAL_VELOCITY_TRIM: int = 15


class PreviewRoutingMode(StrEnum):
    """Hardware routing modes for isolating Ghost Preview from Main PA (REQ-08)."""

    STEREO_SPLIT = "STEREO_SPLIT"
    DUAL_DEVICE = "DUAL_DEVICE"


class BridgeStageName(StrEnum):
    """Four sequential phases of a multi-bar harmonic and arrangement bridge (REQ-09)."""

    STAGE_1_THIN_MIDRANGE = "STAGE_1_THIN_MIDRANGE"
    STAGE_2_PIVOT_CHORDS = "STAGE_2_PIVOT_CHORDS"
    STAGE_3_TEASE_AND_RISER = "STAGE_3_TEASE_AND_RISER"
    STAGE_4_SUBBASS_RESOLVE_AND_DROP = "STAGE_4_SUBBASS_RESOLVE_AND_DROP"


@dataclass(frozen=True)
class ToolCallCommand:
    """Structured agentic tool call emitted by the Conductor (Agentic Capability 2)."""

    tool_name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class GhostPreviewState:
    """Immutable state of the Inner-Ear Ghost Preview Bus (REQ-08)."""

    active: bool
    staged_asset_id: str | None
    preview_bars: int
    gear: ControlGear
    emitted_tool_call: ToolCallCommand | None = None


def build_ghost_stage_state(
    asset_id: str, bars: int = DEFAULT_PREVIEW_BARS
) -> GhostPreviewState:
    """Create GhostPreviewState for staging asset_id in the Ghost Preview Bus."""
    tool_call = ToolCallCommand(
        tool_name="route_to_ghost_cue",
        arguments={"asset_id": asset_id, "bars": bars},
    )
    return GhostPreviewState(
        active=True,
        staged_asset_id=asset_id,
        preview_bars=bars,
        gear=ControlGear.GEAR_2_WHIM_STEERING,
        emitted_tool_call=tool_call,
    )


def build_ghost_idle_state() -> GhostPreviewState:
    """Create idle GhostPreviewState returning control to GEAR_1_AUTOPILOT."""
    return GhostPreviewState(
        active=False,
        staged_asset_id=None,
        preview_bars=0,
        gear=ControlGear.GEAR_1_AUTOPILOT,
        emitted_tool_call=None,
    )


@dataclass(frozen=True)
class BridgeStageSpec:
    """Specification for one stage of the 4-Stage Lookahead Bridge Plan (REQ-09)."""

    stage_name: BridgeStageName
    start_bar: int
    end_bar: int
    note_density_multiplier: float
    stem_gain: float
    target_cc1: int
    tool_calls: tuple[ToolCallCommand, ...]


@dataclass(frozen=True)
class LookaheadBridgePlan:
    """Complete 4-Stage Lookahead Bridge Plan between source and target musical states."""

    target_asset_id: str
    source_mode: ScaleMode
    target_mode: ScaleMode
    total_bars: int
    stages: tuple[BridgeStageSpec, ...]


def _build_stage_1(source_mode: ScaleMode, bars_per_stage: int) -> BridgeStageSpec:
    """Construct Stage 1 (thin midrange) spec."""
    tc = ToolCallCommand("write_midi_clip", {"role": "thin_midrange", "scale": source_mode.value})
    return BridgeStageSpec(
        stage_name=BridgeStageName.STAGE_1_THIN_MIDRANGE,
        start_bar=1,
        end_bar=bars_per_stage,
        note_density_multiplier=STAGE1_DENSITY_MULTIPLIER,
        stem_gain=MIN_UNIT,
        target_cc1=DEFAULT_CC1_MODULATION,
        tool_calls=(tc,),
    )


def _build_stage_2(
    asset: MemoryAsset, source_mode: ScaleMode, target_mode: ScaleMode, bars_per_stage: int
) -> BridgeStageSpec:
    """Construct Stage 2 (pivot chords) spec."""
    args: dict[str, object] = {
        "role": "pivot_chords",
        "asset_id": asset.asset_id,
        "from_scale": source_mode.value,
        "to_scale": target_mode.value,
    }
    return BridgeStageSpec(
        stage_name=BridgeStageName.STAGE_2_PIVOT_CHORDS,
        start_bar=bars_per_stage + 1,
        end_bar=bars_per_stage * 2,
        note_density_multiplier=STAGE2_DENSITY_MULTIPLIER,
        stem_gain=MIN_UNIT,
        target_cc1=DEFAULT_CC1_MODULATION,
        tool_calls=(ToolCallCommand("write_midi_clip", args),),
    )


def _build_stage_3(asset: MemoryAsset, bars_per_stage: int) -> BridgeStageSpec:
    """Construct Stage 3 (tease & riser) spec."""
    calls = (
        ToolCallCommand("warp_stem", {"asset_id": asset.asset_id, "gain": STAGE3_TEASE_GAIN}),
        ToolCallCommand("modulate_filter_cutoff", {"cc1": STAGE3_RISER_CC1}),
    )
    return BridgeStageSpec(
        stage_name=BridgeStageName.STAGE_3_TEASE_AND_RISER,
        start_bar=(bars_per_stage * 2) + 1,
        end_bar=bars_per_stage * 3,
        note_density_multiplier=STAGE2_DENSITY_MULTIPLIER,
        stem_gain=STAGE3_TEASE_GAIN,
        target_cc1=STAGE3_RISER_CC1,
        tool_calls=calls,
    )


def _build_stage_4(
    asset: MemoryAsset, target_mode: ScaleMode, bars_per_stage: int, total_bars: int
) -> BridgeStageSpec:
    """Construct Stage 4 (subbass resolve & drop) spec."""
    calls = (
        ToolCallCommand("warp_stem", {"asset_id": asset.asset_id, "gain": MAX_UNIT}),
        ToolCallCommand("write_midi_clip", {"role": "subbass_resolve", "scale": target_mode.value}),
    )
    return BridgeStageSpec(
        stage_name=BridgeStageName.STAGE_4_SUBBASS_RESOLVE_AND_DROP,
        start_bar=(bars_per_stage * 3) + 1,
        end_bar=total_bars,
        note_density_multiplier=MAX_UNIT,
        stem_gain=MAX_UNIT,
        target_cc1=MAX_MIDI_VAL,
        tool_calls=calls,
    )


def build_lookahead_bridge_plan(
    target_asset: MemoryAsset,
    source_mode: ScaleMode,
    target_mode: ScaleMode,
    total_bars: int = NUM_BRIDGE_STAGES,
) -> LookaheadBridgePlan:
    """Build deterministic 4-Stage Lookahead Bridge Plan (REQ-09)."""
    enforced_bars = max(NUM_BRIDGE_STAGES, total_bars)
    b_stage = enforced_bars // NUM_BRIDGE_STAGES
    stages = (
        _build_stage_1(source_mode, b_stage),
        _build_stage_2(target_asset, source_mode, target_mode, b_stage),
        _build_stage_3(target_asset, b_stage),
        _build_stage_4(target_asset, target_mode, b_stage, enforced_bars),
    )
    return LookaheadBridgePlan(
        target_asset.asset_id, source_mode, target_mode, enforced_bars, stages
    )


@dataclass(frozen=True)
class SpectralCriticResult:
    """Result of Pre-Play Spectral Critic evaluation on a bar of MIDI notes (REQ-09)."""

    adjusted_notes: tuple[ScheduledMidiNote, ...]
    spectral_carve_applied: bool
    carved_note_count: int


def _bands_overlap(band_a: tuple[float, float], band_b: tuple[float, float]) -> bool:
    """Return True if two (low_hz, high_hz) intervals overlap."""
    return max(band_a[0], band_b[0]) < min(band_a[1], band_b[1])


def _carve_note_if_colliding(
    item: ScheduledMidiNote, overlap_active: bool
) -> tuple[ScheduledMidiNote, int]:
    """Shift note down 1 octave and trim velocity by 15 if colliding in 1-3kHz band."""
    if not overlap_active:
        return (item, 0)
    low_note, high_note = MIDI_PRESENCE_NOTE_RANGE
    if not (low_note <= item.note <= high_note):
        return (item, 0)
    shifted = max(MIN_MIDI_VAL, item.note - OCTAVE_SEMITONES)
    trimmed = max(MIN_VELOCITY, item.velocity - SPECTRAL_VELOCITY_TRIM)
    return (
        ScheduledMidiNote(note=shifted, velocity=trimmed, trigger_sample=item.trigger_sample),
        1,
    )


def evaluate_spectral_collision(
    stem_freq_band_hz: tuple[float, float],
    scheduled_notes: tuple[ScheduledMidiNote, ...],
) -> SpectralCriticResult:
    """Inspect active stem presence band (1-3kHz) and carve colliding MIDI notes (REQ-09)."""
    overlap = _bands_overlap(stem_freq_band_hz, PRESENCE_BAND_HZ)
    adjusted: list[ScheduledMidiNote] = []
    carved_count = 0
    for item in scheduled_notes:
        carved_note, delta = _carve_note_if_colliding(item, overlap)
        adjusted.append(carved_note)
        carved_count += delta
    return SpectralCriticResult(
        adjusted_notes=tuple(adjusted),
        spectral_carve_applied=carved_count > 0,
        carved_note_count=carved_count,
    )
