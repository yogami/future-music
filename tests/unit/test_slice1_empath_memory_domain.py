"""Unit tests for Slice 1 (REQ-01): MoodVector, ScaleMode, EpisodicSetMemory, and Emotion Prism."""

import math

import pytest
from future_music.domain.empath_memory import (
    DEFAULT_EPISODIC_WINDOW_MINUTES,
    LEADING_PROJECTION_WEIGHT,
    RECENCY_PENALTY_WEIGHT,
    STAGNATION_BARS_THRESHOLD,
    AutobiographicalEmotionPrism,
    ControlGear,
    EpisodicSetMemory,
    IntentMode,
    MemoryAsset,
    MoodVector,
    ProactiveCoPilotMonitor,
    ScaleMode,
    cosine_similarity_4d,
)


def make_sample_asset(
    asset_id: str,
    embedding: tuple[float, float, float, float],
    mode: ScaleMode = ScaleMode.NATURAL_MINOR,
    modal_lock: bool = False,
) -> MemoryAsset:
    """Create a deterministic MemoryAsset fixture for unit tests."""
    return MemoryAsset(
        asset_id=asset_id,
        title=f"Title {asset_id}",
        emotional_embedding=embedding,
        native_bpm=124.0,
        root_key="F",
        native_mode=mode,
        audio_stem_id=f"{asset_id}.wav",
        symbolic_motif=(65, 68, 72) if not modal_lock else (),
        requires_modal_lock=modal_lock,
        dominant_freq_hz=(1000.0, 3000.0),
    )


def test_mood_vector_validates_bounds_and_rejects_out_of_range_values() -> None:
    """MoodVector enforces valence in [-1, 1] and arousal/tension/nostalgia in [0, 1]."""
    valid = MoodVector(valence=-0.4, arousal=0.7, tension=0.5, nostalgia=0.9)
    assert valid.valence == -0.4
    assert valid.intent_mode == IntentMode.PACING
    with pytest.raises(ValueError, match="valence"):
        MoodVector(valence=1.5, arousal=0.5, tension=0.5, nostalgia=0.5)
    with pytest.raises(ValueError, match="tension"):
        MoodVector(valence=0.0, arousal=0.5, tension=-0.1, nostalgia=0.5)


def test_mood_vector_pacing_vs_leading_amplifies_arousal_and_tension_clamped() -> None:
    """LEADING intent adds LEADING_PROJECTION_WEIGHT (0.25) to arousal/tension, clamped at 1.0."""
    pacing = MoodVector(
        valence=0.2, arousal=0.6, tension=0.85, nostalgia=0.4, intent_mode=IntentMode.PACING
    )
    leading = MoodVector(
        valence=0.2, arousal=0.6, tension=0.85, nostalgia=0.4, intent_mode=IntentMode.LEADING
    )
    assert pacing.effective_embedding() == (0.2, 0.6, 0.85, 0.4)
    assert leading.effective_embedding() == (
        0.2,
        pytest.approx(0.6 + LEADING_PROJECTION_WEIGHT),
        1.0,
        0.4,
    )


def test_scale_mode_exposes_exact_semitone_intervals_for_minor_phrygian_dorian() -> None:
    """ScaleMode defines exact semitone intervals relative to the root key."""
    assert ScaleMode.NATURAL_MINOR.intervals == (0, 2, 3, 5, 7, 8, 10)
    assert ScaleMode.PHRYGIAN.intervals == (0, 1, 3, 5, 7, 8, 10)
    assert ScaleMode.DORIAN.intervals == (0, 2, 3, 5, 7, 9, 10)


def test_cosine_similarity_computes_exact_geometric_dot_product_and_handles_zero_norm() -> None:
    """cosine_similarity_4d computes normalized inner product and returns 0.0 on zero vector."""
    vec_a = (1.0, 0.0, 1.0, 0.0)
    vec_b = (1.0, 1.0, 0.0, 0.0)
    expected = 1.0 / (math.sqrt(2.0) * math.sqrt(2.0))
    assert cosine_similarity_4d(vec_a, vec_b) == pytest.approx(expected)
    assert cosine_similarity_4d((0.0, 0.0, 0.0, 0.0), vec_b) == 0.0


def test_episodic_memory_applies_linear_decay_recency_penalty_within_60_minute_window() -> None:
    """EpisodicSetMemory decays RECENCY_PENALTY_WEIGHT linearly across the 60-minute window."""
    memory = EpisodicSetMemory(window_minutes=DEFAULT_EPISODIC_WINDOW_MINUTES)
    memory.record_play(
        asset_id="robert_miles_children",
        scale_mode=ScaleMode.NATURAL_MINOR,
        transition_type="bridge_4bar",
        timestamp_minutes=10.0,
    )
    penalty_at_10 = memory.compute_recency_penalty("robert_miles_children", current_minutes=10.0)
    penalty_at_40 = memory.compute_recency_penalty("robert_miles_children", current_minutes=40.0)
    assert penalty_at_10 == pytest.approx(RECENCY_PENALTY_WEIGHT)
    assert penalty_at_40 == pytest.approx(RECENCY_PENALTY_WEIGHT * 0.5)


def test_episodic_memory_ignores_plays_outside_60_minute_rolling_window() -> None:
    """Plays older than 60 minutes incur zero recency penalty."""
    memory = EpisodicSetMemory(window_minutes=60.0)
    memory.record_play(
        asset_id="sufi_vocal",
        scale_mode=ScaleMode.PHRYGIAN,
        transition_type="bridge_4bar",
        timestamp_minutes=5.0,
    )
    assert memory.compute_recency_penalty("sufi_vocal", current_minutes=65.0) == 0.0
    assert memory.compute_recency_penalty("sufi_vocal", current_minutes=80.0) == 0.0


def test_episodic_memory_accumulates_penalty_for_repeated_back_to_back_plays() -> None:
    """Playing the same motif multiple times within the window sums the linear penalties."""
    memory = EpisodicSetMemory(window_minutes=60.0)
    memory.record_play("robert_miles_children", ScaleMode.NATURAL_MINOR, "drop", 0.0)
    memory.record_play("robert_miles_children", ScaleMode.NATURAL_MINOR, "drop", 30.0)
    total_penalty = memory.compute_recency_penalty("robert_miles_children", current_minutes=30.0)
    expected = (RECENCY_PENALTY_WEIGHT * 0.5) + RECENCY_PENALTY_WEIGHT
    assert total_penalty == pytest.approx(expected)


def test_autobiographical_prism_ranks_orbs_by_cosine_minus_episodic_penalty() -> None:
    """AutobiographicalEmotionPrism assigns orb slots 1..3 ordered by net score."""
    prism = AutobiographicalEmotionPrism()
    memory = EpisodicSetMemory()
    mood = MoodVector(valence=0.5, arousal=0.5, tension=0.2, nostalgia=0.9)
    a1 = make_sample_asset("robert_miles_children", (0.5, 0.5, 0.2, 0.9))
    a2 = make_sample_asset("sufi_vocal", (0.45, 0.5, 0.25, 0.85), ScaleMode.PHRYGIAN, True)
    a3 = make_sample_asset("berlin_acid_synth", (-0.5, 0.9, 0.9, 0.1))
    orbs = prism.surface_memory_orbs(mood, (a1, a2, a3), memory, current_minutes=20.0)
    assert [orb.orb_slot for orb in orbs] == [1, 2, 3]
    assert [orb.asset.asset_id for orb in orbs] == [
        "robert_miles_children",
        "sufi_vocal",
        "berlin_acid_synth",
    ]


def test_proactive_copilot_triggers_resurface_after_16_bars_of_static_tension_in_gear1() -> None:
    """ProactiveCoPilotMonitor returns True once static tension reaches 16 bars in GEAR_1."""
    monitor = ProactiveCoPilotMonitor(stagnation_bars_threshold=STAGNATION_BARS_THRESHOLD)
    triggered = [
        monitor.on_bar_advanced(tension=0.50, gear=ControlGear.GEAR_1_AUTOPILOT)
        for _ in range(STAGNATION_BARS_THRESHOLD)
    ]
    assert triggered[:-1] == [False] * (STAGNATION_BARS_THRESHOLD - 1)
    assert triggered[-1] is True


def test_proactive_copilot_resets_counter_on_tension_shift_or_non_autopilot_gear() -> None:
    """Significant tension change (>= 0.05) or leaving GEAR_1 resets the stagnation bar counter."""
    monitor = ProactiveCoPilotMonitor(stagnation_bars_threshold=4)
    assert monitor.on_bar_advanced(0.40, ControlGear.GEAR_1_AUTOPILOT) is False
    assert monitor.on_bar_advanced(0.41, ControlGear.GEAR_1_AUTOPILOT) is False
    assert monitor.on_bar_advanced(0.65, ControlGear.GEAR_1_AUTOPILOT) is False
    assert monitor.bars_in_static_tension == 1
    assert monitor.on_bar_advanced(0.65, ControlGear.GEAR_2_WHIM_STEERING) is False
    assert monitor.bars_in_static_tension == 0
