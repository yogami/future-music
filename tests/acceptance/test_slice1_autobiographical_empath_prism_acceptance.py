"""Acceptance tests for Slice 1 (REQ-01): Autobiographical Empath Prism & Set Evolution."""

from future_music.application.memory_surfacing_use_case import SurfaceMemoryOrbsUseCase
from future_music.domain.empath_memory import (
    AutobiographicalEmotionPrism,
    ControlGear,
    EpisodicSetMemory,
    IntentMode,
    MoodVector,
    ProactiveCoPilotMonitor,
)
from future_music.infrastructure.in_memory_vault import create_default_empath_vault


def test_req01_two_unique_empaths_with_identical_ache_channel_distinct_memories() -> None:
    """REQ-01: Two Empaths with identical 2 AM ache surface distinct autobiographical stems."""
    vault = create_default_empath_vault()
    use_case = SurfaceMemoryOrbsUseCase(
        vault=vault,
        prism=AutobiographicalEmotionPrism(),
        episodic_memory=EpisodicSetMemory(),
        copilot_monitor=ProactiveCoPilotMonitor(),
    )
    collective_room_ache = MoodVector(
        valence=0.35,
        arousal=0.55,
        tension=0.30,
        nostalgia=0.95,
        intent_mode=IntentMode.PACING,
    )
    empath_yami_orbs = use_case.surface_orbs(
        empath_id="empath_euro_90s", mood=collective_room_ache, current_minutes=30.0
    )
    empath_sufi_orbs = use_case.surface_orbs(
        empath_id="empath_eastern_mystic", mood=collective_room_ache, current_minutes=30.0
    )

    assert empath_yami_orbs[0].asset.asset_id == "robert_miles_children"
    assert empath_sufi_orbs[0].asset.asset_id == "sufi_vocal"
    assert empath_yami_orbs[0].asset.asset_id != empath_sufi_orbs[0].asset.asset_id


def test_req01_60_minute_set_evolution_prevents_repetition_and_auto_surfaces() -> None:
    """REQ-01: Played motifs are penalized for 60 mins and recover after the 60-min window."""
    vault = create_default_empath_vault()
    use_case = SurfaceMemoryOrbsUseCase(
        vault=vault,
        prism=AutobiographicalEmotionPrism(),
        episodic_memory=EpisodicSetMemory(window_minutes=60.0),
        copilot_monitor=ProactiveCoPilotMonitor(stagnation_bars_threshold=16),
    )
    mood = MoodVector(valence=0.35, arousal=0.55, tension=0.30, nostalgia=0.95)

    first_orbs = use_case.surface_orbs("empath_euro_90s", mood, current_minutes=10.0)
    assert first_orbs[0].asset.asset_id == "robert_miles_children"

    use_case.record_transition_played(
        asset_id="robert_miles_children",
        scale_mode=first_orbs[0].asset.native_mode,
        transition_type="bridge_4bar",
        timestamp_minutes=10.0,
    )
    mid_set_orbs = use_case.surface_orbs("empath_euro_90s", mood, current_minutes=25.0)
    assert mid_set_orbs[0].asset.asset_id != "robert_miles_children"

    late_set_orbs = None
    for _ in range(16):
        late_set_orbs = use_case.on_bar_tick(
            "empath_euro_90s", mood, ControlGear.GEAR_1_AUTOPILOT, current_minutes=75.0
        )
    assert late_set_orbs is not None
    assert late_set_orbs[0].asset.asset_id == "robert_miles_children"
