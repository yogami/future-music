"""Integration tests for Slice 1 (REQ-01): SurfaceMemoryOrbsUseCase with InMemoryPersonalVault."""

from future_music.application.memory_surfacing_use_case import SurfaceMemoryOrbsUseCase
from future_music.domain.empath_memory import (
    AutobiographicalEmotionPrism,
    ControlGear,
    EpisodicSetMemory,
    IntentMode,
    MemoryAsset,
    MoodVector,
    ProactiveCoPilotMonitor,
    ScaleMode,
)
from future_music.infrastructure.in_memory_vault import InMemoryPersonalVault


def build_populated_vault() -> InMemoryPersonalVault:
    """Build an InMemoryPersonalVault with three diverse personal memory assets."""
    vault = InMemoryPersonalVault()
    vault.register_asset(
        "empath_alpha",
        MemoryAsset(
            asset_id="robert_miles_children",
            title="Robert Miles - Children (Dream Piano)",
            emotional_embedding=(0.4, 0.5, 0.3, 0.95),
            native_bpm=124.0,
            root_key="F",
            native_mode=ScaleMode.NATURAL_MINOR,
            audio_stem_id="robert_miles_piano.wav",
            symbolic_motif=(65, 68, 72, 70),
            requires_modal_lock=False,
            dominant_freq_hz=(800.0, 2800.0),
        ),
    )
    vault.register_asset(
        "empath_alpha",
        MemoryAsset(
            asset_id="sufi_vocal",
            title="Qawwali Flute & Vocal in F Phrygian",
            emotional_embedding=(0.35, 0.55, 0.45, 0.90),
            native_bpm=124.0,
            root_key="F",
            native_mode=ScaleMode.PHRYGIAN,
            audio_stem_id="sufi_vocal.wav",
            symbolic_motif=(),
            requires_modal_lock=True,
            dominant_freq_hz=(1000.0, 3000.0),
        ),
    )
    vault.register_asset(
        "empath_alpha",
        MemoryAsset(
            asset_id="berlin_acid_synth",
            title="Berlin 4 AM 303 Acid Line",
            emotional_embedding=(0.1, 0.95, 0.95, 0.2),
            native_bpm=124.0,
            root_key="F",
            native_mode=ScaleMode.PHRYGIAN,
            audio_stem_id="berlin_acid_synth.wav",
            symbolic_motif=(53, 54, 65, 53),
            requires_modal_lock=False,
            dominant_freq_hz=(400.0, 2200.0),
        ),
    )
    return vault


def test_use_case_surfaces_top_three_orbs_from_vault_and_rotates_after_play_recorded() -> None:
    """Recording a play in EpisodicSetMemory demotes that asset on the next orb query."""
    vault = build_populated_vault()
    use_case = SurfaceMemoryOrbsUseCase(
        vault=vault,
        prism=AutobiographicalEmotionPrism(),
        episodic_memory=EpisodicSetMemory(),
        copilot_monitor=ProactiveCoPilotMonitor(),
    )
    mood = MoodVector(valence=0.4, arousal=0.5, tension=0.3, nostalgia=0.95)
    initial_orbs = use_case.surface_orbs("empath_alpha", mood, current_minutes=15.0)
    assert initial_orbs[0].asset.asset_id == "robert_miles_children"

    use_case.record_transition_played(
        asset_id="robert_miles_children",
        scale_mode=ScaleMode.NATURAL_MINOR,
        transition_type="bridge_4bar",
        timestamp_minutes=15.0,
    )
    rotated_orbs = use_case.surface_orbs("empath_alpha", mood, current_minutes=16.0)
    assert rotated_orbs[0].asset.asset_id == "sufi_vocal"


def test_use_case_leading_intent_shifts_top_orb_toward_higher_tension_asset() -> None:
    """Switching MoodVector from PACING to LEADING projects higher arousal and tension."""
    vault = build_populated_vault()
    use_case = SurfaceMemoryOrbsUseCase(
        vault=vault,
        prism=AutobiographicalEmotionPrism(),
        episodic_memory=EpisodicSetMemory(),
        copilot_monitor=ProactiveCoPilotMonitor(),
    )
    leading_mood = MoodVector(
        valence=0.1,
        arousal=0.75,
        tension=0.75,
        nostalgia=0.2,
        intent_mode=IntentMode.LEADING,
    )
    orbs = use_case.surface_orbs("empath_alpha", leading_mood, current_minutes=5.0)
    assert orbs[0].asset.asset_id == "berlin_acid_synth"


def test_use_case_proactive_bar_tick_resurfaces_orbs_at_16_bar_threshold() -> None:
    """Advancing 16 bars in static tension in GEAR_1 automatically returns refreshed MemoryOrbs."""
    vault = build_populated_vault()
    use_case = SurfaceMemoryOrbsUseCase(
        vault=vault,
        prism=AutobiographicalEmotionPrism(),
        episodic_memory=EpisodicSetMemory(),
        copilot_monitor=ProactiveCoPilotMonitor(stagnation_bars_threshold=16),
    )
    mood = MoodVector(valence=0.4, arousal=0.5, tension=0.3, nostalgia=0.95)
    for _ in range(15):
        result = use_case.on_bar_tick(
            "empath_alpha", mood, ControlGear.GEAR_1_AUTOPILOT, current_minutes=20.0
        )
        assert result is None
    refreshed = use_case.on_bar_tick(
        "empath_alpha", mood, ControlGear.GEAR_1_AUTOPILOT, current_minutes=20.5
    )
    assert refreshed is not None
    assert len(refreshed) == 3
    assert refreshed[0].orb_slot == 1
