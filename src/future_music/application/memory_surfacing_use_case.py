"""Slice 1 (REQ-01) application use case for surfacing personal Memory Orbs."""

from dataclasses import dataclass

from future_music.domain.empath_memory import (
    AutobiographicalEmotionPrism,
    ControlGear,
    EpisodicSetMemory,
    MemoryOrb,
    MoodVector,
    ProactiveCoPilotMonitor,
    ScaleMode,
)
from future_music.ports.memory_ports import PersonalVaultPort


@dataclass
class SurfaceMemoryOrbsUseCase:
    """Orchestrates autobiographical orb surfacing, episodic memory, and proactive co-pilot."""

    vault: PersonalVaultPort
    prism: AutobiographicalEmotionPrism
    episodic_memory: EpisodicSetMemory
    copilot_monitor: ProactiveCoPilotMonitor

    def surface_orbs(
        self, empath_id: str, mood: MoodVector, current_minutes: float
    ) -> tuple[MemoryOrb, ...]:
        """Retrieve Empath assets and rank top-3 Memory Orbs for the active mood."""
        assets = self.vault.get_assets(empath_id)
        return self.prism.surface_memory_orbs(
            mood=mood,
            assets=assets,
            episodic_memory=self.episodic_memory,
            current_minutes=current_minutes,
        )

    def record_transition_played(
        self,
        asset_id: str,
        scale_mode: ScaleMode,
        transition_type: str,
        timestamp_minutes: float,
    ) -> None:
        """Record a committed motif transition in the rolling 60-minute episodic memory."""
        self.episodic_memory.record_play(
            asset_id=asset_id,
            scale_mode=scale_mode,
            transition_type=transition_type,
            timestamp_minutes=timestamp_minutes,
        )

    def on_bar_tick(
        self,
        empath_id: str,
        mood: MoodVector,
        gear: ControlGear,
        current_minutes: float,
    ) -> tuple[MemoryOrb, ...] | None:
        """Advance bar monitor and proactively surface orbs when tension stagnates."""
        if self.copilot_monitor.on_bar_advanced(tension=mood.tension, gear=gear):
            return self.surface_orbs(empath_id, mood, current_minutes)
        return None
