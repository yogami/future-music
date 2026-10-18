"""Application use case for Slice 4 (REQ-07, REQ-08, REQ-09): Whim, Ghost Preview & Bridge."""

from dataclasses import dataclass, field

import numpy as np

from future_music.domain.clock_and_midi import ModalScaleEngine, SampleClock, ScheduledMidiNote
from future_music.domain.empath_memory import EpisodicSetMemory, MemoryAsset
from future_music.domain.ghost_and_bridge import (
    DEFAULT_PREVIEW_BARS,
    GhostPreviewState,
    LookaheadBridgePlan,
    SpectralCriticResult,
    build_ghost_idle_state,
    build_ghost_stage_state,
    build_lookahead_bridge_plan,
    evaluate_spectral_collision,
)
from future_music.domain.whim_and_hum import (
    DEFAULT_BRIDGE_BARS,
    HumContourEvent,
    InputGateController,
    QuantizedMidiNote,
    VoiceWhimIntent,
    parse_voice_whim,
    quantize_hum_contour,
)
from future_music.ports.audio_routing_ports import AudioRouterPort, DualBusAudioOutput
from future_music.ports.memory_ports import PersonalVaultPort
from future_music.ports.midi_ports import MidiOutputPort


@dataclass
class WhimAndBridgeUseCase:
    """Orchestrates gated voice/hum whims, Ghost Preview cueing, and 4-Stage Bridge Planner."""

    vault: PersonalVaultPort
    audio_router: AudioRouterPort
    midi_port: MidiOutputPort
    empath_id: str = "empath_euro_90s"
    episodic_memory: EpisodicSetMemory = field(default_factory=EpisodicSetMemory)
    clock: SampleClock = field(default_factory=SampleClock)
    modal_engine: ModalScaleEngine = field(default_factory=ModalScaleEngine)
    input_gate: InputGateController = field(default_factory=InputGateController)
    ghost_preview_state: GhostPreviewState = field(default_factory=build_ghost_idle_state)
    active_bridge_plan: LookaheadBridgePlan | None = None

    def _resolve_asset(self, asset_id: str) -> MemoryAsset:
        """Look up asset_id in the Empath's personal vault or raise KeyError."""
        for asset in self.vault.get_assets(self.empath_id):
            if asset.asset_id == asset_id:
                return asset
        raise KeyError(f"Asset '{asset_id}' not found for empath '{self.empath_id}'")

    def stage_asset_to_ghost_preview(
        self, asset_id: str, bars: int = DEFAULT_PREVIEW_BARS
    ) -> GhostPreviewState:
        """Stage a personal MemoryAsset in the Ghost Preview Bus (REQ-08)."""
        self._resolve_asset(asset_id)
        self.ghost_preview_state = build_ghost_stage_state(asset_id=asset_id, bars=bars)
        return self.ghost_preview_state

    def discard_ghost_preview(self) -> GhostPreviewState:
        """Cancel staged Ghost Preview silently and return to GEAR_1_AUTOPILOT (REQ-08)."""
        self.ghost_preview_state = build_ghost_idle_state()
        return self.ghost_preview_state

    def commit_ghost_preview(
        self, bridge_bars: int = DEFAULT_BRIDGE_BARS, elapsed_minutes: float = 0.0
    ) -> LookaheadBridgePlan:
        """Promote staged Ghost Preview into 4-Stage Lookahead Bridge Planner (REQ-08/09)."""
        staged_id = self.ghost_preview_state.staged_asset_id
        if staged_id is None:
            raise RuntimeError("No asset is currently staged in Ghost Preview")
        asset = self._resolve_asset(staged_id)
        source_mode = self.modal_engine.active_scale_mode
        plan = build_lookahead_bridge_plan(asset, source_mode, asset.native_mode, bridge_bars)
        self.modal_engine.activate_asset(asset)
        self.episodic_memory.record_play(
            asset.asset_id, asset.native_mode, "bridge_commit", elapsed_minutes
        )
        self.ghost_preview_state = build_ghost_idle_state()
        self.active_bridge_plan = plan
        return plan

    def submit_voice_whim(self, transcript: str) -> VoiceWhimIntent:
        """Parse gated voice whim (Hold V) and stage any matched asset to Ghost Preview."""
        self.input_gate.require_voice_gate()
        intent = parse_voice_whim(transcript)
        if intent.target_asset_id is not None:
            self.stage_asset_to_ghost_preview(intent.target_asset_id)
        return intent

    def submit_hum_contour(
        self, events: tuple[HumContourEvent, ...]
    ) -> tuple[QuantizedMidiNote, ...]:
        """Quantize gated hum contour (Hold H) with -80ms lag offset and send MIDI notes."""
        self.input_gate.require_hum_gate()
        quantized = quantize_hum_contour(
            events=events,
            clock=self.clock,
            root_key="F",
            mode=self.modal_engine.active_scale_mode,
        )
        for q_note in quantized:
            self.midi_port.send_note_on(
                note=q_note.midi_note,
                velocity=q_note.velocity,
                trigger_sample=q_note.quantized_sample,
            )
        return quantized

    def render_dual_bus_audio(
        self, main_mono: np.ndarray, ghost_mono: np.ndarray
    ) -> DualBusAudioOutput:
        """Render isolated Main PA and Ghost Preview audio frames (REQ-08)."""
        return self.audio_router.route_signals(
            main_mono=main_mono,
            ghost_mono=ghost_mono,
            ghost_active=self.ghost_preview_state.active,
        )

    def schedule_bar_with_spectral_critic(
        self,
        stem_freq_band_hz: tuple[float, float],
        scheduled_notes: tuple[ScheduledMidiNote, ...],
    ) -> SpectralCriticResult:
        """Run Pre-Play Spectral Critic (REQ-09) and dispatch carved MIDI notes over IAC."""
        result = evaluate_spectral_collision(
            stem_freq_band_hz=stem_freq_band_hz, scheduled_notes=scheduled_notes
        )
        for item in result.adjusted_notes:
            self.midi_port.send_note_on(
                note=item.note, velocity=item.velocity, trigger_sample=item.trigger_sample
            )
        return result
