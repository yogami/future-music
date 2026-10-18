"""Slice 5 (REQ-10): Unified EmpathConductorFacade coordinating all 18 deck actions."""

from dataclasses import dataclass, field

from future_music.application.harmonic_clock_use_case import HarmonicClockUseCase
from future_music.application.memory_surfacing_use_case import SurfaceMemoryOrbsUseCase
from future_music.application.tension_override_use_case import TensionOverrideUseCase
from future_music.application.whim_and_bridge_use_case import WhimAndBridgeUseCase
from future_music.domain.clock_and_midi import (
    GarageBandExpressionState,
    ModalScaleEngine,
    SampleClock,
)
from future_music.domain.dsp_tension_drop import TensionDropController
from future_music.domain.empath_deck import (
    DeckActionCategory,
    DeckKeyCode,
    DeckKeySpec,
    resolve_deck_key,
)
from future_music.domain.empath_memory import (
    AutobiographicalEmotionPrism,
    ControlGear,
    EpisodicSetMemory,
    MemoryOrb,
    MoodVector,
    ProactiveCoPilotMonitor,
    ScaleMode,
)
from future_music.domain.ghost_and_bridge import BridgeStageName
from future_music.domain.whim_and_hum import (
    HumContourEvent,
    InputGateController,
    InputGateMode,
    QuantizedMidiNote,
    VoiceWhimIntent,
)
from future_music.infrastructure.audio_routing_adapter import DualChannelAudioRouterAdapter
from future_music.infrastructure.dsp_stem_mixer_adapter import SampleLockedStemMixerAdapter
from future_music.infrastructure.iac_midi_adapter import GarageBandIacMidiAdapter
from future_music.infrastructure.in_memory_vault import create_default_empath_vault


@dataclass(frozen=True)
class ConductorSnapshot:
    """Complete read-only state snapshot of the Empath Conductor AI DJ (REQ-10)."""

    bpm: float
    current_sample: int
    active_gear: ControlGear
    active_scale_mode: ScaleMode
    modal_lock_active: bool
    cc1_modulation: int
    note_density: float
    drums_hpf_hz: float
    input_gate_mode: InputGateMode
    surfaced_orbs: tuple[MemoryOrb, ...]
    staged_ghost_asset_id: str | None
    active_bridge_stage: BridgeStageName | None


@dataclass
class EmpathConductorFacade:
    """Unified application facade orchestrating all 5 slices and 18 MacBook Deck actions."""

    memory_use_case: SurfaceMemoryOrbsUseCase
    clock_use_case: HarmonicClockUseCase
    tension_use_case: TensionOverrideUseCase
    whim_use_case: WhimAndBridgeUseCase
    empath_id: str = "empath_euro_90s"
    active_gear: ControlGear = ControlGear.GEAR_1_AUTOPILOT
    surfaced_orbs: tuple[MemoryOrb, ...] = field(default_factory=tuple)

    def update_mood(self, mood: MoodVector, current_minutes: float = 0.0) -> tuple[MemoryOrb, ...]:
        """Update Empath MoodVector and surface top-3 autobiographical Memory Orbs."""
        orbs = self.memory_use_case.surface_orbs(self.empath_id, mood, current_minutes)
        self.surfaced_orbs = orbs
        return orbs

    def _exec_orb_select(self, spec: DeckKeySpec) -> None:
        """Stage surfaced Memory Orb corresponding to spec.orb_slot in Ghost Preview."""
        for orb in self.surfaced_orbs:
            if orb.orb_slot == spec.orb_slot:
                self.whim_use_case.stage_asset_to_ghost_preview(orb.asset.asset_id)
                self.active_gear = ControlGear.GEAR_2_WHIM_STEERING
                return
        raise RuntimeError(f"No surfaced Memory Orb at slot {spec.orb_slot}")

    def _exec_input_gate(self, spec: DeckKeySpec) -> None:
        """Open or close Hold V / Hold H microphone gates."""
        gate = self.whim_use_case.input_gate
        actions = {
            DeckKeyCode.HOLD_V: gate.open_voice_gate,
            DeckKeyCode.RELEASE_V: gate.close_gate,
            DeckKeyCode.HOLD_H: gate.open_hum_gate,
            DeckKeyCode.RELEASE_H: gate.close_gate,
        }
        actions[spec.key_code]()
        if spec.target_gear is not None:
            self.active_gear = spec.target_gear

    def _exec_ghost_decision(self, spec: DeckKeySpec, elapsed_minutes: float) -> None:
        """Commit (RETURN) or discard (BACKSPACE) staged Ghost Preview."""
        if spec.key_code == DeckKeyCode.RETURN:
            self.whim_use_case.commit_ghost_preview(elapsed_minutes=elapsed_minutes)
        else:
            self.whim_use_case.discard_ghost_preview()
        self.active_gear = ControlGear.GEAR_1_AUTOPILOT

    def _exec_expression_step(self, spec: DeckKeySpec) -> None:
        """Execute UP/DOWN CC 1 filter morph or LEFT/RIGHT note density step."""
        pos = self.clock_use_case.clock.current_sample
        actions = {
            DeckKeyCode.UP: lambda: self.clock_use_case.step_filter_up(pos),
            DeckKeyCode.DOWN: lambda: self.clock_use_case.step_filter_down(pos),
            DeckKeyCode.LEFT: self.clock_use_case.step_density_left,
            DeckKeyCode.RIGHT: self.clock_use_case.step_density_right,
        }
        actions[spec.key_code]()

    def _exec_tension_override(self, spec: DeckKeySpec, hold_progress: float) -> None:
        """Execute HOLD_SPACE tension sweep or RELEASE_SPACE quantized suck-back drop."""
        if spec.key_code == DeckKeyCode.HOLD_SPACE:
            state = self.tension_use_case.hold_spacebar(progress=hold_progress)
        else:
            state = self.tension_use_case.release_spacebar()
        self.active_gear = state.gear

    def _exec_scale_shift(self, spec: DeckKeySpec) -> None:
        """Shift active scale mode in HarmonicClockUseCase."""
        target = spec.scale_mode if spec.scale_mode is not None else ScaleMode.NATURAL_MINOR
        self.clock_use_case.shift_scale_mode(target)

    def handle_deck_key(
        self, raw_key: str, hold_progress: float = 1.0, elapsed_minutes: float = 0.0
    ) -> ConductorSnapshot:
        """Dispatch any of the 18 MacBook Keyboard Empath Deck actions and return snapshot."""
        spec = resolve_deck_key(raw_key)
        cat = DeckActionCategory
        handlers = {
            cat.ORB_SELECT: lambda: self._exec_orb_select(spec),
            cat.SCALE_SHIFT: lambda: self._exec_scale_shift(spec),
            cat.INPUT_GATE: lambda: self._exec_input_gate(spec),
            cat.GHOST_DECISION: lambda: self._exec_ghost_decision(spec, elapsed_minutes),
            cat.EXPRESSION_STEP: lambda: self._exec_expression_step(spec),
            cat.TENSION_OVERRIDE: lambda: self._exec_tension_override(spec, hold_progress),
        }
        handlers[spec.category]()
        return self.snapshot()

    def submit_voice_whim(self, transcript: str) -> VoiceWhimIntent:
        """Delegate gated voice whim transcript to WhimAndBridgeUseCase."""
        return self.whim_use_case.submit_voice_whim(transcript)

    def submit_hum_contour(
        self, events: tuple[HumContourEvent, ...]
    ) -> tuple[QuantizedMidiNote, ...]:
        """Delegate gated hum contour events to WhimAndBridgeUseCase."""
        return self.whim_use_case.submit_hum_contour(events)

    def snapshot(self) -> ConductorSnapshot:
        """Build a serializable ConductorSnapshot of all live engine state."""
        return _build_snapshot(self)


def _build_snapshot(facade: EmpathConductorFacade) -> ConductorSnapshot:
    """Construct ConductorSnapshot from an EmpathConductorFacade instance."""
    plan = facade.whim_use_case.active_bridge_plan
    stage = plan.stages[0].stage_name if plan is not None else None
    return ConductorSnapshot(
        bpm=facade.clock_use_case.clock.bpm,
        current_sample=facade.clock_use_case.clock.current_sample,
        active_gear=facade.active_gear,
        active_scale_mode=facade.clock_use_case.modal_engine.active_scale_mode,
        modal_lock_active=facade.clock_use_case.modal_engine.modal_lock_active,
        cc1_modulation=facade.clock_use_case.expression.cc1_modulation,
        note_density=facade.clock_use_case.expression.note_density,
        drums_hpf_hz=facade.tension_use_case.controller.drums_hpf_hz,
        input_gate_mode=facade.whim_use_case.input_gate.active_mode,
        surfaced_orbs=facade.surfaced_orbs,
        staged_ghost_asset_id=facade.whim_use_case.ghost_preview_state.staged_asset_id,
        active_bridge_stage=stage,
    )


def _build_whim_use_case(
    empath_id: str,
    episodic: EpisodicSetMemory,
    clock: SampleClock,
    modal: ModalScaleEngine,
    midi: GarageBandIacMidiAdapter,
) -> WhimAndBridgeUseCase:
    """Create WhimAndBridgeUseCase sharing clock, modal engine, and episodic memory."""
    return WhimAndBridgeUseCase(
        vault=create_default_empath_vault(),
        audio_router=DualChannelAudioRouterAdapter(),
        midi_port=midi,
        empath_id=empath_id,
        episodic_memory=episodic,
        clock=clock,
        modal_engine=modal,
        input_gate=InputGateController(),
    )


def build_default_conductor_facade(
    empath_id: str = "empath_euro_90s",
) -> EmpathConductorFacade:
    """Assemble an EmpathConductorFacade with shared clock, modal engine, and adapters."""
    episodic = EpisodicSetMemory(window_minutes=60.0)
    clock = SampleClock(sample_rate=48000, bpm=124.0)
    modal = ModalScaleEngine()
    midi = GarageBandIacMidiAdapter()
    whim_uc = _build_whim_use_case(empath_id, episodic, clock, modal, midi)
    mem_uc = SurfaceMemoryOrbsUseCase(
        whim_uc.vault, AutobiographicalEmotionPrism(), episodic, ProactiveCoPilotMonitor()
    )
    clock_uc = HarmonicClockUseCase(clock, GarageBandExpressionState(), modal, midi)
    tension_uc = TensionOverrideUseCase(
        clock, TensionDropController(), SampleLockedStemMixerAdapter(48000), midi
    )
    facade = EmpathConductorFacade(mem_uc, clock_uc, tension_uc, whim_uc, empath_id=empath_id)
    facade.update_mood(MoodVector(0.35, 0.55, 0.30, 0.95), current_minutes=0.0)
    return facade
