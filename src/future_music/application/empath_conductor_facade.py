"""Slice 5 & 6 (REQ-10, REQ-12): Unified EmpathConductorFacade with live audio transport."""

from dataclasses import dataclass, field

import numpy as np

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
from future_music.domain.stem_synthesis import (
    compute_bridge_stage_for_bar,
    synthesize_default_124bpm_stems,
)
from future_music.domain.whim_and_hum import (
    HumContourEvent,
    InputGateController,
    InputGateMode,
    QuantizedMidiNote,
    VoiceWhimIntent,
)
from future_music.infrastructure.audio_routing_adapter import DualChannelAudioRouterAdapter
from future_music.infrastructure.dsp_stem_mixer_adapter import (
    SampleLockedStemMixerAdapter,
    _extract_looped_slice,
)
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
    """Unified application facade orchestrating all slices, stems, and 18 Deck actions."""

    memory_use_case: SurfaceMemoryOrbsUseCase
    clock_use_case: HarmonicClockUseCase
    tension_use_case: TensionOverrideUseCase
    whim_use_case: WhimAndBridgeUseCase
    empath_id: str = "empath_euro_90s"
    active_gear: ControlGear = ControlGear.GEAR_1_AUTOPILOT
    surfaced_orbs: tuple[MemoryOrb, ...] = field(default_factory=tuple)
    melodic_stems: dict[str, np.ndarray] = field(default_factory=dict)
    melodic_playhead: int = 0
    ghost_playhead: int = 0
    bridge_start_bar: int = 1

    def update_mood(self, mood: MoodVector, current_minutes: float = 0.0) -> tuple[MemoryOrb, ...]:
        """Update Empath MoodVector and surface top-3 autobiographical Memory Orbs."""
        orbs = self.memory_use_case.surface_orbs(self.empath_id, mood, current_minutes)
        self.surfaced_orbs = orbs
        return self.surfaced_orbs

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
            clk = self.clock_use_case.clock
            self.bridge_start_bar = clk.current_bar_and_beat(clk.current_sample)[0]
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

    def render_live_stereo_frame(self, frame_samples: int = 4800) -> np.ndarray:
        """Render stereo audio (Left=Ghost, Right=Main PA) and advance live transport."""
        return _render_live_frame_for_facade(self, frame_samples)

    def snapshot(self) -> ConductorSnapshot:
        """Build a serializable ConductorSnapshot of all live engine state."""
        return _build_snapshot(self)


def _schedule_bar_motif_if_needed(
    facade: EmpathConductorFacade, prev_sample: int, curr_sample: int
) -> None:
    """Dispatch pre-fired bar MIDI motif to GarageBand at bar start or bar crossing."""
    clk = facade.clock_use_case.clock
    prev_bar = clk.current_bar_and_beat(prev_sample)[0]
    curr_bar = clk.current_bar_and_beat(curr_sample)[0]
    if prev_sample == 0:
        facade.clock_use_case.schedule_bar_motif(bar_start_beat=0, velocity=100)
    if curr_bar > prev_bar:
        facade.clock_use_case.schedule_bar_motif(bar_start_beat=(curr_bar - 1) * 4, velocity=100)


def _render_live_frame_for_facade(
    facade: EmpathConductorFacade, frame_samples: int
) -> np.ndarray:
    """Mix drums, bass, active melodic motif (scaled by modal gain), and ghost cue."""
    prev_sample = facade.clock_use_case.clock.current_sample
    rhythm_mono = facade.tension_use_case.render_audio_frame(frame_size=frame_samples)
    _schedule_bar_motif_if_needed(facade, prev_sample, facade.clock_use_case.clock.current_sample)
    modal = facade.clock_use_case.modal_engine
    melodic_id = modal.active_asset.asset_id if modal.active_asset else "robert_miles_children"
    mel_buf = facade.melodic_stems.get(melodic_id, np.zeros(0, dtype=np.float64))
    mel_slice, facade.melodic_playhead = _extract_looped_slice(
        mel_buf, facade.melodic_playhead, frame_samples
    )
    main_mono = rhythm_mono + (mel_slice * modal.current_state().audio_stem_gain)
    staged_id = facade.whim_use_case.ghost_preview_state.staged_asset_id or ""
    ghost_buf = facade.melodic_stems.get(staged_id, np.zeros(0, dtype=np.float64))
    ghost_mono, facade.ghost_playhead = _extract_looped_slice(
        ghost_buf, facade.ghost_playhead, frame_samples
    )
    return facade.whim_use_case.render_dual_bus_audio(main_mono, ghost_mono).stereo_split_frame


def _active_stage_for_facade(facade: EmpathConductorFacade) -> BridgeStageName | None:
    """Compute the active BridgeStageName for facade's current bar position."""
    clk = facade.clock_use_case.clock
    cur_bar = clk.current_bar_and_beat(clk.current_sample)[0]
    return compute_bridge_stage_for_bar(
        facade.whim_use_case.active_bridge_plan, (cur_bar - facade.bridge_start_bar) + 1
    )


def _build_snapshot(facade: EmpathConductorFacade) -> ConductorSnapshot:
    """Construct ConductorSnapshot from an EmpathConductorFacade instance."""
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
        active_bridge_stage=_active_stage_for_facade(facade),
    )


def _build_whim_use_case(
    empath_id: str,
    episodic: EpisodicSetMemory,
    clock: SampleClock,
    modal: ModalScaleEngine,
    midi: GarageBandIacMidiAdapter,
) -> WhimAndBridgeUseCase:
    """Create WhimAndBridgeUseCase and activate default Robert Miles asset on modal."""
    vault = create_default_empath_vault()
    assets = vault.get_assets(empath_id)
    if assets:
        modal.activate_asset(assets[0])
    return WhimAndBridgeUseCase(
        vault, DualChannelAudioRouterAdapter(), midi, empath_id,
        episodic, clock, modal, InputGateController(),
    )


def _build_loaded_stem_mixer(stems: dict[str, np.ndarray]) -> SampleLockedStemMixerAdapter:
    """Create a SampleLockedStemMixerAdapter pre-loaded with 124 BPM drums and bass."""
    mixer = SampleLockedStemMixerAdapter(sample_rate=48000)
    mixer.load_stem_buffer("drums", stems["drums"])
    mixer.load_stem_buffer("bass", stems["bass"])
    return mixer


def build_default_conductor_facade(empath_id: str = "empath_euro_90s") -> EmpathConductorFacade:
    """Assemble an EmpathConductorFacade pre-loaded with 124 BPM stems and CoreMIDI."""
    episodic = EpisodicSetMemory(window_minutes=60.0)
    clock, modal, midi = SampleClock(48000, 124.0), ModalScaleEngine(), GarageBandIacMidiAdapter()
    stems = synthesize_default_124bpm_stems(sample_rate=48000, bpm=124.0)
    whim_uc = _build_whim_use_case(empath_id, episodic, clock, modal, midi)
    mem_uc = SurfaceMemoryOrbsUseCase(
        whim_uc.vault, AutobiographicalEmotionPrism(), episodic, ProactiveCoPilotMonitor()
    )
    clock_uc = HarmonicClockUseCase(clock, GarageBandExpressionState(), modal, midi)
    tension_uc = TensionOverrideUseCase(
        clock, TensionDropController(), _build_loaded_stem_mixer(stems), midi
    )
    facade = EmpathConductorFacade(
        mem_uc, clock_uc, tension_uc, whim_uc, empath_id=empath_id, melodic_stems=stems
    )
    facade.update_mood(MoodVector(0.35, 0.55, 0.30, 0.95), current_minutes=0.0)
    return facade
