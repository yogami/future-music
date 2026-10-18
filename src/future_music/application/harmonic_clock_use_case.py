"""Slice 2 (REQ-02, REQ-03, REQ-04) application use case for clock, MIDI, and modal safety."""

from dataclasses import dataclass

from future_music.domain.clock_and_midi import (
    GarageBandExpressionState,
    ModalRenderState,
    ModalScaleEngine,
    SampleClock,
    ScheduledMidiNote,
)
from future_music.domain.empath_memory import MemoryAsset, ScaleMode
from future_music.ports.midi_ports import MidiOutputPort


@dataclass
class HarmonicClockUseCase:
    """Coordinates SampleClock, GarageBandExpressionState, ModalScaleEngine, and MidiOutputPort."""

    clock: SampleClock
    expression: GarageBandExpressionState
    modal_engine: ModalScaleEngine
    midi_port: MidiOutputPort

    def load_active_asset(self, asset: MemoryAsset) -> ModalRenderState:
        """Activate a MemoryAsset in the modal scale engine."""
        return self.modal_engine.activate_asset(asset)

    def shift_scale_mode(self, target_mode: ScaleMode) -> ModalRenderState:
        """Shift active scale mode, enforcing Modal Lock on microtonal/vocal stems."""
        return self.modal_engine.shift_scale(target_mode)

    def step_filter_up(self, sample_pos: int) -> int:
        """Step GarageBand Alchemy filter cutoff up (+16 CC 1) and dispatch to IAC."""
        event = self.expression.morph_filter_up()
        self.midi_port.send_cc(event.cc_number, event.value, trigger_sample=sample_pos)
        return event.value

    def step_filter_down(self, sample_pos: int) -> int:
        """Step GarageBand Alchemy filter cutoff down (-16 CC 1) and dispatch to IAC."""
        event = self.expression.morph_filter_down()
        self.midi_port.send_cc(event.cc_number, event.value, trigger_sample=sample_pos)
        return event.value

    def step_density_left(self) -> float:
        """Decrease MIDI note density by 0.25 ([LEFT] key)."""
        return self.expression.adjust_note_density_left()

    def step_density_right(self) -> float:
        """Increase MIDI note density by 0.25 ([RIGHT] key)."""
        return self.expression.adjust_note_density_right()

    def schedule_bar_motif(
        self, bar_start_beat: int, velocity: int
    ) -> tuple[ScheduledMidiNote, ...]:
        """Schedule mutated motif notes with -18ms pre-fire offset and density filtering."""
        state = self.modal_engine.current_state()
        filtered = self.expression.filter_notes_by_density(state.mutated_midi_motif)
        scheduled: list[ScheduledMidiNote] = []
        for idx, note in enumerate(filtered):
            trig = self.clock.midi_trigger_sample(bar_start_beat + idx)
            self.midi_port.send_note_on(note=note, velocity=velocity, trigger_sample=trig)
            scheduled.append(ScheduledMidiNote(note=note, velocity=velocity, trigger_sample=trig))
        return tuple(scheduled)
