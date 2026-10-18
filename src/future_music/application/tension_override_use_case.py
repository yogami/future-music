"""Application use case for Slice 3 (REQ-05, REQ-06): Spacebar Tension & Suck-Back Drop."""

from dataclasses import dataclass

import numpy as np

from future_music.domain.clock_and_midi import CC1_MOD_WHEEL, SampleClock
from future_music.domain.dsp_tension_drop import TensionDropController, TensionDropState
from future_music.ports.dsp_ports import StemMixerPort
from future_music.ports.midi_ports import MidiOutputPort


@dataclass
class TensionOverrideUseCase:
    """Orchestrates Gear 3 Spacebar tension hold, stem HPF, and quantized suck-back drop."""

    clock: SampleClock
    controller: TensionDropController
    stem_mixer: StemMixerPort
    midi_port: MidiOutputPort

    def hold_spacebar(self, progress: float) -> TensionDropState:
        """Engage or update [SPACEBAR] hold (REQ-05) and transmit CC 1 over IAC."""
        state = self.controller.engage_tension_hold(progress=progress)
        self.midi_port.send_cc(
            cc_number=CC1_MOD_WHEEL,
            value=state.cc1_modulation,
            trigger_sample=self.clock.current_sample,
        )
        return state

    def release_spacebar(self) -> TensionDropState:
        """Release [SPACEBAR] (REQ-06): on-beat Sample-0 drop or mid-beat suck-back pause."""
        state = self.controller.release_tension_hold(
            release_sample=self.clock.current_sample, clock=self.clock
        )
        if state.playhead_reset_triggered:
            self._execute_drop_reset(state.cc1_modulation, self.clock.current_sample)
        return state

    def _execute_drop_reset(self, cc1_value: int, trigger_sample: int) -> None:
        """Reset stem playheads to Sample 0 and restore baseline CC 1 modulation."""
        self.stem_mixer.reset_playheads_to_zero()
        self.midi_port.send_cc(
            cc_number=CC1_MOD_WHEEL, value=cc1_value, trigger_sample=trigger_sample
        )

    def _render_active_chunk(self, num_samples: int) -> np.ndarray:
        """Advance master sample clock and render mixed stems for num_samples."""
        self.clock.advance_samples(num_samples)
        return self.stem_mixer.render_mixed_chunk(
            num_samples=num_samples,
            drums_gain=self.controller.drums_gain,
            bass_gain=self.controller.bass_gain,
            drums_hpf_hz=self.controller.drums_hpf_hz,
        )

    def _render_suck_back_frame(self, frame_size: int) -> np.ndarray:
        """Render exact zero-amplitude suck-back pause and slam Sample 0 at beat boundary."""
        silence_len = min(frame_size, self.controller.remaining_suck_back_samples)
        silence_buf = np.zeros(silence_len, dtype=np.float64)
        self.clock.advance_samples(silence_len)
        state = self.controller.advance_suck_back(silence_len)
        if state.playhead_reset_triggered:
            self._execute_drop_reset(state.cc1_modulation, self.clock.current_sample)
        remainder = frame_size - silence_len
        if remainder <= 0:
            return silence_buf
        post_drop = self._render_active_chunk(remainder)
        return np.concatenate((silence_buf, post_drop))

    def render_audio_frame(self, frame_size: int) -> np.ndarray:
        """Render one audio callback frame (REQ-05/06) honoring suck-back pause if active."""
        if self.controller.suck_back_active:
            return self._render_suck_back_frame(frame_size)
        return self._render_active_chunk(frame_size)
