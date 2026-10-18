"""Integration tests for Slice 3 (REQ-05, REQ-06): TensionOverrideUseCase and DSP Stem Mixer."""

import math

import numpy as np
import pytest
from future_music.application.tension_override_use_case import TensionOverrideUseCase
from future_music.domain.dsp_tension_drop import TensionDropController
from future_music.infrastructure.dsp_stem_mixer_adapter import SampleLockedStemMixerAdapter

from future_music.domain.clock_and_midi import SampleClock
from future_music.infrastructure.iac_midi_adapter import GarageBandIacMidiAdapter


def build_tension_setup(num_samples: int = 48000) -> tuple[
    TensionOverrideUseCase, SampleLockedStemMixerAdapter, GarageBandIacMidiAdapter
]:
    """Build TensionOverrideUseCase with synthetic 50 Hz kick + 2.5 kHz snare + 55 Hz bass stems."""
    t = np.arange(num_samples, dtype=np.float64) / 48000.0
    drums_buf = 0.7 * np.sin(2.0 * math.pi * 50.0 * t) + 0.3 * np.sin(2.0 * math.pi * 2500.0 * t)
    drums_buf[0] = 1.0
    bass_buf = 0.8 * np.sin(2.0 * math.pi * 55.0 * t)
    bass_buf[0] = 1.0

    mixer = SampleLockedStemMixerAdapter(sample_rate=48000)
    mixer.load_stem_buffer("drums", drums_buf)
    mixer.load_stem_buffer("bass", bass_buf)
    midi = GarageBandIacMidiAdapter()
    use_case = TensionOverrideUseCase(
        clock=SampleClock(sample_rate=48000, bpm=124.0),
        controller=TensionDropController(),
        stem_mixer=mixer,
        midi_port=midi,
    )
    return use_case, mixer, midi


def test_tension_override_use_case_mutes_bass_sweeps_drum_hpf_and_sends_cc1_over_iac() -> None:
    """REQ-05: Holding Spacebar mutes bass buffer, sweeps drum HPF to 350 Hz, and sends CC 1=127."""
    use_case, mixer, midi = build_tension_setup()
    state = use_case.hold_spacebar(progress=1.0)
    assert state.bass_gain == 0.0
    assert state.drums_hpf_hz == pytest.approx(350.0)
    assert midi.sent_messages[-1].control == 1
    assert midi.sent_messages[-1].value == 127

    mixed = use_case.render_audio_frame(frame_size=2048)
    assert mixer.bass_playhead == 2048
    assert float(np.max(np.abs(mixed))) > 0.1


def test_tension_override_use_case_on_beat_release_resets_drum_and_bass_playheads_to_zero() -> None:
    """REQ-06: Releasing Spacebar within 80ms of beat resets drum/bass playheads to Sample 0."""
    use_case, mixer, _ = build_tension_setup()
    use_case.render_audio_frame(frame_size=10000)
    assert mixer.drums_playhead == 10000

    use_case.hold_spacebar(progress=0.8)
    beat_1 = use_case.clock.beat_sample_index(1)
    use_case.clock.current_sample = beat_1 - 500
    state = use_case.release_spacebar()

    assert state.playhead_reset_triggered is True
    assert mixer.drums_playhead == 0
    assert mixer.bass_playhead == 0


def test_tension_override_mid_beat_release_renders_suck_back_zeros_then_sample_zero_drop() -> None:
    """REQ-06: Mid-beat release outputs exact zeros during suck-back then slams Sample 0."""
    use_case, mixer, _ = build_tension_setup()
    use_case.hold_spacebar(progress=1.0)
    beat_1 = use_case.clock.beat_sample_index(1)
    use_case.clock.current_sample = beat_1 - 5000

    state = use_case.release_spacebar()
    assert state.suck_back_active is True
    assert state.remaining_suck_back_samples == 5000

    frame = use_case.render_audio_frame(frame_size=6000)
    suck_back_portion = frame[:5000]
    post_drop_portion = frame[5000:]

    assert float(np.max(np.abs(suck_back_portion))) == 0.0
    assert float(np.max(np.abs(post_drop_portion))) > 0.5
    assert mixer.drums_playhead == 1000
    assert mixer.bass_playhead == 1000
