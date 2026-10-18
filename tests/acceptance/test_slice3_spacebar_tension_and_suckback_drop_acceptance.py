"""Acceptance tests for Slice 3 (REQ-05, REQ-06): Spacebar Tension Hold & Suck-Back Drop."""

import math

import numpy as np
from future_music.application.tension_override_use_case import TensionOverrideUseCase
from future_music.domain.dsp_tension_drop import TensionDropController
from future_music.infrastructure.dsp_stem_mixer_adapter import SampleLockedStemMixerAdapter

from future_music.domain.clock_and_midi import SampleClock
from future_music.infrastructure.iac_midi_adapter import GarageBandIacMidiAdapter


def test_req05_spacebar_hold_strips_50hz_kick_and_mutes_bass_while_keeping_snare_sizzle() -> None:
    """REQ-05: Full tension hold removes >85% of low-end kick/bass energy while keeping snare."""
    sample_rate = 48000
    t = np.arange(9600, dtype=np.float64) / float(sample_rate)
    kick_only = np.sin(2.0 * math.pi * 50.0 * t)
    snare_only = np.sin(2.0 * math.pi * 2500.0 * t)
    bass_only = np.sin(2.0 * math.pi * 55.0 * t)

    mixer = SampleLockedStemMixerAdapter(sample_rate=sample_rate)
    mixer.load_stem_buffer("drums", kick_only + snare_only)
    mixer.load_stem_buffer("bass", bass_only)
    use_case = TensionOverrideUseCase(
        clock=SampleClock(sample_rate=sample_rate, bpm=124.0),
        controller=TensionDropController(),
        stem_mixer=mixer,
        midi_port=GarageBandIacMidiAdapter(),
    )

    normal_out = use_case.render_audio_frame(frame_size=4800)
    normal_rms = float(np.sqrt(np.mean(np.square(normal_out[480:]))))

    use_case.hold_spacebar(progress=1.0)
    tension_out = use_case.render_audio_frame(frame_size=4800)
    tension_rms = float(np.sqrt(np.mean(np.square(tension_out[480:]))))
    snare_rms = float(np.sqrt(np.mean(np.square(snare_only[480:]))))

    assert tension_rms < normal_rms * 0.65
    assert abs(tension_rms - snare_rms) < 0.08


def test_req06_mid_beat_spacebar_release_executes_club_suck_back_pause_and_downbeat_slam() -> None:
    """REQ-06: Releasing mid-beat cuts to dead silence until next beat then slams Sample 0."""
    sample_rate = 48000
    t = np.arange(48000, dtype=np.float64) / float(sample_rate)
    drums = 0.6 * np.sin(2.0 * math.pi * 60.0 * t)
    bass = 0.6 * np.sin(2.0 * math.pi * 55.0 * t)
    drums[0] = 1.0
    bass[0] = 1.0

    mixer = SampleLockedStemMixerAdapter(sample_rate=sample_rate)
    mixer.load_stem_buffer("drums", drums)
    mixer.load_stem_buffer("bass", bass)
    midi = GarageBandIacMidiAdapter()
    use_case = TensionOverrideUseCase(
        clock=SampleClock(sample_rate=sample_rate, bpm=124.0),
        controller=TensionDropController(),
        stem_mixer=mixer,
        midi_port=midi,
    )

    use_case.render_audio_frame(frame_size=12000)
    use_case.hold_spacebar(progress=1.0)
    beat_1 = use_case.clock.beat_sample_index(1)
    suck_back_len = beat_1 - use_case.clock.current_sample
    assert suck_back_len > 4000

    use_case.release_spacebar()
    drop_window = use_case.render_audio_frame(frame_size=suck_back_len + 1000)

    assert np.all(drop_window[:suck_back_len] == 0.0)
    assert drop_window[suck_back_len] > 1.5
    assert midi.sent_messages[-1].control == 1
    assert midi.sent_messages[-1].value == 32
