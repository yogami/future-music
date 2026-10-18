"""Unit tests for Slice 3 (REQ-05, REQ-06): Biquad HPF, Tension Hold, and Suck-Back Drop."""

import math

import numpy as np
import pytest
from future_music.domain.dsp_tension_drop import (
    HPF_MAX_TENSION_HZ,
    HPF_MIN_HZ,
    ON_BEAT_WINDOW_MS,
    TensionDropController,
    apply_biquad_hpf,
    compute_beat_release_plan,
    compute_biquad_hpf_coeffs,
    interpolate_hpf_cutoff,
)

from future_music.domain.clock_and_midi import DEFAULT_CC1_MODULATION, MAX_MIDI_VAL, SampleClock
from future_music.domain.empath_memory import ControlGear


def generate_sine_wave(
    freq_hz: float, sample_rate: int = 48000, num_samples: int = 4800
) -> np.ndarray:
    """Generate a deterministic unit-amplitude sine wave at freq_hz."""
    t = np.arange(num_samples, dtype=np.float64) / float(sample_rate)
    return np.sin(2.0 * math.pi * freq_hz * t)


def rms_energy(signal: np.ndarray) -> float:
    """Compute root-mean-square amplitude of a 1D numpy signal."""
    return float(np.sqrt(np.mean(np.square(signal))))


def test_biquad_hpf_coeffs_at_350hz_attenuate_50hz_kick_while_preserving_2500hz_snare() -> None:
    """REQ-05: Butterworth HPF at 350 Hz strips 50 Hz kick (<10%) and keeps 2.5 kHz (>95%)."""
    coeffs = compute_biquad_hpf_coeffs(cutoff_hz=HPF_MAX_TENSION_HZ, sample_rate=48000)
    kick_50hz = generate_sine_wave(50.0)
    snare_2500hz = generate_sine_wave(2500.0)

    filtered_kick = apply_biquad_hpf(kick_50hz, coeffs)
    filtered_snare = apply_biquad_hpf(snare_2500hz, coeffs)

    kick_ratio = rms_energy(filtered_kick[480:]) / rms_energy(kick_50hz[480:])
    snare_ratio = rms_energy(filtered_snare[480:]) / rms_energy(snare_2500hz[480:])
    assert kick_ratio < 0.10
    assert snare_ratio > 0.95


def test_biquad_hpf_coeffs_at_20hz_preserve_50hz_kick_subbass_energy() -> None:
    """REQ-05: At baseline 20 Hz HPF cutoff, 50 Hz kick passes with >90% RMS amplitude."""
    coeffs = compute_biquad_hpf_coeffs(cutoff_hz=HPF_MIN_HZ, sample_rate=48000)
    kick_50hz = generate_sine_wave(50.0)
    filtered_kick = apply_biquad_hpf(kick_50hz, coeffs)
    kick_ratio = rms_energy(filtered_kick[480:]) / rms_energy(kick_50hz[480:])
    assert kick_ratio > 0.90


def test_tension_controller_hold_transitions_to_gear3_mutes_bass_and_sweeps_hpf_and_cc1() -> None:
    """REQ-05: Holding Spacebar engages GEAR_3, sets bass_gain=0.0, and sweeps HPF + CC 1."""
    ctrl = TensionDropController()
    state_half = ctrl.engage_tension_hold(progress=0.5)
    assert state_half.gear == ControlGear.GEAR_3_TACTILE_OVERRIDE
    assert state_half.bass_gain == 0.0
    assert state_half.drums_hpf_hz == pytest.approx(185.0)
    assert state_half.cc1_modulation == round(DEFAULT_CC1_MODULATION + 0.5 * (127 - 32))

    state_full = ctrl.engage_tension_hold(progress=1.0)
    assert state_full.drums_hpf_hz == pytest.approx(HPF_MAX_TENSION_HZ)
    assert state_full.cc1_modulation == MAX_MIDI_VAL


def test_tension_controller_sweep_progress_clamps_between_0_and_1() -> None:
    """REQ-05: interpolate_hpf_cutoff clamps progress outside [0.0, 1.0]."""
    assert interpolate_hpf_cutoff(-0.5) == pytest.approx(HPF_MIN_HZ)
    assert interpolate_hpf_cutoff(1.8) == pytest.approx(HPF_MAX_TENSION_HZ)


def test_beat_grid_distance_identifies_within_80ms_before_or_after_beat_as_on_beat() -> None:
    """REQ-06: Release within +/-80 ms (3840 samples at 48 kHz) of a beat is on-beat."""
    clock = SampleClock(sample_rate=48000, bpm=124.0)
    beat_2_sample = clock.beat_sample_index(2)
    plan_early = compute_beat_release_plan(
        release_sample=beat_2_sample - 2000, clock=clock, window_ms=ON_BEAT_WINDOW_MS
    )
    plan_late = compute_beat_release_plan(
        release_sample=beat_2_sample + 2000, clock=clock, window_ms=ON_BEAT_WINDOW_MS
    )
    assert plan_early.is_on_beat is True
    assert plan_early.suck_back_samples == 0
    assert plan_late.is_on_beat is True
    assert plan_late.suck_back_samples == 0


def test_beat_grid_distance_identifies_mid_beat_release_and_computes_suck_back_samples() -> None:
    """REQ-06: Release at Beat 2.5 (>80ms from beat) triggers suck-back until Beat 3."""
    clock = SampleClock(sample_rate=48000, bpm=124.0)
    beat_2_sample = clock.beat_sample_index(2)
    beat_3_sample = clock.beat_sample_index(3)
    mid_sample = (beat_2_sample + beat_3_sample) // 2

    plan = compute_beat_release_plan(release_sample=mid_sample, clock=clock)
    assert plan.is_on_beat is False
    assert plan.suck_back_samples == beat_3_sample - mid_sample
    assert plan.drop_target_sample == beat_3_sample


def test_on_beat_release_immediately_resets_playheads_to_sample_zero_without_suck_back() -> None:
    """REQ-06: On-beat release resets playheads to 0, restores bass=1.0 and HPF=20Hz immediately."""
    clock = SampleClock(sample_rate=48000, bpm=124.0)
    ctrl = TensionDropController()
    ctrl.engage_tension_hold(progress=0.8)
    beat_4 = clock.beat_sample_index(4)

    state = ctrl.release_tension_hold(release_sample=beat_4 - 1000, clock=clock)
    assert state.suck_back_active is False
    assert state.remaining_suck_back_samples == 0
    assert state.playhead_reset_triggered is True
    assert state.bass_gain == 1.0
    assert state.drums_hpf_hz == pytest.approx(HPF_MIN_HZ)
    assert state.gear == ControlGear.GEAR_1_AUTOPILOT


def test_mid_beat_release_activates_suck_back_silence_and_slams_sample_zero_on_next_beat() -> None:
    """REQ-06: Mid-beat release enters suck-back silence (drums_gain=0, bass_gain=0)."""
    clock = SampleClock(sample_rate=48000, bpm=124.0)
    ctrl = TensionDropController()
    ctrl.engage_tension_hold(progress=1.0)
    mid_sample = clock.beat_sample_index(1) + 10000

    state = ctrl.release_tension_hold(release_sample=mid_sample, clock=clock)
    assert state.suck_back_active is True
    assert state.drums_gain == 0.0
    assert state.bass_gain == 0.0
    assert state.playhead_reset_triggered is False


def test_suck_back_countdown_transitions_from_silence_to_full_drop_at_exact_boundary() -> None:
    """REQ-06: Advancing through remaining_suck_back_samples triggers Sample 0 phase reset."""
    clock = SampleClock(sample_rate=48000, bpm=124.0)
    ctrl = TensionDropController()
    ctrl.engage_tension_hold(progress=1.0)
    mid_sample = clock.beat_sample_index(2) - 5000
    ctrl.release_tension_hold(release_sample=mid_sample, clock=clock)

    partial = ctrl.advance_suck_back(2000)
    assert partial.suck_back_active is True
    assert partial.remaining_suck_back_samples == 3000

    completed = ctrl.advance_suck_back(3000)
    assert completed.suck_back_active is False
    assert completed.playhead_reset_triggered is True
    assert completed.drums_gain == 1.0
    assert completed.bass_gain == 1.0
    assert completed.drums_hpf_hz == pytest.approx(HPF_MIN_HZ)


def test_hard_cut_override_resets_phase_and_drops_immediately_on_next_beat() -> None:
    """Gear 3 hard cut immediately resets playhead to Sample 0 and restores full club low-end."""
    ctrl = TensionDropController()
    ctrl.engage_tension_hold(progress=0.9)
    state = ctrl.trigger_hard_cut()
    assert state.playhead_reset_triggered is True
    assert state.suck_back_active is False
    assert state.drums_gain == 1.0
    assert state.bass_gain == 1.0
    assert state.drums_hpf_hz == pytest.approx(HPF_MIN_HZ)
