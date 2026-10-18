"""Slice 3 (REQ-05, REQ-06): Real-Time Biquad HPF, Tension Hold, and Suck-Back Drop."""

import math
from dataclasses import dataclass

import numpy as np

from future_music.domain.clock_and_midi import (
    DEFAULT_CC1_MODULATION,
    DEFAULT_SAMPLE_RATE,
    MAX_MIDI_VAL,
    MS_PER_SECOND,
    SampleClock,
)
from future_music.domain.empath_memory import MAX_UNIT, MIN_UNIT, ControlGear

HPF_MIN_HZ: float = 20.0
HPF_MAX_TENSION_HZ: float = 350.0
ON_BEAT_WINDOW_MS: float = 80.0
BUTTERWORTH_Q: float = 1.0 / math.sqrt(2.0)


@dataclass(frozen=True)
class BiquadCoeffs:
    """Normalized 2nd-order digital biquad filter coefficients (a0 = 1.0)."""

    b0: float
    b1: float
    b2: float
    a1: float
    a2: float


def compute_biquad_hpf_coeffs(
    cutoff_hz: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    q_factor: float = BUTTERWORTH_Q,
) -> BiquadCoeffs:
    """Compute Audio EQ Cookbook 2nd-order high-pass biquad filter coefficients."""
    w0 = (2.0 * math.pi * cutoff_hz) / float(sample_rate)
    cos_w0 = math.cos(w0)
    alpha = math.sin(w0) / (2.0 * q_factor)
    a0 = 1.0 + alpha
    half_one_plus_cos = (1.0 + cos_w0) / 2.0
    return BiquadCoeffs(
        b0=half_one_plus_cos / a0,
        b1=-(1.0 + cos_w0) / a0,
        b2=half_one_plus_cos / a0,
        a1=(-2.0 * cos_w0) / a0,
        a2=(1.0 - alpha) / a0,
    )


def apply_biquad_hpf(signal: np.ndarray, coeffs: BiquadCoeffs) -> np.ndarray:
    """Apply normalized 2nd-order Direct Form II Transposed biquad high-pass filter."""
    out = np.empty(signal.shape, dtype=np.float64)
    z1, z2 = 0.0, 0.0
    for idx, sample in enumerate(signal):
        x_val = float(sample)
        y_val = coeffs.b0 * x_val + z1
        z1 = coeffs.b1 * x_val - coeffs.a1 * y_val + z2
        z2 = coeffs.b2 * x_val - coeffs.a2 * y_val
        out[idx] = y_val
    return out


def interpolate_hpf_cutoff(progress: float) -> float:
    """Interpolate drum HPF cutoff frequency in [20.0, 350.0] Hz for clamped progress."""
    clamped = min(MAX_UNIT, max(MIN_UNIT, progress))
    return HPF_MIN_HZ + clamped * (HPF_MAX_TENSION_HZ - HPF_MIN_HZ)


@dataclass(frozen=True)
class BeatReleasePlan:
    """Quantized release decision for Spacebar tension drop (REQ-06)."""

    is_on_beat: bool
    suck_back_samples: int
    drop_target_sample: int


def compute_beat_release_plan(
    release_sample: int,
    clock: SampleClock,
    window_ms: float = ON_BEAT_WINDOW_MS,
) -> BeatReleasePlan:
    """Determine if release is within +/-window_ms of a beat or needs a suck-back pause."""
    window_samples = round((window_ms * clock.sample_rate) / MS_PER_SECOND)
    nearest_beat = round(release_sample / clock.samples_per_beat)
    nearest_sample = clock.beat_sample_index(nearest_beat)
    if abs(release_sample - nearest_sample) <= window_samples:
        return BeatReleasePlan(
            is_on_beat=True, suck_back_samples=0, drop_target_sample=release_sample
        )
    next_beat = math.floor(release_sample / clock.samples_per_beat) + 1
    target_sample = clock.beat_sample_index(next_beat)
    suck_back = max(0, target_sample - release_sample)
    return BeatReleasePlan(
        is_on_beat=False, suck_back_samples=suck_back, drop_target_sample=target_sample
    )


@dataclass(frozen=True)
class TensionDropState:
    """Immutable snapshot of DSP stem gains, HPF cutoff, and suck-back drop state."""

    gear: ControlGear
    drums_gain: float
    bass_gain: float
    drums_hpf_hz: float
    cc1_modulation: int
    suck_back_active: bool
    remaining_suck_back_samples: int
    playhead_reset_triggered: bool


@dataclass
class TensionDropController:
    """Domain state machine for Gear 3 Spacebar Tension Hold and Suck-Back Drop."""

    gear: ControlGear = ControlGear.GEAR_1_AUTOPILOT
    drums_gain: float = MAX_UNIT
    bass_gain: float = MAX_UNIT
    drums_hpf_hz: float = HPF_MIN_HZ
    cc1_modulation: int = DEFAULT_CC1_MODULATION
    suck_back_active: bool = False
    remaining_suck_back_samples: int = 0
    playhead_reset_triggered: bool = False

    def engage_tension_hold(self, progress: float) -> TensionDropState:
        """Hold [SPACEBAR] (REQ-05): mute bass, sweep drum HPF 20->350Hz, ramp CC 1."""
        clamped = min(MAX_UNIT, max(MIN_UNIT, progress))
        self.gear = ControlGear.GEAR_3_TACTILE_OVERRIDE
        self.drums_gain = MAX_UNIT
        self.bass_gain = MIN_UNIT
        self.drums_hpf_hz = interpolate_hpf_cutoff(clamped)
        self.cc1_modulation = round(
            DEFAULT_CC1_MODULATION + clamped * (MAX_MIDI_VAL - DEFAULT_CC1_MODULATION)
        )
        self.suck_back_active = False
        self.remaining_suck_back_samples = 0
        self.playhead_reset_triggered = False
        return self.current_state()

    def release_tension_hold(
        self, release_sample: int, clock: SampleClock
    ) -> TensionDropState:
        """Release [SPACEBAR] (REQ-06): immediate Sample-0 drop if on-beat, else suck-back."""
        plan = compute_beat_release_plan(release_sample=release_sample, clock=clock)
        if plan.is_on_beat:
            return self.trigger_hard_cut()
        self.suck_back_active = True
        self.remaining_suck_back_samples = plan.suck_back_samples
        self.drums_gain = MIN_UNIT
        self.bass_gain = MIN_UNIT
        self.playhead_reset_triggered = False
        return self.current_state()

    def advance_suck_back(self, samples: int) -> TensionDropState:
        """Advance suck-back silence countdown and slam Sample 0 when countdown hits 0."""
        self.remaining_suck_back_samples = max(0, self.remaining_suck_back_samples - samples)
        if self.remaining_suck_back_samples == 0:
            return self.trigger_hard_cut()
        self.playhead_reset_triggered = False
        return self.current_state()

    def trigger_hard_cut(self) -> TensionDropState:
        """Execute Sample-0 transient phase reset and restore full club low-end."""
        self.gear = ControlGear.GEAR_1_AUTOPILOT
        self.drums_gain = MAX_UNIT
        self.bass_gain = MAX_UNIT
        self.drums_hpf_hz = HPF_MIN_HZ
        self.cc1_modulation = DEFAULT_CC1_MODULATION
        self.suck_back_active = False
        self.remaining_suck_back_samples = 0
        self.playhead_reset_triggered = True
        return self.current_state()

    def current_state(self) -> TensionDropState:
        """Return immutable snapshot of current tension and drop state."""
        return TensionDropState(
            gear=self.gear,
            drums_gain=self.drums_gain,
            bass_gain=self.bass_gain,
            drums_hpf_hz=self.drums_hpf_hz,
            cc1_modulation=self.cc1_modulation,
            suck_back_active=self.suck_back_active,
            remaining_suck_back_samples=self.remaining_suck_back_samples,
            playhead_reset_triggered=self.playhead_reset_triggered,
        )
