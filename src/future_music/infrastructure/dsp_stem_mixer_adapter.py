"""Infrastructure adapter for independent drums/bass stem playback and biquad HPF mixing."""

from dataclasses import dataclass, field

import numpy as np

from future_music.domain.clock_and_midi import DEFAULT_SAMPLE_RATE
from future_music.domain.dsp_tension_drop import apply_biquad_hpf, compute_biquad_hpf_coeffs


def _extract_looped_slice(
    buffer: np.ndarray, playhead: int, num_samples: int
) -> tuple[np.ndarray, int]:
    """Extract num_samples from a 1D buffer with modulo wrap-around and return new playhead."""
    if buffer.size == 0 or num_samples <= 0:
        return (np.zeros(max(0, num_samples), dtype=np.float64), playhead)
    indices = (np.arange(num_samples, dtype=np.int64) + playhead) % buffer.size
    new_playhead = (playhead + num_samples) % buffer.size
    return (buffer[indices], int(new_playhead))


@dataclass
class SampleLockedStemMixerAdapter:
    """Renders separate drums and bass stem buffers with real-time biquad HPF (REQ-05/06)."""

    sample_rate: int = DEFAULT_SAMPLE_RATE
    drums_buffer: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    bass_buffer: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    drums_playhead: int = 0
    bass_playhead: int = 0

    def load_stem_buffer(self, stem_name: str, buffer: np.ndarray) -> None:
        """Load a 1D float64 PCM buffer for 'drums' or 'bass'."""
        arr = np.asarray(buffer, dtype=np.float64)
        if stem_name == "drums":
            self.drums_buffer = arr
            self.drums_playhead = 0
            return
        if stem_name == "bass":
            self.bass_buffer = arr
            self.bass_playhead = 0
            return
        raise ValueError(f"Unsupported stem_name '{stem_name}'; expected 'drums' or 'bass'")

    def reset_playheads_to_zero(self) -> None:
        """Reset both drums and bass playheads to Sample 0 for transient alignment."""
        self.drums_playhead = 0
        self.bass_playhead = 0

    def render_mixed_chunk(
        self,
        num_samples: int,
        drums_gain: float,
        bass_gain: float,
        drums_hpf_hz: float,
    ) -> np.ndarray:
        """Render num_samples of HPF-filtered drums + gain-scaled bass and advance playheads."""
        drums_raw, self.drums_playhead = _extract_looped_slice(
            self.drums_buffer, self.drums_playhead, num_samples
        )
        bass_raw, self.bass_playhead = _extract_looped_slice(
            self.bass_buffer, self.bass_playhead, num_samples
        )
        coeffs = compute_biquad_hpf_coeffs(cutoff_hz=drums_hpf_hz, sample_rate=self.sample_rate)
        drums_filtered = apply_biquad_hpf(drums_raw, coeffs) * drums_gain
        bass_scaled = bass_raw * bass_gain
        return drums_filtered + bass_scaled
