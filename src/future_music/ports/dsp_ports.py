"""Secondary port protocol for independent stem DSP mixing and playhead phase reset."""

from typing import Protocol

import numpy as np


class StemMixerPort(Protocol):
    """Port interface for independent drums/bass stem playback and biquad filtering."""

    def load_stem_buffer(self, stem_name: str, buffer: np.ndarray) -> None:
        """Load a 1D float64 PCM buffer for 'drums' or 'bass'."""

    def reset_playheads_to_zero(self) -> None:
        """Reset both 'drums' and 'bass' playheads to Sample 0 for transient alignment."""

    def render_mixed_chunk(
        self,
        num_samples: int,
        drums_gain: float,
        bass_gain: float,
        drums_hpf_hz: float,
    ) -> np.ndarray:
        """Render num_samples of HPF-filtered drums + gain-scaled bass and advance playheads."""
