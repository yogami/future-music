"""Secondary port protocol for Stereo-Split and Dual-Device Ghost Preview audio routing."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class DualBusAudioOutput:
    """Rendered audio frames for STEREO_SPLIT (L=Ghost, R=Main) and DUAL_DEVICE buses."""

    stereo_split_frame: np.ndarray
    main_pa_stereo: np.ndarray
    ghost_preview_stereo: np.ndarray


class AudioRouterPort(Protocol):
    """Port interface isolating Ghost Preview cue from Main PA output (REQ-08)."""

    def route_signals(
        self, main_mono: np.ndarray, ghost_mono: np.ndarray, ghost_active: bool
    ) -> DualBusAudioOutput:
        """Route Main PA and Ghost Preview mono signals into isolated stereo outputs."""
