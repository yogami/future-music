"""Infrastructure adapter for Stereo-Split and Dual-Device Ghost Preview audio routing."""

from dataclasses import dataclass

import numpy as np

from future_music.domain.ghost_and_bridge import PreviewRoutingMode
from future_music.ports.audio_routing_ports import DualBusAudioOutput


@dataclass
class DualChannelAudioRouterAdapter:
    """Routes Ghost Preview to Left channel and Main PA to Right channel (REQ-08)."""

    routing_mode: PreviewRoutingMode = PreviewRoutingMode.STEREO_SPLIT

    def route_signals(
        self, main_mono: np.ndarray, ghost_mono: np.ndarray, ghost_active: bool
    ) -> DualBusAudioOutput:
        """Route Main PA and Ghost Preview mono signals into isolated stereo outputs."""
        main_arr = np.asarray(main_mono, dtype=np.float64)
        ghost_arr = (
            np.asarray(ghost_mono, dtype=np.float64)
            if ghost_active
            else np.zeros_like(main_arr)
        )
        stereo_split = np.column_stack((ghost_arr, main_arr))
        main_stereo = np.column_stack((main_arr, main_arr))
        ghost_stereo = np.column_stack((ghost_arr, ghost_arr))
        return DualBusAudioOutput(
            stereo_split_frame=stereo_split,
            main_pa_stereo=main_stereo,
            ghost_preview_stereo=ghost_stereo,
        )
