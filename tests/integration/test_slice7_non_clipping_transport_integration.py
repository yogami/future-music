"""Integration tests for Slice 7 (REQ-14): Non-Clipping Main PA & Ghost Bus Headroom."""

import numpy as np

from future_music.application.empath_conductor_facade import build_default_conductor_facade


def test_req14_default_facade_renders_audible_non_clipping_main_pa_headroom() -> None:
    """REQ-14: Default Main PA renders RMS > 0.08, peak < 0.99, and click-free start."""
    facade = build_default_conductor_facade()
    stereo = facade.render_live_stereo_frame(frame_samples=12000)
    right_pa = stereo[:, 1]
    assert float(np.sqrt(np.mean(np.square(right_pa)))) > 0.08
    assert float(np.max(np.abs(right_pa))) < 0.99
    assert abs(float(right_pa[0])) < 0.05


def test_req14_consecutive_live_frames_have_no_inter_chunk_biquad_seam_click() -> None:
    """REQ-14: Stateful HPF preserves continuity across consecutive 24k-sample frames."""
    facade = build_default_conductor_facade()
    frame_1 = facade.render_live_stereo_frame(frame_samples=24000)
    frame_2 = facade.render_live_stereo_frame(frame_samples=24000)
    seam_jump = abs(float(frame_1[-1, 1]) - float(frame_2[0, 1]))
    assert seam_jump < 0.03


def test_req14_default_facade_ghost_plus_main_pa_remain_non_clipping_together() -> None:
    """REQ-14: Staged Ghost Cue (Left) + Main PA (Right) stay strictly below clipping."""
    facade = build_default_conductor_facade()
    facade.handle_deck_key("2")
    stereo = facade.render_live_stereo_frame(frame_samples=12000)
    assert float(np.sqrt(np.mean(np.square(stereo[:, 0])))) > 0.05
    assert float(np.max(np.abs(stereo[:, 0]))) < 0.99
    assert float(np.max(np.abs(stereo[:, 1]))) < 0.99
