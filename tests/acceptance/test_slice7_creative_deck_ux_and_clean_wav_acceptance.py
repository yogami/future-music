"""Acceptance tests for Slice 7 (REQ-14): Zero-Clip WAV Stream & Creative-First UX Deck."""

import wave
from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient

from future_music.infrastructure import fastapi_app


def test_req14_wav_stream_has_zero_pcm16_digital_clipping_and_clean_start() -> None:
    """REQ-14: Consecutive WAV frames emit zero +-32767 clipping and seamless chunk seams."""
    client = TestClient(fastapi_app.create_empath_conductor_app())
    resp_1 = client.get("/api/v1/audio/frame.wav?samples=24000")
    resp_2 = client.get("/api/v1/audio/frame.wav?samples=24000")
    assert resp_1.status_code == 200
    assert resp_2.status_code == 200
    with wave.open(BytesIO(resp_1.content), "rb") as wf1:
        pcm_1 = np.frombuffer(wf1.readframes(24000), dtype=np.int16).reshape(-1, 2)
    with wave.open(BytesIO(resp_2.content), "rb") as wf2:
        pcm_2 = np.frombuffer(wf2.readframes(24000), dtype=np.int16).reshape(-1, 2)
    right = pcm_1[:, 1].astype(np.float64)
    assert float(np.sqrt(np.mean(np.square(right)))) > 1500.0
    assert int(np.max(np.abs(pcm_1[:, 1]))) < 32767
    assert int(np.max(np.abs(pcm_2[:, 1]))) < 32767
    assert abs(int(pcm_1[0, 1])) < 1600
    assert abs(int(pcm_1[-1, 1]) - int(pcm_2[0, 1])) < 1000


def test_req14_state_orbs_and_creative_ux_deck_expose_human_metadata_and_controls() -> None:
    """REQ-14: Orbs expose title/native_mode and GET / serves creative-first accessible UI."""
    client = TestClient(fastapi_app.create_empath_conductor_app())
    state = client.get("/api/v1/state").json()
    first_orb = state["surfaced_orbs"][0]
    assert "title" in first_orb
    assert "native_mode" in first_orb

    html = client.get("/").text
    required_ui = (
        "<header",
        "<main",
        'name="color-scheme"',
        'aria-live="polite"',
        'id="volumeSlider"',
        "<details",
        "/api/v1/empath/mood",
        "nextPlayTime",
        "Mix It In",
        "Dreamy 90s",
    )
    for token in required_ui:
        assert token in html, f"Missing creative UX token: {token}"
    assert "onclick=" not in html
