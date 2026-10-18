"""Acceptance tests for Slice 6 (REQ-11, REQ-12, REQ-13): Live WAV Stream, Web Deck, & CLI."""

import wave
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from future_music.infrastructure import fastapi_app


def test_req11_and_req12_live_wav_stream_endpoint_serves_audible_stereo_riff_wave() -> None:
    """REQ-12: GET /api/v1/audio/frame.wav returns 48kHz stereo WAV with audible Main PA."""
    client = TestClient(fastapi_app.create_empath_conductor_app())
    resp = client.get("/api/v1/audio/frame.wav?samples=4800")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"

    with wave.open(BytesIO(resp.content), "rb") as wf:
        assert (wf.getnchannels(), wf.getframerate(), wf.getnframes()) == (2, 48000, 4800)
        pcm = np.frombuffer(wf.readframes(4800), dtype=np.int16).reshape(-1, 2)
    right_rms = float(np.sqrt(np.mean(np.square(pcm[:, 1].astype(np.float64)))))
    assert right_rms > 1500.0


def test_req13_live_web_empath_deck_serves_interactive_18_key_and_webaudio_ui() -> None:
    """REQ-13: GET / serves interactive HTML5 MacBook Keyboard + Mic Live Deck."""
    client = TestClient(fastapi_app.create_empath_conductor_app())
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    html = resp.text
    required_tokens = (
        "HOLD_SPACE",
        "RELEASE_SPACE",
        "HOLD_V",
        "RELEASE_V",
        "HOLD_H",
        "RELEASE_H",
        "/api/v1/whim/voice",
        "/api/v1/whim/hum",
        "onset_ms",
        "quantized_notes",
        "frame.wav",
    )
    for required_token in required_tokens:
        assert required_token in html


def test_req13_module_level_app_and_cli_entrypoint_are_importable_and_configured() -> None:
    """REQ-13: fastapi_app exports 'app' and 'main', and pyproject.toml defines CLI script."""
    assert hasattr(fastapi_app, "app")
    assert callable(getattr(fastapi_app, "main", None))
    pyproject_text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "empath-conductor" in pyproject_text
