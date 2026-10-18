"""Gate 4 automated verifier: Zero-Hollow-Infrastructure and Live Playability smoke checks."""

import sys
import wave
from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient

from future_music.application.empath_conductor_facade import build_default_conductor_facade
from future_music.infrastructure import fastapi_app


def verify_midi_hardware_binding() -> list[str]:
    """Verify CoreMIDI port binding and bar MIDI motif dispatch on transport render."""
    facade = build_default_conductor_facade()
    midi_adapter = facade.clock_use_case.midi_port
    if not getattr(midi_adapter, "active_port_name", None):
        return ["GAP-1: GarageBandIacMidiAdapter did not open a real/virtual CoreMIDI port"]
    facade.render_live_stereo_frame(4800)
    if not getattr(midi_adapter, "sent_messages", []):
        return ["GAP-1: render_live_stereo_frame() did not dispatch bar MIDI motifs"]
    return []


def verify_default_stems_are_audible() -> list[str]:
    """Verify default facade renders RMS > 0.05 and enforces audio_stem_gain on scale shift."""
    facade = build_default_conductor_facade()
    stereo = facade.render_live_stereo_frame(4800)
    right_pa_rms = float(np.sqrt(np.mean(np.square(stereo[:, 1]))))
    if right_pa_rms < 0.05:
        return [f"GAP-2: Default Main PA audio output is silent (RMS={right_pa_rms:.4f})"]
    facade.handle_deck_key("W")
    if facade.clock_use_case.modal_engine.current_state().audio_stem_gain != 0.0:
        return ["GAP-2: Scale shift to Phrygian did not mute incompatible audio stem"]
    return []


def verify_live_web_deck() -> list[str]:
    """Verify module-level app, GET / keyboard deck, and /api/v1/whim/voice + hum wiring."""
    if not hasattr(fastapi_app, "app"):
        return ["GAP-5: fastapi_app.py does not export a module-level 'app' instance"]
    ui_resp = TestClient(fastapi_app.app).get("/")
    has_whim = {"/api/v1/whim/voice", "/api/v1/whim/hum"}.issubset(ui_resp.text.split("'"))
    if ui_resp.status_code != 200 or not has_whim:
        return ["GAP-3: GET / missing Live Deck HTML or /api/v1/whim/voice + hum wiring"]
    return []


def verify_wav_audio_stream() -> list[str]:
    """Verify GET /api/v1/audio/frame.wav returns a valid 48000 Hz stereo WAV frame."""
    client = TestClient(fastapi_app.create_empath_conductor_app())
    wav_resp = client.get("/api/v1/audio/frame.wav")
    if wav_resp.status_code != 200:
        return ["GAP-2: GET /api/v1/audio/frame.wav endpoint is missing or failing"]
    with wave.open(BytesIO(wav_resp.content), "rb") as wf:
        if (wf.getnchannels(), wf.getframerate()) != (2, 48000):
            return ["GAP-2: /api/v1/audio/frame.wav is not 48000 Hz stereo"]
    return []


def main() -> int:
    """Run all Gate 4 Live Playability checks."""
    errors: list[str] = []
    errors.extend(verify_midi_hardware_binding())
    errors.extend(verify_default_stems_are_audible())
    errors.extend(verify_live_web_deck())
    errors.extend(verify_wav_audio_stream())
    for err in errors:
        print(f"GATE-4 ERROR: {err}", file=sys.stderr)
    if not errors:
        print("Gate 4 Live Playability verification passed: CoreMIDI, Stems, and Live Deck ready.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
