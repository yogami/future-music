"""Acceptance tests for Slice 5 (REQ-10): Agent-Ready FastAPI Service and 18-Key Deck."""

from fastapi.testclient import TestClient
from future_music.infrastructure.fastapi_app import create_empath_conductor_app


def create_test_client() -> TestClient:
    """Create a FastAPI TestClient wired to an unmocked EmpathConductorFacade."""
    return TestClient(create_empath_conductor_app())


def test_req10_agent_ready_openapi_docs_and_verified_engine_health_endpoints() -> None:
    """REQ-10 & RULES.md: Service exposes /api/openapi.json, /api/docs, and /api/v1/health."""
    client = create_test_client()
    openapi_resp = client.get("/api/openapi.json")
    docs_resp = client.get("/api/docs")
    health_resp = client.get("/api/v1/health")

    assert openapi_resp.status_code == 200
    assert "/api/v1/deck/key" in openapi_resp.json()["paths"]
    assert docs_resp.status_code == 200
    assert health_resp.status_code == 200
    assert health_resp.json()["engine_tag"] == "[VERIFIED ENGINE]"


def test_req10_mood_update_orb_staging_filter_morph_and_bridge_commit_via_http() -> None:
    """REQ-10: Update mood, stage Orb 1 ('1'), step CC 1 ('UP'), commit ('RETURN'), read /state."""
    client = create_test_client()
    mood_resp = client.post(
        "/api/v1/empath/mood",
        json={"valence": 0.35, "arousal": 0.55, "tension": 0.30, "nostalgia": 0.95},
    )
    assert mood_resp.json()["surfaced_orbs"][0]["asset_id"] == "robert_miles_children"
    assert client.post("/api/v1/deck/key", json={"key_code": "1"}).status_code == 200
    assert client.post("/api/v1/deck/key", json={"key_code": "UP"}).json()["cc1_modulation"] == 48
    assert client.post("/api/v1/deck/key", json={"key_code": "RETURN"}).status_code == 200

    state_resp = client.get("/api/v1/state")
    assert state_resp.status_code == 200
    assert state_resp.json()["active_gear"] == "GEAR_1_AUTOPILOT"
    assert state_resp.json()["bpm"] == 124.0
    assert state_resp.json()["active_bridge_stage"] == "STAGE_1_THIN_MIDRANGE"


def test_req10_modal_lock_conflict_returns_409_and_spacebar_tension_drop_via_http() -> None:
    """REQ-10: Committing Sufi Vocal locks Phrygian (Q -> 409) and Spacebar sweeps HPF."""
    client = create_test_client()
    client.post(
        "/api/v1/empath/mood",
        json={"valence": 0.35, "arousal": 0.55, "tension": 0.30, "nostalgia": 0.95},
    )
    client.post("/api/v1/deck/key", json={"key_code": "2"})
    client.post("/api/v1/deck/key", json={"key_code": "RETURN"})

    conflict_resp = client.post("/api/v1/deck/key", json={"key_code": "Q"})
    assert conflict_resp.status_code == 409

    hold_resp = client.post(
        "/api/v1/deck/key", json={"key_code": "HOLD_SPACE", "hold_progress": 1.0}
    )
    assert hold_resp.json()["drums_hpf_hz"] == 350.0


def test_req10_gated_voice_whim_and_hum_contour_endpoints_enforce_push_to_input() -> None:
    """REQ-10: /api/v1/whim/voice and /api/v1/whim/hum reject IDLE (409) and pass when gated."""
    client = create_test_client()
    idle_resp = client.post("/api/v1/whim/voice", json={"transcript": "drop the drums"})
    assert idle_resp.status_code == 409

    client.post("/api/v1/deck/key", json={"key_code": "HOLD_V"})
    voice_resp = client.post(
        "/api/v1/whim/voice",
        json={"transcript": "bring in the Robert Miles piano in 16 bars"},
    )
    client.post("/api/v1/deck/key", json={"key_code": "RELEASE_V"})
    assert voice_resp.status_code == 200
    assert voice_resp.json()["target_asset_id"] == "robert_miles_children"

    client.post("/api/v1/deck/key", json={"key_code": "HOLD_H"})
    hum_resp = client.post(
        "/api/v1/whim/hum",
        json={"events": [{"onset_ms": 205.0, "f0_hz": 380.0, "velocity": 100}]},
    )
    assert hum_resp.status_code == 200
    assert len(hum_resp.json()["quantized_notes"]) == 1
