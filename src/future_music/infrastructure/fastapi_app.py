"""Slice 5 (REQ-10): Agent-Ready FastAPI Service for the Empath Conductor AI DJ."""

from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from future_music.application.empath_conductor_facade import (
    ConductorSnapshot,
    EmpathConductorFacade,
    build_default_conductor_facade,
)
from future_music.domain.clock_and_midi import ModalLockActiveError
from future_music.domain.empath_memory import MoodVector
from future_music.domain.whim_and_hum import (
    HumContourEvent,
    InputGateConflictError,
    InputGateInactiveError,
)

VERIFIED_ENGINE_TAG: str = "[VERIFIED ENGINE]"
HTTP_CONFLICT_STATUS: int = 409


class MoodUpdateRequest(BaseModel):
    """Request payload for updating the Empath's 4D emotional state."""

    valence: float = Field(ge=-1.0, le=1.0)
    arousal: float = Field(ge=0.0, le=1.0)
    tension: float = Field(ge=0.0, le=1.0)
    nostalgia: float = Field(ge=0.0, le=1.0)
    current_minutes: float = Field(default=0.0, ge=0.0)


class DeckKeyRequest(BaseModel):
    """Request payload for triggering any of the 18 MacBook Empath Deck actions."""

    key_code: str
    hold_progress: float = Field(default=1.0, ge=0.0, le=1.0)
    elapsed_minutes: float = Field(default=0.0, ge=0.0)


class VoiceWhimRequest(BaseModel):
    """Request payload for submitting a gated spoken whim (Hold V)."""

    transcript: str


class HumEventPayload(BaseModel):
    """Single detected hum pitch onset for Hold H quantization."""

    onset_ms: float = Field(ge=0.0)
    f0_hz: float = Field(gt=0.0)
    velocity: int = Field(default=100, ge=1, le=127)


class HumContourRequest(BaseModel):
    """Request payload for submitting a gated hum contour (Hold H)."""

    events: list[HumEventPayload]


def _serialize_orbs(snap: ConductorSnapshot) -> list[dict[str, Any]]:
    """Serialize surfaced Memory Orbs for JSON output."""
    return [
        {"orb_slot": o.orb_slot, "asset_id": o.asset.asset_id, "score": round(o.score, 4)}
        for o in snap.surfaced_orbs
    ]


def _serialize_snapshot(snap: ConductorSnapshot) -> dict[str, Any]:
    """Convert ConductorSnapshot into a JSON-serializable dictionary."""
    stage = snap.active_bridge_stage.value if snap.active_bridge_stage is not None else None
    return {
        "bpm": snap.bpm,
        "current_sample": snap.current_sample,
        "active_gear": snap.active_gear.value,
        "active_scale_mode": snap.active_scale_mode.value,
        "modal_lock_active": snap.modal_lock_active,
        "cc1_modulation": snap.cc1_modulation,
        "note_density": snap.note_density,
        "drums_hpf_hz": snap.drums_hpf_hz,
        "input_gate_mode": snap.input_gate_mode.value,
        "surfaced_orbs": _serialize_orbs(snap),
        "staged_ghost_asset_id": snap.staged_ghost_asset_id,
        "active_bridge_stage": stage,
    }


def _get_facade(request: Request) -> EmpathConductorFacade:
    """Retrieve the active EmpathConductorFacade from FastAPI app state."""
    facade: EmpathConductorFacade = request.app.state.facade
    return facade


router = APIRouter(prefix="/api/v1", tags=["Empath Conductor v1"])


@router.get("/health")
def get_health() -> dict[str, str]:
    """Return service health and [VERIFIED ENGINE] deterministic compliance tag."""
    return {"status": "ok", "engine_tag": VERIFIED_ENGINE_TAG}


@router.get("/state")
def get_conductor_state(request: Request) -> dict[str, Any]:
    """Return live Empath Conductor state snapshot."""
    return _serialize_snapshot(_get_facade(request).snapshot())


@router.post("/empath/mood")
def post_empath_mood(payload: MoodUpdateRequest, request: Request) -> dict[str, Any]:
    """Update Empath MoodVector and surface top-3 autobiographical Memory Orbs."""
    facade = _get_facade(request)
    mood = MoodVector(payload.valence, payload.arousal, payload.tension, payload.nostalgia)
    facade.update_mood(mood=mood, current_minutes=payload.current_minutes)
    return _serialize_snapshot(facade.snapshot())


@router.post("/deck/key")
def post_deck_key(payload: DeckKeyRequest, request: Request) -> dict[str, Any]:
    """Trigger any of the 18 MacBook Empath Deck actions and return updated state."""
    snap = _get_facade(request).handle_deck_key(
        raw_key=payload.key_code,
        hold_progress=payload.hold_progress,
        elapsed_minutes=payload.elapsed_minutes,
    )
    return _serialize_snapshot(snap)


@router.post("/whim/voice")
def post_voice_whim(payload: VoiceWhimRequest, request: Request) -> dict[str, Any]:
    """Submit a gated spoken whim command (requires HOLD_V gate)."""
    intent = _get_facade(request).submit_voice_whim(payload.transcript)
    return {
        "is_recognized": intent.is_recognized,
        "target_asset_id": intent.target_asset_id,
        "target_scale_mode": intent.target_scale_mode.value
        if intent.target_scale_mode is not None
        else None,
        "strip_drums": intent.strip_drums,
        "bridge_bars": intent.bridge_bars,
    }


@router.post("/whim/hum")
def post_hum_contour(payload: HumContourRequest, request: Request) -> dict[str, Any]:
    """Submit a gated hum pitch contour (requires HOLD_H gate)."""
    domain_events = tuple(
        HumContourEvent(onset_ms=ev.onset_ms, f0_hz=ev.f0_hz, velocity=ev.velocity)
        for ev in payload.events
    )
    quantized = _get_facade(request).submit_hum_contour(domain_events)
    return {
        "quantized_notes": [
            {
                "midi_note": q.midi_note,
                "velocity": q.velocity,
                "compensated_onset_ms": q.compensated_onset_ms,
                "quantized_sample": q.quantized_sample,
                "sixteenth_index": q.sixteenth_index,
            }
            for q in quantized
        ]
    }


def _conflict_exception_handler(_req: Request, exc: Exception) -> JSONResponse:
    """Return HTTP 409 Conflict when modal lock or input gate constraints are violated."""
    return JSONResponse(status_code=HTTP_CONFLICT_STATUS, content={"detail": str(exc)})


def create_empath_conductor_app(
    facade: EmpathConductorFacade | None = None,
) -> FastAPI:
    """Create the Agent-Ready FastAPI application exposing /api/openapi.json & /api/docs."""
    app = FastAPI(
        title="Empath Conductor AI DJ",
        version="0.1.0",
        openapi_url="/api/openapi.json",
        docs_url="/api/docs",
    )
    app.state.facade = facade if facade is not None else build_default_conductor_facade()
    for err_cls in (ModalLockActiveError, InputGateInactiveError, InputGateConflictError):
        app.add_exception_handler(err_cls, _conflict_exception_handler)
    app.include_router(router)
    return app
