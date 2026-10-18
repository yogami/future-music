# Future Music: Empath Conductor AI DJ (Prototype 0.1)

A human-coupled live music performance runtime that replaces the mechanical execution of DJing (beatmatching, EQing, folder hunting, harmonic math) with a sample-locked stem weaver and lookahead harmonic bridge compiler. The human performer acts as an emotional antenna and whim-driven conductor using the MacBook keyboard and microphone, while GarageBand (via macOS `IAC Driver`) renders live synthesizer and acoustic patches.

## Governance and Architecture

This repository follows the Berlin AI Automation Studio governance protocol:
- Global Rules: [RULES.md](file:///Users/yamijala/gitprojects/Studio_Second_Brain/03_Resources/RULES.md)
- Service Registry: [Microservices_Catalog.md](file:///Users/yamijala/gitprojects/Studio_Second_Brain/03_Resources/Microservices_Catalog.md)
- Primary Specification: [SPEC.md](./SPEC.md)

### Hexagonal Layout (`src/future_music/`)
- `domain/`: Pure Python and NumPy deterministic math (`[VERIFIED ENGINE]`). Zero external framework imports.
  - `empath_memory.py`: `MoodVector`, `ScaleMode`, `MemoryAsset`, `AutobiographicalEmotionPrism`, 60-minute `EpisodicSetMemory`, and `ProactiveCoPilotMonitor` (`REQ-01`).
  - `clock_and_midi.py`: `SampleClock` (`48000 Hz`, `-18.0 ms` IAC pre-fire advance), `GarageBandExpressionState` (`CC 1`, `CC 64`, velocity, pitch bend), and `ModalScaleEngine` (`REQ-02`, `REQ-03`, `REQ-04`).
  - `dsp_tension_drop.py`: Audio EQ Cookbook 2nd-order Butterworth high-pass biquad filter (`20.0 Hz -> 350.0 Hz`), `TensionDropController`, and quantized `±80.0 ms` beat-grid Suck-Back Pause (`REQ-05`, `REQ-06`).
  - `whim_and_hum.py`: `InputGateController` (`HOLD_V` vs `HOLD_H`), deterministic `VoiceWhimIntent` parser, and `-80.0 ms` lag-compensated `1/16`th-grid pentatonic `quantize_hum_contour` (`REQ-07`).
  - `ghost_and_bridge.py`: `GhostPreviewState`, 4-Stage `LookaheadBridgePlan` with structured tool calls, and Pre-Play Spectral Critic (`1000-3000 Hz` octave carving) (`REQ-08`, `REQ-09`).
  - `empath_deck.py`: Deterministic 18-key MacBook Empath Deck registry (`REQ-10`).
- `ports/`: Protocol interfaces (`memory_ports.py`, `midi_ports.py`, `dsp_ports.py`, `audio_routing_ports.py`).
- `application/`: Use-case orchestrators (`memory_surfacing_use_case.py`, `harmonic_clock_use_case.py`, `tension_override_use_case.py`, `whim_and_bridge_use_case.py`, and `empath_conductor_facade.py`).
- `infrastructure/`: Concrete adapters (`in_memory_vault.py`, `iac_midi_adapter.py`, `dsp_stem_mixer_adapter.py`, `audio_routing_adapter.py`, and `fastapi_app.py`).

## Verification and Quality Gates

Run the linter, AST craftsmanship verifier, and test suite:

```bash
uv run ruff check src/ tests/ scripts/
uv run python scripts/verify_craftsmanship.py
uv run pytest --cov=src/future_music --cov-report=term-missing
```

### Craftsmanship Invariants Enforced by `scripts/verify_craftsmanship.py`
- Cyclomatic Complexity: `<= 3` per function or method
- Method Length: `<= 20` lines per function or method
- Class Length: `<= 120` lines per class
- Testing Pyramid: `54 unit > 17 integration >= 13 acceptance >= 1` (84 total tests, 99% line coverage)
- Zero SUT Mocking: `unittest.mock`, `MagicMock`, `patch`, and `mocker` are banned across `tests/`
