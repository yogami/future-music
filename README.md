# Future Music: Empath Conductor AI DJ (Prototype 0.1)

A human-coupled live music performance runtime that replaces the mechanical execution of DJing (beatmatching, EQing, folder hunting, harmonic math) with a sample-locked stem weaver and lookahead harmonic bridge compiler. The human performer acts as an emotional antenna and whim-driven conductor using the MacBook keyboard and microphone, while GarageBand (via macOS CoreMIDI `IAC Driver` or the `"Empath Conductor IAC"` virtual MIDI port) renders live synthesizer and acoustic patches.

## Quickstart: Live Human Playability

Start the live Empath Conductor server:

```bash
uv run empath-conductor
```

Then open `http://127.0.0.1:8000/` in your browser:
1. Click **Start Live 48kHz Audio Engine** to stream stereo `48000 Hz` PCM audio (`Left = Ghost Cue`, `Right = Main PA`).
2. Open GarageBand on macOS; it automatically receives real-time CoreMIDI `note_on`, `control_change` (`CC 1`, `CC 64`), and `pitchwheel` messages from the `"Empath Conductor IAC"` CoreMIDI port (or `IAC Driver` Bus 1).
3. Perform live from the MacBook keyboard:
   - `[1] [2] [3]`: Stage surfaced autobiographical Memory Orbs into the isolated Left-channel Ghost Preview.
   - `[RETURN]` / `[BACKSPACE]`: Commit the Ghost Preview into the 4-Stage Lookahead Harmonic Bridge or discard it.
   - `[Q] [W] [E]`: Shift scale mode (`F Natural Minor`, `F Phrygian`, `F Dorian`) with automatic Modal Lock and `audio_stem_gain` protection.
   - `[UP] [DOWN] [LEFT] [RIGHT]`: Step GarageBand Alchemy filter cutoff (`CC 1`) and MIDI note density.
   - `[Hold SPACEBAR]`: Gear 3 tactile override (sweeps drum high-pass biquad filter `20 Hz -> 350 Hz` and triggers a beat-quantized Suck-Back Drop on release).
   - `[Hold V]` / `[Hold H]`: Gated Voice Whim and `-80 ms` lag-compensated Hum-to-MIDI quantization.

## Governance and Architecture

This repository follows the Berlin AI Automation Studio 4-Gate governance protocol:
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
  - `stem_synthesis.py`: Deterministic `48000 Hz` `124.0 BPM` (`92,903` samples/bar) audio stem synthesis, 4-stage bridge progression, and 16-bit PCM RIFF WAVE encoding (`REQ-12`).
- `ports/`: Protocol interfaces (`memory_ports.py`, `midi_ports.py`, `dsp_ports.py`, `audio_routing_ports.py`).
- `application/`: Use-case orchestrators (`memory_surfacing_use_case.py`, `harmonic_clock_use_case.py`, `tension_override_use_case.py`, `whim_and_bridge_use_case.py`, and `empath_conductor_facade.py`).
- `infrastructure/`: Concrete hardware and web adapters (`in_memory_vault.py`, `iac_midi_adapter.py`, `dsp_stem_mixer_adapter.py`, `audio_routing_adapter.py`, `live_deck_ui.py`, and `fastapi_app.py`).

## Verification and 4-Gate Quality Suite

Run the linter, Gate 3 AST craftsmanship verifier, Gate 4 live playability verifier, and full test suite:

```bash
uv run ruff check src/ tests/ scripts/
uv run python scripts/verify_craftsmanship.py
uv run python scripts/verify_live_playability.py
uv run pytest --cov=src/future_music --cov-report=term-missing
```

### Invariants Enforced by `verify_craftsmanship.py` (Gate 3) & `verify_live_playability.py` (Gate 4)
- Cyclomatic Complexity: `<= 3` per function or method
- Method Length: `<= 20` lines per function or method
- Class Length: `<= 120` lines per class
- Testing Pyramid: `62 unit > 21 integration >= 16 acceptance >= 1` (99 total tests, 99% line coverage)
- Zero SUT Mocking: `unittest.mock`, `MagicMock`, `patch`, and `mocker` are banned across `tests/`
- Zero Hollow Infrastructure: Real CoreMIDI hardware/virtual port transmission via `mido` + `python-rtmidi`, pre-loaded audible `124.0 BPM` `48000 Hz` stems (`RMS > 0.05`), and live HTML5/WebAudio performance deck at `GET /`.
