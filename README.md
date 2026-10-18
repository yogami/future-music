# Future Music: Empath Conductor AI DJ (Prototype 0.1)

A human-coupled live music performance runtime that replaces the mechanical execution of DJing (beatmatching, EQing, folder hunting, harmonic math) with a sample-locked stem weaver and lookahead harmonic bridge compiler. The human performer acts as an emotional antenna and whim-driven conductor using the MacBook keyboard and microphone, while GarageBand (via macOS `IAC Driver`) renders live synthesizer and acoustic patches.

## Governance and Architecture

This repository follows the Berlin AI Automation Studio governance protocol:
- Global Rules: [RULES.md](file:///Users/yamijala/gitprojects/Studio_Second_Brain/03_Resources/RULES.md)
- Service Registry: [Microservices_Catalog.md](file:///Users/yamijala/gitprojects/Studio_Second_Brain/03_Resources/Microservices_Catalog.md)
- Primary Specification: [SPEC.md](./SPEC.md)

### Hexagonal Layout (`src/future_music/`)
- `domain/`: Pure business logic, music theory value objects, modal scale quantizer, sample-counted clock math, biquad DSP state, and lookahead bridge state machines. Zero external framework dependencies.
- `ports/`: Primary and secondary protocol interfaces (`AudioOutputPort`, `MidiOutputPort`, `StemVaultPort`, `EmotionGraphPort`, `WhimInputPort`).
- `application/`: Use-case orchestration (`EmpathConductorService`, `LookaheadBridgeUseCase`, `GhostPreviewUseCase`, `TensionOverrideUseCase`, `HumToMidiUseCase`).
- `infrastructure/`: Concrete adapters for sample-counted DSP mixing, macOS `IAC Driver` MIDI expression (`CC 1`, velocity, `CC 64`, pitch bend), cosine-similarity personal memory retrieval, keyboard/mic input gating, and the FastAPI Agent-Ready API (`/api/openapi.json` and `/api/docs`).
