# Empath Conductor AI DJ: Prototype 0.1 Specification (`SPEC.md`)

## 1. Purpose and Scope

Prototype 0.1 implements the core runtime of the Empath Conductor AI DJ on macOS using zero custom hardware. The MacBook keyboard acts as the physical performance deck, the Mac microphone captures spoken whims and hummed melodies, a deterministic sample-counted DSP engine weaves isolated audio stems, and GarageBand (driven over the macOS `IAC Driver` virtual MIDI bus) renders live synthesizer and acoustic instrument patches.

This document is the single source of truth for Acceptance Test-Driven Development (ATDD). Every requirement below maps directly to executable unit, integration, and acceptance specifications in `tests/`.

---

## 2. Ubiquitous Language and Domain Model (DDD)

1. **Empath Mood Vector (`MoodVector`)**: Immutable value object representing the performer's affective state across `valence` (`-1.0` to `1.0`), `arousal` (`0.0` to `1.0`), `tension` (`0.0` to `1.0`), `nostalgia` (`0.0` to `1.0`), and `intent_mode` (`PACING` vs. `LEADING`).
2. **Musical Scale and Mode (`ScaleMode`)**: Root pitch class (default `F`, MIDI pitch class `5`) paired with a modal interval structure:
   - `NATURAL_MINOR` (Aeolian): semitone offsets `(0, 2, 3, 5, 7, 8, 10)`
   - `PHRYGIAN`: semitone offsets `(0, 1, 3, 5, 7, 8, 10)`
   - `DORIAN`: semitone offsets `(0, 2, 3, 5, 7, 9, 10)`
3. **Personal Memory Asset (`MemoryAsset`)**: An entry in the Empath's Autobiographical Sonic Vault containing:
   - `asset_id`, `title`, and `emotional_embedding` (normalized 4D vector matching `[valence, arousal, tension, nostalgia]`).
   - `native_bpm` (`124.0`), `root_key` (`F`), and `native_mode` (`NATURAL_MINOR`, `PHRYGIAN`, or `DORIAN`).
   - `audio_stem_id` (reference to pre-warped `124.0 BPM` WAV buffer) and optional `symbolic_motif` (sequence of scale-degree MIDI notes for dynamic modal mutation).
   - `requires_modal_lock` (`bool`): `True` for polyphonic or microtonal vocal recordings (such as `sufi_vocal`) that cannot have individual scale degrees altered without harmonic clashes; `False` for motifs backed by a `symbolic_motif` (such as `robert_miles_children`).
   - `dominant_freq_hz` (`tuple[float, float]`): primary spectral occupancy band (for example, `(1000.0, 3000.0)` for vocal and piano leads).
4. **Memory Orb (`MemoryOrb`)**: One of the top-3 candidate `MemoryAsset` projections surfaced to the Empath's HUD (`orb_slot` `1`, `2`, or `3`) ranked by cosine similarity to the active `MoodVector` with an episodic recency penalty.
5. **Control Gear (`ControlGear`)**: State of the Variable-Grip Conductor:
   - `GEAR_1_AUTOPILOT`: Passive 16-bar phrase evolution, note-density control, and proactive Memory Orb surfacing.
   - `GEAR_2_WHIM_STEERING`: Active Ghost Preview audition, scale mutation, or voice/hum staging.
   - `GEAR_3_TACTILE_OVERRIDE`: Physical Spacebar Tension Hold, Suck-Back Drop, or Hard Cut override.
6. **Voice Whim Intent (`VoiceWhimIntent`)**: Immutable value object parsed from a gated spoken whim (`Hold V`):
   - `raw_transcript`: original spoken command string.
   - `target_asset_id`: `str | None` (resolved personal memory asset, such as `"robert_miles_children"` or `"sufi_vocal"`).
   - `target_scale_mode`: `ScaleMode | None` (resolved modal shift: `NATURAL_MINOR`, `PHRYGIAN`, or `DORIAN`).
   - `strip_drums`: `bool` (`True` if the command requests dropping or stripping drums/hi-hats).
   - `bridge_bars`: `int` (default `4` bars, or `16` bars if extended transition requested).
   - `is_recognized`: `bool` (`False` when the utterance matches no known whim directive).
7. **Agent Tool Call (`AgentToolCall`)**: Structured command emitted by the Lookahead Planner to operate the audio and MIDI engine:
   - `warp_stem(asset_id, target_bpm, target_root_key, gain)`
   - `write_midi_clip(track_name, notes, scale_mode, note_density)`
   - `modulate_filter_cutoff(cc1_value)`
   - `route_to_ghost_cue(asset_id, bars)`

---

## 3. Functional Requirements and Hardened Engineering Mitigations

### REQ-01: Autobiographical Emotion-to-Sound Graph, 60-Minute Episodic Memory & Proactive Co-Pilot Surfacing
- Given an Empath's `MoodVector` and calibrated `PersonalVault`, the system computes cosine similarity against all registered `MemoryAsset` embeddings.
- If `MoodVector.intent_mode == LEADING`, the query vector amplifies target `tension` and `arousal` by a deterministic leading projection factor (`LEADING_PROJECTION_WEIGHT = 0.25`, clamped to `[0.0, 1.0]`) to steer energy ahead of the room; if `PACING`, it matches the current state directly.
- **60-Minute Episodic Set Memory (Agentic Capability 4)**:
  - `EpisodicSetMemory` records every played `asset_id`, `ScaleMode`, and `transition_type` with a session timestamp across a configurable rolling window (`DEFAULT_EPISODIC_WINDOW_MINUTES = 60.0`).
  - Assets played within the active `60.0`-minute window receive a deterministic recency penalty (`RECENCY_PENALTY_WEIGHT = 0.35` per recent play, decayed linearly over the `60.0`-minute window) so the conductor never repeats the same motif or buildup cliché back-to-back.
- **Proactive Co-Pilot Orb Surfacing (Agentic Capability 5)**:
  - When the conductor stays in `GEAR_1_AUTOPILOT` within a static tension band (`abs(delta_tension) < 0.05`) for `>= STAGNATION_BARS_THRESHOLD` (default `16` bars) without an explicit Empath command, the Co-Pilot automatically re-queries the Autobiographical Emotion Graph (applying episodic memory penalties) and updates the 3 floating `MemoryOrb` slots (`1`, `2`, `3`).
- Two different Empath profiles given the exact same `MoodVector` (such as bittersweet 3 AM release) deterministically surface their own personal cultural touchstones (Empath A surfaces `robert_miles_children`; Empath B surfaces `sufi_vocal`).

### REQ-02: Sample-Counted Master Clock & `midi_advance_ms` Pre-Fire Offset (Mitigation 1)
- Master timing is computed strictly from the hardware audio sample counter (`DEFAULT_SAMPLE_RATE = 48000 Hz`, `DEFAULT_BPM = 124.0`), never from wall-clock `time.sleep()`.
- At `124.0 BPM` and `48000 Hz`:
  - `samples_per_beat = 48000 * 60.0 / 124.0 = 23225.8064516...`
  - `samples_per_bar = 4 * samples_per_beat = 92903.2258064...` (4/4 time).
- Scheduled MIDI events destined for GarageBand over the `IAC Driver` are emitted `midi_advance_ms` (default `DEFAULT_MIDI_ADVANCE_MS = 18.0 ms`, `864` samples at `48000 Hz`) prior to the target acoustic beat sample (`midi_trigger_sample = max(0, beat_sample - advance_samples)`) so GarageBand's CoreAudio and Alchemy synth envelope latency aligns with local stem playback.

### REQ-03: Native GarageBand Alchemy MIDI Expression Protocol & Density/Filter Controls (Mitigation 2)
- GarageBand does not support arbitrary MIDI CC mapping (such as `CC 74`).
- The MIDI expression port and adapter strictly emit four supported GarageBand messages:
  1. `CC 1` (Modulation Wheel, value `0..127`, default baseline `32`) for Alchemy Transform Pad and filter cutoff sweeps.
  2. `Note On` / `Note Off` with `note` (`0..127`) and `velocity` (`1..127`).
  3. `CC 64` (Sustain Pedal, value `0` or `127`).
  4. `Pitch Bend` (value `-8192..8191`).
- Any request to modulate filter cutoff is mapped deterministically to `CC 1` (clamped to `0..127`). Attempting to send an unsupported raw CC number (anything other than `1` or `64`) raises an `UnsupportedGarageBandMidiCcError`.
- **Keyboard Filter Morph (`[Up]` / `[Down]`)**:
  - Pressing `[Up]` increments `cc1_modulation` by `CC1_STEP_DELTA = 16` (clamped to `127`) and emits `CC 1`.
  - Pressing `[Down]` decrements `cc1_modulation` by `CC1_STEP_DELTA = 16` (clamped to `0`) and emits `CC 1`.
- **Keyboard MIDI Note Density (`[Left]` / `[Right]`)**:
  - The conductor maintains `note_density: float` in `[0.0, 1.0]` (default `DEFAULT_NOTE_DENSITY = 0.5`).
  - Pressing `[Left]` decreases `note_density` by `NOTE_DENSITY_STEP = 0.25` (clamped to `0.0`, sparse).
  - Pressing `[Right]` increases `note_density` by `NOTE_DENSITY_STEP = 0.25` (clamped to `1.0`, busy).
  - When rendering a symbolic MIDI clip of `N` candidate notes in a bar, `active_note_count = max(1, round(N * note_density))` when `note_density > 0.0` and `0` when `note_density == 0.0`, deterministically selecting the strongest metric positions first.

### REQ-04: Polyphonic Stem Scale-Flip Prevention & Modal Lock (Mitigation 3)
- Pressing `[Q]` selects `F NATURAL_MINOR`, pressing `[W]` selects `F PHRYGIAN`, and pressing `[E]` selects `F DORIAN`.
- When `robert_miles_children` is active in `F NATURAL_MINOR` (`[Q]`), the system plays `robert_miles_piano.wav` (and/or unison MIDI).
- When the Empath shifts the scale to `F PHRYGIAN` (`[W]`) or `F DORIAN` (`[E]`) while `robert_miles_children` is active:
  - The static `robert_miles_piano.wav` stem is muted (`gain = 0.0`) to prevent `G natural` vs. `Gb` minor-second clashes.
  - The transcribed `symbolic_motif` is quantized into the new target `ScaleMode` (flattening the 2nd scale degree from `G` MIDI `67` to `Gb` MIDI `66` in `F PHRYGIAN`, or raising the 6th scale degree from `Db` MIDI `61` to `D` MIDI `62` in `F DORIAN`) and routed to GarageBand over MIDI.
- When `sufi_vocal` (`requires_modal_lock = True`, `native_mode = PHRYGIAN`) is active on the Main Mix:
  - The conductor enforces **Modal Lock** to `F PHRYGIAN`.
  - Requests to switch scale to `NATURAL_MINOR` (`[Q]`) or `DORIAN` (`[E]`) while `sufi_vocal` is active on the Main Mix raise a `ModalLockActiveError` so GarageBand never plays natural-2nd notes against the microtonal Phrygian vocal.

### REQ-05: Separate `drums.wav` + `bass.wav` & Biquad High-Pass Filter Sweep (Mitigation 4)
- `drums` and `bass` are managed as independent stem channels in the DSP mixer.
- Engaging **Tension Hold** (`[Hold Spacebar]`, transitioning to `GEAR_3_TACTILE_OVERRIDE`):
  - Immediately sets `bass` stem gain to `0.0` (muting sub-bass while keeping drums active).
  - Applies a real-time 2nd-order Butterworth high-pass biquad filter to the `drums` stem whose cutoff frequency sweeps linearly from `HPF_MIN_HZ = 20.0 Hz` to `HPF_MAX_TENSION_HZ = 350.0 Hz` as tension progress moves from `0.0` to `1.0`, attenuating low-frequency kick energy (`<100 Hz`) while preserving snare and hi-hat transients (`>1000 Hz`).
  - Ramps GarageBand `CC 1` (Modulation Wheel) linearly toward `127`.

### REQ-06: Phase-Reset & Suck-Back Pause on Spacebar Release (Mitigation 5)
- When the Empath releases Spacebar (`[Release Spacebar]`) to drop the beat at sample `S`:
  - The engine computes distance in milliseconds to the nearest beat boundary (`ON_BEAT_WINDOW_MS = 80.0 ms`, or `3840` samples at `48000 Hz`).
  - **On-Beat Release (`<= 80.0 ms` from nearest beat grid)**: Immediately resets both `drums` and `bass` sample playheads to `Sample 0` (Bar 1, Beat 1), restores `drums` HPF cutoff to `20.0 Hz`, restores `bass` gain to `1.0`, resets `CC 1` to baseline `32`, and returns to `GEAR_1_AUTOPILOT`.
  - **Mid-Beat Release (`> 80.0 ms` from beat grid)**: Enters **Suck-Back Pause** state for the remaining samples until the next beat boundary (`samples_until_next_beat`). During the suck-back window, `drums` and `bass` output dead silence (`0.0` amplitude). At the exact sample of the next beat boundary, `drums` and `bass` reset to `Sample 0` (Bar 1, Beat 1) with full `bass` gain (`1.0`), `20.0 Hz` HPF cutoff, and baseline `CC 1`.

### REQ-07: Input Gating, Deterministic Voice Whim Parser & `-80ms` Lag-Compensated Hum-to-MIDI Quantizer (Mitigation 6)
- Microphone capture requires explicit push-to-input gating:
  - `IDLE`: Microphone audio is rejected (`InputGateInactiveError`).
  - `HOLD_V` (Voice Whim Gate): Audio/transcripts are routed exclusively to the **Voice Whim Intent Parser**; humming pitch extraction is blocked.
  - `HOLD_H` (Hum Melody Gate): Pitch contours are routed exclusively to the **Hum-to-MIDI Quantizer**; speech parsing is blocked.
  - Attempting to activate `HOLD_V` and `HOLD_H` simultaneously, or sending hum data while in `HOLD_V` (or voice data while in `HOLD_H`), raises an `InputGateConflictError`.
- **Deterministic Voice Whim Intent Parser (`Hold V`)**:
  - Parses spoken cues deterministically into `VoiceWhimIntent`:
    - Asset references: phrases containing `"robert miles"`, `"children"`, or `"piano"` resolve `target_asset_id = "robert_miles_children"`; phrases containing `"sufi"`, `"qawwali"`, or `"vocal"` resolve `target_asset_id = "sufi_vocal"`; phrases containing `"synth"` or `"acid"` resolve `target_asset_id = "berlin_acid_synth"`.
    - Scale directives: `"phrygian"`, `"darker"`, or `"hijaz"` resolve `target_scale_mode = ScaleMode.PHRYGIAN`; `"dorian"` or `"brighter"` resolve `target_scale_mode = ScaleMode.DORIAN`; `"natural minor"` or `"aeolian"` resolve `target_scale_mode = ScaleMode.NATURAL_MINOR`.
    - Drum directives: `"drop the drums"`, `"strip the drums"`, or `"strip the hi-hats"` set `strip_drums = True`.
    - Bar count directives: `"in 16 bars"` sets `bridge_bars = 16`; default is `bridge_bars = 4`.
    - Unrecognized utterances return `VoiceWhimIntent(is_recognized=False, ...)` and trigger zero state mutation (safe no-op).
- **Hum-to-MIDI Quantizer (`Hold H`)**:
  - Subtracts `HUMAN_REACTION_LAG_MS = 80.0 ms` from every detected note onset timestamp (clamped at `>= 0.0 ms`) and snaps onsets to the nearest `1/16`th-note grid subdivision (`samples_per_beat / 4`).
  - Converts fundamental frequency (`f0` in Hz) via `69 + 12 * log2(f0 / 440.0)` and snaps each note strictly to the **chord tones or pentatonic subset** of the active `ScaleMode` (for `F NATURAL_MINOR`: semitone offsets `(0, 3, 5, 7, 10)`; for `F PHRYGIAN`: `(0, 1, 3, 7, 8)`; for `F DORIAN`: `(0, 3, 5, 7, 9)`) so a slightly flat or sharp hum always lands on a consonant scale degree.

### REQ-08: Dual-Output / Stereo-Split "Inner-Ear" Ghost Preview Loop (Mitigation 7)
- Pressing `[1]`, `[2]`, or `[3]` selects the corresponding surfaced `MemoryOrb` (`orb_slot` `1`, `2`, or `3`), transitions the conductor to `GEAR_2_WHIM_STEERING`, and emits `route_to_ghost_cue(asset_id, bars=2)` to stage a 2-bar preview in the **Ghost Preview Bus** without altering the **Main PA Bus**.
- In `STEREO_SPLIT` routing mode (for 1-jack Mac hardware), the stereo output frame routes the **Ghost Preview Bus** strictly to the **Left Channel (`channel 0`)** and the **Main PA Bus** strictly to the **Right Channel (`channel 1`)**.
- In `DUAL_DEVICE` routing mode, Main PA and Ghost Preview are returned as two independent stereo buffers.
- Pressing `[Return / Enter]` (`commit_ghost_preview()`) promotes the staged Ghost Preview asset into the 4-Stage Lookahead Bridge Planner scheduled for the next bar boundary.
- Pressing `[Delete / Backspace]` (`discard_ghost_preview()`) clears the Ghost Preview buffer immediately and returns to `GEAR_1_AUTOPILOT` with zero audio disturbance on the Main PA Bus.

### REQ-09: 4-Stage Multi-Bar Lookahead Bridge Planner, Structured Tool Calls & Pre-Play Spectral Critic (Agentic Capabilities 1, 2 & 3)
- Committing a transition between the current groove and a target `MemoryAsset` (or target `ScaleMode`) constructs a deterministic **4-Stage Lookahead Bridge Plan** (spanning `4` bars by default or `16` bars when requested, 4 stages of `bridge_bars // 4` bars each):
  - `STAGE_1_THIN_MIDRANGE`: Reduces conflicting mid-range synth note density (`note_density * 0.5`) and emits `write_midi_clip()`.
  - `STAGE_2_PIVOT_CHORDS`: Generates transitional MIDI pivot chords bridging the source scale and target scale via `write_midi_clip()`.
  - `STAGE_3_TEASE_AND_RISER`: Emits `warp_stem(gain=0.35)` to tease the incoming motif and `modulate_filter_cutoff()` to ramp `CC 1` tension.
  - `STAGE_4_SUBBASS_RESOLVE_AND_DROP`: Resolves the sub-bass root note, locks the target scale (enforcing Modal Lock if `requires_modal_lock=True`), emits `warp_stem(gain=1.0)`, and records the transition in `EpisodicSetMemory`.
- **Pre-Play Spectral Critic (Agentic Capability 3)**:
  - Before staging a bar into the live clock buffer, the Spectral Critic inspects the frequency occupancy of all active audio stems and scheduled MIDI synth notes.
  - If an active vocal or piano stem occupies the `PRESENCE_BAND_HZ = (1000.0, 3000.0)` range and the scheduled MIDI synth clip contains notes colliding in `MIDI_PRESENCE_NOTE_RANGE = (60, 84)` (`C4` to `C6`), the Spectral Critic autonomously shifts the conflicting MIDI synth notes down one octave (`-12` semitones), attenuates their velocity by `SPECTRAL_VELOCITY_TRIM = 15`, and marks `spectral_carve_applied = True` before the bar reaches the master clock.

### REQ-10: Complete MacBook Keyboard Empath Deck & Agent-Ready OpenAPI Service
- All 16 MacBook Keyboard Empath Deck actions map deterministically as specified in the table below:

| Key Code | Deck Control Name | Target Gear / Transition | Deterministic Action & Parameter Delta |
| :--- | :--- | :--- | :--- |
| `"1"` | Memory Orb 1 | `GEAR_2_WHIM_STEERING` | Stage surfaced `orb_slot=1` (e.g. `robert_miles_children`) in Ghost Preview Bus (`2` bars). |
| `"2"` | Memory Orb 2 | `GEAR_2_WHIM_STEERING` | Stage surfaced `orb_slot=2` (e.g. `sufi_vocal`) in Ghost Preview Bus (`2` bars). |
| `"3"` | Memory Orb 3 | `GEAR_2_WHIM_STEERING` | Stage surfaced `orb_slot=3` (e.g. `berlin_acid_synth`) in Ghost Preview Bus (`2` bars). |
| `"Q"` | Scale: Natural Minor | Current Gear | Shift active scale to `F NATURAL_MINOR` (raises `ModalLockActiveError` if Modal Lock active). |
| `"W"` | Scale: Phrygian | Current Gear | Shift active scale to `F PHRYGIAN` (mutates symbolic motif `G -> Gb`, mutes incompatible stem). |
| `"E"` | Scale: Dorian | Current Gear | Shift active scale to `F DORIAN` (mutates symbolic motif `Db -> D`; blocked if Modal Lock active). |
| `"HOLD_V"` | Gate Voice Mic | `GEAR_2_WHIM_STEERING` | Open `HOLD_V` voice whim gate (blocks `HOLD_H` hum gate). |
| `"RELEASE_V"` | Close Voice Mic | Current Gear | Return input gate to `IDLE`. |
| `"HOLD_H"` | Gate Hum Mic | `GEAR_2_WHIM_STEERING` | Open `HOLD_H` hum-to-MIDI gate (blocks `HOLD_V` voice gate). |
| `"RELEASE_H"` | Close Hum Mic | Current Gear | Return input gate to `IDLE`. |
| `"RETURN"` | Commit Ghost Preview | `GEAR_1_AUTOPILOT` | Commit staged Ghost Preview into 4-Stage Lookahead Bridge Planner on next bar boundary. |
| `"BACKSPACE"` | Discard Ghost Preview | `GEAR_1_AUTOPILOT` | Cancel staged Ghost Preview silently; Main PA continues uninterrupted. |
| `"UP"` | Filter Morph Up | Current Gear | Increase GarageBand Alchemy `CC 1` by `+16` (clamped to `0..127`) and emit MIDI `CC 1`. |
| `"DOWN"` | Filter Morph Down | Current Gear | Decrease GarageBand Alchemy `CC 1` by `-16` (clamped to `0..127`) and emit MIDI `CC 1`. |
| `"LEFT"` | Note Density Sparse | Current Gear | Decrease `note_density` by `-0.25` (clamped to `0.0..1.0`). |
| `"RIGHT"` | Note Density Busy | Current Gear | Increase `note_density` by `+0.25` (clamped to `0.0..1.0`). |
| `"HOLD_SPACE"` | Tension Hold Override | `GEAR_3_TACTILE_OVERRIDE` | Mute `bass` (`0.0`), sweep `drums` HPF `20Hz -> 350Hz`, ramp `CC 1` toward `127`. |
| `"RELEASE_SPACE"` | Drop Release | `GEAR_1_AUTOPILOT` | If `<= 80ms` from beat: instant Sample 0 reset; if `> 80ms`: Suck-Back silence then Sample 0 slam. |

- **Agent-Ready FastAPI Service**:
  - `GET /api/openapi.json` and `GET /api/docs` (Agent-Ready API compliance per `RULES.md`).
  - `GET /api/v1/health`: Returns service health and engine verification status (`[VERIFIED ENGINE]`).
  - `GET /api/v1/state`: Current BPM, sample position, active gear, active scale mode, modal lock state, `cc1_modulation`, `note_density`, surfaced Memory Orbs, Ghost Preview state, and active lookahead bridge stage.
  - `POST /api/v1/deck/key`: Trigger any MacBook Empath Deck key event from the table above.
  - `POST /api/v1/whim/voice`: Submit a gated spoken whim command (`Hold V`).
  - `POST /api/v1/whim/hum`: Submit a gated hummed pitch contour (`Hold H`) for `-80ms` lag-compensated quantization.
  - `POST /api/v1/empath/mood`: Update the Empath's `MoodVector` and retrieve the newly surfaced 3 Memory Orbs.

---

## 4. Software Craftsmanship, `[VERIFIED ENGINE]` Determinism & Testing Pyramid Rules

1. **Hexagonal Architecture (`src/future_music/`)**:
   - `domain/`: Pure Python + NumPy deterministic math only. Zero imports from `ports`, `application`, `infrastructure`, `fastapi`, `starlette`, `pydantic`, `mido`, `sounddevice`, or `random`.
   - `ports/`: Abstract Protocol definitions (`typing.Protocol`).
   - `application/`: Use-case orchestrators coordinating `domain` objects and `ports`.
   - `infrastructure/`: Concrete DSP, MIDI (`IAC Driver`), Stem Vault, Keyboard/Mic adapters, and FastAPI web layer.
2. **`[VERIFIED ENGINE]` Determinism & Zero-Magic-Number Rule (`RULES.md` Section 7)**:
   - All production logic is deterministic and mathematically grounded (`[VERIFIED ENGINE]`).
   - Zero calls to `random` or `np.random` are permitted anywhere in `src/future_music/`.
   - All acoustic, DSP, MIDI, and scoring constants (`48000 Hz`, `124.0 BPM`, `18.0 ms`, `80.0 ms`, `20.0 Hz`, `350.0 Hz`, `60.0 min`, `16` CC step, `0.25` density step) must be named constants or explicit configuration fields, never undocumented magic literals.
3. **Code Craftsmanship Constraints**:
   - **Cyclomatic Complexity**: `<= 3` per function/method (enforced via `ruff C90` and `scripts/verify_craftsmanship.py`).
   - **Method Length**: `<= 20` lines per function/method.
   - **Class Length**: `<= 120` lines per class.
   - **Coverage**: `>= 80%` line coverage across `src/future_music/`.
4. **Testing Pyramid & Zero-Mocking Invariant**:
   - **Unit Tests (`tests/unit/`)**: Fast, pure tests of all domain entities, value objects, music theory, biquad DSP math, sample-clock math, suck-back phase reset, lag-compensated pitch quantizer, cosine vector prism, spectral critic, and 4-stage bridge planner.
   - **Integration Tests (`tests/integration/`)**: Tests wiring real `application` use cases + real `domain` + real `infrastructure` adapters.
   - **Acceptance Tests (`tests/acceptance/`)**: End-to-end ATDD scenario tests verifying every requirement (`REQ-01` through `REQ-10`) through the full unmocked application stack and FastAPI HTTP interface.
   - **Pyramid Invariant**: `count(unit_tests) > count(integration_tests) >= count(acceptance_tests) >= 1` (enforced by `scripts/verify_craftsmanship.py`).
   - **Zero Mocking Invariant**: `unittest.mock`, `MagicMock`, `AsyncMock`, `patch`, and `mocker` are strictly forbidden in `tests/` (enforced via AST check in `scripts/verify_craftsmanship.py`). Every test function must contain substantive `assert` statements or `pytest.raises` checks.

---

## 5. Iterative Delivery Slices (ATDD / TDD Cycle with Independent Review)

- **Slice 1**: Domain Foundation: `MoodVector`, `ScaleMode`, `MemoryAsset`, Autobiographical Cosine Emotion Prism, 60-Minute `EpisodicSetMemory`, and Proactive Memory Orb Surfacing (`REQ-01`).
- **Slice 2**: Deterministic Clock, Native GarageBand MIDI Expression (`CC 1`, `Velocity`, `CC 64`, `Pitch Bend`), Filter/Density Step Controls, `midi_advance_ms` Pre-Fire Offset, and Modal Lock / Symbolic Motif Mutation (`REQ-02`, `REQ-03`, `REQ-04`).
- **Slice 3**: Sample-Locked DSP Stem Mixer, Separate `drums`/`bass` Channels, Real-Time Biquad High-Pass Filter (`20Hz -> 350Hz`), Spacebar Tension Hold, and Suck-Back Pause + Sample-0 Phase Reset Drop (`REQ-05`, `REQ-06`).
- **Slice 4**: Input Gating (`Hold V` vs. `Hold H`), `-80ms` Lag-Compensated Hum-to-MIDI Chord-Tone Quantizer, Deterministic Voice Whim Parser, Dual-Channel Ghost Preview (`STEREO_SPLIT` & `DUAL_DEVICE`), 4-Stage Lookahead Bridge Planner with Structured Tool Calls, and Pre-Play Spectral Critic (`REQ-07`, `REQ-08`, `REQ-09`).
- **Slice 5**: MacBook Keyboard Empath Deck Controller (all 18 key/gate actions), Full Conductor Application Facade, and Agent-Ready FastAPI Service (`/api/openapi.json`, `/api/docs`, `/api/v1/*`) + Complete Acceptance Suite (`REQ-10`).
