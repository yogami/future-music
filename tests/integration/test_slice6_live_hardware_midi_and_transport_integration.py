"""Integration tests for Slice 6 (REQ-11, REQ-12): Live CoreMIDI Port & Audio Transport."""

import numpy as np

from future_music.application.empath_conductor_facade import build_default_conductor_facade
from future_music.domain.ghost_and_bridge import BridgeStageName
from future_music.infrastructure.iac_midi_adapter import GarageBandIacMidiAdapter


def test_garageband_iac_midi_adapter_opens_real_or_virtual_coremidi_port_and_transmits() -> None:
    """REQ-11: GarageBandIacMidiAdapter opens a live CoreMIDI port and transmits messages."""
    adapter = GarageBandIacMidiAdapter()
    assert adapter.active_port_name != ""
    assert adapter.is_hardware_connected is True

    adapter.send_note_on(note=65, velocity=100, trigger_sample=0)
    adapter.send_cc(cc_number=1, value=64, trigger_sample=480)
    assert len(adapter.sent_messages) == 2
    assert adapter.sent_messages[0].note == 65
    assert adapter.sent_messages[1].value == 64
    adapter.close()


def test_default_conductor_facade_preloads_124bpm_stems_and_renders_audible_main_pa() -> None:
    """REQ-12: Default facade renders audible Main PA (Right) and isolated Ghost Cue (Left)."""
    facade = build_default_conductor_facade()
    idle_stereo = facade.render_live_stereo_frame(frame_samples=4800)
    left_idle_rms = float(np.sqrt(np.mean(np.square(idle_stereo[:, 0]))))
    right_pa_rms = float(np.sqrt(np.mean(np.square(idle_stereo[:, 1]))))
    assert left_idle_rms == 0.0
    assert right_pa_rms > 0.08

    facade.handle_deck_key("1")
    ghost_stereo = facade.render_live_stereo_frame(frame_samples=4800)
    left_ghost_rms = float(np.sqrt(np.mean(np.square(ghost_stereo[:, 0]))))
    assert left_ghost_rms > 0.05


def test_live_stereo_frame_transport_steps_bridge_stages_across_bar_boundaries() -> None:
    """REQ-12: Rendering across bar boundaries advances active_bridge_stage 1 -> 2 -> 3 -> 4."""
    facade = build_default_conductor_facade()
    facade.handle_deck_key("1")
    snap_1 = facade.handle_deck_key("RETURN")
    assert snap_1.active_bridge_stage == BridgeStageName.STAGE_1_THIN_MIDRANGE

    facade.render_live_stereo_frame(frame_samples=92904)
    assert facade.snapshot().active_bridge_stage == BridgeStageName.STAGE_2_PIVOT_CHORDS
    facade.render_live_stereo_frame(frame_samples=92904)
    assert facade.snapshot().active_bridge_stage == BridgeStageName.STAGE_3_TEASE_AND_RISER
    facade.render_live_stereo_frame(frame_samples=92904)
    assert (
        facade.snapshot().active_bridge_stage
        == BridgeStageName.STAGE_4_SUBBASS_RESOLVE_AND_DROP
    )


def test_phrygian_scale_shift_mutes_melodic_audio_stem_and_dispatches_mutated_midi() -> None:
    """REQ-04 & REQ-12: Pressing W mutes Natural Minor audio stem and sends Gb4 (66) MIDI motif."""
    facade_natural = build_default_conductor_facade()
    natural_frame = facade_natural.render_live_stereo_frame(frame_samples=4800)

    facade_phrygian = build_default_conductor_facade()
    facade_phrygian.handle_deck_key("RIGHT")
    facade_phrygian.handle_deck_key("RIGHT")
    facade_phrygian.handle_deck_key("W")
    assert facade_phrygian.clock_use_case.modal_engine.current_state().audio_stem_gain == 0.0
    phrygian_frame = facade_phrygian.render_live_stereo_frame(frame_samples=4800)

    diff_rms = float(np.sqrt(np.mean(np.square(natural_frame[:, 1] - phrygian_frame[:, 1]))))
    assert diff_rms > 0.05
    midi_adapter = facade_phrygian.clock_use_case.midi_port
    sent_notes = [m.note for m in getattr(midi_adapter, "sent_messages", []) if m.note is not None]
    assert 66 in sent_notes
    assert 67 not in sent_notes
