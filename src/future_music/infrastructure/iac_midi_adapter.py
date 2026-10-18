"""macOS IAC Driver MIDI adapter enforcing native GarageBand Alchemy controls."""

from dataclasses import dataclass, field

import mido

from future_music.domain.clock_and_midi import (
    CC64_SUSTAIN_PEDAL,
    MAX_MIDI_VAL,
    MIN_MIDI_VAL,
    validate_garageband_cc,
    validate_midi_note_and_velocity,
    validate_pitch_bend,
)


@dataclass(frozen=True)
class SentMidiRecord:
    """Immutable record of a validated MIDI message dispatched over the IAC bus."""

    message_type: str
    trigger_sample: int
    note: int | None = None
    velocity: int | None = None
    control: int | None = None
    value: int | None = None
    pitch: int | None = None


@dataclass
class GarageBandIacMidiAdapter:
    """Concrete MidiOutputPort adapter building validated mido messages for GarageBand."""

    port_name: str = "IAC Driver Bus 1"
    sent_messages: list[SentMidiRecord] = field(default_factory=list)

    def send_note_on(self, note: int, velocity: int, trigger_sample: int) -> None:
        """Validate and dispatch a MIDI note_on message via mido."""
        validate_midi_note_and_velocity(note=note, velocity=velocity)
        msg = mido.Message("note_on", note=note, velocity=velocity)
        self.sent_messages.append(
            SentMidiRecord(
                message_type=msg.type,
                trigger_sample=trigger_sample,
                note=msg.note,
                velocity=msg.velocity,
            )
        )

    def send_cc(self, cc_number: int, value: int, trigger_sample: int) -> None:
        """Validate native GarageBand CC (1 or 64) and dispatch control_change."""
        validate_garageband_cc(cc_number=cc_number, value=value)
        msg = mido.Message("control_change", control=cc_number, value=value)
        self.sent_messages.append(
            SentMidiRecord(
                message_type=msg.type,
                trigger_sample=trigger_sample,
                control=msg.control,
                value=msg.value,
            )
        )

    def send_sustain_pedal(self, sustain_on: bool, trigger_sample: int) -> None:
        """Send MIDI CC 64 (127 when sustain_on is True, 0 when False)."""
        val = MAX_MIDI_VAL if sustain_on else MIN_MIDI_VAL
        self.send_cc(cc_number=CC64_SUSTAIN_PEDAL, value=val, trigger_sample=trigger_sample)

    def send_pitch_bend(self, pitch_bend: int, trigger_sample: int) -> None:
        """Validate and dispatch a 14-bit pitchwheel message."""
        validate_pitch_bend(pitch_bend=pitch_bend)
        msg = mido.Message("pitchwheel", pitch=pitch_bend)
        self.sent_messages.append(
            SentMidiRecord(
                message_type=msg.type,
                trigger_sample=trigger_sample,
                pitch=msg.pitch,
            )
        )
