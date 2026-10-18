"""macOS IAC Driver MIDI adapter enforcing native GarageBand Alchemy controls (REQ-03, REQ-11)."""

from dataclasses import dataclass, field
from typing import Any

import mido

from future_music.domain.clock_and_midi import (
    CC64_SUSTAIN_PEDAL,
    MAX_MIDI_VAL,
    MIN_MIDI_VAL,
    validate_garageband_cc,
    validate_midi_note_and_velocity,
    validate_pitch_bend,
)

VIRTUAL_IAC_PORT_NAME: str = "Empath Conductor IAC"
_SHARED_PORT_CACHE: dict[str, Any] = {}


def _find_online_iac_port() -> str | None:
    """Return the first online macOS CoreMIDI output port matching 'IAC', or None."""
    for name in mido.get_output_names():
        if "IAC" in name.upper():
            return str(name)
    return None


def _open_fresh_coremidi_port() -> tuple[Any, str]:
    """Open an online IAC Driver port or create a virtual CoreMIDI port."""
    iac_name = _find_online_iac_port()
    if iac_name is not None:
        return (mido.open_output(iac_name), iac_name)
    return (mido.open_output(VIRTUAL_IAC_PORT_NAME, virtual=True), VIRTUAL_IAC_PORT_NAME)


def _acquire_coremidi_output() -> tuple[Any, str]:
    """Return cached active CoreMIDI output port or open a fresh one."""
    cached = _SHARED_PORT_CACHE.get("port")
    if not getattr(cached, "closed", True):
        return (cached, str(_SHARED_PORT_CACHE["name"]))
    port, resolved_name = _open_fresh_coremidi_port()
    _SHARED_PORT_CACHE["port"] = port
    _SHARED_PORT_CACHE["name"] = resolved_name
    return (port, resolved_name)


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
    """Concrete MidiOutputPort adapter transmitting real CoreMIDI messages to GarageBand."""

    port_name: str = "IAC Driver Bus 1"
    sent_messages: list[SentMidiRecord] = field(default_factory=list)
    active_port_name: str = field(init=False, default="")
    is_hardware_connected: bool = field(init=False, default=False)
    _hw_port: Any = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        """Bind to macOS CoreMIDI IAC Driver or virtual CoreMIDI port on startup."""
        self._hw_port, self.active_port_name = _acquire_coremidi_output()
        self.is_hardware_connected = self._hw_port is not None

    def _transmit(self, msg: mido.Message, record: SentMidiRecord) -> None:
        """Send mido.Message over the live CoreMIDI port and append audit record."""
        if not getattr(self._hw_port, "closed", True):
            self._hw_port.send(msg)
        self.sent_messages.append(record)

    def send_note_on(self, note: int, velocity: int, trigger_sample: int) -> None:
        """Validate and dispatch a MIDI note_on message via CoreMIDI."""
        validate_midi_note_and_velocity(note=note, velocity=velocity)
        msg = mido.Message("note_on", note=note, velocity=velocity)
        rec = SentMidiRecord(msg.type, trigger_sample, note=msg.note, velocity=msg.velocity)
        self._transmit(msg, rec)

    def send_cc(self, cc_number: int, value: int, trigger_sample: int) -> None:
        """Validate native GarageBand CC (1 or 64) and dispatch control_change via CoreMIDI."""
        validate_garageband_cc(cc_number=cc_number, value=value)
        msg = mido.Message("control_change", control=cc_number, value=value)
        rec = SentMidiRecord(msg.type, trigger_sample, control=msg.control, value=msg.value)
        self._transmit(msg, rec)

    def send_sustain_pedal(self, sustain_on: bool, trigger_sample: int) -> None:
        """Send MIDI CC 64 (127 when sustain_on is True, 0 when False)."""
        val = MAX_MIDI_VAL if sustain_on else MIN_MIDI_VAL
        self.send_cc(cc_number=CC64_SUSTAIN_PEDAL, value=val, trigger_sample=trigger_sample)

    def send_pitch_bend(self, pitch_bend: int, trigger_sample: int) -> None:
        """Validate and dispatch a 14-bit pitchwheel message via CoreMIDI."""
        validate_pitch_bend(pitch_bend=pitch_bend)
        msg = mido.Message("pitchwheel", pitch=pitch_bend)
        rec = SentMidiRecord(msg.type, trigger_sample, pitch=msg.pitch)
        self._transmit(msg, rec)

    def close(self) -> None:
        """Close the active CoreMIDI output port."""
        if self._hw_port is not None:
            self._hw_port.close()
        self.is_hardware_connected = False
