"""Secondary port protocol for GarageBand / macOS IAC Driver MIDI output."""

from typing import Protocol


class MidiOutputPort(Protocol):
    """Port interface for sending sample-accurate MIDI notes and native GarageBand CCs."""

    def send_note_on(self, note: int, velocity: int, trigger_sample: int) -> None:
        """Dispatch a MIDI Note-On message scheduled at trigger_sample."""

    def send_cc(self, cc_number: int, value: int, trigger_sample: int) -> None:
        """Dispatch a validated GarageBand MIDI CC message (CC 1 or CC 64)."""

    def send_sustain_pedal(self, sustain_on: bool, trigger_sample: int) -> None:
        """Dispatch a MIDI CC 64 Sustain Pedal event (127 for on, 0 for off)."""

    def send_pitch_bend(self, pitch_bend: int, trigger_sample: int) -> None:
        """Dispatch a 14-bit MIDI Pitch Bend event (-8192..8191)."""
