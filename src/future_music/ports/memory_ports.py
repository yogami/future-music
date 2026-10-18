"""Secondary port protocols for Personal Stem & Memory Vault access."""

from typing import Protocol

from future_music.domain.empath_memory import MemoryAsset


class PersonalVaultPort(Protocol):
    """Port interface for registering and retrieving an Empath's autobiographical stems."""

    def register_asset(self, empath_id: str, asset: MemoryAsset) -> None:
        """Store a personal memory asset under the given empath profile ID."""

    def get_assets(self, empath_id: str) -> tuple[MemoryAsset, ...]:
        """Return all personal memory assets calibrated for the given empath profile ID."""
