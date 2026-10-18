"""In-memory repository adapter for PersonalVaultPort with default Empath profiles."""

from dataclasses import dataclass, field

from future_music.domain.empath_memory import MemoryAsset, ScaleMode, Vector4D

DEFAULT_BPM: float = 124.0
ROBERT_MILES_MOTIF: tuple[int, ...] = (65, 68, 72, 70, 68, 67, 65, 63)
ACID_SYNTH_MOTIF: tuple[int, ...] = (53, 54, 65, 53, 68, 66, 53, 53)
VOCAL_FREQ_BAND: tuple[float, float] = (1000.0, 3000.0)
PIANO_FREQ_BAND: tuple[float, float] = (800.0, 2800.0)
SYNTH_FREQ_BAND: tuple[float, float] = (400.0, 2200.0)


@dataclass
class InMemoryPersonalVault:
    """In-memory implementation of PersonalVaultPort storing per-Empath memory crates."""

    _storage: dict[str, list[MemoryAsset]] = field(default_factory=dict)

    def register_asset(self, empath_id: str, asset: MemoryAsset) -> None:
        """Append or replace a personal memory asset for empath_id."""
        bucket = self._storage.setdefault(empath_id, [])
        filtered = [existing for existing in bucket if existing.asset_id != asset.asset_id]
        filtered.append(asset)
        self._storage[empath_id] = filtered

    def get_assets(self, empath_id: str) -> tuple[MemoryAsset, ...]:
        """Return all personal memory assets registered for empath_id."""
        return tuple(self._storage.get(empath_id, []))


def _make_robert_miles(embedding: Vector4D) -> MemoryAsset:
    """Create Robert Miles Children piano motif asset with given autobiographical embedding."""
    return MemoryAsset(
        asset_id="robert_miles_children",
        title="Robert Miles - Children (Dream Piano Motif)",
        emotional_embedding=embedding,
        native_bpm=DEFAULT_BPM,
        root_key="F",
        native_mode=ScaleMode.NATURAL_MINOR,
        audio_stem_id="robert_miles_piano.wav",
        symbolic_motif=ROBERT_MILES_MOTIF,
        requires_modal_lock=False,
        dominant_freq_hz=PIANO_FREQ_BAND,
    )


def _make_sufi_vocal(embedding: Vector4D) -> MemoryAsset:
    """Create Sufi Qawwali vocal & ney flute asset with given autobiographical embedding."""
    return MemoryAsset(
        asset_id="sufi_vocal",
        title="Qawwali Vocal & Ney Flute in F Phrygian",
        emotional_embedding=embedding,
        native_bpm=DEFAULT_BPM,
        root_key="F",
        native_mode=ScaleMode.PHRYGIAN,
        audio_stem_id="sufi_vocal.wav",
        symbolic_motif=(),
        requires_modal_lock=True,
        dominant_freq_hz=VOCAL_FREQ_BAND,
    )


def _make_acid_synth(embedding: Vector4D) -> MemoryAsset:
    """Create Berlin 303 acid lead asset with given autobiographical embedding."""
    return MemoryAsset(
        asset_id="berlin_acid_synth",
        title="Berlin 4 AM 303 Acid Lead",
        emotional_embedding=embedding,
        native_bpm=DEFAULT_BPM,
        root_key="F",
        native_mode=ScaleMode.PHRYGIAN,
        audio_stem_id="berlin_acid_synth.wav",
        symbolic_motif=ACID_SYNTH_MOTIF,
        requires_modal_lock=False,
        dominant_freq_hz=SYNTH_FREQ_BAND,
    )


def create_default_empath_vault() -> InMemoryPersonalVault:
    """Create a vault pre-calibrated with distinct Empath autobiographical profiles."""
    vault = InMemoryPersonalVault()
    euro_assets = (
        _make_robert_miles((0.35, 0.55, 0.30, 0.95)),
        _make_sufi_vocal((0.10, 0.70, 0.65, 0.60)),
        _make_acid_synth((-0.30, 0.95, 0.90, 0.15)),
    )
    mystic_assets = (
        _make_sufi_vocal((0.35, 0.55, 0.30, 0.95)),
        _make_robert_miles((0.65, 0.40, 0.15, 0.50)),
        _make_acid_synth((-0.30, 0.95, 0.90, 0.15)),
    )
    for asset in euro_assets:
        vault.register_asset("empath_euro_90s", asset)
    for asset in mystic_assets:
        vault.register_asset("empath_eastern_mystic", asset)
    return vault
