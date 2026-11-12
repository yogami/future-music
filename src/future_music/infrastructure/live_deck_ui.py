"""Self-contained HTML5 + WebAudio + Creative Performance Empath Deck UI (REQ-13, REQ-14)."""

from pathlib import Path

_HTML_PATH: Path = Path(__file__).with_name("live_deck.html")


def get_live_deck_html() -> str:
    """Return the layman-friendly interactive HTML5/WebAudio Empath Deck page."""
    return _HTML_PATH.read_text(encoding="utf-8")


LIVE_DECK_HTML: str = get_live_deck_html()
