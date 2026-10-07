"""Side-by-side rendering of sanitization modes for a few fixtures."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from ..nb import esc
from ..transforms import TokenizerProtocol, render_modes

MODES = ("redact", "label", "label_number", "label_token")


def side_by_side(
    fixture: Mapping,
    predictions: Mapping[str, Sequence[dict]],
    *,
    modes: Iterable[str] = MODES,
    token_vault: TokenizerProtocol | None = None,
) -> str:
    """Markdown for one fixture: rows = detectors, columns = modes.

    predictions: {detector: spans for this fixture}.
    """
    modes = list(modes)
    parts = [
        f"### Example `{fixture['id']}`",
        "",
        f"> {esc(fixture['text'])}",
        "",
        "| detector | " + " | ".join(modes) + " |",
        "|" + "|".join(["---"] * (len(modes) + 1)) + "|",
    ]
    for detector, spans in predictions.items():
        rendered = render_modes(
            fixture["text"], list(spans), modes=modes, token_vault_client=token_vault
        )
        cells = [detector] + [esc(rendered.get(m, "")) for m in modes]
        parts.append("| " + " | ".join(cells) + " |")
    return "\n".join(parts)
