"""Deterministic target-cardinality policy for REST3D object phrases."""

from __future__ import annotations

import re


_EXPLICIT_GROUP_RE = re.compile(
    r"^(?:a\s+)?(?:pair|pairs|group|groups|set|sets|several|multiple|many|"
    r"two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\b",
    re.IGNORECASE,
)

# Everything after one of these relations describes a landmark, location, or an
# attached component rather than changing the cardinality of the target head.
_RELATION_RE = re.compile(
    r"\b(?:in front of|next to|with|inside|on|in|behind|beside|near|under|"
    r"below|above|against|at|around|along|across|by|between|facing|"
    r"surrounding|of)\b",
    re.IGNORECASE,
)

_PLURAL_TARGET_NOUNS = {
    "appliances", "armchairs", "beds", "benches", "bookcases", "books",
    "bottles", "boxes", "branches", "cabinets", "candles", "chairs",
    "coats", "containers", "couches", "cups", "cushions", "desks",
    "dishes", "drawers", "fixtures", "flowers", "frames", "items", "lamps",
    "lights", "monitors", "nightstands", "objects", "ottomans", "pictures",
    "pillows", "plants", "pots", "racks", "recliners", "shelves", "sideboards",
    "sofas", "stools", "tables", "units", "vases", "wardrobes",
}

_SINGULAR_S_ENDINGS = {
    "bass", "bus", "canvas", "cactus", "chassis", "glass", "grass", "lens",
    "mattress", "news", "series", "species", "status", "truss",
}


def classify_target_cardinality(target: str) -> dict[str, str]:
    """Classify a v5 canonical target as one instance or a plural instance group.

    This intentionally handles the compact noun phrases produced by the object-list
    prompt rather than attempting general English parsing. The decision and reason are
    both persisted so a run can be audited without recreating parser state.
    """
    text = re.sub(r"\s+", " ", str(target or "")).strip().casefold()
    if not text:
        return {"cardinality": "single", "reason": "empty target"}

    if _EXPLICIT_GROUP_RE.search(text):
        return {"cardinality": "plural_group", "reason": "explicit group/count word"}

    if re.search(
        r"\bbuilt[ -]in\b.*\b(?:cabinet|cabinets|shelf|shelves|shelving|"
        r"wardrobe|wardrobes|closet|closets)\b",
        text,
    ):
        return {"cardinality": "single", "reason": "built-in unified installation"}

    head = _RELATION_RE.split(text, maxsplit=1)[0].strip()
    words = re.findall(r"[a-z0-9]+", head)
    if not words:
        return {"cardinality": "single", "reason": "no target-head token"}

    # The object-list contract treats built-in cabinetry and a singular named system as
    # one functional object even when the phrase mentions multiple compartments.
    if "system" in words:
        return {"cardinality": "single", "reason": "singular unified system"}
    plural_nouns = [word for word in words if word in _PLURAL_TARGET_NOUNS]
    if plural_nouns:
        return {
            "cardinality": "plural_group",
            "reason": f"plural target noun: {plural_nouns[0]}",
        }

    # Cover novel regular plurals without making singular words such as ``glass`` or
    # ``mattress`` plural. Trailing participles are skipped for phrases like
    # ``coats hanging on coat rack``.
    meaningful = [
        word for word in words
        if word not in {"hanging", "arranged", "grouped", "located", "placed", "standing"}
    ]
    last = meaningful[-1] if meaningful else words[-1]
    if last.endswith("s") and last not in _SINGULAR_S_ENDINGS:
        return {"cardinality": "plural_group", "reason": f"regular plural head: {last}"}

    return {"cardinality": "single", "reason": f"singular target head: {last}"}


def is_plural_group_target(target: str) -> bool:
    """Return whether v5 should preserve selected masks as separate instances."""
    return classify_target_cardinality(target)["cardinality"] == "plural_group"
