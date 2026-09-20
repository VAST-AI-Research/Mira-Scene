"""Selection-level verification for the SAM3 grounding agent.

The grounding agent may need several SAM3 prompts to find one object.  A selector is
therefore allowed to return several candidate masks, but every returned mask must belong
to the same physical instance.  This module contains the model-independent contract and
the defensive response parsing used by the online verifier in ``agent_core.py``.
"""

from __future__ import annotations

import json
import re
from typing import Any


VERIFIER_SYSTEM_PROMPT = """\
You are a strict segmentation selection verifier.  The original query defines the target.
Several masks are valid only when they are visible parts of the same physical instance.
Never combine separate instances, even when the query uses a plural or group phrase.
Every selected mask must belong to the original target rather than an independently
nameable neighbouring object outside it.

Be especially strict for free-standing furniture and other large reconstructable objects:
chairs, armchairs, sofas, tables, desks, beds, cabinets, shelves, benches, stools and
similar furnishings.  Two separately reconstructable instances of any such object are
different_object, even when they have the same semantic name, material or color, touch,
overlap in the image, or one is partly occluded by the other.  For example, a rear gray
armchair is not a visible part of a front gray armchair, and a neighbouring chair back is
not that chair's armrest.  In that situation keep at most the candidate for the one
canonical instance; do not union the other instance into it.  The other candidate may be
considered later by a separate unclaimed-candidate/recycle pass, but it is not membership
in this selection.

Inspect the raw image, numbered candidate board, individual context crops, mask-only RGB
cutouts and silhouettes, and finally the proposed union.  First identify the independently
nameable object represented by the actual pixels inside each mask, without assuming it is
the requested target.  White pixels in a mask-only cutout are outside the candidate and
must not influence its identity.  Only after this independent identification should you
compare candidates to the original target and to one another.

A candidate that is a chair, table, wall, floor, or any other separately reconstructable
object must be dropped even if it touches or overlaps the target.  Similar color, proximity,
one shared union outline, or background visible inside a context bounding box is not proof
of shared identity.  Do not reject a candidate merely because it is a disjoint visible part
of the target object when its mask-only pixels provide positive evidence of that identity.

The confidence field is an audit signal for later review; it must not override a
candidate's relation. A low-confidence candidate is still valid when its relation is
explicitly same_target. Return JSON only with this schema:
{
  "verdict": "pass" | "repair" | "retry" | "uncertain",
  "keep_ids": [<candidate numbers>],
  "candidates": [
    {"id": <number>, "relation": "same_target" | "different_object" | "uncertain",
     "identity": "<short name>", "confidence": "high" | "low",
     "reason": "<short concrete reason>"}
  ],
  "reason": "<short overall reason>"
}

Use "pass" only when all proposed candidates are the same physical instance. Use
"repair" when a strict subset in keep_ids fixes the proposal. Use "retry" only when you
need the selector to reconsider the same candidates. Use "uncertain" when you cannot
make a reliable attribution. Membership is determined by relation, not confidence:
relation="same_target" is sufficient, including when confidence="low". Never invent a
candidate number.
"""


PLURAL_MEMBERSHIP_VERIFIER_SYSTEM_PROMPT = """\
You are a strict plural-target segmentation membership verifier. The canonical query
describes a GROUP of objects, so different valid members must remain separate output
masks; never union them and never reject a candidate merely because it is a different
instance of the same group. You must audit EVERY proposed candidate independently.

For each candidate, require all of the following before accepting it:
1. group_membership=yes: the pixels inside the mask are actually the requested target
   class, not a neighbouring object, support surface, object part, or background;
2. instance_status=single_instance: the mask is one independently identifiable physical
   instance. Reject fragments such as only a chair back, table top, legs, or foliage;
   reject a mask containing two merged instances; reject duplicate masks for the same
   instance (use duplicate_of when applicable);
3. border_status=not_cropped: the target is not cut off by the image border. Objects may
   be occluded by other scene objects when their identity and instance remain clear,
   but an object visibly truncated by the left, right, top, or bottom image edge must
   be rejected. A mask touching an image edge is a warning, not proof by itself; inspect
   the raw image and the mask-only pixels to decide whether the physical object is
   actually cropped;
4. visibility must be fully_visible or scene_occluded, never unidentifiable.

Inspect the raw image, the numbered candidate board, and for EVERY candidate its context
crop and mask-only RGB cutout plus silhouette. The mask-only cutout's white pixels are
outside the candidate and must not influence identity. Candidate IDs are 1-based and
come from the latest candidate board. Do not use area, score, or largest-area fallback
as evidence. Do not accept a candidate just because it is near another target or shares
its color. A scene-occluded but clearly identifiable whole chair/table/plant/etc. is
allowed; an image-border fragment is not.

Return JSON only:
{
  "verdict": "pass" | "repair" | "uncertain",
  "keep_ids": [<candidate numbers that satisfy ALL requirements>],
  "candidates": [
    {"id": <number>,
     "group_membership": "yes" | "no" | "uncertain",
     "instance_status": "single_instance" | "fragment" | "multiple_instances" | "duplicate" | "uncertain",
     "visibility": "fully_visible" | "scene_occluded" | "unidentifiable",
     "border_status": "not_cropped" | "cropped_by_image" | "uncertain",
     "duplicate_of": null | <candidate number>,
     "identity": "<short name>",
     "confidence": "high" | "low",
     "reason": "<short concrete reason>"}
  ],
  "reason": "<short overall reason>"
}

Use verdict=pass only if every proposed candidate passes all requirements. Use repair
when at least one candidate passes and one or more proposed candidates fail. Use
uncertain when no candidate has enough evidence to pass. Membership in keep_ids is valid
only when the candidate record independently satisfies every requirement; never infer a
missing field, and never invent a candidate number.
"""


# The verifier answers membership within one canonical object. A plural/group query has
# different semantics: its selected masks are separate instances and must remain separate
# outputs. Keep this deliberately small and conservative; unknown phrases stay singular.
_PLURAL_HEADS = {
    "chairs", "armchairs", "recliners", "stools", "benches", "sofas", "couches",
    "tables", "desks", "beds", "cabinets", "shelves", "bookcases", "wardrobes",
    "lamps", "lights", "fixtures", "books", "pillows", "cushions", "plants",
    "vases", "flowers", "branches", "candles", "cups", "bottles", "boxes",
    "pictures", "frames", "chairs", "objects", "items", "dishes", "containers",
}


def is_plural_target(text: str) -> bool:
    """Return whether a canonical target asks for a collection of instances."""
    # The target noun occurs before its spatial/container qualifier: ``books on
    # table`` is plural while ``bench behind chairs`` is singular. ``with`` is
    # included because a chair with cushions remains one chair.
    head = re.split(r"\b(?:on|in|with|behind|beside|under|above|at|near)\b",
                    str(text or "").lower(), maxsplit=1)[0]
    words = re.findall(r"[a-z0-9]+", head)
    if not words:
        return False
    if any(w in {"pair", "pairs", "group", "groups", "set", "sets", "several", "multiple"}
           for w in words):
        return True
    return words[-1] in _PLURAL_HEADS


def _json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from a model response."""
    if not text:
        return None
    candidate = text.strip()
    # Models sometimes wrap JSON in markdown fences or a short explanation.
    candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.I)
    try:
        value = json.loads(candidate)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for start, char in enumerate(candidate):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(candidate[start:])
        except json.JSONDecodeError:
            continue
        return value if isinstance(value, dict) else None
    return None


def parse_verifier_response(text: str, available_ids: set[int]) -> dict[str, Any]:
    """Normalize a verifier response and reject unsafe/invalid candidate IDs."""
    raw = _json_object(text)
    if raw is None:
        return {"verdict": "uncertain", "keep_ids": [], "raw": text,
                "error": "response was not a JSON object"}

    verdict = str(raw.get("verdict", "uncertain")).lower().strip()
    if verdict not in {"pass", "repair", "retry", "uncertain"}:
        verdict = "uncertain"
    keep_raw = raw.get("keep_ids", [])
    if not isinstance(keep_raw, list):
        keep_raw = []
    keep_ids = []
    for value in keep_raw:
        try:
            idx = int(value)
        except (TypeError, ValueError):
            continue
        if idx in available_ids and idx not in keep_ids:
            keep_ids.append(idx)
    out = dict(raw)
    out["verdict"] = verdict
    out["keep_ids"] = sorted(keep_ids)
    out["raw"] = text
    return out


_PLURAL_VALID_INSTANCE_STATUSES = {"single_instance"}
_PLURAL_VALID_VISIBILITY = {"fully_visible", "scene_occluded"}
_PLURAL_VALID_BORDER = {"not_cropped"}


def parse_plural_membership_response(text: str, available_ids: set[int]) -> dict[str, Any]:
    """Normalize the strict per-candidate plural membership response."""
    raw = _json_object(text)
    if raw is None:
        return {"verdict": "uncertain", "keep_ids": [], "candidates": [],
                "raw": text, "error": "response was not a JSON object"}

    verdict = str(raw.get("verdict", "uncertain")).lower().strip()
    if verdict not in {"pass", "repair", "uncertain"}:
        verdict = "uncertain"
    records = raw.get("candidates", [])
    if not isinstance(records, list):
        records = []
    normalized_records = []
    by_id = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        try:
            idx = int(record.get("id"))
        except (TypeError, ValueError):
            continue
        if idx not in available_ids or idx in by_id:
            continue
        membership = str(record.get("group_membership", "uncertain")).lower().strip()
        if membership not in {"yes", "no", "uncertain"}:
            membership = "uncertain"
        instance_status = str(record.get("instance_status", "uncertain")).lower().strip()
        if instance_status not in {
            "single_instance", "fragment", "multiple_instances", "duplicate", "uncertain",
        }:
            instance_status = "uncertain"
        visibility = str(record.get("visibility", "unidentifiable")).lower().strip()
        if visibility in {"partially_occluded", "partially_visible", "occluded"}:
            # Common synonym used by VLMs; the contract's scene_occluded value
            # means the same thing and remains distinct from border cropping.
            visibility = "scene_occluded"
        if visibility not in {"fully_visible", "scene_occluded", "unidentifiable"}:
            visibility = "unidentifiable"
        border_status = str(record.get("border_status", "uncertain")).lower().strip()
        if border_status not in {"not_cropped", "cropped_by_image", "uncertain"}:
            border_status = "uncertain"
        # ``duplicate_of`` is mandatory evidence.  An omitted or malformed field
        # must not silently become ``None`` (which would otherwise look like an
        # explicit statement that this candidate is not a duplicate).
        duplicate_field_valid = "duplicate_of" in record
        duplicate_raw = record.get("duplicate_of")
        duplicate_of = None
        if duplicate_raw is not None:
            try:
                duplicate_of = int(duplicate_raw)
            except (TypeError, ValueError):
                duplicate_field_valid = False
                duplicate_of = None
            if duplicate_of not in available_ids:
                duplicate_field_valid = False
                duplicate_of = None
        normalized = dict(record)
        normalized.update({
            "id": idx,
            "group_membership": membership,
            "instance_status": instance_status,
            "visibility": visibility,
            "border_status": border_status,
            "duplicate_of": duplicate_of,
            "duplicate_field_valid": duplicate_field_valid,
            "confidence": str(record.get("confidence", "low")).lower().strip()
            if str(record.get("confidence", "low")).lower().strip() in {"high", "low"}
            else "low",
        })
        normalized_records.append(normalized)
        by_id[idx] = normalized

    keep_raw = raw.get("keep_ids", [])
    if not isinstance(keep_raw, list):
        keep_raw = []
    requested_keep = []
    for value in keep_raw:
        try:
            idx = int(value)
        except (TypeError, ValueError):
            continue
        if idx in available_ids and idx not in requested_keep:
            requested_keep.append(idx)
    # The record fields, rather than keep_ids, are authoritative. This prevents a model
    # from accidentally accepting a candidate while omitting the required evidence.
    evidence_keep = []
    for idx, record in by_id.items():
        if (
            record["group_membership"] == "yes"
            and record["instance_status"] in _PLURAL_VALID_INSTANCE_STATUSES
            and record["visibility"] in _PLURAL_VALID_VISIBILITY
            and record["border_status"] in _PLURAL_VALID_BORDER
            and record["duplicate_field_valid"]
            and record["duplicate_of"] is None
        ):
            evidence_keep.append(idx)
    keep_ids = sorted(set(requested_keep) & set(evidence_keep))
    out = dict(raw)
    out.update({
        "verdict": verdict,
        "keep_ids": keep_ids,
        "candidates": sorted(normalized_records, key=lambda item: item["id"]),
        "raw": text,
        "requested_keep_ids": sorted(requested_keep),
    })
    return out


def resolve_plural_membership_selection(selected_ids: list[int], response: dict[str, Any]) -> tuple[list[int], str]:
    """Keep only candidates with complete positive plural-membership evidence."""
    original = sorted(set(selected_ids))
    keep = sorted(set(response.get("keep_ids", [])) & set(original))
    if keep == original and original:
        return keep, "plural_verifier_pass"
    if keep:
        return keep, "plural_verifier_repair"
    return [], "plural_verifier_uncertain_no_selection"


def _same_target_ids(response: dict[str, Any]) -> set[int]:
    """Return IDs backed by an explicit same-target judgment."""
    candidates = response.get("candidates", [])
    if not isinstance(candidates, list):
        return set()
    supported = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        try:
            idx = int(candidate.get("id"))
        except (TypeError, ValueError):
            continue
        relation = str(candidate.get("relation", "")).lower().strip()
        if relation == "same_target":
            supported.add(idx)
    return supported


def resolve_verified_selection(selected_ids: list[int], areas: dict[int, int],
                               response: dict[str, Any]) -> tuple[list[int], str]:
    """Apply the conservative selection contract to a normalized verifier response.

    Candidate membership is driven by explicit same-target relations; confidence is kept
    for later audit but does not decide membership. A pass keeps the exact original set,
    while a repair can only remove candidates. If no usable same-target relation exists,
    return no mask rather than guessing a core candidate by area.
    """
    original = sorted(set(selected_ids))
    proposed = sorted(set(response.get("keep_ids", [])))
    supported = _same_target_ids(response)
    relation_keep = sorted(set(proposed) & set(original) & supported)
    if proposed == original and set(original) <= supported:
        return original, "verifier_pass"
    if relation_keep and set(relation_keep) < set(original):
        return relation_keep, "verifier_repair"
    return [], "verifier_uncertain_no_selection"
