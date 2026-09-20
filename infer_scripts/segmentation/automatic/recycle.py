"""Recycle SAM3 candidates that no delivered instance claimed -- the "missed foreground" pass.

Motivation (3dfuture_scene_6): the VLM listed one armchair twice ("gray"/"blue"), both agents
grounded to the SAME left chair, dedup correctly deleted one, and the right-hand sofa was
never claimed. But SAM3 HAD produced a mask for it: candidate 'couch' #1 / 'sofa' #1 covered
~67% of it. The agent only ever chose among ONE phrase's candidates, so masks other phrases
left behind were thrown away. This pass reads them back off disk -- no SAM3 rerun.

Gate order matters; each step is here because measurement said so, not by intuition:

  0. dedup candidates against each other (IoU > 0.85). 'sofa' and 'couch' returned the same
     three masks in scene_6, so raw candidate counts triple-count.
  1. covered-by-one-instance >= 0.80  -> duplicate. Same shape as stage1's own dedup rule.
  2. covered-by-union >= 0.50         -> a *group* mask spanning several delivered objects,
                                         not a new object.
  3. head noun in BG_HEADS            -> background. Must be the HEAD noun, not a substring:
                                         'floor lamp', 'ceiling fan', 'wall decoration',
                                         'floor mirror' are real foreground. Verified over all
                                         455 distinct phrases in this batch: head-noun matching
                                         flags 25 phrases, all genuinely background, and the 14
                                         phrases that contain a background word but survive are
                                         13 real objects + 'floor area' (listed explicitly).
  4. area < MIN_FRAC of image          -> fragment. Keep this low enough for heavily occluded
                                         furniture: in flux2_013_dining_room a valid rear-chair
                                         candidate is only 0.32% of the image. Candidates that
                                         pass this gate still require the VLM whole-object check.
  5. NO upper area cap. Measured against the 567 delivered non-floor instances: a desk is 54%
     of its image, beds are 26/20/17%. Any cap low enough to catch background slabs also
     deletes real furniture, so background is rejected by phrase (step 3), never by size.
  6. survivors -> one VLM yes/no per candidate. Judging by eye over the 56 orphans >=0.5%:
     ~20 are real missed foreground, ~3 background, and ~30 are ambiguous plants/book stacks
     where it is genuinely unclear whether the blob is a new instance or part of a delivered
     one. Auto-accepting all of them would inject junk, so a cheap confirmation runs instead
     (one four-image query per candidate, still much cheaper than an agent loop).

Deliberate asymmetry, flagged rather than hidden: rugs/carpets are rejected here as background,
yet stage1 already delivers some as foreground ('area_rug_000' 12% of image,
'round_black_adidas_rug_000' 10%). That follows the instruction to treat carpet/floor/wall as
background for recycling; it does mean the two sides do not use one rug convention.

The CLI still writes an offline audit plan.  ``apply_unclaimed_recycle`` is the online
entry point used by stage1: it confirms candidates, writes accepted masks, and records a
manifest next to the scene output.
"""
import argparse
import json
import os
import shutil
import sys
from glob import glob

import numpy as np
import pycocotools.mask as mu
from PIL import Image

from ..utils.rendering.vis import save_seg_obj
from .policies.vlm_policy import object_profile_violation_reason

sys.path.insert(0, os.path.abspath("."))

OUT_ROOT = "exp_seg100/output"
SRC = os.getenv("REST3D_RECYCLE_SOURCE_ROOT", "./data")

# The seg15 benchmark keeps each scene's render inside the scene directory instead of one
# flat <scene>.png per image, so the source image cannot be derived from SRC alone. Both
# layouts are tried in order; --src / --src_glob override.
SRC_SEG15 = os.getenv("REST3D_RECYCLE_SEG15_SOURCE_ROOT", SRC)
SRC_PATTERNS = ("{src}/{scene}.png", "{src}/{scene}/renders/camera_0/scene.png")


def find_source_image(src: str, scene: str, extra: str | None = None) -> str:
    pats = ((extra,) if extra else ()) + SRC_PATTERNS
    for p in pats:
        cand = p.format(src=src, scene=scene)
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        f"no source image for {scene!r} under {src!r}; tried {[p.format(src=src, scene=scene) for p in pats]}")

DUP_COVER = 0.80
GROUP_COVER = 0.50
CAND_IOU = 0.85
# 0.3% admits the occluded dining-chair candidate in flux2_013 while still filtering the
# next smaller 0.22% generic "seat" candidate. This is only a pre-VLM gate, not acceptance.
MIN_FRAC = 0.003
UPGRADE_OLD_COVER = 0.95
UPGRADE_MAX_OLD_SHARE = 0.80
UPGRADE_MIN_AREA_RATIO = 1.15
UPGRADE_MIN_EXTRA_FRAC = 0.0005
UPGRADE_MAX_EXTRA_CONFLICT = 0.10
CROP_CONTEXT_SCALE = 1.5
CROP_UPSCALE = 1.5
CROP_MAX_EDGE = 1024
VERIFIER_MODES = (
    "legacy_joint", "identity_geometry", "identity_upgrade", "shadow_compare",
)
DEFAULT_VERIFIER_MODE = "identity_upgrade"
OBJECT_PROFILES = ("standard", "major", "major_v3", "major_v4", "major_v5", "major_v6")
DEFAULT_OBJECT_PROFILE = "standard"

# Head nouns that denote background. Applied to the LAST word of the phrase only.
BG_HEADS = {
    "floor", "flooring", "ground", "carpet", "rug", "wall", "walls",
    "ceiling", "surface", "panel", "backdrop", "background", "tile", "tiles",
}
# Phrases whose head noun is innocuous but which name a background region anyway.
BG_PHRASES = {"floor area"}

PROMPT = """You are checking one segmentation mask in an indoor scene photo.

Image 1: the full scene. The region under consideration is tinted {tint}.
Image 2: the same scene with every object we have ALREADY segmented outlined in cyan.

The {tint} region was produced by a detector for the phrase "{phrase}" but no object in our
current output claims it.

Answer whether the {tint} region is ONE distinct foreground object that we are MISSING.

Answer NO if the {tint} region is:
 - floor, wall, ceiling, a rug/carpet, or any flat background surface
 - a part of an object already outlined in cyan (e.g. a cushion on an outlined sofa,
   a drawer of an outlined cabinet)
 - several separate objects lumped together
 - mostly empty space or a sliver of another object

Answer YES only if it is a single, whole, physically distinct foreground object that no
cyan outline covers.

Reply with exactly one line:
VERDICT: YES|NO
OBJECT: <a short noun phrase naming it, or "-" if NO>
WHY: <one short sentence>"""

NEUTRAL_PROMPT = """You are checking one candidate segmentation mask in an indoor scene.

You receive four images in this exact order:
1. A full-scene location view: the candidate has a thin yellow boundary.
2. A full-scene comparison view: yellow is the candidate and cyan marks objects already
   delivered by the segmentation system.
3. A moderately enlarged RGB context crop around the candidate, with NO annotations.
4. The same crop with only the candidate-mask pixels retaining their original RGB; all
   pixels outside the candidate are white.

Yellow and cyan are annotations, not object colors. Use Images 1-2 only to understand
location, occlusion, and whether cyan already claims the instance. Use Images 3-4 to see
what the candidate actually depicts without the yellow boundary obscuring its appearance.
White pixels in Image 4 mean "outside this mask", not white scene content.

Decide whether the candidate's visible pixels belong to ONE physically distinct foreground
object that is MISSING from the delivered cyan objects.

IMPORTANT OCCLUSION RULE: a separate large furniture instance may be heavily occluded by a
table, another piece of furniture, or the image boundary. Its candidate mask does NOT need
to be a complete silhouette, and its visible portions may be separated by occluders. Answer
YES when the remaining visible shape is still coherent enough to identify one independent
instance, such as a chair, armchair, dining chair, stool, bench, ottoman, sofa, couch,
loveseat, chaise, table, desk, console, cabinet, sideboard, wardrobe, dresser, bookcase,
shelving unit, TV stand, bed, daybed, or bunk bed. A recognizable chair back, armrest, seat,
leg set, tabletop, cabinet face, bed frame, or headboard can be sufficient evidence when the
parts jointly belong to that one missing instance.

Answer NO if the candidate is:
- floor, wall, ceiling, rug/carpet, background, shadow, or empty space
- merely an internal component of an already-delivered object, such as its cushion, pillow,
  drawer, cabinet door, mattress, or tabletop
- pixels from two or more separate physical instances, even if they share the same category
- unrelated fragments scattered across different objects
- too ambiguous to assign to one independent physical instance

Do not reject a candidate merely because the independent object is partially visible,
truncated, or heavily occluded. The key distinction is one occluded independent instance
versus a fragment of an existing instance or a mixture of multiple instances.

Reply with exactly one line per field:
VERDICT: YES|NO
OBJECT: <short noun phrase, or ->
WHY: <one short sentence>"""

IDENTITY_PROMPT = """You are judging ONLY the identity and instance coherence of one candidate mask.
Duplicate or previously-delivered status has already been checked using exact mask geometry
and is NOT part of this question. No delivered-object comparison image is provided.

Decide whether the DOMINANT area and structure of the candidate plausibly belongs to one
independent foreground instance. This includes furniture and coherent decorative arrangements.
The mask need not be a complete silhouette.
Visible parts may be disconnected because a table, another object, or the image boundary
occludes the object. A small minority of outlier pixels may be ignored. Answer YES when the
main components form one recognizable independent chair, sofa, table, cabinet, bed, or
similar furniture instance. Also answer YES for one complete container-and-content arrangement,
such as a decorative vase with plants, flowers, or branches; a pot or planter with its plant;
or a floral arrangement with its container. Treat that intended combination as one foreground
instance. Answer NO for flowers, leaves, branches, or container fragments alone when the intended
compound object is visibly incomplete, and when no physical instance clearly dominates or
substantial components belong to different instances.

Use the unannotated crop to map each mask-only component back into the scene. Check whether
occluders explain the gaps and whether upper, middle, and lower components align as plausible
parts of one object. Do not classify a component only from its height or nearby objects.

{object_policy}

Detector phrase aliases for near-identical SAM3 masks are included below as WEAK identity
clues. They are recall hypotheses, not proof: accept only when the visible pixels and geometry
support one of them. Contradictory aliases should make you inspect the images more carefully,
not combine multiple objects.

{aliases}

Reply with exactly one line per field:
VERDICT: YES|NO
OBJECT: <short noun phrase, or ->
WHY: <one short sentence>"""

MAJOR_IDENTITY_POLICY = """MAJOR-OBJECT PROFILE:
Answer YES only for a missing room-layout object: independent furniture, a large appliance or
functional machine, a large television, a freestanding floor lamp, a large floor-standing plant
with its pot, a freestanding full-length mirror, a room divider, or a major bathroom fixture.

Answer NO for any standalone pillow, cushion, mattress, sheet, duvet, comforter, quilt, blanket,
bedding, or throw. These soft items belong inside a bed/sofa/chair target or are omitted. Also
answer NO for curtains, blinds, rugs, mats, artwork, pictures, frames, mirrors attached to a wall,
ceiling/wall/table lights, vases, flowers, tabletop plants, books, papers, dishes, bottles, toys,
clothes, luggage, baskets, bins, trays, desktop electronics, shelf/cabinet contents, ornaments,
and other loose or decorative objects. Candidate size alone does not make an object major."""

MAJOR_V3_IDENTITY_POLICY = """MAJOR-V3 FOREGROUND PROFILE:
Answer YES for a missing independent foreground object allowed by the major-object profile. In
addition to furniture, equipment, large floor pot+plant compounds, and major fixtures, this
version also allows one coherent light fixture (including pendant, chandelier, ceiling, wall,
table, desk, or floor lamp) and one complete substantial freestanding rocking/ride-on play object
such as a wooden rocking horse.

Answer NO for standalone soft furnishings, curtains, blinds, rugs, wall artwork, mirrors attached
to a wall, vases, flower arrangements, tabletop plants, books, papers, dishes, bottles, small toys,
plush toys, toy vehicles, bead mazes, clothes, luggage, baskets, bins, trays, desktop electronics,
shelf/cabinet contents, ornaments, and other loose clutter. A light fixture or complete rocking
horse is allowed; small toys and isolated toy fragments are not."""

MAJOR_V4_IDENTITY_POLICY = """MAJOR-V4 FOREGROUND PROFILE:
Answer YES only for one independently meaningful foreground object that is physically
substantial or functionally important enough to belong in a scene-level inventory. Main
furniture, meaningful light fixtures, appliances/equipment, large plants with their pots,
and other coherent large functional or play objects can qualify. A heavily occluded important
object can still qualify even when its visible mask is small.

Answer NO for architectural background, small loose clutter, minor decorative accessories,
or standalone pillows, cushions, mattresses, bedding, and throws that belong to a bed, sofa,
or chair. Judge importance using both object role and physical scale in the scene; do not use
one long category allowlist and do not promote an object merely because it is floor-standing."""

UPGRADE_PROMPT = """You are comparing an existing segmentation mask with a larger SAM3 candidate.

The existing mask is named "{old_name}". Exact geometry already established that the candidate
contains at least 95% of the existing mask and adds substantial pixels that are not claimed by
other delivered objects. Your task is NOT to decide whether the candidate merely resembles some
foreground object. Decide whether it is a MORE COMPLETE AND STILL CLEAN segmentation of the SAME
physical instance as the existing mask.

You receive five images in this exact order:
1. Full-scene comparison: existing mask outlined in cyan, candidate outlined in yellow, and
   candidate-only added pixels tinted magenta.
2. Unannotated RGB context crop shared by both masks.
3. Existing-mask-only RGB crop; outside-mask pixels are white.
4. Candidate-mask-only RGB crop; outside-mask pixels are white.
5. Candidate-only added pixels; outside the added region is white.

Answer REPLACE only when all important added regions belong to the same physical instance and the
candidate is visibly more complete without introducing another object or substantial background.
For occluded furniture, disconnected added parts are allowed when the scene geometry clearly
explains that they are the same chair, sofa, table, cabinet, bed, shelf, bench, or similar object.
For a vase, pot, or planter, added flowers, branches, foliage, and the associated container may
likewise form one intended compound instance when the candidate cleanly captures the arrangement.

Answer KEEP when the candidate adds any independently nameable neighbouring object, mixes two
instances of the same category, adds substantial table/floor/wall/background, merely changes the
mask without improving it, or the same-instance assignment is uncertain. The safe default is KEEP.

Detector phrase aliases are weak recall clues, not proof:
{aliases}

Reply with exactly one line per field:
VERDICT: REPLACE|KEEP
OBJECT: <short noun phrase for the existing physical instance, or ->
WHY: <one short sentence>"""


def decode(js, shape):
    try:
        with open(js, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return []
    h, w = d.get("orig_img_h"), d.get("orig_img_w")
    if (h, w) != shape:
        return []
    out = []
    for rle in d.get("pred_masks", []):
        c = rle.encode("utf-8") if isinstance(rle, str) else rle
        try:
            m = mu.decode({"counts": c, "size": [h, w]}).astype(bool)
        except Exception:
            continue
        if m.sum():
            out.append(m)
    return out


def iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 0.0


def is_bg(phrase):
    p = phrase.strip().lower()
    return p in BG_PHRASES or (p.split() and p.split()[-1] in BG_HEADS)


TINTS = {"red": (255, 30, 30), "green": (30, 255, 30)}


def overlay(rgb, mask, finals=None, tint="red"):
    """Tinted candidate; optionally cyan outlines of delivered instances.

    The tint colour is a knob because the first run came back naming things "red sofa",
    "red chair" -- the VLM was reading the tint as the object's own colour. A control run
    with a different tint tells whether the tint drove the VERDICT too (fatal) or only
    contaminated the name (cosmetic).
    """
    import cv2
    col = TINTS[tint]
    img = rgb.copy()
    if finals is not None:
        img = (img * 0.65).astype(np.uint8)
        for f in finals:
            cs, _ = cv2.findContours(f.astype(np.uint8), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cs, -1, (0, 230, 230), 3)
    img[mask] = (0.4 * img[mask] + 0.6 * np.array(col, float)).astype(np.uint8)
    return Image.fromarray(img)


def neutral_overlay(rgb, mask, finals=None):
    """Render an annotation without tinting pixels enough to change object identity."""
    import cv2
    img = np.asarray(rgb).copy()
    if finals is not None:
        img = (img.astype(np.float32) * 0.72).astype(np.uint8)
        for f in finals:
            contours, _ = cv2.findContours(f.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, (0, 220, 220), 2)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (255, 220, 0), 3)
    return Image.fromarray(img)


def candidate_crop_box(mask, context_scale=CROP_CONTEXT_SCALE):
    """Return a centered context box around a non-empty candidate mask."""
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0, 0, width, height

    mask_width = int(xs.max() - xs.min() + 1)
    mask_height = int(ys.max() - ys.min() + 1)
    crop_width = min(width, max(mask_width, int(round(mask_width * context_scale))))
    crop_height = min(height, max(mask_height, int(round(mask_height * context_scale))))
    center_x = (float(xs.min()) + float(xs.max()) + 1.0) / 2.0
    center_y = (float(ys.min()) + float(ys.max()) + 1.0) / 2.0
    left = min(max(0, int(round(center_x - crop_width / 2.0))), width - crop_width)
    top = min(max(0, int(round(center_y - crop_height / 2.0))), height - crop_height)
    return left, top, left + crop_width, top + crop_height


def candidate_detail_images(rgb, mask, context_scale=CROP_CONTEXT_SCALE,
                            upscale=CROP_UPSCALE, max_edge=CROP_MAX_EDGE):
    """Create an unannotated context crop and an exact mask-only RGB crop.

    The crop includes modest surrounding context and is enlarged at most ``upscale`` times.
    The mask is resized separately with nearest-neighbor sampling so pixels outside it remain
    exactly white rather than acquiring annotation-like edge colors.
    """
    rgb = np.asarray(rgb).astype(np.uint8)
    left, top, right, bottom = candidate_crop_box(mask, context_scale=context_scale)
    crop_rgb = rgb[top:bottom, left:right]
    crop_mask = mask[top:bottom, left:right].astype(np.uint8) * 255

    scale = min(float(upscale), float(max_edge) / max(crop_rgb.shape[:2]))
    out_width = max(1, int(round(crop_rgb.shape[1] * scale)))
    out_height = max(1, int(round(crop_rgb.shape[0] * scale)))
    size = (out_width, out_height)
    context = Image.fromarray(crop_rgb).resize(size, Image.Resampling.LANCZOS)
    resized_mask = np.asarray(
        Image.fromarray(crop_mask).resize(size, Image.Resampling.NEAREST)
    ) > 127
    context_rgb = np.asarray(context)
    mask_only = np.full_like(context_rgb, 255)
    mask_only[resized_mask] = context_rgb[resized_mask]
    return context, Image.fromarray(mask_only)


def recycle_verifier_messages(candidate, rgb, finals):
    """Build the original joint identity + delivered-status verifier input."""
    candidate_img = neutral_overlay(rgb, candidate["mask"])
    delivered_img = neutral_overlay(rgb, candidate["mask"], [m for _, m in finals])
    context_crop, mask_only_crop = candidate_detail_images(rgb, candidate["mask"])
    return [{"role": "user", "content": [
        {"type": "text", "text": NEUTRAL_PROMPT},
        {"type": "text", "text": "Image 1: full-scene candidate location view."},
        {"type": "image", "image": candidate_img},
        {"type": "text", "text": "Image 2: full-scene delivered-object comparison view."},
        {"type": "image", "image": delivered_img},
        {"type": "text", "text": "Image 3: unannotated enlarged RGB context crop."},
        {"type": "image", "image": context_crop},
        {"type": "text", "text": "Image 4: mask-only RGB crop; outside-mask pixels are white."},
        {"type": "image", "image": mask_only_crop},
    ]}]


def candidate_alias_phrases(candidate):
    """Return case-insensitively unique NMS source phrases, representative first."""
    phrases = []
    seen = set()
    for phrase in [candidate.get("phrase", "")] + [
        alias.get("phrase", "") for alias in candidate.get("aliases", [])
    ]:
        phrase = str(phrase or "").strip()
        key = phrase.casefold()
        if phrase and key not in seen:
            phrases.append(phrase)
            seen.add(key)
    return phrases


def identity_verifier_messages(candidate, rgb, object_profile=DEFAULT_OBJECT_PROFILE):
    """Build an identity-only input; delivered status remains a geometric decision."""
    object_profile = normalize_object_profile(object_profile)
    candidate_img = neutral_overlay(rgb, candidate["mask"])
    context_crop, mask_only_crop = candidate_detail_images(rgb, candidate["mask"])
    aliases = candidate_alias_phrases(candidate)
    alias_text = "Detector phrase aliases:\n" + "\n".join(f"- {p}" for p in aliases)
    object_policy = {
        "major": MAJOR_IDENTITY_POLICY,
        "major_v3": MAJOR_V3_IDENTITY_POLICY,
        "major_v4": MAJOR_V4_IDENTITY_POLICY,
        "major_v5": MAJOR_V4_IDENTITY_POLICY,
    }.get(object_profile, "")
    return [{"role": "user", "content": [
        {"type": "text", "text": IDENTITY_PROMPT.format(
            aliases=alias_text,
            object_policy=object_policy,
        )},
        {"type": "text", "text": "Image 1: full-scene candidate location view."},
        {"type": "image", "image": candidate_img},
        {"type": "text", "text": "Image 2: unannotated enlarged RGB context crop."},
        {"type": "image", "image": context_crop},
        {"type": "text", "text": "Image 3: mask-only RGB crop; outside-mask pixels are white."},
        {"type": "image", "image": mask_only_crop},
    ]}]


def upgrade_comparison_image(rgb, old_mask, candidate_mask, finals, old_name):
    """Full-scene view for an old-vs-candidate replacement decision."""
    import cv2

    img = np.asarray(rgb).astype(np.uint8).copy()
    img = (img.astype(np.float32) * 0.82).astype(np.uint8)
    # Other delivered objects are only context.  The old object has a unique cyan outline.
    for name, mask in finals:
        if name == old_name:
            continue
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (110, 110, 110), 1)
    extra = candidate_mask & ~old_mask
    img[extra] = (0.45 * img[extra] + 0.55 * np.array((230, 40, 220))).astype(np.uint8)
    old_contours, _ = cv2.findContours(old_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
    candidate_contours, _ = cv2.findContours(candidate_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, candidate_contours, -1, (255, 220, 0), 3)
    cv2.drawContours(img, old_contours, -1, (0, 230, 230), 3)
    return Image.fromarray(img)


def shared_mask_detail_images(rgb, old_mask, candidate_mask):
    """Context plus aligned old/candidate/added RGB-only crops for an upgrade audit."""
    combined = old_mask | candidate_mask
    left, top, right, bottom = candidate_crop_box(combined)
    crop_rgb = np.asarray(rgb).astype(np.uint8)[top:bottom, left:right]
    scale = min(float(CROP_UPSCALE), float(CROP_MAX_EDGE) / max(crop_rgb.shape[:2]))
    size = (max(1, int(round(crop_rgb.shape[1] * scale))),
            max(1, int(round(crop_rgb.shape[0] * scale))))
    context = Image.fromarray(crop_rgb).resize(size, Image.Resampling.LANCZOS)
    context_rgb = np.asarray(context)

    def mask_only(mask):
        crop_mask = mask[top:bottom, left:right].astype(np.uint8) * 255
        resized = np.asarray(
            Image.fromarray(crop_mask).resize(size, Image.Resampling.NEAREST)
        ) > 127
        out = np.full_like(context_rgb, 255)
        out[resized] = context_rgb[resized]
        return Image.fromarray(out)

    return context, mask_only(old_mask), mask_only(candidate_mask), \
        mask_only(candidate_mask & ~old_mask)


def upgrade_verifier_messages(candidate, old_name, old_mask, rgb, finals):
    """Build the explicit same-instance replacement verifier input."""
    comparison = upgrade_comparison_image(rgb, old_mask, candidate["mask"], finals, old_name)
    context, old_only, candidate_only, added_only = shared_mask_detail_images(
        rgb, old_mask, candidate["mask"]
    )
    aliases = candidate_alias_phrases(candidate)
    alias_text = "Detector phrase aliases:\n" + "\n".join(f"- {p}" for p in aliases)
    return [{"role": "user", "content": [
        {"type": "text", "text": UPGRADE_PROMPT.format(
            old_name=old_name, aliases=alias_text,
        )},
        {"type": "text", "text": "Image 1: full-scene old/candidate comparison."},
        {"type": "image", "image": comparison},
        {"type": "text", "text": "Image 2: unannotated shared RGB context crop."},
        {"type": "image", "image": context},
        {"type": "text", "text": "Image 3: existing-mask-only RGB crop; outside is white."},
        {"type": "image", "image": old_only},
        {"type": "text", "text": "Image 4: candidate-mask-only RGB crop; outside is white."},
        {"type": "image", "image": candidate_only},
        {"type": "text", "text": "Image 5: candidate-only added pixels; outside is white."},
        {"type": "image", "image": added_only},
    ]}]


def parse_verifier_response(raw):
    verdict, obj, why = "NO", "-", ""
    for line in str(raw or "").splitlines():
        upper = line.strip().upper()
        if upper.startswith("VERDICT:"):
            verdict = "YES" if "YES" in upper else "NO"
        elif upper.startswith("OBJECT:"):
            obj = line.split(":", 1)[1].strip() or "-"
        elif upper.startswith("WHY:"):
            why = line.split(":", 1)[1].strip()
    return {"verdict": verdict, "object": obj, "why": why, "raw": str(raw or "")}


def enforce_major_identity_gate(result, object_profile):
    """Reject a forbidden recycle identity even when the VLM says YES."""
    if object_profile not in {"major", "major_v3", "major_v4", "major_v5"} or result.get("verdict") != "YES":
        return result
    reason = object_profile_violation_reason(result.get("object", ""), object_profile)
    if not reason:
        return result
    gated = dict(result)
    gated.update({
        "verdict": "NO",
        "vlm_verdict": "YES",
        "major_gate_reason": reason,
        "why": (
            f"Rejected by deterministic major-object gate ({reason}); "
            f"VLM identity was {result.get('object', '-')}."
        ),
    })
    return gated


def parse_upgrade_response(raw):
    verdict, obj, why = "KEEP", "-", ""
    for line in str(raw or "").splitlines():
        upper = line.strip().upper()
        if upper.startswith("VERDICT:"):
            verdict = "REPLACE" if "REPLACE" in upper else "KEEP"
        elif upper.startswith("OBJECT:"):
            obj = line.split(":", 1)[1].strip() or "-"
        elif upper.startswith("WHY:"):
            why = line.split(":", 1)[1].strip()
    return {"verdict": verdict, "object": obj, "why": why, "raw": str(raw or "")}


def save_verifier_trace(trace_root, kind, candidate, messages, result, extra=None):
    """Persist the exact prompt/images sent to a recycle verifier for HTML auditing."""
    if not trace_root:
        return None
    safe = "".join(c if c.isalnum() or c in "-_" else "_"
                   for c in str(candidate.get("phrase", "candidate")))[:72]
    out_dir = os.path.join(trace_root, f"{kind}_{safe}_{candidate.get('idx', 0):03d}")
    os.makedirs(out_dir, exist_ok=True)
    prompt_parts = []
    image_paths = []
    image_index = 1
    for message in messages:
        content = message.get("content", [])
        content = [{"type": "text", "text": content}] if isinstance(content, str) else content
        for part in content:
            if part.get("type") == "text":
                prompt_parts.append(str(part.get("text", "")))
            elif part.get("type") == "image":
                filename = f"image_{image_index}.png"
                image = part.get("image")
                if isinstance(image, Image.Image):
                    image.save(os.path.join(out_dir, filename))
                elif isinstance(image, str) and os.path.exists(image):
                    shutil.copy2(image, os.path.join(out_dir, filename))
                else:
                    continue
                image_paths.append(filename)
                image_index += 1
    with open(os.path.join(out_dir, "prompt.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(prompt_parts))
    record = {
        "kind": kind,
        "phrase": candidate.get("phrase"),
        "idx": candidate.get("idx"),
        "aliases": candidate.get("aliases", []),
        "images": image_paths,
        "result": result,
    }
    if extra:
        record.update(extra)
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return out_dir


def normalize_verifier_mode(mode=None):
    mode = mode or os.getenv("REST3D_RECYCLE_VERIFIER_MODE") or DEFAULT_VERIFIER_MODE
    if mode not in VERIFIER_MODES:
        raise ValueError(f"unknown recycle verifier mode {mode!r}; expected one of {VERIFIER_MODES}")
    return mode


def normalize_object_profile(profile=None):
    profile = profile or os.getenv("REST3D_OBJECT_PROFILE") or DEFAULT_OBJECT_PROFILE
    if profile not in OBJECT_PROFILES:
        raise ValueError(f"unknown object profile {profile!r}; expected one of {OBJECT_PROFILES}")
    return profile


def collect_unclaimed_candidates(root, shape, min_frac=MIN_FRAC, include_upgrades=False):
    """Return raw SAM3 candidates that are not already represented in stage1 outputs."""
    seg = sorted(glob(os.path.join(root, "segemented_obj", "*.png")))
    finals, floor = [], None
    for path in seg:
        arr = np.array(Image.open(path))
        if arr.ndim != 3 or arr.shape[-1] != 4:
            continue
        mask = arr[..., 3] > 127
        name = os.path.splitext(os.path.basename(path))[0]
        if name.startswith("the_floor"):
            floor = mask if floor is None else (floor | mask)
        elif mask.any():
            finals.append((name, mask))
    if not finals:
        return [], finals, floor
    H, W = shape
    if floor is None:
        floor = np.zeros((H, W), dtype=bool)
    named = finals + [("the_floor", floor)]
    delivered = floor.copy()
    for _, mask in finals:
        delivered |= mask

    raw = []
    sam_root = os.path.join(root, "segment_agent_out", "sam_out")
    for cand_dir in glob(os.path.join(sam_root, "*")):
        if not os.path.isdir(cand_dir):
            continue
        for js in sorted(glob(os.path.join(cand_dir, "*.json"))):
            phrase = os.path.splitext(os.path.basename(js))[0]
            for idx, mask in enumerate(decode(js, shape), 1):
                raw.append((phrase, idx, mask, js))

    # Candidate-level NMS removes duplicate returns from synonyms before coverage gates.
    # Keep provenance: the smaller duplicate's phrase can be a better identity clue than the
    # area-max representative phrase (flux2_013 raw chair #6 is represented by a "bench" mask).
    unique = []
    for phrase, idx, mask, js in sorted(raw, key=lambda x: -int(x[2].sum())):
        alias = {"phrase": phrase, "idx": idx, "source_json": js}
        matches = [(iou(mask, item["mask"]), item) for item in unique]
        best_iou, best = max(matches, default=(0.0, None), key=lambda x: x[0])
        if best_iou > CAND_IOU:
            best["aliases"].append({**alias, "representative_iou": best_iou})
            continue
        unique.append({
            "phrase": phrase,
            "idx": idx,
            "mask": mask,
            "source_json": js,
            "aliases": [{**alias, "representative_iou": 1.0}],
        })

    kept = []
    upgrades = []
    for item in unique:
        mask = item["mask"]
        area = int(mask.sum())
        if area / float(H * W) < min_frac:
            continue
        cover, nearest = max((((mask & fm).sum() / area, nm) for nm, fm in named),
                             key=lambda x: x[0])
        delivered_cover = (mask & delivered).sum() / float(area)
        # A larger candidate that almost contains one delivered instance is not a new orphan.
        # Keep it in a separate upgrade pool so a VLM can decide whether it is a clean replacement.
        upgrade_old = None
        upgrade_old_cov = 0.0
        upgrade_old_share = 0.0
        if include_upgrades:
            for nm, fm in finals:
                inter = int((mask & fm).sum())
                old_cov = inter / float(fm.sum())
                cand_share = inter / float(area)
                if (old_cov >= UPGRADE_OLD_COVER
                        and cand_share < UPGRADE_MAX_OLD_SHARE
                        and area >= int(UPGRADE_MIN_AREA_RATIO * fm.sum())):
                    extra = mask & ~fm
                    other_union = np.zeros_like(mask)
                    for other_nm, other_fm in finals:
                        if other_nm != nm:
                            other_union |= other_fm
                    extra_conflict = (extra & other_union).sum() / float(max(1, extra.sum()))
                    if extra.sum() / float(H * W) >= UPGRADE_MIN_EXTRA_FRAC and \
                            extra_conflict <= UPGRADE_MAX_EXTRA_CONFLICT and old_cov > upgrade_old_cov:
                        upgrade_old, upgrade_old_cov, upgrade_old_share = nm, old_cov, cand_share
            if upgrade_old is not None:
                item.update({
                    "px": area, "frac": area / float(H * W),
                    "covered": delivered_cover, "nearest": nearest, "nearest_cov": cover,
                    "upgrade_old": upgrade_old, "upgrade_old_cov": upgrade_old_cov,
                    "upgrade_old_share": upgrade_old_share,
                })
                upgrades.append(item)
                continue
        if cover >= DUP_COVER or delivered_cover >= GROUP_COVER:
            continue
        if is_bg(item["phrase"]):
            continue
        item.update({
            "px": area,
            "frac": area / float(H * W),
            "covered": delivered_cover,
            "nearest": nearest,
            "nearest_cov": cover,
        })
        kept.append(item)
    if include_upgrades:
        return kept, finals, floor, upgrades
    return kept, finals, floor


def confirm_unclaimed_candidate(candidate, rgb, finals, generate_fn,
                                verifier_mode=DEFAULT_VERIFIER_MODE,
                                object_profile=DEFAULT_OBJECT_PROFILE):
    """Run the selected verifier policy while keeping legacy behavior available."""
    verifier_mode = normalize_verifier_mode(verifier_mode)
    object_profile = normalize_object_profile(object_profile)

    def run(messages):
        return parse_verifier_response(generate_fn(messages))

    if verifier_mode == "legacy_joint":
        legacy = run(recycle_verifier_messages(candidate, rgb, finals))
        return {**legacy, "verifier_mode": verifier_mode, "legacy": legacy}

    if verifier_mode == "identity_geometry":
        identity = run(identity_verifier_messages(candidate, rgb, object_profile))
        return {**identity, "verifier_mode": verifier_mode, "identity": identity}

    legacy = run(recycle_verifier_messages(candidate, rgb, finals))
    try:
        identity = run(identity_verifier_messages(candidate, rgb, object_profile))
    except Exception as exc:
        # A shadow observer must never change the active legacy decision, including when
        # the extra call times out or its backend is temporarily unavailable.
        identity = {
            "verdict": "ERROR", "object": "-", "why": str(exc)[:200], "raw": "",
        }
    return {
        **legacy,
        "verifier_mode": verifier_mode,
        "active_policy": "legacy_joint",
        "legacy": legacy,
        "identity": identity,
        "disagreement": (
            identity["verdict"] in ("YES", "NO")
            and legacy["verdict"] != identity["verdict"]
        ),
        "shadow_error": identity["verdict"] == "ERROR",
    }


def confirm_upgrade_candidate(candidate, old_name, old_mask, rgb, finals, generate_fn):
    """Ask whether a larger candidate cleanly replaces one delivered instance."""
    messages = upgrade_verifier_messages(candidate, old_name, old_mask, rgb, finals)
    return parse_upgrade_response(generate_fn(messages))


def replace_delivered_mask(root, image, old_name, new_mask):
    """Overwrite an existing delivered-instance PNG only after an upgrade verdict."""
    path = os.path.join(root, "segemented_obj", old_name + ".png")
    if not os.path.exists(path):
        raise FileNotFoundError(f"upgrade target disappeared: {old_name}")
    backup_dir = os.path.join(root, "recycle_verifier", "upgrade_backups")
    os.makedirs(backup_dir, exist_ok=True)
    shutil.copy2(path, os.path.join(backup_dir, old_name + ".png"))
    save_seg_obj(image, new_mask, out_path=path)


def run_upgrade_pass(root, image, source_rgb, generate_fn, logger=None, min_frac=MIN_FRAC):
    """Replace incomplete delivered masks before orphan recovery; never creates an instance."""
    H, W = source_rgb.shape[:2]
    _, finals, _, upgrades = collect_unclaimed_candidates(
        root, (H, W), min_frac=min_frac, include_upgrades=True,
    )
    results = []
    trace_root = os.path.join(root, "recycle_verifier")
    for candidate in upgrades:
        old_name = candidate["upgrade_old"]
        old_mask = next((mask for name, mask in finals if name == old_name), None)
        if old_mask is None:
            candidate.update({
                "verdict": "KEEP", "object": "-",
                "why": "the delivered mask selected for upgrade was unavailable", "raw": "",
            })
            results.append(candidate)
            continue
        try:
            messages = upgrade_verifier_messages(
                candidate, old_name, old_mask, source_rgb, finals,
            )
            result = parse_upgrade_response(generate_fn(messages))
        except Exception as exc:
            messages = []
            result = {
                "verdict": "KEEP", "object": "-", "why": str(exc)[:200], "raw": "",
            }
        candidate.update(result)
        candidate["upgrade_action"] = "replaced" if result["verdict"] == "REPLACE" else "kept_old"
        trace = save_verifier_trace(trace_root, "upgrade", candidate, messages, result, {
            "old_name": old_name,
            "old_covered_by_candidate": candidate["upgrade_old_cov"],
            "candidate_covered_by_old": candidate["upgrade_old_share"],
        })
        if trace:
            candidate["trace_dir"] = os.path.relpath(trace, root)
        if logger is not None:
            logger.info("    upgrade candidate '%s' #%d -> %s (%s)",
                        candidate["phrase"], candidate["idx"],
                        result["verdict"], result["object"])
        if result["verdict"] == "REPLACE":
            replace_delivered_mask(root, image, old_name, candidate["mask"])
            # Keep the in-memory comparison set accurate for later candidates.
            finals = [(name, candidate["mask"] if name == old_name else mask)
                      for name, mask in finals]
        results.append(candidate)
    return results


def apply_unclaimed_recycle(root, image, source_rgb, generate_fn, logger=None,
                            min_frac=MIN_FRAC, verifier_mode=None,
                            object_profile=DEFAULT_OBJECT_PROFILE):
    """Confirm and write unclaimed raw candidates into an existing stage1 directory."""
    verifier_mode = normalize_verifier_mode(verifier_mode)
    object_profile = normalize_object_profile(object_profile)
    H, W = source_rgb.shape[:2]
    upgrades = []
    if verifier_mode == "identity_upgrade":
        upgrades = run_upgrade_pass(
            root, image, source_rgb, generate_fn, logger=logger, min_frac=min_frac,
        )
    # The upgrade pass may have replaced an old PNG. Rebuild all geometry from disk before
    # deciding which raw candidates are genuine new-instance orphans.
    if verifier_mode == "identity_upgrade":
        candidates, finals, _, _ = collect_unclaimed_candidates(
            root, (H, W), min_frac=min_frac, include_upgrades=True,
        )
    else:
        candidates, finals, _ = collect_unclaimed_candidates(
            root, (H, W), min_frac=min_frac,
        )
    manifest_path = os.path.join(root, "recycle_manifest.json")
    # A resumed shard sees the already-recycled masks as delivered, so the raw candidate
    # set is empty. Preserve the prior audit record instead of replacing it with {}.
    if not candidates and os.path.exists(manifest_path):
        return []
    applied = []
    accepted_masks = []
    trace_root = os.path.join(root, "recycle_verifier")
    for candidate in candidates:
        # Candidate-level IoU NMS is intentionally conservative.  Before writing, also
        # enforce stage1's containment rule against earlier recovered instances.
        if any(max(
            (candidate["mask"] & old).sum() / float(candidate["px"]),
            (candidate["mask"] & old).sum() / float(old.sum()),
        ) >= DUP_COVER for old in accepted_masks):
            candidate.update({
                "verdict": "DUPLICATE_RECYCLED",
                "object": "-",
                "why": "overlaps an earlier accepted recycle candidate",
                "raw": "",
            })
            continue
        try:
            if verifier_mode == "identity_upgrade":
                messages = identity_verifier_messages(candidate, source_rgb, object_profile)
                result = parse_verifier_response(generate_fn(messages))
                result.update({
                    "verifier_mode": verifier_mode,
                    "object_profile": object_profile,
                    "identity": dict(result),
                })
            else:
                messages = None
                result = confirm_unclaimed_candidate(
                    candidate, source_rgb, finals, generate_fn, verifier_mode=verifier_mode,
                    object_profile=object_profile,
                )
        except Exception as exc:
            messages = []
            result = {
                "verdict": "ERROR", "object": "-", "why": str(exc)[:200], "raw": "",
                "verifier_mode": verifier_mode,
            }
        result = enforce_major_identity_gate(result, object_profile)
        candidate.update(result)
        if verifier_mode == "identity_upgrade":
            trace = save_verifier_trace(
                trace_root, "identity", candidate, messages, result,
                {"nearest": candidate.get("nearest"), "nearest_cov": candidate.get("nearest_cov")},
            )
            if trace:
                candidate["trace_dir"] = os.path.relpath(trace, root)
        if logger is not None:
            logger.info("    recycle candidate '%s' #%d [%s] -> %s (%s)",
                        candidate["phrase"], candidate["idx"], verifier_mode,
                        candidate["verdict"], candidate["object"])
            if verifier_mode == "shadow_compare" and candidate.get("disagreement"):
                logger.info("      shadow disagreement: legacy=%s identity=%s",
                            candidate["legacy"]["verdict"], candidate["identity"]["verdict"])
        if result["verdict"] != "YES":
            continue
        safe = candidate["phrase"].replace("/", "_").replace(" ", "_")
        out_name = f"recycled_{safe}_{candidate['idx']:03d}"
        out_path = os.path.join(root, "segemented_obj", out_name + ".png")
        save_seg_obj(image, candidate["mask"], out_path=out_path)
        accepted_masks.append(candidate["mask"])
        applied.append({
            "name": out_name,
            "phrase": candidate["phrase"],
            "idx": candidate["idx"],
            "source_json": candidate["source_json"],
            "px": candidate["px"],
            "frac": candidate["frac"],
            "covered": candidate["covered"],
            "object": candidate["object"],
            "why": candidate["why"],
            "aliases": candidate.get("aliases", []),
            "verifier_mode": verifier_mode,
            "object_profile": object_profile,
        })
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "verifier_mode": verifier_mode,
            "object_profile": object_profile,
            "active_policy": "legacy_joint" if verifier_mode == "shadow_compare" else verifier_mode,
            "upgrades": [{k: v for k, v in c.items() if k != "mask"} for c in upgrades],
            "applied": applied, "considered": [
            {k: v for k, v in c.items() if k != "mask"} for c in candidates
        ]}, f, indent=2, ensure_ascii=False)
    return applied


def finalize_recycle_manifest(root):
    """Record which accepted recycle masks survived the normal stage1 dedup pass."""
    manifest_path = os.path.join(root, "recycle_manifest.json")
    if not os.path.exists(manifest_path):
        return []
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    final = []
    for item in manifest.get("applied", []):
        path = os.path.join(root, "segemented_obj", item["name"] + ".png")
        item["survived_dedup"] = os.path.exists(path)
        if item["survived_dedup"]:
            final.append(item)
    manifest["final"] = final
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", help="default: all")
    ap.add_argument("--no_vlm", action="store_true",
                    help="run the geometric/phrase gates only, skip confirmation")
    ap.add_argument("--out", default="exp_seg100/recycle_plan.json")
    # exp_seg15's 15 scenes were never covered by this pass, yet they are the ONLY scenes
    # with GT. the-breakfast-room is the worked example: two listing phrases ("front left"
    # / "front center") grounded to a bit-identical mask, dedup dropped one, and the fourth
    # chair went unclaimed -- while `white chair` #2 covers it at IoU 0.885. Exactly the
    # failure this pass exists to fix, on a batch it never ran over.
    ap.add_argument("--out_root", default=OUT_ROOT,
                    help="stage1 output tree, e.g. exp_seg15/output")
    ap.add_argument("--src", default=SRC,
                    help="source image root; seg15 uses SRC_SEG15")
    ap.add_argument("--src_glob", default=None,
                    help="extra source-image pattern, e.g. '{src}/{scene}/im.png'")
    ap.add_argument("--debug_dir", default=None,
                    help="where overlays go (default exp_seg100/recycle_debug_<tint>)")
    ap.add_argument("--tint", default="red", choices=sorted(TINTS),
                    help="control knob: rerun with a different tint to check the verdicts "
                         "are not being driven by the tint colour itself")
    ap.add_argument("--verifier_mode",
                    default=os.getenv("REST3D_RECYCLE_VERIFIER_MODE", DEFAULT_VERIFIER_MODE),
                    choices=VERIFIER_MODES,
                    help="shadow_compare records both policies but applies legacy_joint")
    ap.add_argument("--object_profile",
                    default=os.getenv("REST3D_OBJECT_PROFILE", DEFAULT_OBJECT_PROFILE),
                    choices=OBJECT_PROFILES,
                    help="major narrows identity recovery to room-layout objects")
    ap.add_argument("--vlm_backend",
                    default=os.getenv("REST3D_VLM_BACKEND", "gpt"),
                    choices=["gpt", "gpt4o", "gemini", "anthropic"])
    args = ap.parse_args()

    if not args.no_vlm:
        from .policies.vlm_policy import generate_vlm_response, set_vlm_backend
        set_vlm_backend(args.vlm_backend)

    out_root = args.out_root
    scenes = args.scenes or sorted(
        os.path.basename(d) for d in glob(os.path.join(out_root, "*")) if os.path.isdir(d))

    plan, tally = {}, dict(cand=0, uniq=0, dup=0, group=0, bg=0, small=0,
                           asked=0, yes=0, no=0, err=0)

    for sc in scenes:
        root = os.path.join(out_root, sc, "stage1")
        seg = sorted(glob(os.path.join(root, "segemented_obj", "*.png")))
        if not seg:
            continue
        finals, floor = [], None
        for f in seg:
            m = np.array(Image.open(f))[..., 3] > 127
            if os.path.basename(f).startswith("the_floor"):
                floor = m if floor is None else (floor | m)
            else:
                finals.append((os.path.splitext(os.path.basename(f))[0], m))
        if not finals:
            continue
        H, W = finals[0][1].shape
        if floor is None:
            floor = np.zeros((H, W), bool)
        named = finals + [("the_floor", floor)]
        delivered = floor.copy()
        for _, m in finals:
            delivered |= m

        cands = []
        for d in glob(os.path.join(root, "segment_agent_out", "sam_out", "*")):
            if not os.path.isdir(d):
                continue
            for js in sorted(glob(os.path.join(d, "*.json"))):
                ph = os.path.splitext(os.path.basename(js))[0]
                for i, m in enumerate(decode(js, (H, W)), 1):
                    cands.append((ph, i, m))
        tally["cand"] += len(cands)

        uniq = []
        for ph, i, m in sorted(cands, key=lambda x: -x[2].sum()):
            alias = {"phrase": ph, "idx": i}
            matches = [(iou(m, item["mask"]), item) for item in uniq]
            best_iou, best = max(matches, default=(0.0, None), key=lambda x: x[0])
            if best_iou > CAND_IOU:
                best["aliases"].append({**alias, "representative_iou": best_iou})
                continue
            uniq.append({
                "phrase": ph, "idx": i, "mask": m,
                "aliases": [{**alias, "representative_iou": 1.0}],
            })
        tally["uniq"] += len(uniq)

        keep = []
        for item in uniq:
            ph, i, m = item["phrase"], item["idx"], item["mask"]
            a = int(m.sum())
            bcov, bname = max(((m & fm).sum() / a, nm) for nm, fm in named)
            if bcov >= DUP_COVER:
                tally["dup"] += 1
                continue
            if (m & delivered).sum() / a >= GROUP_COVER:
                tally["group"] += 1
                continue
            if is_bg(ph):
                tally["bg"] += 1
                continue
            if a / (H * W) < MIN_FRAC:
                tally["small"] += 1
                continue
            keep.append(dict(phrase=ph, idx=i, px=a, frac=round(a / (H * W), 4),
                             covered=round(float((m & delivered).sum() / a), 3),
                             nearest=bname, nearest_cov=round(float(bcov), 3),
                             aliases=item["aliases"],
                             rle=mu.encode(np.asfortranarray(m.astype(np.uint8))),
                             _mask=m))

        if not keep:
            print(f"{sc:28s} uniq={len(uniq):3d} -> 0 candidates survive the gates",
                  flush=True)
            continue

        rgb = np.array(Image.open(
            find_source_image(args.src, sc, args.src_glob)).convert("RGB"))
        # Not under output/<scene>/stage1/: that tree was produced in the pod as root and is
        # not writable as liutianjia.
        dbg = os.path.join(args.debug_dir
                           or os.path.join("exp_seg100", f"recycle_debug_{args.tint}"), sc)
        os.makedirs(dbg, exist_ok=True)

        for k in keep:
            k["rle"]["counts"] = k["rle"]["counts"].decode("ascii")
            safe = k["phrase"].replace("/", "_").replace(" ", "_")
            candidate = {**k, "mask": k["_mask"]}
            legacy_msgs = recycle_verifier_messages(candidate, rgb, finals)
            identity_msgs = identity_verifier_messages(candidate, rgb, args.object_profile)
            message_sets = [("legacy", legacy_msgs)]
            if args.verifier_mode in ("identity_geometry", "shadow_compare"):
                message_sets.append(("identity", identity_msgs))
            for prefix, message_set in message_sets:
                evidence = [part["image"] for part in message_set[0]["content"]
                            if part["type"] == "image"]
                for image_idx, image_part in enumerate(evidence, 1):
                    image_part.save(os.path.join(
                        dbg, f"{safe}_{k['idx']}_{prefix}_{image_idx}.png"
                    ))

            if args.no_vlm:
                k["verdict"] = "SKIPPED"
                del k["_mask"]
                continue

            tally["asked"] += 1
            try:
                result = confirm_unclaimed_candidate(
                    candidate, rgb, finals, generate_vlm_response,
                    verifier_mode=args.verifier_mode,
                    object_profile=args.object_profile,
                )
            except Exception as e:
                k["verdict"], k["why"] = "ERROR", str(e)[:200]
                tally["err"] += 1
                del k["_mask"]
                continue
            k.update(result)
            tally["yes" if result["verdict"] == "YES" else "no"] += 1
            del k["_mask"]

        n_yes = sum(1 for k in keep if k["verdict"] == "YES")
        plan[sc] = keep
        print(f"{sc:28s} uniq={len(uniq):3d} gated->{len(keep):2d} "
              f"YES={n_yes}" + (f"  {[k.get('object', k['phrase']) for k in keep if k['verdict']=='YES']}"
                                if n_yes else ""), flush=True)

    plan["_summary"] = dict(
        tally=tally, min_frac=MIN_FRAC, dup_cover=DUP_COVER, group_cover=GROUP_COVER,
        cand_iou=CAND_IOU, tint=args.tint, bg_heads=sorted(BG_HEADS), bg_phrases=sorted(BG_PHRASES),
        verifier_mode=args.verifier_mode,
        object_profile=args.object_profile,
        n_scenes=len(scenes),
        n_recovered=sum(1 for s, v in plan.items() if s != "_summary"
                        for k in v if k["verdict"] == "YES"))
    json.dump(plan, open(args.out, "w"), indent=1)
    print(f"\n{json.dumps(tally)}")
    print(f"recovered {plan['_summary']['n_recovered']} objects -> {args.out}")


if __name__ == "__main__":
    main()
