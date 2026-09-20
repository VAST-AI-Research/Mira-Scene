"""Generate per-scene LDR equirectangular environment maps from image context.

Input layout (for each ``<scene>``)::

    <data_dir>/<scene>/input/
        scene.png                 # source RGB image
        scene_fg.png              # foreground cutout/mask
        floor_mask.png            # floor mask

Output layout::

    <output_dir>/<scene>/environment/
        background_reference.png          # masked image sent for review/generation
        background_reference_preview.png  # small preview of the condition image
        environment_equirect.png         # final 2:1 LDR sRGB panorama
        environment_metadata.json        # generation settings and provenance

Example command (prepare conditions only, no API request)::

    python infer_scripts/6_generate_environment_map.py \
        --output_dir Mira_Scene_Demo/data \
        --data_dir Mira_Scene_Demo/data \
        --prepare_only \
        --environment_size 2048x1024

Example command (generate/rebuild panoramas)::

    python infer_scripts/6_generate_environment_map.py \
        --output_dir Mira_Scene_Demo/data \
        --data_dir Mira_Scene_Demo/data \
        --environment_size 2048x1024 \
        --overwrite

The source image contains both reconstructed foreground objects and the floor.
This script removes those two regions using ``scene_fg.png`` and
``floor_mask.png``, writes the resulting background condition for review, and
uses Lumina's OpenAI-compatible GPT Image endpoint to complete an equirectangular
environment image.  It deliberately writes an LDR sRGB panorama instead of
mislabeling an up-converted PNG as a physically measured HDRI.

The generated panorama is an appearance/lighting prior: a single perspective
image cannot observe the camera rear hemisphere or occluded environment.

Generated panoramas are post-aligned by rendering rectilinear yaw candidates
and matching the retained reference background with SIFT/RANSAC.  The selected
longitude is circularly shifted to image ``width / 2`` and the original is
kept as ``environment_equirect_unaligned.png``.  Existing panoramas can be
processed without another generation request via ``--align_existing``.
"""

import argparse
import base64
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm


LUMINA_DEFAULT_BASE_URL = "https://lumina.tripo3d.com/v1"
LUMINA_API_KEY_ENVS = ("CODEX_API_KEY", "LUMINA_API_KEY")
DEFAULT_ENV_FILE = None
ALIGNMENT_METHOD = "sift_rectilinear_yaw_search_v1"


def load_codex_api_key(env_file):
    """Credentials are accepted only from the inherited process environment."""
    return bool(os.environ.get("CODEX_API_KEY") or os.environ.get("LUMINA_API_KEY"))


def ensure_lumina_no_proxy(base_url):
    """Route direct Lumina traffic around a proxy known to reject this endpoint."""
    host = urlparse(base_url).hostname
    if not host:
        return
    entries, known = [], set()
    for key in ("NO_PROXY", "no_proxy"):
        for value in os.environ.get(key, "").split(","):
            value = value.strip()
            if value and value.lower() not in known:
                entries.append(value)
                known.add(value.lower())
    if host.lower() not in known:
        entries.append(host)
    os.environ["NO_PROXY"] = ",".join(entries)
    os.environ["no_proxy"] = os.environ["NO_PROXY"]


def binary_mask(image, threshold):
    """Return a foreground mask from an RGB cutout or a grayscale mask."""
    values = np.asarray(image)
    if values.ndim == 3:
        values = values[..., :3].max(axis=-1)
    return values > threshold


def dilate(mask, radius):
    """Dilate a binary mask in image pixels without requiring OpenCV."""
    if radius <= 0:
        return mask
    size = radius * 2 + 1
    return np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(size))
    ) > 0


def make_background_condition(scene_path, foreground_path, floor_path, threshold, dilate_radius):
    """Create transparent and human-readable background-only conditioning images."""
    scene = Image.open(scene_path).convert("RGB")
    foreground = Image.open(foreground_path).convert("RGB")
    floor = Image.open(floor_path).convert("L")
    if foreground.size != scene.size:
        foreground = foreground.resize(scene.size, Image.Resampling.NEAREST)
    if floor.size != scene.size:
        floor = floor.resize(scene.size, Image.Resampling.NEAREST)

    foreground_mask = binary_mask(foreground, threshold)
    floor_mask = binary_mask(floor, threshold)
    excluded = dilate(foreground_mask | floor_mask, dilate_radius)
    keep = ~excluded

    rgb = np.asarray(scene, dtype=np.uint8)
    rgba = np.empty((*rgb.shape[:2], 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = keep.astype(np.uint8) * 255

    # This preview makes the conditioning intent visible in ordinary image
    # viewers, while the RGBA image passed to the model retains only background
    # pixels as opaque content.
    preview = rgb.copy()
    if np.any(keep):
        fill = np.median(rgb[keep], axis=0).astype(np.uint8)
    else:
        fill = np.array([127, 127, 127], dtype=np.uint8)
    preview[excluded] = fill
    return (
        Image.fromarray(rgba, "RGBA"),
        Image.fromarray(preview, "RGB"),
        foreground_mask,
        floor_mask,
        keep,
    )


def build_environment_prompt(scene_name):
    """Prompt with explicit panoramic projection and orientation invariants."""
    return (
        "Use case: photorealistic-natural. Asset type: LDR environment panorama for a 3D scene. "
        "Primary request: create one seamless 2:1 equirectangular (latitude-longitude, 360-degree) "
        f"environment map consistent with the visible background of the reference image for scene '{scene_name}'. "
        "Input image: opaque pixels are the observed background; transparent pixels are removed foreground objects "
        "and floor and must not be reproduced as objects or a floor texture. "
        "Projection and orientation: this must be a full spherical panorama, not a perspective photograph, cubemap, "
        "collage, fisheye, or a framed image. Place the observed input-camera direction at the horizontal center of "
        "the panorama; preserve its horizon height and vertical architecture. The top edge is the upward direction. "
        "Complete only unobserved directions in a spatially plausible, style-consistent way. "
        "Constraints: no foreground furniture or removable objects from masked areas, no dominant ground plane, "
        "no text, logos, watermark, borders, or UI; continuous left/right seam; natural illumination consistent with "
        "the observed background."
    )


def _circular_delta(values, center, period):
    return (values - center + period / 2.0) % period - period / 2.0


def _densest_circular_cluster(values, period, radius):
    """Return the densest robust circular cluster and its circular center."""
    best = None
    for candidate in values:
        residuals = np.abs(_circular_delta(values, candidate, period))
        members = residuals <= radius
        if not np.any(members):
            continue
        local = residuals[members]
        key = (int(members.sum()), -float(np.median(local)), -float(local.mean()))
        if best is None or key > best[0]:
            best = (key, members, candidate)
    if best is None:
        return np.zeros(len(values), dtype=bool), None
    members, anchor = best[1], best[2]
    unwrapped = anchor + _circular_delta(values[members], anchor, period)
    center = float(np.median(unwrapped) % period)
    # One refinement rejects matches displaced by generated/changed content.
    residuals = np.abs(_circular_delta(values, center, period))
    refined = residuals <= radius
    if np.any(refined):
        unwrapped = center + _circular_delta(values[refined], center, period)
        center = float(np.median(unwrapped) % period)
    return refined, center


def render_panorama_perspective(panorama_rgb, yaw_rad, fov_x_rad, output_size, cv2):
    """Render a pinhole view from an equirectangular panorama at zero pitch."""
    output_width, output_height = output_size
    focal = output_width / (2.0 * np.tan(fov_x_rad / 2.0))
    x = (np.arange(output_width, dtype=np.float32) + 0.5 - output_width / 2.0) / focal
    y = -(np.arange(output_height, dtype=np.float32) + 0.5 - output_height / 2.0) / focal
    ray_x, ray_y = np.meshgrid(x, y)
    ray_z = np.ones_like(ray_x)
    longitude = yaw_rad + np.arctan2(ray_x, ray_z)
    latitude = np.arctan2(ray_y, np.sqrt(ray_x * ray_x + ray_z * ray_z))
    panorama_height, panorama_width = panorama_rgb.shape[:2]
    map_x = ((longitude / (2.0 * np.pi) + 0.5) * panorama_width).astype(np.float32)
    map_y = ((0.5 - latitude / np.pi) * panorama_height).astype(np.float32)
    return cv2.remap(
        panorama_rgb, map_x, map_y, interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )


def _score_perspective_candidate(
    reference_keypoints, reference_descriptors, candidate_rgb, sift, matcher,
    ratio_test, cv2,
):
    gray = cv2.cvtColor(candidate_rgb, cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None or len(keypoints) < 2:
        return None
    pairs = matcher.knnMatch(reference_descriptors, descriptors, k=2)
    matches = [
        pair[0] for pair in pairs
        if len(pair) == 2 and pair[0].distance < ratio_test * pair[1].distance
    ]
    if len(matches) >= 4:
        source = np.float32(
            [reference_keypoints[item.queryIdx].pt for item in matches]
        ).reshape(-1, 1, 2)
        target = np.float32([keypoints[item.trainIdx].pt for item in matches]).reshape(-1, 1, 2)
        _, inlier_mask = cv2.findHomography(source, target, cv2.RANSAC, 6.0)
        inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    else:
        inliers = 0
    distances = np.array([item.distance for item in matches], dtype=np.float64)
    median_distance = float(np.median(distances)) if len(distances) else float("inf")
    return {
        "keypoint_count": len(keypoints),
        "match_count": len(matches),
        "inlier_count": inliers,
        "median_descriptor_distance": median_distance,
        "score": (inliers, len(matches), -median_distance),
    }


def _dense_masked_similarity(reference_rgb, candidate_rgb, mask, cv2):
    """Low-frequency colour/gradient correlation for feature-poor scenes."""
    scores = []
    valid = mask > 0
    if int(valid.sum()) < 64:
        return -1.0
    reference_blur = cv2.GaussianBlur(reference_rgb, (0, 0), 3.0).astype(np.float32)
    candidate_blur = cv2.GaussianBlur(candidate_rgb, (0, 0), 3.0).astype(np.float32)
    for channel in range(3):
        first = reference_blur[..., channel][valid]
        second = candidate_blur[..., channel][valid]
        if float(first.std()) < 2.0 or float(second.std()) < 2.0:
            continue
        scores.append(float(np.corrcoef(first, second)[0, 1]))
    reference_gray = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    candidate_gray = cv2.cvtColor(candidate_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    reference_gradient = cv2.magnitude(
        cv2.Sobel(reference_gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(reference_gray, cv2.CV_32F, 0, 1, ksize=3),
    )[valid]
    candidate_gradient = cv2.magnitude(
        cv2.Sobel(candidate_gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(candidate_gray, cv2.CV_32F, 0, 1, ksize=3),
    )[valid]
    if float(reference_gradient.std()) >= 1.0 and float(candidate_gradient.std()) >= 1.0:
        scores.append(float(np.corrcoef(reference_gradient, candidate_gradient)[0, 1]))
    finite = [value for value in scores if np.isfinite(value)]
    return float(np.mean(finite)) if finite else -1.0


def align_panorama_to_reference(panorama, reference, keep_mask, fov_x_rad, args):
    """Roll a panorama so the reference camera direction lands at width / 2.

    The panorama is rendered into rectilinear candidate views at many yaw
    angles so SIFT never has to match perspective pixels directly against a
    distorted equirectangular image.  A coarse-to-fine yaw search and RANSAC
    homography consensus locate the reference optical direction.
    """
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("panorama alignment requires an OpenCV build with SIFT") from error
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("panorama alignment requires cv2.SIFT_create")

    panorama_rgb = np.asarray(panorama.convert("RGB"), dtype=np.uint8)
    reference_rgb = np.asarray(reference.convert("RGB"), dtype=np.uint8)
    mask = np.asarray(keep_mask, dtype=bool)
    if mask.shape != reference_rgb.shape[:2]:
        mask = np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                reference.size, Image.Resampling.NEAREST
            )
        ) > 0
    reference_max = max(reference.size)
    reference_scale = min(1.0, args.alignment_reference_max_size / max(reference_max, 1))
    ref_size = (
        max(1, round(reference.width * reference_scale)),
        max(1, round(reference.height * reference_scale)),
    )
    panorama_scale = min(1.0, args.alignment_panorama_max_width / max(panorama.width, 1))
    pano_size = (
        max(1, round(panorama.width * panorama_scale)),
        max(1, round(panorama.height * panorama_scale)),
    )
    ref_small = cv2.resize(reference_rgb, ref_size, interpolation=cv2.INTER_AREA)
    pano_small = cv2.resize(panorama_rgb, pano_size, interpolation=cv2.INTER_AREA)
    mask_small = cv2.resize(
        mask.astype(np.uint8) * 255, ref_size, interpolation=cv2.INTER_NEAREST
    )
    # Erode the mask so feature patches never straddle transparent cutout edges.
    mask_small = cv2.erode(mask_small, np.ones((5, 5), np.uint8), iterations=1)
    ref_gray = cv2.cvtColor(ref_small, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(
        nfeatures=args.alignment_max_features,
        contrastThreshold=args.alignment_contrast_threshold,
    )
    ref_keypoints, ref_descriptors = sift.detectAndCompute(ref_gray, mask_small)
    if ref_descriptors is None or len(ref_keypoints) < 2:
        raise RuntimeError("not enough SIFT features in the visible reference background")
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    coarse_step = np.deg2rad(args.alignment_coarse_step_degrees)
    coarse_yaws = np.arange(-np.pi, np.pi, coarse_step)

    def evaluate(yaw):
        candidate = render_panorama_perspective(
            pano_small, float(yaw), fov_x_rad, ref_size, cv2
        )
        score = _score_perspective_candidate(
            ref_keypoints, ref_descriptors, candidate, sift, matcher,
            args.alignment_ratio_test, cv2,
        )
        if score is not None:
            score["yaw_rad"] = float((yaw + np.pi) % (2.0 * np.pi) - np.pi)
        return score

    coarse_results = [result for result in (evaluate(yaw) for yaw in coarse_yaws) if result]
    fine_step = np.deg2rad(args.alignment_fine_step_degrees)
    fine_offsets = np.arange(-coarse_step, coarse_step + fine_step * 0.5, fine_step)
    if coarse_results:
        coarse_best = max(coarse_results, key=lambda item: item["score"])
        fine_results = [
            result for result in
            (evaluate(coarse_best["yaw_rad"] + offset) for offset in fine_offsets)
            if result
        ]
        best = max(fine_results, key=lambda item: item["score"])
    else:
        best = {
            "yaw_rad": 0.0, "match_count": 0, "inlier_count": 0,
            "median_descriptor_distance": None,
        }
    inlier_count = best["inlier_count"]
    alignment_status = "aligned"
    fallback_reason = None
    if best["match_count"] < args.alignment_min_matches or inlier_count < args.alignment_min_inliers:
        dense_width = min(192, ref_size[0])
        dense_height = max(1, round(ref_size[1] * dense_width / ref_size[0]))
        dense_size = (dense_width, dense_height)
        dense_reference = cv2.resize(ref_small, dense_size, interpolation=cv2.INTER_AREA)
        dense_mask = cv2.resize(mask_small, dense_size, interpolation=cv2.INTER_NEAREST)

        def evaluate_dense(yaw):
            candidate = render_panorama_perspective(
                pano_small, float(yaw), fov_x_rad, dense_size, cv2
            )
            return {
                "yaw_rad": float((yaw + np.pi) % (2.0 * np.pi) - np.pi),
                "similarity": _dense_masked_similarity(
                    dense_reference, candidate, dense_mask, cv2
                ),
            }

        dense_coarse = [evaluate_dense(yaw) for yaw in coarse_yaws]
        dense_best_coarse = max(dense_coarse, key=lambda item: item["similarity"])
        dense_fine = [
            evaluate_dense(dense_best_coarse["yaw_rad"] + offset)
            for offset in fine_offsets
        ]
        dense_ranked = sorted(dense_fine, key=lambda item: item["similarity"], reverse=True)
        dense_best = dense_ranked[0]
        separated = [
            item for item in dense_coarse
            if abs(np.rad2deg(_circular_delta(
                item["yaw_rad"], dense_best["yaw_rad"], 2.0 * np.pi
            ))) >= args.alignment_coarse_step_degrees * 1.5
        ]
        runner_up = max(
            (item["similarity"] for item in separated), default=-1.0
        )
        margin = dense_best["similarity"] - runner_up
        if (
            dense_best["similarity"] < args.alignment_dense_min_similarity
            or margin < args.alignment_dense_min_margin
        ):
            fallback_reason = (
                f"best yaw has {inlier_count}/{best['match_count']} consistent SIFT "
                f"matches and ambiguous dense score {dense_best['similarity']:.3f} "
                f"(margin {margin:.3f})"
            )
            if args.require_verified_alignment:
                raise RuntimeError(fallback_reason)
            # The generation prompt explicitly requests the input direction at
            # panorama centre.  When output and condition share too little
            # observable content to verify yaw, preserve that centre prior
            # instead of applying an arbitrary low-confidence rotation.
            best = {
                **best, "yaw_rad": 0.0,
                "dense_similarity": dense_best["similarity"],
                "dense_margin": margin,
            }
            match_strategy = "prompt_center_prior_unverified"
            alignment_status = "unverified_center_prior"
        else:
            best = {
                **best,
                "yaw_rad": dense_best["yaw_rad"],
                "dense_similarity": dense_best["similarity"],
                "dense_margin": margin,
            }
            match_strategy = "masked_dense_colour_gradient_fallback"
    else:
        match_strategy = "rectilinear_candidates_lowe_ratio_ransac_homography"

    center_full = ((best["yaw_rad"] / (2.0 * np.pi) + 0.5) % 1.0) * panorama.width
    shift_pixels = int(round(panorama.width / 2.0 - center_full))
    # Choose the equivalent smallest circular roll and record it explicitly.
    shift_pixels = int((shift_pixels + panorama.width // 2) % panorama.width - panorama.width // 2)
    aligned = Image.fromarray(np.roll(panorama_rgb, shift_pixels, axis=1), "RGB")
    return aligned, {
        "status": alignment_status,
        "method": ALIGNMENT_METHOD,
        "reference_fov_x_degrees": float(np.rad2deg(fov_x_rad)),
        "reference_feature_count": len(ref_keypoints),
        "match_strategy": match_strategy,
        "coarse_step_degrees": args.alignment_coarse_step_degrees,
        "fine_step_degrees": args.alignment_fine_step_degrees,
        "candidate_match_count": best["match_count"],
        "consensus_inlier_count": inlier_count,
        "consensus_inlier_ratio": float(inlier_count / max(1, best["match_count"])),
        "median_descriptor_distance": best["median_descriptor_distance"],
        "dense_similarity": best.get("dense_similarity"),
        "dense_margin": best.get("dense_margin"),
        "matched_yaw_degrees_before": float(np.rad2deg(best["yaw_rad"])),
        "matched_center_x_before": float(center_full),
        "target_center_x": panorama.width / 2.0,
        "circular_shift_pixels": shift_pixels,
        "yaw_correction_degrees": float(shift_pixels / panorama.width * 360.0),
        "fallback_reason": fallback_reason,
    }


def atomic_save_png(image, path):
    path = Path(path)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    image.save(temporary, format="PNG")
    os.replace(temporary, path)


def generate_environment(reference, scene_name, args):
    """Generate one 2:1 panorama using Lumina's OpenAI-compatible image edit API."""
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("environment generation requires the openai Python package") from error

    api_key = next((os.environ.get(name) for name in LUMINA_API_KEY_ENVS if os.environ.get(name)), None)
    if not api_key:
        raise RuntimeError("environment generation requires CODEX_API_KEY or LUMINA_API_KEY")
    ensure_lumina_no_proxy(args.environment_base_url)
    payload = io.BytesIO()
    reference.save(payload, format="PNG")
    payload.name = "background_reference.png"
    payload.seek(0)
    prompt = build_environment_prompt(scene_name)
    client = OpenAI(
        api_key=api_key,
        base_url=args.environment_base_url.rstrip("/"),
        max_retries=0,
        timeout=args.environment_timeout,
    )
    response = client.images.edit(
        model=args.environment_model,
        image=payload,
        prompt=prompt,
        n=1,
        size=args.environment_size,
        quality=args.environment_quality,
        output_format="png",
        response_format="b64_json",
        background="opaque",
    )
    encoded = response.data[0].b64_json
    if not encoded:
        raise RuntimeError("Lumina returned no inline environment image bytes")
    image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
    expected_width, expected_height = (int(value) for value in args.environment_size.split("x"))
    returned_width, returned_height = image.size
    # Lumina occasionally returns a different absolute resolution despite the
    # requested API size (for example 1774x887 for a 2048x1024 request).  A
    # 2:1 image is still a valid equirectangular panorama, so normalize it to
    # the pipeline's requested standard instead of discarding a valid result.
    # Do not silently reshape a non-panoramic image: that usually signals the
    # generator returned a perspective image despite the projection prompt.
    if not np.isclose(returned_width / returned_height, 2.0, rtol=0.0, atol=0.01):
        raise RuntimeError(
            f"Lumina returned non-2:1 image {returned_width}x{returned_height}; expected equirectangular 2:1"
        )
    normalized = image.size != (expected_width, expected_height)
    if normalized:
        image = image.resize((expected_width, expected_height), Image.Resampling.LANCZOS)
    return image, prompt, {
        "returned_resolution": f"{returned_width}x{returned_height}",
        "output_resolution": f"{expected_width}x{expected_height}",
        "resolution_normalized": normalized,
    }


def process_scene(args, scene_name):
    input_dir = os.path.join(args.data_dir, scene_name, "input")
    required = {
        "scene": os.path.join(input_dir, "scene.png"),
        "foreground": os.path.join(input_dir, "scene_fg.png"),
        "floor": os.path.join(input_dir, "floor_mask.png"),
    }
    missing = [label for label, path in required.items() if not os.path.isfile(path)]
    if missing:
        return False, "missing " + ", ".join(missing) + " input(s)"

    output_dir = os.path.join(args.output_dir, scene_name, args.environment_dir_name)
    panorama_path = os.path.join(output_dir, "environment_equirect.png")
    raw_panorama_path = os.path.join(output_dir, "environment_equirect_unaligned.png")
    if os.path.exists(panorama_path) and not args.overwrite and not args.align_existing:
        return True, "environment exists (use --overwrite to regenerate)"
    os.makedirs(output_dir, exist_ok=True)

    reference, preview, foreground_mask, floor_mask, keep = make_background_condition(
        required["scene"],
        required["foreground"],
        required["floor"],
        args.mask_threshold,
        args.mask_dilate_pixels,
    )
    reference.save(os.path.join(output_dir, "background_reference.png"))
    preview.save(os.path.join(output_dir, "background_reference_preview.png"))
    Image.fromarray(keep.astype(np.uint8) * 255, "L").save(
        os.path.join(output_dir, "background_keep_mask.png")
    )
    Image.fromarray(foreground_mask.astype(np.uint8) * 255, "L").save(
        os.path.join(output_dir, "foreground_mask.png")
    )
    Image.fromarray(floor_mask.astype(np.uint8) * 255, "L").save(
        os.path.join(output_dir, "floor_mask.png")
    )

    prompt = None
    generation_info = None
    mode = "prepared_only"
    existing_metadata = {}
    metadata_path = os.path.join(output_dir, "environment_metadata.json")
    if os.path.isfile(metadata_path):
        try:
            with open(metadata_path, encoding="utf-8") as handle:
                existing_metadata = json.load(handle)
        except (OSError, ValueError):
            existing_metadata = {}
    if args.align_existing:
        if not os.path.isfile(panorama_path):
            return False, "missing environment_equirect.png for --align_existing"
        if not os.path.isfile(raw_panorama_path):
            atomic_save_png(Image.open(panorama_path).convert("RGB"), raw_panorama_path)
        environment = Image.open(raw_panorama_path).convert("RGB")
        prompt = existing_metadata.get("prompt")
        generation_info = {
            "returned_resolution": existing_metadata.get("returned_resolution"),
            "output_resolution": existing_metadata.get("output_resolution")
                                 or f"{environment.width}x{environment.height}",
            "resolution_normalized": existing_metadata.get("resolution_normalized"),
        }
        mode = existing_metadata.get("environment_mode", "lumina_generated")
    elif not args.prepare_only:
        environment, prompt, generation_info = generate_environment(reference, scene_name, args)
        atomic_save_png(environment, raw_panorama_path)
        mode = "lumina_generated"

    alignment_info = None
    if mode != "prepared_only" and not args.no_align_panorama:
        fov_x_rad = np.deg2rad(args.alignment_fov_degrees)
        environment, alignment_info = align_panorama_to_reference(
            environment, reference, keep, fov_x_rad, args
        )
        alignment_info["reference_fov_source"] = "environment.alignment_fov_degrees"
        alignment_info["unaligned_file"] = "environment_equirect_unaligned.png"
        atomic_save_png(environment, panorama_path)
    elif mode != "prepared_only":
        alignment_info = {"status": "disabled", "method": None}
        atomic_save_png(environment, panorama_path)

    metadata = dict(existing_metadata) if args.align_existing else {}
    metadata.update({
        "scene_name": scene_name,
        "source_scene": required["scene"],
        "source_scene_fg": required["foreground"],
        "source_floor_mask": required["floor"],
        "background_definition": "NOT (scene_fg non-black OR floor_mask non-black), then exclusion dilation",
        "mask_threshold": args.mask_threshold,
        "mask_dilate_pixels": args.mask_dilate_pixels,
        "foreground_fraction": float(foreground_mask.mean()),
        "floor_fraction": float(floor_mask.mean()),
        "background_keep_fraction": float(keep.mean()),
        "environment_mode": mode,
        "environment_file": "environment_equirect.png" if mode == "lumina_generated" else None,
        "projection": "equirectangular (latitude-longitude), 2:1",
        "returned_resolution": generation_info["returned_resolution"] if generation_info else None,
        "output_resolution": generation_info["output_resolution"] if generation_info else None,
        "resolution_normalized": generation_info["resolution_normalized"] if generation_info else None,
        "color_space": "sRGB",
        "dynamic_range": "LDR; generated appearance/lighting prior, not measured HDRI",
        "orientation": {
            "panorama_center": "input scene camera forward direction",
            "panorama_top": "up direction; intended to align with floor-aligned +Y",
            "center_alignment": alignment_info,
        },
        "model": args.environment_model if mode == "lumina_generated" else None,
        "base_url": args.environment_base_url if mode == "lumina_generated" else None,
        "prompt": prompt,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    temporary_metadata_path = metadata_path + ".tmp"
    with open(temporary_metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    os.replace(temporary_metadata_path, metadata_path)
    alignment_message = ""
    if alignment_info and alignment_info.get("status") == "aligned":
        alignment_message = (
            f", center shift={alignment_info['circular_shift_pixels']:+d}px "
            f"({alignment_info['yaw_correction_degrees']:+.1f}deg), "
            f"matches={alignment_info['consensus_inlier_count']}/"
            f"{alignment_info['candidate_match_count']}"
        )
    elif alignment_info and alignment_info.get("status") == "unverified_center_prior":
        alignment_message = ", center prior kept (match unverified)"
    return True, f"background={keep.mean():.1%}, mode={mode}{alignment_message}"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate LDR equirectangular environment maps from scene backgrounds")
    parser.add_argument("--output_dir", required=True, help="Root containing per-scene output directories")
    parser.add_argument("--data_dir", required=True, help="Dataset root containing {scene}/input files")
    parser.add_argument("--scene_filter", default=None, help="Only process scene names containing this string")
    parser.add_argument("--case", action="append",
                        help="Exact case name; repeat as needed")
    parser.add_argument("--max_cases", type=int, default=-1)
    parser.add_argument("--environment_dir_name", default="environment")
    parser.add_argument(
        "--mask_threshold", type=int, default=0,
        help="Pixels greater than this value are considered mask foreground (0-255).",
    )
    parser.add_argument("--mask_dilate_pixels", type=int, default=2)
    parser.add_argument(
        "--prepare_only", action="store_true",
        help="Write and inspect background conditions without an image-generation API request.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Regenerate an existing environment panorama")
    parser.add_argument(
        "--align_existing", action="store_true",
        help="Align existing panoramas without making an image-generation API request.",
    )
    parser.add_argument(
        "--no_align_panorama", action="store_true",
        help="Disable post-generation reference-direction centering.",
    )
    parser.add_argument(
        "--require_verified_alignment", action="store_true",
        help="Fail instead of keeping the prompt-centred prior when matching is ambiguous.",
    )
    parser.add_argument(
        "--alignment_fov_degrees",
        "--alignment_fov_fallback_degrees",
        dest="alignment_fov_degrees",
        type=float,
        default=60.0,
        help=(
            "fixed horizontal FOV used for panorama alignment; the legacy "
            "--alignment_fov_fallback_degrees spelling is retained as an alias"
        ),
    )
    parser.add_argument("--alignment_reference_max_size", type=int, default=768)
    parser.add_argument("--alignment_panorama_max_width", type=int, default=2048)
    parser.add_argument("--alignment_max_features", type=int, default=6000)
    parser.add_argument("--alignment_contrast_threshold", type=float, default=0.015)
    parser.add_argument("--alignment_ratio_test", type=float, default=0.82)
    parser.add_argument("--alignment_min_matches", type=int, default=6)
    parser.add_argument("--alignment_min_inliers", type=int, default=5)
    parser.add_argument("--alignment_coarse_step_degrees", type=float, default=15.0)
    parser.add_argument("--alignment_fine_step_degrees", type=float, default=2.0)
    parser.add_argument("--alignment_dense_min_similarity", type=float, default=0.12)
    parser.add_argument("--alignment_dense_min_margin", type=float, default=0.02)
    parser.add_argument("--environment_model", default="gpt-image-2")
    parser.add_argument("--environment_base_url", default=LUMINA_DEFAULT_BASE_URL)
    parser.add_argument("--environment_size", default="2048x1024", help="Must be a 2:1 image size")
    parser.add_argument("--environment_quality", default="high", choices=["low", "medium", "high", "auto"])
    parser.add_argument("--environment_timeout", type=float, default=300.0)
    parser.add_argument("--env_file", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def validate_args(args):
    try:
        width, height = (int(value) for value in args.environment_size.split("x"))
    except ValueError as error:
        raise ValueError("--environment_size must be WIDTHxHEIGHT") from error
    if (
        width != height * 2
        or width > 3840
        or height <= 0
        or width % 16
        or height % 16
        or width * height < 655_360
        or width * height > 8_294_400
        or not 0 <= args.mask_threshold <= 255
        or args.mask_dilate_pixels < 0
        or args.environment_timeout <= 0
        or not 10 <= args.alignment_fov_degrees <= 170
        or args.alignment_reference_max_size <= 0
        or args.alignment_panorama_max_width <= 0
        or args.alignment_max_features <= 0
        or args.alignment_contrast_threshold <= 0
        or not 0 < args.alignment_ratio_test < 1
        or args.alignment_min_matches < 2
        or args.alignment_min_inliers < 2
        or not 1 <= args.alignment_coarse_step_degrees <= 90
        or not 0 < args.alignment_fine_step_degrees <= args.alignment_coarse_step_degrees
        or not -1 <= args.alignment_dense_min_similarity <= 1
        or not 0 <= args.alignment_dense_min_margin <= 1
    ):
        raise ValueError("invalid environment-map parameters; size must be a supported 2:1 GPT Image size")


def main():
    args = parse_args()
    validate_args(args)
    load_codex_api_key(args.env_file)
    if not os.path.isdir(args.data_dir):
        raise ValueError(f"data directory does not exist: {args.data_dir}")
    if args.case:
        scene_names = list(dict.fromkeys(args.case))
        missing = [
            name for name in scene_names
            if not os.path.isdir(os.path.join(args.data_dir, name))
        ]
        if missing:
            raise FileNotFoundError("missing case directories: " + ", ".join(missing))
    else:
        scene_names = sorted(
            name for name in os.listdir(args.data_dir)
            if os.path.isdir(os.path.join(args.data_dir, name))
            and (
                not args.align_existing
                or os.path.isfile(os.path.join(
                    args.output_dir, name, args.environment_dir_name,
                    "environment_equirect.png",
                ))
            )
        )
    if args.scene_filter:
        scene_names = [name for name in scene_names if args.scene_filter in name]
    if args.max_cases > 0:
        scene_names = scene_names[:args.max_cases]
    print(f"Generating environment conditions for {len(scene_names)} scene(s)")

    failures = []
    for scene_name in tqdm(scene_names, desc="Environment map"):
        try:
            success, message = process_scene(args, scene_name)
        except Exception as error:
            success, message = False, f"{type(error).__name__}: {error}"
        if success:
            print(f"  {scene_name}: {message}")
        else:
            failures.append((scene_name, message))
    if failures:
        print(f"Failed {len(failures)} scene(s):")
        for scene_name, message in failures:
            print(f"  {scene_name}: {message}")
        raise SystemExit(1)


if __name__ == "__main__":
    from core.stage_logging import run_logged
    run_logged(main, "06_environment.log", primary_root_flags=("--output_dir",))
