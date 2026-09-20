import os
import json
import re
from PIL import Image
import base64
from io import BytesIO

# Global variable to store selected VLM backend. ``gpt4o`` remains an alias for
# historical commands and cached output names; new runs should use ``gpt``.
_VLM_BACKEND = "gpt"  # options: gpt, gemini, anthropic
_VLM_CLIENT = None


def set_vlm_client(client):
    """Inject Mira's configured OpenAI-compatible client into the frozen flow."""
    global _VLM_CLIENT
    _VLM_CLIENT = client

SCENE_PLANNER_PROMPT = """You are a scene-mode router for an image segmentation pipeline.

Choose exactly one mode for the entire image:
- ROOM: the image is primarily a room/interior view whose important targets are furniture,
  built-in systems, appliances, lighting, beds, sofas, chairs, plants, or other room-scale objects.
- TABLETOP: the image is primarily a close view of a desk, table, workbench, counter,
  shelf top, cabinet top, or bedside surface, and the small independent objects on that
  surface are the main subjects.

When uncertain, choose ROOM. Do not list objects. Return JSON only:
{"scene_mode":"ROOM" or "TABLETOP", "reason":"one concise visual reason"}
"""


def plan_scene_mode(image, save_dir=None):
    """Route one image to the stable room prompt or the opt-in tabletop prompt."""
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image.convert("RGB")},
            {"type": "text", "text": SCENE_PLANNER_PROMPT},
        ],
    }]
    response = generate_vlm_response(messages, save_dir, "scene_mode_planner")
    text = str(response or "").strip()
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    parsed = None
    try:
        parsed = json.loads(clean)
    except (TypeError, json.JSONDecodeError):
        match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = None
    mode = str((parsed or {}).get("scene_mode", "")).strip().upper()
    if mode not in {"ROOM", "TABLETOP"}:
        mode = "ROOM"
        reason = "planner response was invalid; conservative ROOM default"
        planner_status = "fallback_room"
    else:
        reason = str((parsed or {}).get("reason", "")).strip() or "planner returned no reason"
        planner_status = "ok"
    result = {
        "scene_mode": mode,
        "reason": reason,
        "planner_status": planner_status,
        "raw_response": text,
        "prompt": SCENE_PLANNER_PROMPT,
    }
    if save_dir is not None:
        with open(os.path.join(save_dir, "scene_mode_manifest.json"), "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
    return result


def normalize_sam3_prompt(object_phrase):
    """Add high-value part/whole cues to the initial SAM3 grounding phrase.

    The object-list phrase remains the canonical target used by the agent and verifier;
    this helper only changes the first search phrase sent to SAM3.
    """
    phrase = re.sub(r"\s+", " ", str(object_phrase or "")).strip()
    if not phrase:
        return phrase
    lower = phrase.casefold()

    def has_any(*terms):
        return any(re.search(rf"\b{re.escape(term)}\b", lower) for term in terms)

    def is_target_class(*terms):
        # Object-list phrases put descriptors before the target noun.  A class
        # mentioned after a spatial preposition ("table behind armchair") is a
        # landmark, not the target whose attached parts should be requested.
        head = re.split(
            r"\b(?:behind|beside|near|next to|in front of|under|below|above|on|against|of)\b",
            lower,
            maxsplit=1,
        )[0]
        return any(re.search(rf"\b{re.escape(term)}\b", head) for term in terms)

    def add_component(component):
        return f"{phrase} and {component}" if has_any("with") else f"{phrase} with {component}"

    if is_target_class("potted plant"):
        return re.sub(r"\bpotted\s+plant\b", "pot with plant", phrase, flags=re.IGNORECASE)
    if is_target_class("pot", "flowerpot", "planter", "plant pot", "potted"):
        if has_any(
            "plant", "plants", "flower", "flowers", "foliage",
            "tree", "trees", "branch", "branches",
        ):
            return phrase
        return add_component("plant")
    if is_target_class("vase", "urn", "jar"):
        if has_any("plant", "plants", "flower", "flowers", "branches", "foliage"):
            return phrase
        return add_component("flowers")
    if is_target_class("sofa", "couch", "loveseat") and not has_any(
        "pillow", "pillows", "cushion", "cushions"
    ):
        return add_component("pillows")
    if is_target_class("chair", "armchair", "recliner", "stool") and not has_any(
        "cushion", "cushions", "pillow", "pillows"
    ):
        return add_component("cushion")
    if is_target_class("bed", "bunk bed", "daybed") and not has_any(
        "mattress", "mattresses"
    ):
        return add_component("mattress")
    return phrase


_MAJOR_HOST_RE = re.compile(
    r"\b(?:bed|daybed|bunk bed|sofa|couch|sectional|loveseat|chaise|chair|"
    r"armchair|recliner|bench|window seat)s?\b",
    re.IGNORECASE,
)
_MAJOR_SOFT_RE = re.compile(
    r"\b(?:pillow|cushion|mattress|sheet|duvet|comforter|quilt|blanket|bedding|throw)s?\b",
    re.IGNORECASE,
)
_MAJOR_EXCLUDED_PATTERNS = (
    ("architectural/decor", re.compile(
        r"\b(?:curtain|drape|blind|shutter|rug|carpet|mat|painting|artwork|poster|"
        r"photograph|picture frame|tapestry|wall mirror|clock|chandelier|"
        r"pendant (?:light|fixture)|"
        r"ceiling light|wall light|sconce|track light|recessed light|ceiling fan)s?\b",
        re.IGNORECASE,
    )),
    ("loose/clutter", re.compile(
        r"\b(?:book|magazine|paper|cup|dish|bottle|candle|vase|flower|sculpture|"
        r"ornament|toy|rocking horse|clothes|shoe|bag|suitcase|luggage|basket|bin|"
        r"tray|umbrella|umbrella stand|step ladder|bicycle|laptop|monitor|keyboard)s?\b",
        re.IGNORECASE,
    )),
    ("non-floor light", re.compile(
        r"\b(?:table lamp|desk lamp|ceiling lamp|wall lamp)s?\b",
        re.IGNORECASE,
    )),
)

# A forbidden loose item can also appear as a modifier of a legitimate room-layout
# object (for example, ``bean bag chair`` or ``plant shelving unit``).  Remove only
# these narrow, established compound uses before applying the deny patterns below.
_MAJOR_PROTECTED_MODIFIERS = (
    re.compile(r"\bbean\s+bag(?=\s+chairs?\b)", re.IGNORECASE),
    re.compile(r"\bclothes(?=\s+dryers?\b)", re.IGNORECASE),
    re.compile(r"\bbook(?=\s+display\s+tables?\b)", re.IGNORECASE),
    re.compile(r"\bhall\s+tree\b", re.IGNORECASE),
    re.compile(
        r"\b(?:book|plant|clothes|shoe|toy)s?\b(?=\s+(?:display\s+)?"
        r"(?:racks?|shelf|shelves|shelving(?:\s+units?)?|carts?|storage(?:\s+units?)?)\b)",
        re.IGNORECASE,
    ),
)


def major_target_violation_reason(object_phrase):
    """Return why a VLM target is forbidden by the opt-in major-object profile."""
    text = re.sub(r"\s+", " ", str(object_phrase or "")).strip()
    if not text or text.casefold() == "the floor":
        return None

    # Landmarks after a spatial relation do not define the target's identity.
    head = re.split(
        r"\b(?:on|inside|behind|beside|near|next to|in front of|under|below|above|against|at)\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    check_head = head
    for pattern in _MAJOR_PROTECTED_MODIFIERS:
        check_head = pattern.sub("", check_head)
    host = _MAJOR_HOST_RE.search(head)
    soft = _MAJOR_SOFT_RE.search(head)
    if soft and (not host or soft.start() < host.start()):
        return "standalone soft furnishing"

    if re.search(r"\bfloor lamps?\b", head, re.IGNORECASE):
        return None
    if (re.search(r"\b(?:pot|planter)s?\b", head, re.IGNORECASE)
            or re.search(r"\bpotted\b", head, re.IGNORECASE)) and re.search(
        r"\b(?:plant|tree)s?\b", head, re.IGNORECASE
    ):
        if re.search(r"\b(?:tabletop|on (?:a |the )?(?:table|desk|shelf|cabinet|counter))\b",
                     text, re.IGNORECASE):
            return "tabletop plant/decor"
        return None
    if re.search(r"\bmirror\b", head, re.IGNORECASE):
        if re.search(r"\b(?:freestanding|full-length|floor|vanity table)\b", head,
                     re.IGNORECASE):
            return None
        return "wall/decor mirror"

    for reason, pattern in _MAJOR_EXCLUDED_PATTERNS:
        if pattern.search(check_head):
            return reason
    if re.search(r"\blamp\b", head, re.IGNORECASE):
        return "non-floor light"
    if re.search(r"\b(?:plant|tree)s?\b", check_head, re.IGNORECASE):
        return "uncontained plant/decor"
    return None


_MAJOR_V3_LIGHT_RE = re.compile(
    r"\b(?:floor lamp|table lamp|desk lamp|ceiling lamp|wall lamp|pendant light|"
    r"pendant fixture|light fixture|chandelier|sconce|track light|recessed light)s?\b",
    re.IGNORECASE,
)
_MAJOR_V3_PLAY_RE = re.compile(r"\b(?:rocking horse|ride-on toy)s?\b", re.IGNORECASE)

_LEADING_EXACT_COUNT_RE = re.compile(
    r"^(?:(?:a\s+)?pair\s+of|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|\d+)\s+",
    re.IGNORECASE,
)


def normalize_major_v4_target_phrase(object_phrase):
    """Remove unreliable leading visual counts from v4 canonical targets."""
    text = re.sub(r"\s+", " ", str(object_phrase or "")).strip()
    return _LEADING_EXACT_COUNT_RE.sub("", text, count=1).strip()


def major_v4_target_violation_reason(object_phrase):
    """Keep v4's hard gate minimal; importance/scale requires image context."""
    text = re.sub(r"\s+", " ", str(object_phrase or "")).strip()
    if not text or text.casefold() == "the floor":
        return None
    head = re.split(
        r"\b(?:on|inside|behind|beside|near|next to|in front of|under|below|above|against|at)\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    host = _MAJOR_HOST_RE.search(head)
    soft = _MAJOR_SOFT_RE.search(head)
    if soft and (not host or soft.start() < host.start()):
        return "standalone soft furnishing"
    return None


def object_profile_violation_reason(object_phrase, object_profile="major"):
    """Return a deny reason for one versioned object-granularity profile."""
    if object_profile == "standard":
        return None
    if object_profile == "tabletop":
        return None
    if object_profile not in {"major", "major_v3", "major_v4", "major_v5", "major_v6"}:
        raise ValueError(f"unknown object profile {object_profile!r}")
    if object_profile in {"major_v4", "major_v5", "major_v6"}:
        return major_v4_target_violation_reason(object_phrase)
    if object_profile == "major_v3":
        text = re.sub(r"\s+", " ", str(object_phrase or "")).strip()
        head = re.split(
            r"\b(?:on|inside|behind|beside|near|next to|in front of|under|below|above|against|at)\b",
            text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        if _MAJOR_V3_LIGHT_RE.search(head) or _MAJOR_V3_PLAY_RE.search(head):
            return None
    return major_target_violation_reason(object_phrase)


def filter_major_object_targets(objects, object_profile="major"):
    """Apply the deterministic deny-only gate for the opt-in major profile."""
    kept, removed = [], []
    seen = set()
    for object_phrase in objects:
        if object_profile in {"major_v4", "major_v5", "major_v6"}:
            object_phrase = normalize_major_v4_target_phrase(object_phrase)
            if not object_phrase:
                continue
        reason = object_profile_violation_reason(object_phrase, object_profile)
        if reason:
            removed.append({"target": object_phrase, "reason": reason})
        elif object_phrase.casefold() not in seen:
            kept.append(object_phrase)
            seen.add(object_phrase.casefold())
    return kept, removed


def set_vlm_backend(backend):
    """Set which VLM backend to use: 'gpt', 'gemini' or 'anthropic'."""
    global _VLM_BACKEND
    _VLM_BACKEND = "gpt" if backend == "gpt4o" else backend


def image_to_base64(image):
    """Convert PIL Image to base64 string for API calls"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def generate_vlm_response_gpt(messages, save_dir=None, save_name="test",
                              model=None, max_retries=5):
    """Generate a response through an OpenAI-compatible chat-completions API."""
    import random
    import requests
    import time

    api_key = (os.getenv("REST3D_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
               or os.getenv("ANTHROPIC_AUTH_TOKEN"))
    if not api_key:
        raise ValueError(
            "REST3D_OPENAI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_AUTH_TOKEN "
            "environment variable not set"
        )

    base_url = (os.getenv("REST3D_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL")
                or "https://lumina.tripo3d.com/v1")
    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint += "/chat/completions"
    model = (model or os.getenv("REST3D_OPENAI_MODEL") or os.getenv("OPENAI_MODEL")
             or "gpt-5.6-sol")
    max_edge = int(os.getenv("REST3D_VLM_MAX_IMAGE_EDGE", "1568"))

    gpt_messages = []
    for msg in messages:
        content = msg["content"]
        parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
        content_list = []
        for item in parts:
            if item.get("type") == "text":
                content_list.append({"type": "text", "text": item["text"]})
            elif item.get("type") == "image":
                img = item["image"]
                if isinstance(img, str):
                    img = Image.open(img)
                img = img.convert("RGB")
                if max_edge and max(img.size) > max_edge:
                    scale = max_edge / max(img.size)
                    img = img.resize(
                        (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
                        Image.Resampling.LANCZOS,
                    )
                base64_image = image_to_base64(img)
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                })
            else:
                raise ValueError(f"Unknown content type: {item.get('type')}")
        gpt_messages.append({"role": msg["role"], "content": content_list})

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": gpt_messages,
        "max_tokens": int(os.getenv("REST3D_VLM_MAX_TOKENS", "4096")),
        "temperature": 0,
    }

    max_retries = int(os.getenv("REST3D_VLM_MAX_RETRIES", max_retries))
    timeout = float(os.getenv("REST3D_VLM_TIMEOUT", "600"))
    output_text = ""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            result = response.json()
            output_text = result["choices"][0]["message"]["content"]
            if isinstance(output_text, list):
                output_text = "".join(
                    part.get("text", "") for part in output_text
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"OpenAI-compatible API call failed after {max_retries} attempts: {e}"
                )
            wait = min(20 * (2 ** attempt), 300) + random.uniform(0, 15)
            print(f"⚠️ API call failed (attempt {attempt+1}/{max_retries}), "
                  f"waiting {wait:.0f}s... {type(e).__name__}: {e}")
            time.sleep(wait)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f'{save_name}.txt'), 'w') as file:
            file.write(output_text)

    return output_text


def generate_vlm_response_gemini(messages, save_dir=None, save_name="test",
                                 model="gemini-3-flash-preview", max_retries=5):
    """Generate response using Gemini via Google API (uses requests, no google-generativeai package needed)"""
    from google import genai
    import time
    from google.genai import types

    # Get API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    project = os.getenv("GOOGLE_CLOUD_PROJECT", "phsyscene")
    os.environ["GOOGLE_CLOUD_PROJECT"] = project

    client = genai.Client(api_key=api_key)

    # Build a single user turn from your messages (you can extend to multi-turn later)
    parts = []
    for msg in messages:
        for item in msg["content"]:
            if item["type"] == "text":
                parts.append(types.Part.from_text(text=item["text"]))
            elif item["type"] == "image":
                img = item["image"]  # PIL Image
                b64 = image_to_base64(img)  # base64 str, no prefix
                img_bytes = base64.b64decode(b64)
                parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))
            else:
                raise ValueError(f"Unknown content type: {item['type']}")

    contents = [types.Content(role="user", parts=parts)]

    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=8192,
                ),
            )
            output_text = resp.text or ""
            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Gemini API call failed after {max_retries} attempts: {e}")
            is_rate_limit = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
            wait = (60 * (2 ** attempt)) if is_rate_limit else 10
            print(f"⚠️ API call failed (attempt {attempt+1}/{max_retries}), waiting {wait}s... {e}")
            time.sleep(wait)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"{save_name}.txt"), "w") as f:
            f.write(output_text)

    return output_text


def generate_vlm_response_anthropic(messages, save_dir=None, save_name="test",
                                    model=None, max_retries=5):
    """Generate response using Claude via the Anthropic Messages API.

    Handles three format differences from the internal message format:
      - ``role: "system"`` turns must be hoisted into the top-level ``system`` param
      - image parts use ``{"type": "image", "source": {...base64...}}``
      - consecutive same-role turns must be merged (the API requires alternation)
    """
    import random
    import time
    import anthropic

    # Cap the long edge before encoding. The API downsamples anything above 1568px
    # server-side, so this loses nothing -- but it keeps the payload small. Scene-tree
    # parent queries send the full scene plus one overlay per nearby object, so at
    # 2560x2560 that is ~30 MB of base64 in a single request and the gateway stalls
    # indefinitely (no error, no timeout fired -- measured 11+ min on wild_idea_table).
    max_edge = int(os.getenv("REST3D_VLM_MAX_IMAGE_EDGE", "1568"))

    api_key = os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_AUTH_TOKEN (or ANTHROPIC_API_KEY) environment variable not set")

    if model is None:
        model = os.getenv("REST3D_ANTHROPIC_MODEL") or os.getenv("ANTHROPIC_MODEL") or "claude-opus-4-8"

    client = anthropic.Anthropic(
        api_key=api_key,
        base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
        timeout=float(os.getenv("REST3D_VLM_TIMEOUT", "600")),
        max_retries=0,   # retries handled below so we can log them
    )
    max_retries = int(os.getenv("REST3D_VLM_MAX_RETRIES", max_retries))

    system_chunks = []
    turns = []
    for msg in messages:
        content = msg["content"]
        # Normalise to a list of parts
        parts = [{"type": "text", "text": content}] if isinstance(content, str) else content

        if msg["role"] == "system":
            for item in parts:
                if item.get("type") == "text":
                    system_chunks.append(item["text"])
            continue

        new_parts = []
        for item in parts:
            if item.get("type") == "text":
                new_parts.append({"type": "text", "text": item["text"]})
            elif item.get("type") == "image":
                img = item["image"]
                if isinstance(img, str):
                    img = Image.open(img)
                img = img.convert("RGB")
                if max_edge and max(img.size) > max_edge:
                    s = max_edge / max(img.size)
                    img = img.resize((max(1, round(img.width * s)),
                                      max(1, round(img.height * s))), Image.LANCZOS)
                new_parts.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_to_base64(img),
                    },
                })
            else:
                raise ValueError(f"Unknown content type: {item.get('type')}")

        # Merge into the previous turn when the role repeats (API requires alternation)
        if turns and turns[-1]["role"] == msg["role"]:
            turns[-1]["content"].extend(new_parts)
        else:
            turns.append({"role": msg["role"], "content": new_parts})

    # The API requires the conversation to open with a user turn
    if turns and turns[0]["role"] != "user":
        turns.insert(0, {"role": "user", "content": [{"type": "text", "text": "Proceed."}]})

    kwargs = dict(
        model=model,
        max_tokens=int(os.getenv("REST3D_VLM_MAX_TOKENS", "4096")),
        temperature=0,
        messages=turns,
    )
    if system_chunks:
        kwargs["system"] = "\n\n".join(system_chunks)

    # Retry policy: the gateway degrades under concurrency (many workers each POSTing a
    # 518x518 image), and connection errors are the dominant failure -- they are transient
    # and worth waiting out, so back off exponentially instead of a flat 10s. A jitter term
    # keeps N workers from retrying in lockstep after a shared blip.
    output_text = ""
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(**kwargs)
            output_text = "".join(b.text for b in resp.content if b.type == "text")
            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Anthropic API call failed after {max_retries} attempts: {e}")
            wait = min(20 * (2 ** attempt), 300) + random.uniform(0, 15)
            print(f"⚠️ API call failed (attempt {attempt+1}/{max_retries}), "
                  f"waiting {wait:.0f}s... {type(e).__name__}: {e}")
            time.sleep(wait)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"{save_name}.txt"), "w") as f:
            f.write(output_text)

    return output_text


def generate_vlm_response(messages, save_dir=None, save_name='test', save=True):
    """
    Generate VLM response using the selected backend.

    Args:
        messages: List of message dicts with role and content
        save_dir: Directory to save response
        save_name: Name for saved file
        save: Whether to save (kept for backwards compatibility)

    Returns:
        str: VLM response text
    """
    if _VLM_CLIENT is not None:
        output_text = _VLM_CLIENT.complete_messages(messages)
        if save and save_dir:
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, f"{save_name}.txt"), "w", encoding="utf-8") as handle:
                handle.write(output_text)
        return output_text
    if _VLM_BACKEND == "gpt":
        return generate_vlm_response_gpt(messages, save_dir, save_name)
    elif _VLM_BACKEND == "gemini":
        return generate_vlm_response_gemini(messages, save_dir, save_name)
    elif _VLM_BACKEND == "anthropic":
        return generate_vlm_response_anthropic(messages, save_dir, save_name)
    else:
        raise ValueError(f"Unknown VLM backend: {_VLM_BACKEND}. Choose from: gpt, gemini, anthropic")


def analyze_scene_object_lists(image, save_dir=None, vlm_prompt_file=None,
                               object_profile="standard"):
    """
    Use VLM to identify all salient objects in an image.

    Args:
        image: PIL Image or file path
        save_dir: Directory to save vlm_objects.json
        vlm_prompt_file: Path to txt file containing VLM prompt. If None,
            defaults to the bundled ``prompts/list_objects.txt`` shipped with
            this package. Relative paths are resolved against ``PROMPTS_DIR``.
        object_profile: ``standard`` keeps the raw parsed list. ``major`` preserves
            the v2 behavior. ``major_v3`` additionally admits foreground lights and
            substantial freestanding rocking/ride-on play objects. ``major_v4`` uses
            the simplified original-REST3D-style importance/scale policy; ``major_v5``
            keeps that list policy and changes only downstream target cardinality.

    Returns:
        list[str]: Object description prompts, one per object
    """
    from .. import PROMPTS_DIR

    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    # Resolve prompt file path
    if vlm_prompt_file is None:
        vlm_prompt_file = os.path.join(PROMPTS_DIR, "list_objects.txt")
    elif not os.path.isabs(vlm_prompt_file):
        vlm_prompt_file = os.path.join(PROMPTS_DIR, vlm_prompt_file)
    with open(vlm_prompt_file, "r", encoding="utf-8") as f:
        vlm_prompt = f.read().strip()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image.convert("RGB")},
                {"type": "text", "text": vlm_prompt},
            ],
        }
    ]

    response = generate_vlm_response(messages, save_dir, 'scene_object_lists')
    print(f"VLM response:\n{response}")

    # Parse: one object per line, strip numbering prefixes
    objects = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'^[\d]+[.\)]\s*', '', line)
        line = re.sub(r'^[-*]\s*', '', line)
        line = line.strip()
        if line and len(line) < 100:
            objects.append(line)

    if object_profile not in {"standard", "tabletop", "major", "major_v3", "major_v4", "major_v5", "major_v6"}:
        raise ValueError(f"unknown object profile {object_profile!r}")
    if object_profile in {"major", "major_v3", "major_v4", "major_v5", "major_v6"}:
        objects, removed = filter_major_object_targets(objects, object_profile)
    else:
        removed = []
    if save_dir is not None:
        with open(os.path.join(save_dir, "scene_object_lists_raw.txt"), "w",
                  encoding="utf-8") as handle:
            handle.write(response)
        with open(os.path.join(save_dir, "scene_object_lists.txt"), "w",
                  encoding="utf-8") as handle:
            handle.write("\n".join(objects))
        with open(os.path.join(save_dir, "scene_object_list_filter.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"object_profile": object_profile, "removed": removed}, handle,
                      indent=2, ensure_ascii=False)
    objects.append("the floor")

    return objects
