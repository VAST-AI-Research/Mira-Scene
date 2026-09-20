"""OpenAI-compatible vision calls used by segmentation and scene graphs."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

LUMINA_DEFAULT_BASE_URL = "https://lumina.tripo3d.com/v1"
LUMINA_API_KEY_ENVS = ("CODEX_API_KEY", "LUMINA_API_KEY")


class VLMClient:
    def __init__(self, model: str, base_url: str = LUMINA_DEFAULT_BASE_URL,
                 timeout: float = 300, max_tokens: int = 8192):
        self.model, self.base_url = model, base_url.rstrip("/")
        self.timeout, self.max_tokens = timeout, max_tokens

    def _key(self) -> str:
        for name in LUMINA_API_KEY_ENVS:
            if os.environ.get(name):
                return str(os.environ[name])
        raise RuntimeError("set CODEX_API_KEY or LUMINA_API_KEY")

    def complete(self, prompt: str, images: list[Path] | None = None) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in images or []:
            content.append(self._image_part(path))
        return self._request([{"role": "user", "content": content}])

    @staticmethod
    def _image_part(value: Any) -> dict[str, Any]:
        if isinstance(value, (str, Path)):
            path = Path(value)
            mime = mimetypes.guess_type(path.name)[0] or "image/png"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        else:
            import io
            output = io.BytesIO()
            value.convert("RGB").save(output, format="PNG")
            mime = "image/png"
            encoded = base64.b64encode(output.getvalue()).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}

    def complete_messages(self, messages: list[dict[str, Any]]) -> str:
        """Send the frozen Stage-1 multi-turn text/image message format."""
        converted = []
        for message in messages:
            raw = message.get("content", "")
            parts = [{"type": "text", "text": raw}] if isinstance(raw, str) else raw
            content = []
            for item in parts:
                if item.get("type") == "text":
                    content.append({"type": "text", "text": str(item.get("text", ""))})
                elif item.get("type") == "image":
                    content.append(self._image_part(item.get("image")))
                else:
                    raise ValueError(f"unsupported VLM content type: {item.get('type')}")
            converted.append({"role": str(message.get("role", "user")), "content": content})
        return self._request(converted)

    def _request(self, messages: list[dict[str, Any]]) -> str:
        payload = json.dumps({"model": self.model, "messages": messages,
                              "max_tokens": self.max_tokens, "temperature": 0}).encode()
        request = urllib.request.Request(self.base_url + "/chat/completions", data=payload,
            headers={"Authorization": f"Bearer {self._key()}", "Content-Type": "application/json"}, method="POST")
        try:
            # Lumina rejects the cluster's CONNECT proxy (HTTP 403/tunnel
            # failures).  Use an opener with an explicit empty proxy map so
            # segmentation remains reliable even when the parent shell exports
            # http_proxy/https_proxy.  This is the programmatic equivalent of
            # unsetting those variables for the Lumina request only.
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:1000]
            raise RuntimeError(f"VLM HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"VLM connection failed: {exc.reason}") from exc
        return str(result["choices"][0]["message"].get("content") or "")


def parse_json(text: str) -> Any:
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S)
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"[\[{].*[\]}]", clean, flags=re.S)
        if not match:
            raise ValueError("VLM did not return JSON")
        return json.loads(match.group(0))
