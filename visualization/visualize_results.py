#!/usr/bin/env python3
"""Serve a read-only browser for Mira-Scene inference results.

The server has no third-party Python dependencies.  It discovers cases and
constructed scenes at request time, so a running viewer also sees newly
finished pipeline outputs after the page is refreshed.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import re
import shutil
import socket
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote, unquote, urlsplit


VISUALIZATION_ROOT = Path(__file__).resolve().parent


def _first_file(paths: Iterable[Path]) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def _is_case(path: Path) -> bool:
    return any(
        candidate.is_file()
        for candidate in (
            path / "input" / "scene.png",
            path / "input" / "source.png",
            path / "case.json",
        )
    )


def discover_cases(result_root: Path) -> list[Path]:
    """Return case directories for either an output root or one case path."""
    if _is_case(result_root):
        return [result_root]
    return sorted(
        (path for path in result_root.iterdir() if path.is_dir() and _is_case(path)),
        key=lambda path: path.name,
    )


def _asset_url(result_root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    relative = path.resolve().relative_to(result_root)
    return "/assets/" + quote(relative.as_posix(), safe="/")


def _pretty_token(value: str) -> str:
    aliases = {"sam3d": "SAM3D", "trellis2": "TRELLIS.2", "ppd": "PPD", "moge": "MoGe"}
    words = re.split(r"[_-]+", value)
    return " ".join(aliases.get(word.lower(), word.capitalize()) for word in words if word)


def discover_scenes(case_dir: Path, result_root: Path) -> list[dict[str, Any]]:
    """Discover backend-specific and legacy constructed scene layouts."""
    scene_root = case_dir / "scene"
    if not scene_root.is_dir():
        return []

    options: list[dict[str, Any]] = []
    for directory in sorted(path for path in scene_root.rglob("*") if path.is_dir()):
        asset = _first_file((directory / "scene_with_floor.glb", directory / "scene.glb"))
        if asset is None:
            continue
        parts = directory.relative_to(scene_root).parts
        if not parts:
            continue
        if len(parts) == 1:
            backend = "legacy"
            method = parts[0]
            label = f"{_pretty_token(method)} · legacy"
        else:
            backend = parts[0]
            method = " / ".join(parts[1:])
            label = f"{_pretty_token(backend)} · {_pretty_token(method)}"
        if asset.name == "scene_with_floor.glb":
            label += " · floor"
        options.append(
            {
                "id": "/".join(parts),
                "label": label,
                "url": _asset_url(result_root, asset),
                "backend": backend,
                "method": method,
            }
        )
    return options


def build_manifest(result_root: Path) -> dict[str, Any]:
    """Build the frontend manifest without changing the result directory."""
    results: dict[str, list[dict[str, Any]]] = {}
    for case_dir in discover_cases(result_root):
        image = _first_file(
            (
                case_dir / "input" / "scene.png",
                case_dir / "input" / "source.png",
                case_dir / "input" / "scene_fg.png",
            )
        )
        segmentation = _first_file(
            (
                case_dir / "CCM" / "rgb_mask.png",
                case_dir / "input" / "rgb_mask.png",
            )
        )
        scenes = discover_scenes(case_dir, result_root)
        environment = _first_file(
            (
                case_dir / "environment" / "environment_equirect.png",
                case_dir / "environment" / "environment_equirect.jpg",
                case_dir / "environment" / "environment_equirect.jpeg",
                case_dir / "environment" / "environment_equirect.webp",
            )
        )
        scene_url = scenes[0]["url"] if scenes else None
        fields = [
            {
                "label": "Input image",
                "kind": "image",
                "url": _asset_url(result_root, image),
                "note": "input/scene.png",
            },
            {
                "label": "Segmentation",
                "kind": "mask",
                "url": _asset_url(result_root, segmentation),
                "crop": "right_half" if segmentation and segmentation.name == "rgb_mask.png" else None,
                "note": "instance ID palette",
                "legend": "Producer palette · right half of rgb_mask.png" if segmentation else None,
            },
            {
                "label": "Interactive 3D scene",
                "kind": "glb",
                "url": scene_url,
                "options": scenes,
                "environment_url": _asset_url(result_root, environment),
                "environment_rotation": 90,
                "source_up": "y",
                "note": (
                    f"Y-up source/display · {len(scenes)} variant{'s' if len(scenes) != 1 else ''}"
                    + (" · environment map" if environment else " · no environment map")
                ),
            },
        ]
        results[case_dir.name] = [
            {
                "source": case_dir.name,
                "source_label": case_dir.name,
                "fields": fields,
            }
        ]

    return {
        "version": 1,
        "project_title": "Mira-Scene Results",
        "result_root": str(result_root),
        "viewer": {"max_active_3d": 6, "azimuth": 0, "elevation": 10},
        "results": results,
    }


def discover_server_ip() -> str | None:
    """Best-effort address that another machine can use to reach this host."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # UDP connect selects the outbound interface without sending traffic.
        sock.connect(("8.8.8.8", 80))
        address = sock.getsockname()[0]
        return address if address and not address.startswith("127.") else None
    except OSError:
        try:
            address = socket.gethostbyname(socket.gethostname())
            return address if address and not address.startswith("127.") else None
        except OSError:
            return None
    finally:
        sock.close()


class ResultRequestHandler(BaseHTTPRequestHandler):
    server_version = "MiraSceneResultViewer/1.0"

    def __init__(self, *args: Any, result_root: Path, **kwargs: Any) -> None:
        self.result_root = result_root
        super().__init__(*args, **kwargs)

    def _json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK, *, head: bool = False) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not head:
            self.wfile.write(data)

    def _error(self, status: HTTPStatus, message: str, *, head: bool = False) -> None:
        self._json({"error": message}, status, head=head)

    @staticmethod
    def _resolve_below(root: Path, relative: str) -> Path | None:
        try:
            target = (root / unquote(relative).lstrip("/")).resolve()
            target.relative_to(root)
        except (OSError, ValueError):
            return None
        return target

    def _file(self, root: Path, relative: str, *, head: bool = False) -> None:
        path = self._resolve_below(root, relative)
        if path is None or not path.is_file():
            self._error(HTTPStatus.NOT_FOUND, "Asset does not exist", head=head)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix.lower() == ".glb":
            content_type = "model/gltf-binary"
        elif path.suffix.lower() == ".js":
            content_type = "text/javascript"
        stat = path.stat()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(stat.st_size))
        self.send_header("Last-Modified", self.date_time_string(stat.st_mtime))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if not head:
            with path.open("rb") as source:
                shutil.copyfileobj(source, self.wfile)

    def _handle(self, *, head: bool = False) -> None:
        path = urlsplit(self.path).path
        if path in {
            "/",
            "/visualization",
            "/visualization/",
            "/visualization/index.html",
        }:
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", "/visualization/result_vis.html")
            self.end_headers()
        elif path == "/api/health":
            self._json(
                {"ok": True, "result_root": str(self.result_root), "cases": len(discover_cases(self.result_root))},
                head=head,
            )
        elif path in {"/api/manifest", "/visualization/manifest.json"}:
            self._json(build_manifest(self.result_root), head=head)
        elif path.startswith("/assets/"):
            self._file(self.result_root, path.removeprefix("/assets/"), head=head)
        elif path.startswith("/visualization/"):
            self._file(
                VISUALIZATION_ROOT,
                path.removeprefix("/visualization/"),
                head=head,
            )
        else:
            self._error(HTTPStatus.NOT_FOUND, "Unknown route", head=head)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        self._handle()

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib handler API
        self._handle(head=True)


class ResultHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_directory", nargs="?", type=Path, help="Pipeline result root or one case directory")
    parser.add_argument("--result-dir", dest="result_directory_option", type=Path, help="Alias for the positional result directory")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0, accessible through the server IP)")
    parser.add_argument("--port", default=8000, type=int, help="TCP port (default: 8000)")
    args = parser.parse_args()
    selected = args.result_directory_option or args.result_directory
    if selected is None:
        parser.error("a result directory is required")
    if args.result_directory_option and args.result_directory:
        if args.result_directory_option.resolve() != args.result_directory.resolve():
            parser.error("positional result directory and --result-dir refer to different paths")
    args.result_directory = selected.resolve()
    if not args.result_directory.is_dir():
        parser.error(f"result directory does not exist: {args.result_directory}")
    if not discover_cases(args.result_directory):
        parser.error(f"no Mira-Scene cases found under: {args.result_directory}")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    return args


def main() -> int:
    args = parse_args()
    handler = partial(ResultRequestHandler, result_root=args.result_directory)
    server = ResultHTTPServer((args.host, args.port), handler)
    print(f"Mira-Scene result viewer: {args.result_directory}")
    if args.host in {"0.0.0.0", "::"}:
        server_ip = discover_server_ip()
        print(f"Local URL:  http://127.0.0.1:{args.port}/")
        if server_ip:
            print(f"Remote URL: http://{server_ip}:{args.port}/")
        else:
            print(f"Remote URL: http://<server-ip>:{args.port}/")
    else:
        print(f"Open http://{args.host}:{args.port}/")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping viewer.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
