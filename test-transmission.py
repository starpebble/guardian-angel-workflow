#!/usr/bin/env python3
"""POST one SALT transmission with five triage rows and a scene image (hex-encoded).

Reads ``salt-scene-test.png`` from the repository root (next to this script). The image is
**resized and saved as JPEG** (via Pillow) so the ``picture`` field stays under Firestore’s
**~1 MiB per document** limit — full-resolution PNG hex routinely exceeds that and causes
``500 Failed to persist transmission``.

The result is ``hex:<lowercase hex>`` of JPEG bytes so :func:`guardian_angel.hex_image.hex_to_image_data_uri`
can show a thumbnail on ``/transmissions``. Payload shape matches ``demo_data.demo_victims()`` —
one individual per SALT color (red, yellow, green, gray, black) in a **single** transmission.

Requires ``GUARDIAN_ANGEL_API_SECRET`` (or ``GUARDIAN_ANGEL_SHARED_SECRET``) in ``.env`` or
the environment. Uses the local API by default.

Usage::

    # Place salt-scene-test.png beside this file, then:
    uv run python test-transmission.py
    uv run python test-transmission.py --image /path/to/other.png
"""

from __future__ import annotations

import argparse
import binascii
import io
import json
import os
import sys
from pathlib import Path

# Firestore document max ~1 MiB; leave room for triage JSON + metadata.
_MAX_HEX_LEN = 750_000

_ROOT = Path(__file__).resolve().parent

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = _ROOT / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)


def _picture_as_hex(path: Path, *, max_edge: int, jpeg_quality: int) -> str:
    """Load ``path``, downscale and JPEG-compress, then ``hex:``-encode for Firestore-sized payloads."""

    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            "Pillow is required to resize/compress the scene image. "
            "Install with:  uv sync   or   pip install pillow"
        ) from e

    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        longest = max(w, h)
        if longest > max_edge:
            scale = max_edge / longest
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        quality = jpeg_quality
        for _ in range(12):
            buf.seek(0)
            buf.truncate()
            im.save(buf, format="JPEG", quality=quality, optimize=True)
            raw = buf.getvalue()
            hex_len = 4 + len(raw) * 2  # "hex:" + hexlify
            if hex_len <= _MAX_HEX_LEN or quality <= 45:
                break
            quality = max(45, quality - 7)

        if 4 + len(raw) * 2 > _MAX_HEX_LEN:
            raise RuntimeError(
                f"Scene JPEG still too large for Firestore (~{4 + len(raw) * 2} hex chars after "
                f"resize max_edge={max_edge}). Try --max-edge 960 or a smaller source image."
            )

    return "hex:" + binascii.hexlify(raw).decode("ascii")


def _payload(image_path: Path, *, max_edge: int, jpeg_quality: int) -> dict:
    """Single transmission: five victims, one per SALT color (same text as ``demo_data`` rows)."""

    # Order matches ``TRIAGE_COLOR_ORDER``: red, yellow, green, gray, black.
    rows: list[dict[str, str | None]] = [
        {
            "id": "T1V1",
            "color": "red",
            "description": "Bleeding, impaled object in leg; responsive.",
        },
        {
            "id": "T1V3",
            "color": "yellow",
            "description": "Altered mental status; possible head injury.",
        },
        {
            "id": "T1V2",
            "color": "green",
            "description": "Awake, breathing, ambulatory; no visible hemorrhage.",
        },
        {
            "id": "T1V4",
            "color": "gray",
            "description": "Unable to assess — obscured by debris.",
        },
        {
            "id": "T1V5",
            "color": "black",
            "description": "No spontaneous respirations after airway opening.",
        },
    ]

    triage = []
    for i, row in enumerate(rows):
        triage.append(
            {
                "id": row["id"],
                "color": row["color"],
                "description": row["description"],
                "boundingbox": {"width": 120, "height": 90, "x": 40 + i * 30, "y": 20 + i * 12},
            }
        )

    return {
        "triage_system": "SALT",
        "picture": _picture_as_hex(image_path, max_edge=max_edge, jpeg_quality=jpeg_quality),
        "geo": "37.7749 -122.4194",
        "triage": triage,
    }


def main() -> int:
    _load_dotenv()

    parser = argparse.ArgumentParser(
        description="POST one transmission with demo-style triage + resized JPEG scene (hex-encoded)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("GUARDIAN_ANGEL_BASE_URL", "http://127.0.0.1:8000"),
        help="API base URL (default: env GUARDIAN_ANGEL_BASE_URL or http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=_ROOT / "salt-scene-test.png",
        help="Path to scene PNG (default: ./salt-scene-test.png next to this script)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print JSON metadata only (picture field shortened); do not POST",
    )
    parser.add_argument(
        "--max-edge",
        type=int,
        default=1280,
        metavar="PX",
        help="Longest edge in pixels before JPEG encode (default: 1280)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=82,
        metavar="1-95",
        help="Initial JPEG quality; lowered automatically if the payload would be too large (default: 82)",
    )
    args = parser.parse_args()

    image_path = args.image.resolve()
    if not image_path.is_file():
        print(f"Image not found: {image_path}", file=sys.stderr)
        return 1

    secret = os.environ.get("GUARDIAN_ANGEL_API_SECRET") or os.environ.get(
        "GUARDIAN_ANGEL_SHARED_SECRET"
    )
    if not secret and not args.dry_run:
        print(
            "Missing GUARDIAN_ANGEL_API_SECRET (or GUARDIAN_ANGEL_SHARED_SECRET) in .env or environment.",
            file=sys.stderr,
        )
        return 1

    try:
        payload = _payload(
            image_path,
            max_edge=args.max_edge,
            jpeg_quality=min(95, max(1, args.jpeg_quality)),
        )
    except ImportError as e:
        print(str(e), file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1

    if args.dry_run:
        preview = {**payload, "picture": f"<hex omitted, {len(payload['picture'])} chars>"}
        print(json.dumps(preview, indent=2, ensure_ascii=False))
        return 0

    try:
        import httpx
    except ImportError:
        print("Install httpx:  pip install httpx   or   uv sync", file=sys.stderr)
        return 1

    base = args.base_url.rstrip("/")
    url = f"{base}/api/v1/transmissions"
    r = httpx.post(
        url,
        json=payload,
        headers={"Authorization": f"Bearer {secret}"},
        timeout=120.0,
    )
    print(f"{r.status_code}  POST {url}")
    try:
        print(json.dumps(r.json(), indent=2))
    except Exception:
        print(r.text)
    return 0 if r.is_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
