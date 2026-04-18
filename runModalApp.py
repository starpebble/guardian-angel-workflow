"""Run Modal SAM3 inference, SALT triage via OpenAI, POST to Guardian Angel API.

Processes one image from ``images/``, saves Modal outputs, calls OpenAI with the SALT
prompt + annotated image + Modal JSON, then POSTs ``/api/v1/transmissions`` once per run.

Requires ``OPENAI_API_KEY``, ``GUARDIAN_ANGEL_API_SECRET`` (or ``GUARDIAN_ANGEL_SHARED_SECRET``),
and optional ``GUARDIAN_ANGEL_BASE_URL``, ``GUARDIAN_ANGEL_GEO``, ``OPENAI_MODEL`` in ``.env``.
"""

from __future__ import annotations

import binascii
import io
import json
import os
import re
import sys
from pathlib import Path

import httpx
import modal
from openai import OpenAI

_ROOT = Path(__file__).resolve().parent

# Firestore document max ~1 MiB; leave room for triage JSON + metadata (see test-transmission.py).
_MAX_HEX_LEN = 750_000

SALT_TRIAGE_PROMPT = """
# Role
You are an expert Emergency Medical Consultant specializing in Disaster Medicine and Mass Casualty Incident (MCI) management.

# Task
Analyze the attached image of victims and perform a preliminary triage using the SALT (Sort, Assess, Lifesaving Interventions, Treatment/Transport) protocol.

# Process Requirements
1. **Identify Individuals:** Assign a unique ID to each visible victim (e.g., Victim_01, Victim_02).
2. **Global Sorting:** Identify if victims are "Walking," "Waving/Moving," or "Still."
3. **Assessment:** Look for "Obvious Life Threats" (e.g., major hemorrhage, respiratory distress, non-responsiveness).
4. **Categorization:** Assign one of the 5 SALT categories:
   - RED (Immediate)
   - YELLOW (Delayed)
   - GREEN (Minimal)
   - GRAY (Expectant)
   - BLACK (Dead)

# Output Format
Return the analysis strictly as a JSON document. Use the following structure:

{
  "incident_summary": "Brief description of the scene and total victim count identified.",
  "victims": [
    {
      "victim_id": "string",
      "visual_observations": "Description of posture, visible injuries, and movement state",
      "salt_step_1_sort": "Walking | Waving/Purposeful Movement | Still",
      "obvious_life_threat": "Yes/No (Describe if Yes)",
      "recommended_lsi": "Suggested Life Saving Intervention (e.g., Tourniquet, Airway positioning)",
      "triage_category": "RED | YELLOW | GREEN | GRAY | BLACK",
      "confidence_score": "1-10 based on visual clarity"
    }
  ],
  "resource_note": "Note on which victims should be prioritized for immediate transport based on the RED category."
}
""".strip()


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    p = _ROOT / ".env"
    if p.is_file():
        load_dotenv(p, override=False)


def _picture_as_hex(
    path: Path, *, max_edge: int = 1280, jpeg_quality: int = 82
) -> str:
    """Load image, downscale and JPEG-compress, then ``hex:``-encode for Firestore-sized payloads."""
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        longest = max(w, h)
        if longest > max_edge:
            scale = max_edge / longest
            im = im.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                Image.Resampling.LANCZOS,
            )

        buf = io.BytesIO()
        quality = jpeg_quality
        for _ in range(12):
            buf.seek(0)
            buf.truncate()
            im.save(buf, format="JPEG", quality=quality, optimize=True)
            raw = buf.getvalue()
            hex_len = 4 + len(raw) * 2
            if hex_len <= _MAX_HEX_LEN or quality <= 45:
                break
            quality = max(45, quality - 7)

        if 4 + len(raw) * 2 > _MAX_HEX_LEN:
            raise RuntimeError(
                f"Scene JPEG still too large for Firestore (~{4 + len(raw) * 2} hex chars). "
                f"Try a smaller max_edge or source image."
            )

    return "hex:" + binascii.hexlify(raw).decode("ascii")


def _parse_json_from_llm(text: str) -> dict:
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    return json.loads(text)


def _salt_category_to_api_color(triage_category: str) -> str:
    """Map SALT label to Guardian Angel API color (lowercase)."""
    s = (triage_category or "").upper()
    for name in ("RED", "YELLOW", "GREEN", "GRAY", "BLACK"):
        if name in s:
            return name.lower()
    return "green"


def run_openai_salt_triage(
    *,
    annotated_image_path: Path,
    modal_json_path: Path,
    model: str | None = None,
) -> dict:
    """Upload annotated image + Modal JSON; return parsed SALT JSON from the model."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set (add to .env or environment).")

    client = OpenAI(api_key=api_key)
    model_name = model or os.environ.get("OPENAI_MODEL", "gpt-4o")

    with open(annotated_image_path, "rb") as f:
        uploaded_image = client.files.create(file=f, purpose="vision")
    with open(modal_json_path, "rb") as f:
        uploaded_json = client.files.create(file=f, purpose="user_data")

    response = client.responses.create(
        model=model_name,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": SALT_TRIAGE_PROMPT},
                    {"type": "input_image", "file_id": uploaded_image.id},
                    {"type": "input_file", "file_id": uploaded_json.id},
                ],
            }
        ],
    )

    out = getattr(response, "output_text", None) or ""
    return _parse_json_from_llm(out)


def build_triage_rows(
    salt_analysis: dict, objects: list[dict]
) -> list[dict[str, object]]:
    """Merge OpenAI ``victims`` with Modal ``objects`` (bounding boxes) for API ``triage``."""
    victims = salt_analysis.get("victims") or []
    rows: list[dict[str, object]] = []

    for i, obj in enumerate(objects):
        v = victims[i] if i < len(victims) else {}
        vid = v.get("victim_id") or f"Victim_{i + 1:02d}"
        color = _salt_category_to_api_color(str(v.get("triage_category", "GREEN")))
        # Single field for server ``description``: visual observations per user request.
        desc = str(v.get("visual_observations") or "").strip() or "(no description)"
        row: dict[str, object] = {
            "id": vid,
            "color": color,
            "description": desc,
            "boundingbox": {
                "width": float(obj.get("width", 0)),
                "height": float(obj.get("height", 0)),
                "x": float(obj.get("x", 0)),
                "y": float(obj.get("y", 0)),
            },
        }
        rows.append(row)

    # If the model listed victims not matched 1:1 to boxes, append rows without boxes.
    for j in range(len(objects), len(victims)):
        v = victims[j]
        vid = v.get("victim_id") or f"Victim_{j + 1:02d}"
        color = _salt_category_to_api_color(str(v.get("triage_category", "GREEN")))
        desc = str(v.get("visual_observations") or "").strip() or "(no description)"
        rows.append(
            {
                "id": vid,
                "color": color,
                "description": desc,
                "boundingbox": {"width": 0.0, "height": 0.0, "x": 0.0, "y": 0.0},
            }
        )

    return rows


def post_guardian_transmission(payload: dict) -> httpx.Response:
    secret = os.environ.get("GUARDIAN_ANGEL_API_SECRET") or os.environ.get(
        "GUARDIAN_ANGEL_SHARED_SECRET"
    )
    if not secret:
        raise RuntimeError(
            "GUARDIAN_ANGEL_API_SECRET (or GUARDIAN_ANGEL_SHARED_SECRET) is not set."
        )

    base = (
        os.environ.get("GUARDIAN_ANGEL_BASE_URL", "http://127.0.0.1:8000")
        .rstrip("/")
    )
    url = f"{base}/api/v1/transmissions"
    return httpx.post(
        url,
        json=payload,
        headers={"Authorization": f"Bearer {secret}"},
        timeout=120.0,
    )


def main() -> None:
    _load_dotenv()

    image_name = "image_001"
    image_ending = ".png"
    prompt = "Label the injured soliders"

    image_path = _ROOT / "images" / f"{image_name}{image_ending}"
    print("INPUT IMAGE PATH: ", image_path)

    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    image_bytes = image_path.read_bytes()

    print("MAKING MODEL")
    Sam3 = modal.Cls.from_name("sam3-infer", "Sam3")
    model = Sam3()

    result_bytes = model.infer.remote(image_bytes, prompt)

    output_path = _ROOT / "images" / f"output_{image_name}{image_ending}"
    output_path.write_bytes(result_bytes["image_with_boxes"])

    json_data = result_bytes["json_data"]
    json_output_path = _ROOT / "images" / f"output_{image_name}.json"
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    print("SAVED OUTPUTS")

    print("OPENAI SALT TRIAGE")
    salt_analysis = run_openai_salt_triage(
        annotated_image_path=output_path,
        modal_json_path=json_output_path,
    )

    objects = json_data.get("objects") or []
    triage = build_triage_rows(salt_analysis, objects)

    # Analyzed scene for the server: annotated output (boxes + labels), JPEG-compressed.
    picture_hex = _picture_as_hex(output_path)

    geo = os.environ.get("GUARDIAN_ANGEL_GEO", "37.7749 -122.4194")

    payload: dict = {
        "triage_system": "SALT",
        "picture": picture_hex,
        "geo": geo,
        "triage": triage,
    }
    # Full structured LLM output (disable with GUARDIAN_INCLUDE_SALT_ANALYSIS=0 if API rejects extra keys).
    if os.environ.get("GUARDIAN_INCLUDE_SALT_ANALYSIS", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    ):
        payload["salt_analysis"] = salt_analysis

    print("POST GUARDIAN ANGEL")
    r = post_guardian_transmission(payload)
    print(f"{r.status_code}  POST /api/v1/transmissions")
    try:
        print(json.dumps(r.json(), indent=2))
    except Exception:
        print(r.text)
    if not r.is_success:
        sys.exit(1)


if __name__ == "__main__":
    main()
