"""Vision-LLM page filter: real semantic understanding of plan sheets.

For each page we send a thumbnail to Claude (or any vision LLM) with a
strict-JSON prompt asking:

  1. Is this page an architectural floor plan / elevation / something else?
  2. If a floor plan: what is the bounding box of the BUILDING ONLY,
     excluding title block, schedules, legends, dimensions, key plans?
  3. Should we run wall takeoff on this page?

Cropping subsequent steps to that bbox eliminates the "wall detector
picks up notes/legend lines" problem at the source.

Activation: set ANTHROPIC_API_KEY (or in Streamlit secrets). When unset,
the function returns the full image bbox so the rest of the pipeline
behaves exactly as before.
"""

from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class PageDecision:
    is_floor_plan: bool
    bbox: tuple[int, int, int, int]  # (x, y, w, h) in original image px
    page_kind: str                    # "floor_plan" | "elevation" | "schedule" | "skip"
    confidence: float
    raw: str


_PROMPT = (
    "You are inspecting a single page from an architectural construction set.\n"
    "Decide:\n"
    "  - page_kind: one of \"floor_plan\", \"elevation\", \"schedule\", \"skip\".\n"
    "    Use \"floor_plan\" only for plan-view drawings of a building (top-down with rooms/walls).\n"
    "    Use \"skip\" for cover sheets, code/general notes, accessibility detail sheets, manufacturer specs.\n"
    "  - building_bbox: bounding box of ONLY the building drawing — exclude the title block,\n"
    "    schedules, legends, key plans, dimension strings outside the building, and notes.\n"
    "    Coordinates are fractions of width/height in [0, 1].\n"
    "  - confidence: 0..1.\n"
    "Reply with strict JSON only, no markdown:\n"
    '{"page_kind": "...", "building_bbox": [x, y, w, h], "confidence": 0.0}'
)


def _encode_thumbnail(image_bgr: np.ndarray, max_side: int = 1024) -> str:
    h, w = image_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        thumb = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        thumb = image_bgr
    ok, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise RuntimeError("thumbnail encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _api_key() -> Optional[str]:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    # Streamlit secrets fallback
    try:
        import streamlit as st  # type: ignore

        return st.secrets.get("ANTHROPIC_API_KEY")  # type: ignore
    except Exception:
        return None


def classify_page(image_bgr: np.ndarray, model: str = "claude-haiku-4-5-20251001") -> PageDecision:
    """Send thumbnail to Claude vision; parse strict-JSON response.

    Returns a PageDecision with the building bbox in absolute pixel coords.
    Falls back to the full image bbox when the API key isn't configured.
    """
    H, W = image_bgr.shape[:2]
    full_bbox = (0, 0, W, H)
    key = _api_key()
    if not key:
        return PageDecision(True, full_bbox, "floor_plan", 0.0, "no api key")

    try:
        import anthropic  # type: ignore
    except ImportError:
        return PageDecision(True, full_bbox, "floor_plan", 0.0, "anthropic sdk not installed")

    img_b64 = _encode_thumbnail(image_bgr)
    client = anthropic.Anthropic(api_key=key)
    msg = client.messages.create(
        model=model,
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                    {"type": "text", "text": _PROMPT},
                ],
            }
        ],
    )

    text = msg.content[0].text if msg.content else ""
    try:
        data = json.loads(text.strip().split("```")[0])
    except Exception:
        return PageDecision(True, full_bbox, "floor_plan", 0.0, text)

    kind = str(data.get("page_kind", "skip"))
    fbox = data.get("building_bbox", [0, 0, 1, 1])
    if not (isinstance(fbox, list) and len(fbox) == 4):
        fbox = [0, 0, 1, 1]
    x = int(max(0, min(1, fbox[0])) * W)
    y = int(max(0, min(1, fbox[1])) * H)
    w = int(max(0, min(1, fbox[2])) * W)
    h = int(max(0, min(1, fbox[3])) * H)
    if w <= 10 or h <= 10:
        x, y, w, h = full_bbox
    return PageDecision(
        is_floor_plan=(kind == "floor_plan"),
        bbox=(x, y, w, h),
        page_kind=kind,
        confidence=float(data.get("confidence", 0.0)),
        raw=text,
    )


def crop_to_bbox(image_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    return image_bgr[y : y + h, x : x + w].copy()
