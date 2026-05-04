"""Extract ground-truth wall/stucco regions from completed-exhibit PDFs.

Estimators mark up exhibit PDFs to show what gets stuccoed:
  - red translucent fills on plan views    -> wall runs (linear takeoff)
  - cyan/blue translucent fills on elevations -> facade area (sq-ft takeoff)

This module produces per-page binary masks for each color so the wall
detector can be calibrated against real labeled output.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import cv2
import numpy as np

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None  # type: ignore


@dataclass
class ExhibitPage:
    page_index: int
    image_bgr: np.ndarray
    red_mask: np.ndarray        # plan-view wall highlights
    cyan_mask: np.ndarray       # elevation-view facade fills
    red_pixel_count: int
    cyan_pixel_count: int


# HSV color bands tuned for typical translucent-marker overlays.
# OpenCV HSV: H 0–179, S 0–255, V 0–255. Red wraps around 0/180.
RED_HSV_BANDS = [
    ((0,   90, 100), (10,  255, 255)),
    ((170, 90, 100), (179, 255, 255)),
]
CYAN_HSV_BAND = ((85, 60, 140), (110, 255, 255))


def _mask_for_band(hsv: np.ndarray, lo: tuple[int, int, int], hi: tuple[int, int, int]) -> np.ndarray:
    return cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))


def extract_color_masks(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (red_mask, cyan_mask) as uint8 0/255 arrays."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    red = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in RED_HSV_BANDS:
        red = cv2.bitwise_or(red, _mask_for_band(hsv, lo, hi))

    cyan = _mask_for_band(hsv, *CYAN_HSV_BAND)

    # close small holes from anti-aliased edges, drop noise
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, k, iterations=2)
    cyan = cv2.morphologyEx(cyan, cv2.MORPH_CLOSE, k, iterations=2)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, k, iterations=1)
    cyan = cv2.morphologyEx(cyan, cv2.MORPH_OPEN, k, iterations=1)
    return red, cyan


def render_exhibit_pages(pdf_path: str, dpi: int = 150) -> list[ExhibitPage]:
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is required to render exhibit PDFs")
    doc = fitz.open(pdf_path)
    pages: list[ExhibitPage] = []
    try:
        for i in range(doc.page_count):
            pix = doc[i].get_pixmap(dpi=dpi, alpha=False)
            buf = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            bgr = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
            red, cyan = extract_color_masks(bgr)
            pages.append(ExhibitPage(
                page_index=i,
                image_bgr=bgr,
                red_mask=red,
                cyan_mask=cyan,
                red_pixel_count=int((red > 0).sum()),
                cyan_pixel_count=int((cyan > 0).sum()),
            ))
    finally:
        doc.close()
    return pages


def classify_page(page: ExhibitPage, area_threshold: int = 500) -> str:
    """Heuristic page-type classifier for downstream pairing.

    Returns one of: 'plan' (mostly red highlights), 'elevation'
    (large cyan fill), 'mixed', or 'none'.
    """
    r = page.red_pixel_count
    c = page.cyan_pixel_count
    if r < area_threshold and c < area_threshold:
        return "none"
    if c > 5 * max(r, 1):
        return "elevation"
    if r > 5 * max(c, 1):
        return "plan"
    return "mixed"


def red_segments_to_walls(red_mask: np.ndarray, min_area: int = 200) -> list[tuple[int, int, int, int]]:
    """Bounding boxes of contiguous red regions = highlighted wall runs."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(red_mask, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            out.append((int(x), int(y), int(w), int(h)))
    return out
