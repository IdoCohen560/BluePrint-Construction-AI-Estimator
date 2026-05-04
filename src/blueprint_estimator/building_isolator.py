"""Pick the floor-plan / building region from a multi-region drawing sheet.

A construction sheet always has multiple ink regions:

  - the building drawing (target)
  - title block / sheet stamp (tables of text, regular grid)
  - schedule blocks (door / window / finish tables — uniform parallel lines + text)
  - notes / general-information blocks (paragraphs of text + horizontal rules)
  - key plan / north arrow / vicinity map (small sub-drawings)
  - dimension strings outside the building

The previous "biggest connected component" heuristic happily picked title
blocks and schedules. This module scores each large component by how
floor-plan-like it is, then returns a binary mask of the best one only.

Floor-plan-likeness signals:

  +  many Hough segments at varied lengths (rooms have walls of every
     size; tables have one or two repeated lengths)
  +  diverse segment positions (high spatial entropy)
  +  high ink area relative to bbox (table cells are mostly white)
  +  mix of perpendicular T-junctions (walls meet at T's; tables meet
     at crosses)
  -  extreme bbox aspect ratio (>4:1 = a banner / dimension string)
  -  bbox covers the entire page (= sheet border)
"""

from __future__ import annotations

import math
import cv2
import numpy as np


def _hough_segments(gray_patch: np.ndarray) -> np.ndarray:
    if gray_patch.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    blur = cv2.GaussianBlur(gray_patch, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=40,
        minLineLength=20, maxLineGap=10,
    )
    if lines is None:
        return np.zeros((0, 4), dtype=np.float32)
    return lines.reshape(-1, 4).astype(np.float32)


def _floor_plan_score(gray: np.ndarray, x: int, y: int, w: int, h: int, area: int) -> tuple[float, dict]:
    """Higher score = more floor-plan-like."""
    H, W = gray.shape
    # full-page candidates are sheet borders, never buildings
    if w >= W * 0.92 and h >= H * 0.92:
        return 0.0, {"reason": "page_border"}
    aspect = w / max(h, 1)
    # Real multi-unit apartment buildings can be 6:1 or more. A title-block
    # banner is usually >12:1 or extremely thin. Be permissive.
    if aspect > 12.0 or aspect < 0.08:
        return 0.0, {"reason": "extreme_aspect", "aspect": aspect}
    if area < 30_000:
        return 0.0, {"reason": "too_small", "area": area}

    patch = gray[y : y + h, x : x + w]
    segs = _hough_segments(patch)
    n = len(segs)
    if n < 20:
        return 0.0, {"reason": "too_few_segments", "n": n}

    lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    len_mean = float(lengths.mean())
    len_std = float(lengths.std())
    len_cv = len_std / max(len_mean, 1.0)  # coefficient of variation

    # spatial entropy — split bbox into a 4x4 grid, count segments per cell
    mids_x = (segs[:, 0] + segs[:, 2]) / 2
    mids_y = (segs[:, 1] + segs[:, 3]) / 2
    cell_x = np.clip((mids_x / max(w, 1) * 4).astype(int), 0, 3)
    cell_y = np.clip((mids_y / max(h, 1) * 4).astype(int), 0, 3)
    hist = np.zeros((4, 4))
    for cx, cy in zip(cell_x, cell_y):
        hist[cy, cx] += 1
    p = hist / max(hist.sum(), 1)
    p_nz = p[p > 0]
    spatial_entropy = float(-(p_nz * np.log2(p_nz)).sum()) if p_nz.size else 0.0

    # ink density (paper between walls is white; tables are mostly white between
    # cells too, so this alone isn't enough — used as a multiplier)
    ink_density = float(((patch < 200).sum()) / max(patch.size, 1))

    score = (
        math.log1p(n)            # plenty of line work
        * (0.5 + len_cv)          # length diversity
        * spatial_entropy         # not concentrated in one corner
        * (0.2 + ink_density)     # has some ink
    )
    return float(score), {
        "n": n, "len_cv": len_cv, "entropy": spatial_entropy,
        "density": ink_density, "aspect": aspect, "area": int(area),
    }


def isolate_building(image_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return a binary mask (255 = inside building region, 0 = elsewhere) and
    metadata about which connected component was selected.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    ink = (gray < 200).astype(np.uint8) * 255
    # Use a SMALL closing kernel so title block / schedules stay separate
    # from the floor plan. The previous (15,15) iters=4 fused them into one
    # mega-component.
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, k, iterations=2)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

    best_idx = -1
    best_score = 0.0
    best_meta = {}
    candidates = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        score, meta = _floor_plan_score(gray, x, y, w, h, area)
        candidates.append({"label": int(i), "bbox": (int(x), int(y), int(w), int(h)),
                           "score": score, **meta})
        if score > best_score:
            best_score = score
            best_idx = i
            best_meta = meta

    info = {
        "n_components": int(n - 1),
        "best_label": int(best_idx),
        "best_score": float(best_score),
        "best_meta": best_meta,
        "candidates": sorted(candidates, key=lambda c: -c["score"])[:6],
    }

    if best_idx < 0:
        return np.zeros_like(gray), info

    # Filled bounding rectangle of the winning component, slightly padded.
    # Used as an INSIDE/OUTSIDE test for segment midpoints (which sit in
    # white space between walls), so it must be a solid rectangle, not
    # the ink trace.
    x = int(stats[best_idx, 0])
    y = int(stats[best_idx, 1])
    w = int(stats[best_idx, 2])
    h = int(stats[best_idx, 3])
    pad = 25
    final = np.zeros_like(gray)
    final[max(0, y - pad) : min(H, y + h + pad),
          max(0, x - pad) : min(W, x + w + pad)] = 255
    info["best_bbox"] = (x, y, w, h)
    return final, info


def reject_text_underlines(segments, mser_text_mask: np.ndarray, samples: int = 7):
    """Conservatively drop segments that are clearly text underlines or
    leader lines, while preserving real walls that merely brush a room
    label.

    Rules (must satisfy all to be dropped):
      - >=70% of sampled points lie in MSER-detected text regions, AND
      - the segment is short (<120 px) — long structural walls passing
        through several labels are kept, AND
      - the segment is nearly horizontal (|dy| < 4 px) OR the segment is
        short enough that orientation doesn't matter (<60 px).
    """
    if mser_text_mask is None:
        return segments
    h, w = mser_text_mask.shape
    out = []
    for s in segments:
        xs = np.linspace(s.x1, s.x2, samples).astype(int).clip(0, w - 1)
        ys = np.linspace(s.y1, s.y2, samples).astype(int).clip(0, h - 1)
        in_text = float((mser_text_mask[ys, xs] == 0).mean())
        if in_text < 0.7:
            out.append(s)
            continue
        length = math.hypot(s.x2 - s.x1, s.y2 - s.y1)
        dy = abs(s.y2 - s.y1)
        is_underline = (length < 120) and (dy < 4 or length < 60)
        if not is_underline:
            out.append(s)
    return out
