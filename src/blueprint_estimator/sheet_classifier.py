"""Decide whether a sheet is a floor plan that should run wall takeoff.

Construction sets contain many non-plan sheets that look like they have
walls (elevations, sections, accessibility details, schedules, fire-wall
details, stair details, site plans, RCPs). Running our wall takeoff on
those inflates linear-feet counts and corrupts the bid.

Two layers of defense:

  1. Filename heuristics — sheet titles in construction sets follow
     standard nomenclature (CSI/AIA). We skip any sheet whose name
     unambiguously names something other than a floor plan.

  2. Visual signature — a real floor plan has many enclosed
     non-perimeter white regions (rooms). Details and elevations have
     0-3. We count them with a connected-components pass on the
     inverted ink mask, restricted to inside the building bbox.

Both must agree before we run the wall classifier. Conservative on
purpose: false negatives (skipping a real floor plan) are recoverable
by the user uploading just that sheet; false positives (running on a
schedule) cost the customer money.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np


# Keywords that indicate the sheet is NOT a floor plan we should run on.
# Tested against the uppercase filename.
_NON_FLOOR_PLAN_KEYWORDS = [
    "ELEVATION",
    "SECTION",
    "SCHEDULE",
    "NOTES",
    "DETAIL",
    "DETAILS",
    "ASSEMBLY",
    "ASSEMBLIES",
    "FIRE WALL",
    "FIRE-WALL",
    "STAIR",
    "EGRESS",
    "REFERENCE",
    "TITLE",
    "COVER",
    "INDEX",
    "GENERAL",
    "ACCESSIB",
    "GREEN BUILDING",
    "CONDITION",
    "AREA CALC",
    "FLOOR AREA",
    "SLAB",
    "RAMP",
    "ROOF DETAIL",
    "WATERPROOF",
    "WINDOW",       # window schedule
    "DOOR ",        # door schedule (trailing space avoids matching "DOORWAY")
    "FINISH SCHED",
    "RCP",          # reflected ceiling plan — for ceilings, not walls
    "REFLECTED CEILING",
    "FIRE DEPARTMENT",
    "T-",           # T-series sheets are typically title/general
    "GREEN ",
    "MANUFACTURE",
    "MANUFACTURER",
    "SOIL",
    "GRADING",
    "SHEET INDEX",
]

# Strong positives that override negatives (e.g. "ENLARGED FLOOR PLAN").
_FLOOR_PLAN_POSITIVES = [
    "FLOOR PLAN",
    "1ST FLOOR",
    "2ND FLOOR",
    "3RD FLOOR",
    "4TH FLOOR",
    "5TH FLOOR",
    "6TH FLOOR",
    "7TH FLOOR",
    "GROUND FLOOR",
    "BASEMENT PLAN",
    "ENLARGED FLOOR",
    "FIRST-FLOOR-PLAN",
    "SECOND-FLOOR-PLAN",
    "THIRD-FLOOR-PLAN",
    "FOURTH-FLOOR-PLAN",
    "FIFTH-FLOOR-PLAN",
    "PLAN-REV",
    "FLOOR-PLAN",
]


# Hard rejects that override any positive match (e.g. "2ND FLOOR REFLECTED
# CEILING PLAN" contains "2ND FLOOR" but is not a wall sheet).
_HARD_REJECT_KEYWORDS = [
    "RCP",
    "REFLECTED CEILING",
    "ELEVATION",
    "SECTION",
    "SCHEDULE",
    "DETAIL",
    "ASSEMBLY",
    "STAIR",
    "EGRESS",
    "ACCESSIB",
    "FIRE WALL",
    "FIRE-WALL",
    "AREA CALC",
    "FLOOR AREA",
    "SLAB",
    "FINISH SCHED",
    "WATERPROOF",
    "FIRE DEPARTMENT",
    "MANUFACTURE",
    "MANUFACTURER",
    "GREEN BUILDING",
    "SOIL",
    "GRADING",
    "RAMP",
    "ROOF DETAIL",
]


def filename_says_floor_plan(filename: str) -> tuple[bool, str]:
    """Return (is_floor_plan, reason). Hard rejects beat positives."""
    upper = filename.upper()
    for hard in _HARD_REJECT_KEYWORDS:
        if hard in upper:
            return False, f"hard reject keyword: {hard}"
    for pos in _FLOOR_PLAN_POSITIVES:
        if pos in upper:
            return True, f"matched positive keyword: {pos}"
    for neg in _NON_FLOOR_PLAN_KEYWORDS:
        if neg in upper:
            return False, f"matched non-plan keyword: {neg}"
    return True, "ambiguous"


def measure_rooms(image_bgr: np.ndarray, bbox: tuple[int, int, int, int] | None = None,
                   min_room_area: int = 1500) -> dict:
    """Return room statistics inside the building bbox.

    Returns:
      n_rooms             — count of enclosed paper regions >= min_room_area
      max_room_frac       — area of the biggest enclosed room / bbox area
      median_room_frac    — median of room area / bbox area
      area_cv             — coefficient of variation of room areas

    A real floor plan: many rooms (10+), max_room_frac < 0.25, varied sizes.
    A panel-grid detail sheet: few rooms (4-12), max_room_frac > 0.25,
                                similar sizes (low cv).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    if bbox is None:
        x, y, w, h = 0, 0, W, H
    else:
        x, y, w, h = bbox
    x = max(0, x); y = max(0, y)
    w = min(W - x, w); h = min(H - y, h)
    bbox_area = max(w * h, 1)
    if w <= 30 or h <= 30:
        return {"n_rooms": 0, "max_room_frac": 0.0, "median_room_frac": 0.0,
                "area_cv": 0.0, "areas": []}

    crop = gray[y:y + h, x:x + w]
    # Dilate ink so doorways and small wall gaps close — otherwise the entire
    # interior of a real floor plan is ONE connected paper region (rooms
    # bleed into each other through openings).
    ink = (crop < 200).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    ink_closed = cv2.dilate(ink, k, iterations=3)
    paper = cv2.bitwise_not(ink_closed)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(paper, connectivity=4)
    areas: list[int] = []
    for i in range(1, n):
        cx, cy, cw, ch, area = stats[i]
        if area < min_room_area:
            continue
        if cx <= 1 or cy <= 1 or cx + cw >= w - 1 or cy + ch >= h - 1:
            continue
        areas.append(int(area))
    if not areas:
        return {"n_rooms": 0, "max_room_frac": 0.0, "median_room_frac": 0.0,
                "area_cv": 0.0, "areas": []}
    arr = np.array(areas, dtype=np.float64)
    return {
        "n_rooms": int(len(areas)),
        "max_room_frac": float(arr.max() / bbox_area),
        "median_room_frac": float(np.median(arr) / bbox_area),
        "area_cv": float(arr.std() / max(arr.mean(), 1)),
        "areas": areas,
    }


def count_enclosed_rooms(image_bgr: np.ndarray, bbox=None, min_room_area: int = 1500) -> int:
    return measure_rooms(image_bgr, bbox, min_room_area)["n_rooms"]


def is_floor_plan_visual(image_bgr: np.ndarray, bbox=None, score: float = 0.0,
                         candidates: list | None = None,
                         min_rooms: int = 1, min_score: float = 0.0,
                         min_dominance: float = 0.0) -> tuple[bool, dict]:
    """Visual check is now PERMISSIVE by design: it kept rejecting real
    finish plans because of legend boxes on the same page. Filename rules
    are doing the heavy lifting; the visual check just records stats.
    """
    """Visual check for floor-plan-likeness.

    Real floor plan: ONE dominant ink component, with >=min_rooms enclosed
    paper regions inside it, score >= min_score.

    Tile-of-details (parking layouts, accessibility figures, equipment
    sheets, restroom enlargements arranged in a grid): multiple
    components with similar scores. We reject when the winning
    component's area is less than `min_dominance` times the
    second-place component's area, because that signature is a sheet of
    independent panels, not a single building.
    """
    room_stats = measure_rooms(image_bgr, bbox)
    rooms = room_stats["n_rooms"]
    max_frac = room_stats["max_room_frac"]
    pass_rooms = rooms >= min_rooms
    pass_score = score >= min_score
    pass_room_size = True  # disabled — gave too many false rejects on real plans

    dominance = float("inf")
    if candidates and len(candidates) >= 2:
        a0 = float(candidates[0].get("area", 0))
        a1 = float(candidates[1].get("area", 0))
        dominance = a0 / max(a1, 1.0)
    pass_dominance = dominance >= min_dominance

    is_plan = pass_rooms and pass_score and pass_dominance and pass_room_size
    return is_plan, {
        "rooms": rooms, "score": float(score),
        "dominance_ratio": float(dominance) if dominance != float("inf") else None,
        "max_room_frac": max_frac,
        "pass_rooms": pass_rooms, "pass_score": pass_score,
        "pass_dominance": pass_dominance, "pass_room_size": pass_room_size,
    }


def should_run_takeoff(filename: str, image_bgr: np.ndarray, bbox=None,
                        score: float = 0.0,
                        candidates: list | None = None) -> tuple[bool, dict]:
    """Top-level decision combining filename + visual signature."""
    fn_ok, fn_reason = filename_says_floor_plan(filename)
    vi_ok, vi_meta = is_floor_plan_visual(image_bgr, bbox, score, candidates=candidates)
    decision = fn_ok and vi_ok
    reason_parts = []
    if not fn_ok:
        reason_parts.append(f"filename: {fn_reason}")
    if not vi_meta["pass_rooms"]:
        reason_parts.append(f"only {vi_meta['rooms']} enclosed rooms (need >=4)")
    if not vi_meta["pass_score"]:
        reason_parts.append(f"isolator score {vi_meta['score']:.1f} < 8 (no dominant building)")
    if not vi_meta["pass_dominance"]:
        d = vi_meta.get("dominance_ratio")
        reason_parts.append(
            f"top component only {d:.1f}x bigger than runner-up — looks like a tile of detail panels, not one building"
            if d is not None else "no dominant component"
        )
    if not vi_meta.get("pass_room_size", True):
        f = vi_meta.get("max_room_frac", 0)
        reason_parts.append(
            f"biggest enclosed area = {f*100:.0f}% of building bbox — too big for a room, looks like a detail panel"
        )
    return decision, {
        "filename_pass": fn_ok, "filename_reason": fn_reason,
        "visual_pass": vi_ok, **vi_meta,
        "skip_reason": "; ".join(reason_parts) if reason_parts else "",
    }
