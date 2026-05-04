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


def count_enclosed_rooms(image_bgr: np.ndarray, bbox: tuple[int, int, int, int] | None = None,
                          min_room_area: int = 1500) -> int:
    """Connected white regions inside the building bbox that don't touch
    the bbox boundary. Floor plans have many; details have 0–3.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    if bbox is None:
        x, y, w, h = 0, 0, W, H
    else:
        x, y, w, h = bbox
    x = max(0, x); y = max(0, y)
    w = min(W - x, w); h = min(H - y, h)
    if w <= 30 or h <= 30:
        return 0

    crop = gray[y:y + h, x:x + w]
    # ink = dark, paper = bright. We want enclosed paper regions.
    paper = (crop > 200).astype(np.uint8) * 255
    n, lab, stats, _ = cv2.connectedComponentsWithStats(paper, connectivity=4)
    rooms = 0
    for i in range(1, n):
        cx, cy, cw, ch, area = stats[i]
        if area < min_room_area:
            continue
        # exclude regions touching the crop border (= exterior whitespace)
        if cx <= 1 or cy <= 1 or cx + cw >= w - 1 or cy + ch >= h - 1:
            continue
        rooms += 1
    return rooms


def is_floor_plan_visual(image_bgr: np.ndarray, bbox=None, score: float = 0.0,
                         min_rooms: int = 4, min_score: float = 8.0) -> tuple[bool, dict]:
    """Visual check for floor-plan-likeness.

    Floor plan must have >=min_rooms enclosed rooms inside the building
    bbox AND a building-isolator score >= min_score.
    """
    rooms = count_enclosed_rooms(image_bgr, bbox)
    is_plan = rooms >= min_rooms and score >= min_score
    return is_plan, {"rooms": rooms, "score": float(score)}


def should_run_takeoff(filename: str, image_bgr: np.ndarray, bbox=None,
                        score: float = 0.0) -> tuple[bool, dict]:
    """Top-level decision combining filename + visual signature."""
    fn_ok, fn_reason = filename_says_floor_plan(filename)
    vi_ok, vi_meta = is_floor_plan_visual(image_bgr, bbox, score)
    decision = fn_ok and vi_ok
    return decision, {
        "filename_pass": fn_ok, "filename_reason": fn_reason,
        "visual_pass": vi_ok, "rooms": vi_meta["rooms"],
        "iso_score": vi_meta["score"],
    }
