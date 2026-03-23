from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from blueprint_estimator.schemas import IngestResult, Segment


def _line(img: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], w: int = 3) -> None:
    cv2.line(img, p1, p2, (0, 0, 0), w, lineType=cv2.LINE_AA)


def generate_rect_floorplan(
    size: tuple[int, int] = (800, 600),
    margin: int = 80,
    thickness: int = 3,
) -> tuple[np.ndarray, list[Segment]]:
    """
    Simple rectangular room + one partition (L-shaped interior).
    Returns BGR image (white bg, black walls) and ground-truth segments in pixel coords.
    """
    h, w = size[1], size[0]
    img = np.ones((h, w, 3), dtype=np.uint8) * 255

    x0, y0 = margin, margin
    x1, y1 = w - margin, h - margin
    mid_x = (x0 + x1) // 2

    segments: list[Segment] = []

    def add_seg(a: tuple[int, int], b: tuple[int, int]) -> None:
        _line(img, a, b, thickness)
        segments.append(
            Segment(float(a[0]), float(a[1]), float(b[0]), float(b[1]), meta={"gt": True})
        )

    # Outer rectangle (clockwise)
    add_seg((x0, y0), (x1, y0))
    add_seg((x1, y0), (x1, y1))
    add_seg((x1, y1), (x0, y1))
    add_seg((x0, y1), (x0, y0))
    # Interior partition
    add_seg((mid_x, y0 + (y1 - y0) // 3), (mid_x, y1))

    return img, segments


def synthetic_ingest(
    size: tuple[int, int] = (800, 600),
) -> IngestResult:
    img, segs = generate_rect_floorplan(size=size)
    return IngestResult(
        source="synthetic",
        image_bgr=img,
        segments_draw=segs,
        meta={"scale_note": "pixel units; supply drawing_units_per_ft for real quantities"},
    )


def total_gt_length_px(segments: list[Segment]) -> float:
    return sum(s.length_px() for s in segments)
