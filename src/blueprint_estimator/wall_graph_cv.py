"""Classical CV: line detection, merge, and wall graph construction."""

from __future__ import annotations

import math
import cv2
import numpy as np

from blueprint_estimator.schemas import Segment, WallGraph


def _snap(p: tuple[float, float], tol: float) -> tuple[float, float]:
    return (round(p[0] / tol) * tol, round(p[1] / tol) * tol)


def segments_from_image_hough(
    image_bgr: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    hough_threshold: int = 40,
    min_line_length: int = 30,
    max_line_gap: int = 10,
) -> list[Segment]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, canny_low, canny_high)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    out: list[Segment] = []
    if lines is None:
        return out
    for ln in lines[:, 0]:
        x1, y1, x2, y2 = ln.tolist()
        out.append(
            Segment(
                float(x1),
                float(y1),
                float(x2),
                float(y2),
                meta={"source": "hough"},
            )
        )
    return out


def _angle_deg(s: Segment) -> float:
    return math.degrees(math.atan2(s.y2 - s.y1, s.x2 - s.x1))


def _merge_collinear(segments: list[Segment], angle_tol_deg: float = 10.0, dist_tol: float = 25.0) -> list[Segment]:
    """
    Best-effort merge of parallel segments whose endpoints align (reduces Hough fragmentation).
    If nothing merges cleanly, returns the original list.
    """
    if len(segments) < 2:
        return segments

    def para(a: float, b: float) -> bool:
        d = abs(a - b) % 180.0
        return min(d, 180.0 - d) < angle_tol_deg

    out: list[Segment] = []
    used = [False] * len(segments)
    for i, s in enumerate(segments):
        if used[i]:
            continue
        ax1, ay1, ax2, ay2 = s.x1, s.y1, s.x2, s.y2
        ang_s = _angle_deg(s)
        pts = [(ax1, ay1), (ax2, ay2)]
        for j in range(i + 1, len(segments)):
            if used[j]:
                continue
            t = segments[j]
            if not para(ang_s, _angle_deg(t)):
                continue
            # any endpoint of t near any accumulated endpoint?
            near = False
            for (px, py) in pts:
                for (qx, qy) in ((t.x1, t.y1), (t.x2, t.y2)):
                    if math.hypot(px - qx, py - qy) < dist_tol:
                        near = True
                        break
                if near:
                    break
            if near:
                used[j] = True
                pts.extend([(t.x1, t.y1), (t.x2, t.y2)])
        # farthest pair among pts = merged segment
        best_d, p1, p2 = -1.0, pts[0], pts[1]
        for a in range(len(pts)):
            for b in range(a + 1, len(pts)):
                d = math.hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1])
                if d > best_d:
                    best_d, p1, p2 = d, pts[a], pts[b]
        out.append(Segment(p1[0], p1[1], p2[0], p2[1], meta={**s.meta, "merged": True}))
        used[i] = True

    return out if len(out) <= len(segments) else segments


def segments_to_wall_graph(segments: list[Segment], snap_tol: float = 5.0) -> WallGraph:
    """Snap endpoints to a grid and build edges with Euclidean lengths."""
    g = WallGraph()
    for s in segments:
        a = _snap((s.x1, s.y1), snap_tol)
        b = _snap((s.x2, s.y2), snap_tol)
        if a == b:
            continue
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        g.nodes.add(a)
        g.nodes.add(b)
        g.edges.append((a, b, {"length": length, "meta": dict(s.meta)}))

    return g


def image_to_wall_segments(
    image_bgr: np.ndarray,
    merge: bool = True,
) -> tuple[list[Segment], WallGraph]:
    raw = segments_from_image_hough(image_bgr)
    segs = _merge_collinear(raw) if merge and raw else raw
    graph = segments_to_wall_graph(segs)
    return segs, graph


def total_segment_length(segments: list[Segment]) -> float:
    return sum(s.length_px() for s in segments)
