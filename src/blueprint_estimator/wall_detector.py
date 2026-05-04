"""Wall vs. non-wall discrimination via image recognition.

Walls in blueprints come in two conventions:
  - Double-line: two parallel strokes at wall thickness (residential CAD).
  - Single-line: one bold stroke (schematic, MEP, quick floor plans).

Pure geometry cannot tell a single-line wall from a dimension line. So this
detector extracts visual features around each candidate segment and trains
an in-image RandomForest using pseudo-labels bootstrapped from priors:

  Positive priors:
    - Segments belonging to a confident double-line pair.
    - Long, axis-aligned segments whose endpoints meet other long segments
      (walls form closed rooms; dimensions don't).

  Negative priors:
    - Segments inside text regions (MSER mask).
    - Very short segments / hatching strokes.
    - Segments terminating at arrowheads (gradient blob at endpoint).

Features per segment (visual + geometric):
    - perpendicular intensity profile: peak depth, FWHM (stroke thickness)
    - parallel-partner score (presence and quality of a double-line partner)
    - endpoint junction degree (how many other segments meet at endpoints)
    - neighborhood ink density vs. ink-density gradient (walls border rooms)
    - length, axis-alignment, in-text-mask flag

Empty result when no segment exceeds the wall-probability threshold.
"""

from __future__ import annotations

import math
import cv2
import numpy as np

from blueprint_estimator.schemas import Segment, WallGraph
from blueprint_estimator.wall_graph_cv import (
    segments_from_image_hough,
    segments_to_wall_graph,
)


# Physical priors (real-world inches/feet — converted to pixels per image).
WALL_THICKNESS_MIN_INCHES = 2.0   # interior partition lower bound
WALL_THICKNESS_MAX_INCHES = 14.0  # exterior wall upper bound
MIN_WALL_LEN_FEET = 1.5           # walls span at least ~18"
JUNCTION_RADIUS_INCHES = 4.0      # endpoints meeting within ~4" = same junction

# Pixel fallbacks (used when no scale info available; assume 150 DPI, 1/4"=1' plan).
PARALLEL_ANGLE_TOL_DEG = 6.0
OVERLAP_MIN_FRAC = 0.4
WALL_PROB_THRESHOLD = 0.55


def physical_thresholds(
    feet_per_pixel: float | None,
    image_shape: tuple[int, int] | None = None,
) -> dict:
    """Convert physical wall priors to per-image pixel thresholds.

    feet_per_pixel comes from ScaleConfig.resolved_feet_per_pixel() — the
    pipeline already computes it from PDF scale text or OCR'd title block.
    Falls back to assuming 1/4"=1' at 150 DPI when missing.
    """
    if feet_per_pixel is None or feet_per_pixel <= 0:
        # 1 drawing inch = 4 real feet; 150 px = 1 drawing inch → 4 ft / 150 px
        feet_per_pixel = 4.0 / 150.0

    px_per_inch = 1.0 / (feet_per_pixel * 12.0)
    px_per_foot = px_per_inch * 12.0

    th_min = max(2.0, WALL_THICKNESS_MIN_INCHES * px_per_inch)
    th_max = max(th_min + 4.0, WALL_THICKNESS_MAX_INCHES * px_per_inch)
    min_len = max(20.0, MIN_WALL_LEN_FEET * px_per_foot)
    junc_r = max(3.0, JUNCTION_RADIUS_INCHES * px_per_inch)

    return {
        "wall_thickness_min_px": float(th_min),
        "wall_thickness_max_px": float(th_max),
        "min_len_px": float(min_len),
        "junction_radius_px": float(junc_r),
        "px_per_inch": float(px_per_inch),
        "px_per_foot": float(px_per_foot),
        "feet_per_pixel": float(feet_per_pixel),
    }


# ----------------------------- text masking ---------------------------------

def mask_text_regions(image_bgr: np.ndarray) -> np.ndarray:
    """Return uint8 mask: 255 keep, 0 = likely text. Uses MSER."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    keep = np.full((h, w), 255, dtype=np.uint8)
    try:
        mser = cv2.MSER_create(delta=5, min_area=20, max_area=2000)
        regions, _ = mser.detectRegions(gray)
    except Exception:
        return keep
    for pts in regions:
        x, y, rw, rh = cv2.boundingRect(pts.reshape(-1, 1, 2))
        if rh < 4 or rh > 60:
            continue
        aspect = rw / max(rh, 1)
        if 0.1 < aspect < 8.0 and rw * rh < 4000:
            cv2.rectangle(keep, (x - 1, y - 1), (x + rw + 1, y + rh + 1), 0, -1)
    return keep


# ---------------------------- segment helpers -------------------------------

def _angle_deg(s: Segment) -> float:
    return math.degrees(math.atan2(s.y2 - s.y1, s.x2 - s.x1))


def _length(s: Segment) -> float:
    return math.hypot(s.x2 - s.x1, s.y2 - s.y1)


def _axis_aligned(s: Segment, tol_deg: float = 8.0) -> bool:
    a = abs(_angle_deg(s)) % 180.0
    return a < tol_deg or abs(a - 90.0) < tol_deg or abs(a - 180.0) < tol_deg


def _parallel(a: Segment, b: Segment, tol_deg: float = PARALLEL_ANGLE_TOL_DEG) -> bool:
    d = abs(_angle_deg(a) - _angle_deg(b)) % 180.0
    return min(d, 180.0 - d) < tol_deg


def _perp_distance_and_overlap(a: Segment, b: Segment) -> tuple[float, float]:
    ax, ay = a.x2 - a.x1, a.y2 - a.y1
    L = math.hypot(ax, ay)
    if L < 1e-6:
        return float("inf"), 0.0
    ux, uy = ax / L, ay / L
    nx, ny = -uy, ux
    bmx, bmy = (b.x1 + b.x2) * 0.5, (b.y1 + b.y2) * 0.5
    perp = abs((bmx - a.x1) * nx + (bmy - a.y1) * ny)

    def proj(px: float, py: float) -> float:
        return (px - a.x1) * ux + (py - a.y1) * uy

    t_b1, t_b2 = sorted((proj(b.x1, b.y1), proj(b.x2, b.y2)))
    inter = max(0.0, min(L, t_b2) - max(0.0, t_b1))
    return perp, inter / L


# --------------------------- visual feature extraction ----------------------

def _perpendicular_profile(gray: np.ndarray, s: Segment, half: int = 12) -> np.ndarray:
    """Sample intensities perpendicular to the segment at its midpoint.

    Dark (low intensity) = ink. A wall stroke produces a clear dark dip.
    """
    h, w = gray.shape
    cx = (s.x1 + s.x2) * 0.5
    cy = (s.y1 + s.y2) * 0.5
    dx, dy = s.x2 - s.x1, s.y2 - s.y1
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return np.full(2 * half + 1, 255.0)
    nx, ny = -dy / L, dx / L  # perpendicular unit
    samples = []
    for k in range(-half, half + 1):
        x = int(round(cx + k * nx))
        y = int(round(cy + k * ny))
        if 0 <= x < w and 0 <= y < h:
            samples.append(float(gray[y, x]))
        else:
            samples.append(255.0)
    return np.array(samples, dtype=np.float64)


def _stroke_metrics(profile: np.ndarray) -> tuple[float, float, float]:
    """Return (peak_depth, fwhm_px, profile_contrast) for the dark dip."""
    bg = float(np.percentile(profile, 90))  # paper white
    inv = bg - profile  # ink darkness
    peak = float(inv.max())
    if peak <= 5:
        return 0.0, 0.0, 0.0
    half = peak * 0.5
    above = inv >= half
    if not above.any():
        fwhm = 0.0
    else:
        idx = np.where(above)[0]
        fwhm = float(idx[-1] - idx[0] + 1)
    contrast = float(inv.std())
    return peak, fwhm, contrast


def _endpoint_junctions(
    s: Segment,
    others: list[Segment],
    radius: float = 6.0,
    min_other_len: float = 50.0,
) -> int:
    """Count endpoints of other LONG segments meeting either endpoint.

    Walls form rooms with other walls. Dimension-line tick marks are short
    and must not count, otherwise dimensions look like walls.
    """
    cnt = 0
    eps = ((s.x1, s.y1), (s.x2, s.y2))
    for o in others:
        if o is s:
            continue
        if _length(o) < min_other_len:
            continue
        for ep in ((o.x1, o.y1), (o.x2, o.y2)):
            for me in eps:
                if math.hypot(ep[0] - me[0], ep[1] - me[1]) <= radius:
                    cnt += 1
                    break
    return cnt


def _neighborhood_ink(gray: np.ndarray, s: Segment, box: int = 30) -> tuple[float, float]:
    """Ink fraction near segment vs. just outside — walls separate rooms."""
    h, w = gray.shape
    cx = int((s.x1 + s.x2) * 0.5)
    cy = int((s.y1 + s.y2) * 0.5)
    dx, dy = s.x2 - s.x1, s.y2 - s.y1
    L = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / L, dx / L

    def ink_box(ox: float, oy: float) -> float:
        x0 = max(0, int(cx + ox - box / 2))
        y0 = max(0, int(cy + oy - box / 2))
        x1 = min(w, int(cx + ox + box / 2))
        y1 = min(h, int(cy + oy + box / 2))
        if x1 <= x0 or y1 <= y0:
            return 0.0
        patch = gray[y0:y1, x0:x1]
        return float((patch < 200).mean())

    side_a = ink_box(nx * (box * 0.7), ny * (box * 0.7))
    side_b = ink_box(-nx * (box * 0.7), -ny * (box * 0.7))
    return side_a, side_b


def building_footprint_mask(gray: np.ndarray) -> np.ndarray:
    """Approximate building footprint: the largest connected blob of ink.

    Architectural plans always have ONE dominant figure (the building) on
    a white sheet with title block, schedules, and notes around it. We
    threshold ink, dilate to fill interior, then keep the biggest blob.
    """
    h, w = gray.shape
    ink = (gray < 200).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, k, iterations=4)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if n <= 1:
        return np.zeros_like(gray)
    # exclude the background (label 0) and pick largest interior blob whose
    # bounding box doesn't cover the whole sheet (= page border) and is big
    best_i, best_area = -1, 0
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bw >= w * 0.95 and bh >= h * 0.95:
            continue
        if area > best_area:
            best_area = int(area)
            best_i = i
    if best_i < 0:
        return np.zeros_like(gray)
    return (lab == best_i).astype(np.uint8) * 255


def _footprint_distance_features(footprint: np.ndarray, segments: list[Segment]) -> np.ndarray:
    """For each segment: (midpoint inside footprint?, normalized distance from
    midpoint to footprint boundary). Exterior walls sit near the boundary.
    """
    if footprint is None or footprint.max() == 0:
        return np.zeros((len(segments), 2), dtype=np.float64)
    # signed distance: positive inside, zero on boundary; we want distance
    # from boundary regardless of inside/outside.
    inv = (footprint == 0).astype(np.uint8) * 255
    dist_in = cv2.distanceTransform(footprint, cv2.DIST_L2, 3)
    dist_out = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    H, W = footprint.shape
    diag = float(np.hypot(H, W))
    out = np.zeros((len(segments), 2), dtype=np.float64)
    for i, s in enumerate(segments):
        x = int(np.clip((s.x1 + s.x2) / 2, 0, W - 1))
        y = int(np.clip((s.y1 + s.y2) / 2, 0, H - 1))
        inside = footprint[y, x] > 0
        d = (dist_in[y, x] if inside else dist_out[y, x])
        out[i, 0] = 1.0 if inside else 0.0
        out[i, 1] = float(d / diag)
    return out


def _in_text_mask(s: Segment, text_keep: np.ndarray) -> float:
    """Fraction of the segment that lies in masked-text regions (0=keep, 1=text)."""
    h, w = text_keep.shape
    n = max(8, int(_length(s)))
    xs = np.linspace(s.x1, s.x2, n).astype(int)
    ys = np.linspace(s.y1, s.y2, n).astype(int)
    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)
    samples = text_keep[ys, xs]
    return float((samples == 0).mean())


def segment_features(
    s: Segment,
    gray: np.ndarray,
    others: list[Segment],
    text_keep: np.ndarray,
    parallel_partner_score: float,
    junction_radius: float = 6.0,
    min_other_len: float = 50.0,
) -> np.ndarray:
    profile = _perpendicular_profile(gray, s)
    peak, fwhm, contrast = _stroke_metrics(profile)
    junc = _endpoint_junctions(s, others, radius=junction_radius, min_other_len=min_other_len)
    ink_a, ink_b = _neighborhood_ink(gray, s)
    text_frac = _in_text_mask(s, text_keep)
    length = _length(s)
    axis = 1.0 if _axis_aligned(s) else 0.0
    return np.array([
        length,
        axis,
        peak,
        fwhm,
        contrast,
        float(junc),
        ink_a,
        ink_b,
        abs(ink_a - ink_b),         # asymmetry → wall borders different rooms
        text_frac,
        parallel_partner_score,
    ], dtype=np.float64)


# ----------------------- pseudo-labels from priors --------------------------

def _is_paper_between(gray: np.ndarray, a: Segment, b: Segment) -> bool:
    """True if the gap between two parallel segments is mostly white paper.

    Filters out anti-aliased single-stroke edges that appear as a parallel pair.
    """
    h, w = gray.shape
    n = 12
    samples = []
    for t in np.linspace(0.2, 0.8, n):
        ax = a.x1 + t * (a.x2 - a.x1)
        ay = a.y1 + t * (a.y2 - a.y1)
        # closest point on b
        bx = b.x1 + t * (b.x2 - b.x1)
        by = b.y1 + t * (b.y2 - b.y1)
        # midpoint between a(t) and b(t)
        mx = int(round((ax + bx) * 0.5))
        my = int(round((ay + by) * 0.5))
        if 0 <= mx < w and 0 <= my < h:
            samples.append(float(gray[my, mx]))
    if not samples:
        return False
    return float(np.mean(samples)) > 200  # bright = paper


def _parallel_partner_scores(
    segs: list[Segment],
    text_keep: np.ndarray | None = None,
    min_partner_len: float = 50.0,
    gray: np.ndarray | None = None,
    thickness_min: float = 2.0,
    thickness_max: float = 22.0,
) -> tuple[list[float], list[int]]:
    """For each segment: best partner score in [0,1], and partner index (-1 if none).

    Pairs where either segment is too short or lies in text regions are ignored —
    text rows form parallel patterns that mimic double-line walls.
    """
    n = len(segs)
    scores = [0.0] * n
    partner = [-1] * n
    in_text = [
        (_in_text_mask(s, text_keep) if text_keep is not None else 0.0) for s in segs
    ]
    lengths = [_length(s) for s in segs]
    for i in range(n):
        if lengths[i] < min_partner_len or in_text[i] >= 0.3:
            continue
        for j in range(i + 1, n):
            if lengths[j] < min_partner_len or in_text[j] >= 0.3:
                continue
            if not _parallel(segs[i], segs[j]):
                continue
            perp, overlap = _perp_distance_and_overlap(segs[i], segs[j])
            if not (thickness_min <= perp <= thickness_max):
                continue
            if overlap < OVERLAP_MIN_FRAC:
                continue
            if gray is not None and not _is_paper_between(gray, segs[i], segs[j]):
                continue
            band_center = (thickness_min + thickness_max) / 2
            thickness_score = 1.0 - min(1.0, abs(perp - band_center) / band_center)
            score = 0.6 * overlap + 0.4 * thickness_score
            if score > scores[i]:
                scores[i] = score
                partner[i] = j
            if score > scores[j]:
                scores[j] = score
                partner[j] = i
    return scores, partner


def _bootstrap_labels(
    segs: list[Segment],
    feats: np.ndarray,
    partner_scores: list[float],
    min_len_px: float = 25.0,
) -> np.ndarray:
    """Return labels: 1=positive, 0=negative, -1=unlabeled."""
    n = len(segs)
    labels = np.full(n, -1, dtype=np.int64)
    # column indices match segment_features() order
    LEN, AXIS, PEAK, FWHM, CONTR, JUNC, INKA, INKB, ASYM, TEXT, PARP = range(11)

    for i in range(n):
        # POSITIVE: confident double-line pair OR (long + axis-aligned + thick stroke + junctions + not in text)
        if partner_scores[i] >= 0.55:
            labels[i] = 1
            continue
        if (
            feats[i, LEN] >= 60
            and feats[i, AXIS] == 1.0
            and feats[i, PEAK] >= 60
            and feats[i, FWHM] >= 1.5
            and feats[i, JUNC] >= 2
            and feats[i, TEXT] < 0.2
        ):
            labels[i] = 1
            continue
        # NEGATIVE: in text, or very short, or no junctions and weak stroke
        if feats[i, TEXT] >= 0.5:
            labels[i] = 0
            continue
        if feats[i, LEN] < min_len_px:
            labels[i] = 0
            continue
        if feats[i, JUNC] == 0 and feats[i, PEAK] < 40:
            labels[i] = 0
            continue
        if feats[i, FWHM] > 0 and feats[i, FWHM] < 1.0 and feats[i, JUNC] == 0:
            labels[i] = 0
    return labels


# --------------------------------- detector ---------------------------------

_GLOBAL_STUCCO_MODEL = None


def _load_global_stucco_model():
    global _GLOBAL_STUCCO_MODEL
    if _GLOBAL_STUCCO_MODEL is not None:
        return _GLOBAL_STUCCO_MODEL
    import os
    import pickle
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "..", "data", "stucco_rf.pkl"),
        os.path.join(here, "..", "..", "..", "data", "stucco_rf.pkl"),
    ]
    for p in candidates:
        p = os.path.abspath(p)
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    _GLOBAL_STUCCO_MODEL = pickle.load(f)
                return _GLOBAL_STUCCO_MODEL
            except Exception:
                pass
    _GLOBAL_STUCCO_MODEL = False
    return False


def detect_walls(
    image_bgr: np.ndarray,
    use_text_mask: bool = True,
    threshold: float = WALL_PROB_THRESHOLD,
    scale_config=None,
    feet_per_pixel: float | None = None,
    use_stucco_model: bool = True,
) -> tuple[list[Segment], WallGraph, dict]:
    """Hybrid geometric + learned image-recognition wall detector.

    Pass `scale_config` (from blueprint_estimator.scale_qty.ScaleConfig) or
    `feet_per_pixel` directly so thickness/length thresholds scale with the
    drawing's real-world units instead of being hardcoded for ~150 DPI.
    """
    if feet_per_pixel is None and scale_config is not None:
        try:
            feet_per_pixel = float(scale_config.resolved_feet_per_pixel())
        except Exception:
            feet_per_pixel = None
    th = physical_thresholds(feet_per_pixel, image_bgr.shape[:2])

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    text_keep = mask_text_regions(image_bgr) if use_text_mask else np.full(gray.shape, 255, np.uint8)

    # Run Hough on a copy with text painted out, but keep features on original.
    img_for_hough = image_bgr
    if use_text_mask:
        masked = gray.copy()
        masked[text_keep == 0] = 255
        img_for_hough = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)

    raw = segments_from_image_hough(img_for_hough)
    raw = [s for s in raw if _length(s) >= th["min_len_px"]]

    if not raw:
        return [], WallGraph(), {
            "raw_segments": 0, "wall_segments": 0, "mean_confidence": 0.0,
            "text_masked": use_text_mask, "method": "no-segments",
            "physical_thresholds": th,
        }

    partner_scores, _ = _parallel_partner_scores(
        raw, text_keep=text_keep, gray=gray,
        min_partner_len=max(40.0, th["min_len_px"] * 0.8),
        thickness_min=th["wall_thickness_min_px"],
        thickness_max=th["wall_thickness_max_px"],
    )
    feats = np.stack([
        segment_features(s, gray, raw, text_keep, partner_scores[i],
                         junction_radius=th["junction_radius_px"],
                         min_other_len=th["min_len_px"])
        for i, s in enumerate(raw)
    ])

    labels = _bootstrap_labels(raw, feats, partner_scores, min_len_px=th["min_len_px"])
    pos_n = int((labels == 1).sum())
    neg_n = int((labels == 0).sum())

    method = "rules"
    probs = np.zeros(len(raw), dtype=np.float64)

    if pos_n >= 4 and neg_n >= 4:
        try:
            from sklearn.ensemble import RandomForestClassifier
            mask = labels >= 0
            clf = RandomForestClassifier(
                n_estimators=120, max_depth=8, min_samples_leaf=2,
                class_weight="balanced", random_state=42,
            )
            clf.fit(feats[mask], labels[mask])
            probs = clf.predict_proba(feats)[:, list(clf.classes_).index(1)]
            method = f"rf(pos={pos_n},neg={neg_n})"
        except Exception:
            method = "rules-fallback"

    if method.startswith("rules"):
        # Hand-tuned scorer when not enough labels for a model.
        # Scale features into [0,1] roughly and weight.
        LEN, AXIS, PEAK, FWHM, CONTR, JUNC, INKA, INKB, ASYM, TEXT, PARP = range(11)
        s_len = np.clip(feats[:, LEN] / 200.0, 0, 1)
        s_peak = np.clip(feats[:, PEAK] / 120.0, 0, 1)
        s_fwhm = np.clip(feats[:, FWHM] / 6.0, 0, 1)
        s_junc = np.clip(feats[:, JUNC] / 4.0, 0, 1)
        s_par = feats[:, PARP]
        s_text = feats[:, TEXT]
        probs = (
            0.18 * s_len + 0.12 * feats[:, AXIS] + 0.15 * s_peak + 0.10 * s_fwhm
            + 0.15 * s_junc + 0.30 * s_par
        ) - 1.0 * s_text
        # hard gates: too short, mostly-in-text, OR isolated (no junction & no partner) → never a wall
        probs[feats[:, LEN] < th["min_len_px"]] = 0.0
        probs[feats[:, TEXT] >= 0.3] = 0.0
        probs[(feats[:, JUNC] == 0) & (feats[:, PARP] < 0.4)] = 0.0
        probs = np.clip(probs, 0.0, 1.0)

    # Optional second pass: trained stucco classifier (from completed exhibits)
    stucco_probs = None
    if use_stucco_model:
        bundle = _load_global_stucco_model()
        if bundle:
            try:
                H, W = image_bgr.shape[:2]
                pos_feats = np.array([
                    [
                        ((s.x1 + s.x2) / 2) / W,
                        ((s.y1 + s.y2) / 2) / H,
                        min(((s.x1 + s.x2) / 2) / W, 1 - ((s.x1 + s.x2) / 2) / W),
                        min(((s.y1 + s.y2) / 2) / H, 1 - ((s.y1 + s.y2) / 2) / H),
                    ]
                    for s in raw
                ])
                footprint = building_footprint_mask(gray)
                fp_feats = _footprint_distance_features(footprint, raw)
                feats_full = np.hstack([feats, pos_feats, fp_feats]) if pos_feats.size else feats
                expected = getattr(bundle["model"], "n_features_in_", None)
                if expected is not None:
                    if expected == feats.shape[1]:
                        feats_full = feats
                    elif expected == feats.shape[1] + pos_feats.shape[1]:
                        feats_full = np.hstack([feats, pos_feats])
                stucco_probs = bundle["model"].predict_proba(feats_full)[
                    :, list(bundle["model"].classes_).index(1)
                ]
                method = method + "+stucco_rf"
            except Exception:
                stucco_probs = None

    walls: list[Segment] = []
    for i, (s, p) in enumerate(zip(raw, probs)):
        if p >= threshold:
            meta = dict(s.meta)
            meta.update({"confidence": float(p), "source": "wall_detector"})
            if stucco_probs is not None:
                meta["stucco_probability"] = float(stucco_probs[i])
            walls.append(Segment(s.x1, s.y1, s.x2, s.y2, meta=meta))

    graph = segments_to_wall_graph(walls)
    info = {
        "raw_segments": len(raw),
        "wall_segments": len(walls),
        "mean_confidence": float(np.mean([s.meta.get("confidence", 0.0) for s in walls])) if walls else 0.0,
        "text_masked": use_text_mask,
        "method": method,
        "threshold": threshold,
        "pos_pseudo_labels": pos_n,
        "neg_pseudo_labels": neg_n,
        "physical_thresholds": th,
    }
    return walls, graph, info
