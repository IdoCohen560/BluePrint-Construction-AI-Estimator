"""End-to-end supervised training: stucco-wall classifier from completed exhibits.

Each completed Exhibit A page = (plan image) + (red overlay = ground truth
for stucco walls). We:

  1. Render every exhibit page across all projects.
  2. Erase the red overlay so we have the underlying plan image.
  3. Run the wall detector on the cleaned image to get candidate segments.
  4. Label each segment by the fraction of its rasterization that lies in
     the red mask: >= POS_OVERLAP -> positive, otherwise negative.
  5. Aggregate features across every page in every project.
  6. Train a RandomForest globally; pickle it to data/stucco_rf.pkl.
  7. Report per-project precision / recall / IoU on a held-out split.
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from blueprint_estimator.exhibit_mask import (  # noqa: E402
    classify_page,
    render_exhibit_pages,
)
from blueprint_estimator.wall_detector import (  # noqa: E402
    building_footprint_mask,
    detect_walls,
    _footprint_distance_features,
    mask_text_regions,
    segment_features,
    _length,
    _parallel_partner_scores,
    physical_thresholds,
    segments_from_image_hough,
)


POS_OVERLAP = 0.30
NEG_OVERLAP_MAX = 0.03
RASTER_THICKNESS = 6
RENDER_DPI = 220
MIN_HIGHLIGHT_COMPONENT_AREA = 600   # ignore tiny color blobs (legend dots, stamps)
INSIDE_REGION_FRACTION = 0.6         # segment is positive if 60%+ of its midpoint
                                     # neighborhood lies inside one component bbox


def rasterize_segment(segment, shape) -> np.ndarray:
    h, w = shape[:2]
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.line(m, (int(segment.x1), int(segment.y1)), (int(segment.x2), int(segment.y2)),
             255, RASTER_THICKNESS)
    return m


def overlap_fraction(seg_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    a = seg_mask > 0
    b = gt_mask > 0
    inter = np.logical_and(a, b).sum()
    total = a.sum()
    return float(inter / total) if total else 0.0


def _build_region_mask(highlight_mask: np.ndarray) -> np.ndarray:
    """Drop tiny color noise (legend swatches, stamp ink) and return the
    cleaned per-pixel highlight mask used as positive ground truth.
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(highlight_mask, connectivity=8)
    out = np.zeros_like(highlight_mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_HIGHLIGHT_COMPONENT_AREA:
            out[lab == i] = 255
    # generous dilation so we capture line work that BORDERS the highlight region
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    return cv2.dilate(out, k, iterations=3)


def page_features_and_labels(image_bgr: np.ndarray, highlight_mask: np.ndarray):
    """Return (X, y, segments) for one exhibit page."""
    th = physical_thresholds(None)
    text_keep = mask_text_regions(image_bgr)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    masked = gray.copy()
    masked[text_keep == 0] = 255
    raw = segments_from_image_hough(cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR))
    raw = [s for s in raw if _length(s) >= th["min_len_px"]]
    if not raw:
        return np.zeros((0, 11)), np.zeros((0,), dtype=np.int64), []

    partner_scores, _ = _parallel_partner_scores(
        raw, text_keep=text_keep, gray=gray,
        thickness_min=th["wall_thickness_min_px"],
        thickness_max=th["wall_thickness_max_px"],
    )
    H, W = image_bgr.shape[:2]
    base = np.stack([
        segment_features(
            s, gray, raw, text_keep, partner_scores[i],
            junction_radius=th["junction_radius_px"],
            min_other_len=th["min_len_px"],
        )
        for i, s in enumerate(raw)
    ])
    # add normalized positional features: midpoint x/y as fraction of image,
    # and signed distance from page edge (helps model learn "near building
    # boundary" without leaking labels)
    pos = np.array([
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
    feats = np.hstack([base, pos, fp_feats])

    gt = _build_region_mask(highlight_mask)
    h, w = gt.shape

    labels = np.full(len(raw), -1, dtype=np.int64)
    for i, s in enumerate(raw):
        # sample N points along the segment and check fraction inside gt
        n_samples = 9
        xs = np.linspace(s.x1, s.x2, n_samples).astype(int).clip(0, w - 1)
        ys = np.linspace(s.y1, s.y2, n_samples).astype(int).clip(0, h - 1)
        inside_frac = float((gt[ys, xs] > 0).mean())
        # also rasterize-overlap as a stronger positive signal
        sm = rasterize_segment(s, image_bgr.shape)
        ov = overlap_fraction(sm, gt)
        if inside_frac >= INSIDE_REGION_FRACTION or ov >= POS_OVERLAP:
            labels[i] = 1
        elif inside_frac <= 0.05 and ov <= NEG_OVERLAP_MAX:
            labels[i] = 0
    keep = labels >= 0
    return feats[keep], labels[keep], [raw[i] for i in range(len(raw)) if keep[i]]


def main() -> int:
    exhibits_dir = ROOT / "data" / "real" / "all_exhibits"
    pdfs = sorted(exhibits_dir.glob("*.pdf"))
    print(f"projects: {len(pdfs)}")

    X_all = []
    y_all = []
    project_for_row = []
    project_summaries = []

    t0 = time.time()
    for pdf in pdfs:
        project = pdf.stem.replace(" Exhibit A", "").replace(" - High Res", "").strip()
        print(f"\n[{project}]")
        try:
            pages = render_exhibit_pages(str(pdf), dpi=RENDER_DPI)
        except Exception as e:
            print(f"  render failed: {e}")
            continue
        proj_pos = proj_neg = 0
        for page in pages:
            # use any page with meaningful plan-side highlights, regardless of cyan presence
            if page.red_pixel_count < 4000:
                continue
            clean = page.image_bgr.copy()
            clean[page.red_mask > 0] = (255, 255, 255)
            clean[page.cyan_mask > 0] = (255, 255, 255)
            X, y, _segs = page_features_and_labels(clean, page.red_mask)
            if len(y) == 0:
                continue
            X_all.append(X)
            y_all.append(y)
            project_for_row.extend([project] * len(y))
            n_pos = int((y == 1).sum())
            n_neg = int((y == 0).sum())
            proj_pos += n_pos
            proj_neg += n_neg
            print(f"  page {page.page_index+1:02d} segs={len(y):4d} pos={n_pos:3d} neg={n_neg:3d}")
        project_summaries.append((project, proj_pos, proj_neg))

    if not X_all:
        print("no labeled data")
        return 1

    X_all = [x for x in X_all if x.shape[0] > 0]
    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    project_for_row = np.array(project_for_row)
    print(f"\ntotal segments: {len(y)}  pos={int((y==1).sum())} neg={int((y==0).sum())}  ({time.time()-t0:.1f}s)")
    for p, pos, neg in project_summaries:
        print(f"  {p:35s} pos={pos:4d} neg={neg:4d}")

    # leave-one-project-out evaluation with gradient boosting (imbalance friendly)
    from sklearn.ensemble import GradientBoostingClassifier
    print("\n=== LOPO eval ===")
    projects = sorted(set(project_for_row))
    f1_records = []
    for held in projects:
        test = project_for_row == held
        train = ~test
        if (y[train] == 1).sum() < 5 or (y[test] == 1).sum() < 1:
            continue
        # class-weight via sample_weight: positives weighted up by inverse frequency
        n_pos = (y[train] == 1).sum()
        n_neg = (y[train] == 0).sum()
        w_pos = n_neg / max(n_pos, 1)
        sw = np.where(y[train] == 1, w_pos, 1.0)
        clf = GradientBoostingClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05, random_state=42,
        )
        clf.fit(X[train], y[train], sample_weight=sw)
        prob = clf.predict_proba(X[test])[:, list(clf.classes_).index(1)]
        for thresh in (0.5, 0.6, 0.7):
            pred = (prob >= thresh).astype(int)
            tp = ((pred == 1) & (y[test] == 1)).sum()
            fp = ((pred == 1) & (y[test] == 0)).sum()
            fn = ((pred == 0) & (y[test] == 1)).sum()
            p = tp / max(tp + fp, 1)
            r = tp / max(tp + fn, 1)
            f1 = 2 * p * r / max(p + r, 1e-9)
            print(f"  hold={held:35s} thr={thresh}  P={p:.3f} R={r:.3f} F1={f1:.3f}  (n_test={test.sum()})")

    # final model on all data
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    w_pos = n_neg / max(n_pos, 1)
    sw = np.where(y == 1, w_pos, 1.0)
    clf = GradientBoostingClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.05, random_state=42,
    )
    clf.fit(X, y, sample_weight=sw)
    out = ROOT / "data" / "stucco_rf.pkl"
    with open(out, "wb") as f:
        pickle.dump({"model": clf, "feature_names": [
            "length", "axis_aligned", "stroke_peak", "stroke_fwhm",
            "stroke_contrast", "junctions", "ink_left", "ink_right",
            "ink_asymmetry", "in_text_frac", "parallel_partner",
            "x_norm", "y_norm", "x_edge_dist", "y_edge_dist",
            "inside_footprint", "footprint_boundary_dist",
        ]}, f)
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
