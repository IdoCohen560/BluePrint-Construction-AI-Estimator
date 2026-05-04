"""Score the wall detector against ground-truth completed-exhibit masks.

Usage:
    python3 scripts/calibrate_walls.py <project_dir>

`project_dir` must contain:
    plans/        # raw architectural PDFs (one per sheet, or multi-sheet)
    exhibit/      # completed Exhibit A PDF with red/cyan highlights

For now the script aligns by page index when plan and exhibit page counts
match; otherwise it processes only the exhibit pages classified as 'plan'
and reports per-page stats.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from blueprint_estimator.exhibit_mask import (  # noqa: E402
    classify_page,
    render_exhibit_pages,
    red_segments_to_walls,
)
from blueprint_estimator.wall_detector import detect_walls  # noqa: E402


def rasterize_wall_segments(segments, shape, thickness: int = 6) -> np.ndarray:
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for s in segments:
        cv2.line(mask, (int(s.x1), int(s.y1)), (int(s.x2), int(s.y2)), 255, thickness)
    return mask


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a > 0
    b = b > 0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def precision_recall(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    pred = pred > 0
    gt = gt > 0
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    p = float(tp / (tp + fp)) if (tp + fp) else 0.0
    r = float(tp / (tp + fn)) if (tp + fn) else 0.0
    return p, r


def main(project_dir: str) -> int:
    proj = Path(project_dir)
    exhibit_pdfs = list((proj / "exhibit").glob("*.pdf"))
    if not exhibit_pdfs:
        print("no exhibit PDF found", file=sys.stderr)
        return 1

    out_dir = proj / "calibration_out"
    out_dir.mkdir(exist_ok=True)

    pages = render_exhibit_pages(str(exhibit_pdfs[0]), dpi=150)
    print(f"exhibit pages: {len(pages)}")

    page_summary = []
    for page in pages:
        page_type = classify_page(page)
        info_line = f"page {page.page_index+1:02d} type={page_type:9s} red_px={page.red_pixel_count:>7d} cyan_px={page.cyan_pixel_count:>7d}"

        if page_type != "plan":
            print(info_line)
            page_summary.append((page.page_index + 1, page_type, None, None, None))
            continue

        # Run detector on the exhibit image itself (overlays are translucent
        # — underlying line work is intact). Erase highlight colors first so
        # they don't bias the perpendicular profile.
        clean = page.image_bgr.copy()
        clean[page.red_mask > 0] = (255, 255, 255)
        clean[page.cyan_mask > 0] = (255, 255, 255)
        walls, _, info = detect_walls(clean)
        pred_mask = rasterize_wall_segments(walls, page.image_bgr.shape)

        # Ground truth = red mask, dilated to span wall thickness
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        gt_dilated = cv2.dilate(page.red_mask, k, iterations=2)
        i = iou(pred_mask, gt_dilated)
        p, r = precision_recall(pred_mask, gt_dilated)

        # Save overlay for visual inspection
        vis = page.image_bgr.copy()
        vis[pred_mask > 0] = (0, 255, 0)            # detector = green
        vis[(gt_dilated > 0) & (pred_mask == 0)] = (0, 0, 255)  # missed GT = red
        cv2.imwrite(str(out_dir / f"page_{page.page_index+1:02d}_overlay.png"), vis)

        print(f"{info_line} | walls={len(walls):3d} IoU={i:.3f} P={p:.3f} R={r:.3f} ({info['method']})")
        page_summary.append((page.page_index + 1, page_type, i, p, r))

    print("\n=== summary ===")
    plan_rows = [row for row in page_summary if row[1] == "plan"]
    if plan_rows:
        ious = [row[2] for row in plan_rows]
        precs = [row[3] for row in plan_rows]
        recs = [row[4] for row in plan_rows]
        print(f"plan pages: {len(plan_rows)}  mean IoU={np.mean(ious):.3f}  P={np.mean(precs):.3f}  R={np.mean(recs):.3f}")
    else:
        print("no 'plan' pages classified — check exhibit color thresholds")
    print(f"overlays saved to {out_dir}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
