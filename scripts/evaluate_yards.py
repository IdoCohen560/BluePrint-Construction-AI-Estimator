"""Compare AI-predicted stucco wall yards vs ground-truth from exhibits.

Ground-truth wall yards per page = (linear feet of red highlight) * ceiling / 9
where linear feet is approximated by the perimeter of red components and
ceiling defaults to 9 ft. This is the bid-relevant comparison: how close
is the AI's takeoff to what the human estimator marked.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from blueprint_estimator.exhibit_mask import render_exhibit_pages  # noqa: E402
from blueprint_estimator.scale_qty import ScaleConfig  # noqa: E402
from blueprint_estimator.wall_detector import detect_walls  # noqa: E402


def gt_linear_feet_from_mask(mask: np.ndarray, feet_per_pixel: float, min_area: int = 600) -> float:
    """Total perimeter of large highlight components, halved (each highlight
    region has two long edges = single wall run), converted to feet.
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    perim_px = 0.0
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        comp = (lab == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            perim_px += float(cv2.arcLength(c, True))
    return perim_px / 2.0 * feet_per_pixel


def predicted_linear_feet(walls, feet_per_pixel: float, threshold: float = 0.5) -> float:
    total_px = 0.0
    for s in walls:
        if s.meta.get("stucco_probability", 0.0) < threshold:
            continue
        total_px += float(np.hypot(s.x2 - s.x1, s.y2 - s.y1))
    return total_px * feet_per_pixel


def main(threshold: float = 0.5, ceiling_ft: float = 9.0) -> int:
    exhibits_dir = ROOT / "data" / "real" / "all_exhibits"
    pdfs = sorted(exhibits_dir.glob("*.pdf"))
    sc = ScaleConfig(dpi=220, drawing_feet_per_drawing_inch=4.0)
    feet_per_pixel = sc.resolved_feet_per_pixel()

    print(f"{'project':<30} {'pages':>5}  {'GT_yds':>8}  {'AI_yds':>8}  {'err%':>7}")
    grand_gt = grand_ai = 0.0
    for pdf in pdfs:
        project = pdf.stem.replace(" Exhibit A", "").replace(" - High Res", "").strip()
        try:
            pages = render_exhibit_pages(str(pdf), dpi=220)
        except Exception as e:
            print(f"{project:<30}  render failed: {e}")
            continue
        gt_lf = ai_lf = 0.0
        n_pages = 0
        for p in pages:
            if p.red_pixel_count < 4000:
                continue
            n_pages += 1
            gt_lf += gt_linear_feet_from_mask(p.red_mask, feet_per_pixel)
            clean = p.image_bgr.copy()
            clean[p.red_mask > 0] = (255, 255, 255)
            clean[p.cyan_mask > 0] = (255, 255, 255)
            walls, _, _ = detect_walls(clean, scale_config=sc)
            ai_lf += predicted_linear_feet(walls, feet_per_pixel, threshold)
        gt_yds = gt_lf * ceiling_ft / 9.0
        ai_yds = ai_lf * ceiling_ft / 9.0
        err = abs(ai_yds - gt_yds) / max(gt_yds, 1e-6) * 100
        grand_gt += gt_yds
        grand_ai += ai_yds
        print(f"{project:<30} {n_pages:>5}  {gt_yds:>8.1f}  {ai_yds:>8.1f}  {err:>6.1f}%")

    err = abs(grand_ai - grand_gt) / max(grand_gt, 1e-6) * 100
    print(f"\n{'TOTAL':<30} {'':>5}  {grand_gt:>8.1f}  {grand_ai:>8.1f}  {err:>6.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
