"""End-to-end validation across every plan PDF and completed exhibit.

For every plan path (file or folder) we:
  1. Open the PDF with PyMuPDF.
  2. Render up to MAX_PAGES pages at RENDER_DPI.
  3. Run detect_walls; record building-isolator score, wall count,
     classifier method, and whether the overlay stays inside the
     building bbox.
  4. Save a per-page overlay PNG so it can be eyeballed.

For every completed exhibit (which already has labeled stucco walls)
we additionally compute IoU between the predicted-wall mask and the
red highlight mask.

Output: data/validation_out/SUMMARY.csv + thumbnail gallery + per-page
overlays. Print a per-project table to stdout.
"""

from __future__ import annotations

import csv
import gc
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from blueprint_estimator.exhibit_mask import render_exhibit_pages  # noqa: E402
from blueprint_estimator.scale_qty import ScaleConfig  # noqa: E402
from blueprint_estimator.wall_detector import detect_walls  # noqa: E402

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

WIN_BASE = "/mnt/c/Users/idoth/Desktop/Blueprint Material Estimator AI"

PLAN_PATHS = [
    f"{WIN_BASE}/Plans/939 Kingsly Arch",
    f"{WIN_BASE}/Plans/3950 Ingraham Arch",
    f"{WIN_BASE}/Plans/5814 Vineland Arch",
    f"{WIN_BASE}/Plans/10400 Santa Monica ARCHITECTURAL",
    f"{WIN_BASE}/Plans/Saerom Architectural",
    f"{WIN_BASE}/Plans/139 OK - Architectural Construction Set - Newest.pdf",
    f"{WIN_BASE}/Plans/LADBS Stamped Set FIG.pdf",
    f"{WIN_BASE}/Plans/MAD - Architectural - Construction Set - REV J.pdf",
    f"{WIN_BASE}/Plans/Vanowen - ARCH - PC Resubmittal - 20251222.pdf",
]
EXHIBIT_DIR = f"{WIN_BASE}/Completed Exhibits"

OUT_DIR = ROOT / "data" / "validation_out"
RENDER_DPI = 100
MAX_PAGES_PER_PDF = 4
MAX_FILE_MB = 700
SCALE = ScaleConfig(dpi=RENDER_DPI, drawing_feet_per_drawing_inch=4.0)


def safe_label(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name)[:60]


def overlay_walls(image_bgr: np.ndarray, walls, bbox=None) -> np.ndarray:
    out = image_bgr.copy()
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 80, 80), 2)
    for s in walls:
        cv2.line(out, (int(s.x1), int(s.y1)), (int(s.x2), int(s.y2)),
                 (0, 200, 0), 2, cv2.LINE_AA)
    return out


def process_pdf(pdf_path: Path, project_label: str, max_pages: int, exhibits_for_iou=None) -> list[dict]:
    rows = []
    size_mb = pdf_path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        return [{"project": project_label, "file": pdf_path.name, "page": -1,
                 "status": "skipped_too_big", "size_mb": round(size_mb, 1)}]
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        return [{"project": project_label, "file": pdf_path.name, "page": -1,
                 "status": f"open_failed: {e}", "size_mb": round(size_mb, 1)}]

    total = doc.page_count
    pages_to_try = list(range(min(total, max_pages)))
    project_dir = OUT_DIR / safe_label(project_label)
    project_dir.mkdir(parents=True, exist_ok=True)
    file_label = safe_label(pdf_path.stem)

    for i in pages_to_try:
        try:
            pix = doc[i].get_pixmap(dpi=RENDER_DPI, alpha=False)
            buf = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            bgr = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
        except Exception as e:
            rows.append({"project": project_label, "file": pdf_path.name, "page": i + 1,
                         "status": f"render_failed: {e}"})
            continue

        try:
            walls, _, info = detect_walls(bgr, scale_config=SCALE)
        except Exception as e:
            rows.append({"project": project_label, "file": pdf_path.name, "page": i + 1,
                         "status": f"detect_failed: {e}"})
            continue

        bi = info.get("building_isolator", {})
        bbox = bi.get("best_bbox")
        n_walls = len(walls)
        score = float(bi.get("best_score", 0.0))
        method = info.get("method", "")

        outside = 0
        if bbox is not None:
            x, y, w, h = bbox
            for s in walls:
                cx = (s.x1 + s.x2) / 2
                cy = (s.y1 + s.y2) / 2
                if not (x - 30 <= cx <= x + w + 30 and y - 30 <= cy <= y + h + 30):
                    outside += 1
        outside_pct = (outside / n_walls * 100) if n_walls else 0.0

        ovl = overlay_walls(bgr, walls, bbox)
        long = max(ovl.shape[:2])
        if long > 1400:
            scale = 1400 / long
            ovl = cv2.resize(ovl, (int(ovl.shape[1] * scale), int(ovl.shape[0] * scale)))
        cv2.imwrite(str(project_dir / f"{file_label}_p{i+1:02d}.jpg"),
                    ovl, [cv2.IMWRITE_JPEG_QUALITY, 78])

        rows.append({
            "project": project_label, "file": pdf_path.name, "page": i + 1,
            "status": "ok", "walls": n_walls, "iso_score": round(score, 2),
            "outside_bbox_pct": round(outside_pct, 1),
            "method": method, "size_mb": round(size_mb, 1),
            "page_w": pix.width, "page_h": pix.height, "total_pages": total,
        })
        del bgr, ovl, walls, pix, buf
        gc.collect()
    doc.close()
    return rows


def discover(path_str: str) -> list[Path]:
    p = Path(path_str)
    if not p.exists():
        return []
    if p.is_file() and p.suffix.lower() == ".pdf":
        return [p]
    if p.is_dir():
        return sorted([f for f in p.iterdir() if f.suffix.lower() == ".pdf"])
    return []


def main():
    if fitz is None:
        print("PyMuPDF not installed", file=sys.stderr)
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    t0 = time.time()

    for path_str in PLAN_PATHS:
        label = Path(path_str).stem if Path(path_str).suffix else Path(path_str).name
        pdfs = discover(path_str)
        if not pdfs:
            print(f"[{label}] no PDFs found at {path_str}")
            rows.append({"project": label, "file": "(none)", "page": -1,
                         "status": "missing"})
            continue
        print(f"[{label}] {len(pdfs)} PDFs")
        for pdf in pdfs:
            r = process_pdf(pdf, label, MAX_PAGES_PER_PDF)
            for row in r:
                tag = row.get("status", "")
                walls = row.get("walls", "-")
                outside = row.get("outside_bbox_pct", "-")
                print(f"  {pdf.name[:55]:<55} p{row.get('page', '-'):>2}: "
                      f"status={tag} walls={walls} outside%={outside}")
            rows.extend(r)

    # Completed exhibits with IoU vs ground truth
    print("\n=== Completed exhibits IoU eval ===")
    if Path(EXHIBIT_DIR).exists():
        for pdf in sorted(Path(EXHIBIT_DIR).glob("*.pdf")):
            label = pdf.stem
            try:
                pages = render_exhibit_pages(str(pdf), dpi=RENDER_DPI)
            except Exception as e:
                rows.append({"project": label, "file": pdf.name, "page": -1,
                             "status": f"exhibit_render_fail: {e}"})
                continue
            project_dir = OUT_DIR / safe_label(label)
            project_dir.mkdir(parents=True, exist_ok=True)
            for p in pages[:6]:
                if p.red_pixel_count < 4000:
                    continue
                clean = p.image_bgr.copy()
                clean[p.red_mask > 0] = (255, 255, 255)
                clean[p.cyan_mask > 0] = (255, 255, 255)
                walls, _, info = detect_walls(clean, scale_config=SCALE)
                bi = info.get("building_isolator", {})
                bbox = bi.get("best_bbox")
                # rasterize predicted walls
                pred = np.zeros(clean.shape[:2], np.uint8)
                for s in walls:
                    cv2.line(pred, (int(s.x1), int(s.y1)), (int(s.x2), int(s.y2)), 255, 6)
                gt = cv2.dilate(p.red_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), 2)
                inter = np.logical_and(pred > 0, gt > 0).sum()
                union = np.logical_or(pred > 0, gt > 0).sum()
                iou = float(inter / union) if union else 0.0
                ovl = overlay_walls(p.image_bgr, walls, bbox)
                cv2.imwrite(str(project_dir / f"exhibit_p{p.page_index+1:02d}.jpg"),
                            ovl, [cv2.IMWRITE_JPEG_QUALITY, 78])
                rows.append({
                    "project": label, "file": pdf.name, "page": p.page_index + 1,
                    "status": "exhibit_ok", "walls": len(walls),
                    "iso_score": round(float(bi.get("best_score", 0.0)), 2),
                    "iou_vs_gt": round(iou, 4), "method": info.get("method", ""),
                })
                print(f"  {label[:35]:<35} p{p.page_index+1:>2}: walls={len(walls):3d} "
                      f"score={bi.get('best_score', 0):.1f} IoU={iou:.3f}")

    # Summary CSV
    csv_path = OUT_DIR / "SUMMARY.csv"
    keys = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n=== summary ({time.time() - t0:.0f}s) ===")
    print(f"rows: {len(rows)}; csv: {csv_path}")
    n_ok = sum(1 for r in rows if r.get("status") in ("ok", "exhibit_ok"))
    n_skip = sum(1 for r in rows if r.get("status", "").startswith("skipped"))
    n_fail = sum(1 for r in rows if r.get("status", "") not in ("ok", "exhibit_ok") and not r.get("status", "").startswith("skipped"))
    print(f"ok pages: {n_ok}  skipped: {n_skip}  failed: {n_fail}")
    # per-project quality summary
    by_proj: dict[str, list[dict]] = {}
    for r in rows:
        by_proj.setdefault(r["project"], []).append(r)
    print("\nproject              ok pages   mean walls   median outside%   mean IoU")
    for proj in sorted(by_proj):
        ok_rows = [r for r in by_proj[proj] if r.get("status") in ("ok", "exhibit_ok")]
        if not ok_rows:
            print(f"{proj[:25]:<25}  0          -            -                 -")
            continue
        walls = [r.get("walls", 0) or 0 for r in ok_rows]
        outs = [r.get("outside_bbox_pct", 0) or 0 for r in ok_rows if r.get("status") == "ok"]
        ious = [r.get("iou_vs_gt", 0) or 0 for r in ok_rows if r.get("status") == "exhibit_ok"]
        mean_iou = round(sum(ious) / len(ious), 3) if ious else "-"
        med_out = round(sorted(outs)[len(outs)//2], 1) if outs else "-"
        print(f"{proj[:25]:<25}  {len(ok_rows):<3}        {sum(walls)//max(len(walls),1):<5}        "
              f"{med_out!s:<14}    {mean_iou!s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
