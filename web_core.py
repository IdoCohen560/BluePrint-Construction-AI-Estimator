"""Shared blueprint analysis: batch uploads, per-file scale inference, project totals."""

from __future__ import annotations

import functools
import io
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

SUPPORTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".json", ".dxf"}
import os
# Lower default on resource-constrained hosts (Streamlit Cloud free = 1 GB RAM).
# Override via BPE_DEFAULT_DPI env var.
DEFAULT_RASTER_DPI = int(os.environ.get("BPE_DEFAULT_DPI", "120"))
# Hard cap on PDF size to keep workers from OOMing. Override via BPE_MAX_PDF_MB.
MAX_PDF_MB = int(os.environ.get("BPE_MAX_PDF_MB", "60"))
# Only render up to this many pages from a single PDF. Override via BPE_MAX_PAGES.
MAX_PDF_PAGES = int(os.environ.get("BPE_MAX_PAGES", "30"))
# Line-only DOCX/PDF lists (no type column): all rows get this ``material_type``.
DEFAULT_LINE_MATERIAL_TYPE = "misc"


def ensure_src_path() -> None:
    import sys

    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def draw_segments_overlay(bgr: np.ndarray, segments, color=(0, 200, 0), thickness: int = 2) -> np.ndarray:
    out = bgr.copy()
    for s in segments:
        cv2.line(
            out,
            (int(s.x1), int(s.y1)),
            (int(s.x2), int(s.y2)),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
    return out


def png_bytes_bgr(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()


def high_res_overlay_png(image_bgr: np.ndarray, segments, target_long_side: int = 1920,
                         color=(0, 200, 0), thickness: int = 3) -> bytes:
    """Render the wall overlay at >=1080p (default 1920 long side, ~2 MP).

    Up-samples the source image so that fine walls remain sharp on zoom.
    """
    if image_bgr is None or image_bgr.size == 0:
        return b""
    h, w = image_bgr.shape[:2]
    long = max(h, w)
    scale = max(1.0, target_long_side / float(long))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    big = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    for s in segments:
        cv2.line(
            big,
            (int(s.x1 * scale), int(s.y1 * scale)),
            (int(s.x2 * scale), int(s.y2 * scale)),
            color,
            max(2, int(thickness * (scale / 2))),
            lineType=cv2.LINE_AA,
        )
    ok, buf = cv2.imencode(".png", big, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise RuntimeError("hi-res PNG encode failed")
    return buf.tobytes()


def overlay_pdf_bytes(image_bgr: np.ndarray, segments, target_long_side: int = 2400) -> bytes:
    """Wrap the high-res overlay in a single-page PDF that opens in Acrobat,
    Preview, browsers, or any PDF viewer. Embeds JPEG for compact size.
    """
    if image_bgr is None or image_bgr.size == 0:
        return b""
    h, w = image_bgr.shape[:2]
    long = max(h, w)
    scale = max(1.0, target_long_side / float(long))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    big = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    for s in segments:
        cv2.line(
            big,
            (int(s.x1 * scale), int(s.y1 * scale)),
            (int(s.x2 * scale), int(s.y2 * scale)),
            (0, 200, 0),
            max(2, int(3 * (scale / 2))),
            lineType=cv2.LINE_AA,
        )
    ok, jpg = cv2.imencode(".jpg", big, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    jpg = jpg.tobytes()
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise RuntimeError("PyMuPDF required for PDF export") from e
    pdf = fitz.open()
    # 72 dpi PDF user units == 1 px per unit; page size = pixel size keeps zoom crisp
    page = pdf.new_page(width=new_w, height=new_h)
    page.insert_image(fitz.Rect(0, 0, new_w, new_h), stream=jpg)
    out = pdf.tobytes()
    pdf.close()
    return out


def expand_zip_archive(zip_name: str, data: bytes) -> list[tuple[str, bytes]]:
    out: list[tuple[str, bytes]] = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            if "__MACOSX" in info.filename or info.filename.startswith("."):
                continue
            suf = Path(info.filename).suffix.lower()
            if suf not in SUPPORTED_SUFFIXES:
                continue
            out.append((info.filename.replace("\\", "/"), zf.read(info)))
    return out


def normalize_uploads(entries: list[tuple[str, bytes]]) -> list[tuple[str, bytes]]:
    """Expand ZIPs, filter extensions, dedupe by (name, size)."""
    merged: list[tuple[str, bytes]] = []
    seen: set[tuple[str, int]] = set()
    for name, data in entries:
        if name.lower().endswith(".zip"):
            for n2, d2 in expand_zip_archive(name, data):
                key = (n2, len(d2))
                if key not in seen:
                    seen.add(key)
                    merged.append((n2, d2))
            continue
        suf = Path(name).suffix.lower()
        if suf not in SUPPORTED_SUFFIXES:
            continue
        key = (name, len(data))
        if key in seen:
            continue
        seen.add(key)
        merged.append((name, data))
    return merged


@functools.lru_cache(maxsize=1)
def load_trained_rf():
    ensure_src_path()
    from blueprint_estimator.features import build_pair_features, load_materials_table
    from blueprint_estimator.train_trees import train_random_forest

    materials = load_materials_table(ROOT / "data" / "materials" / "materials_sample.csv")
    labels = pd.read_csv(ROOT / "data" / "labels" / "labels_sample.csv")
    X, y, _ = build_pair_features(materials, labels)
    cat = ["material_type_req", "material_type"]
    result = train_random_forest(X, y, cat)
    return result["model"], materials


def rank_materials_for_job(
    model,
    materials: pd.DataFrame,
    linear_ft: float,
    ceiling_ft: float,
    material_type_req: str,
) -> pd.DataFrame:
    rows = []
    for _, m in materials.iterrows():
        Xrow = pd.DataFrame(
            [
                {
                    "linear_ft_required": linear_ft,
                    "ceiling_height_ft": ceiling_ft,
                    "size_value": float(m.get("size_value", 0) or 0),
                    "unit_cost_usd": float(m.get("unit_cost_usd", 0) or 0),
                    "material_type_req": material_type_req,
                    "material_type": str(m.get("material_type", "unknown")),
                }
            ]
        )
        pr = model.predict_proba(Xrow)[0]
        p_match = float(pr[1]) if len(pr) > 1 else float(pr[0])
        unit_cost = float(m.get("unit_cost_usd", 0) or 0)
        unit = str(m.get("unit", "lf")).lower()
        if unit in ("lf", "linear_ft", "ft"):
            est = linear_ft * unit_cost
        elif unit in ("sqft", "sf"):
            est = linear_ft * ceiling_ft * unit_cost
        else:
            est = unit_cost
        rows.append(
            {
                "material_id": m["material_id"],
                "material_type": m["material_type"],
                "p_match": p_match,
                "unit": m.get("unit", ""),
                "unit_cost_usd": unit_cost,
                "rough_line_usd": round(est, 2),
            }
        )
    return pd.DataFrame(rows).sort_values("p_match", ascending=False).reset_index(drop=True)


def rank_materials_multi_trade(
    model,
    materials: pd.DataFrame,
    linear_ft: float,
    ceiling_ft: float,
) -> pd.DataFrame:
    """
    Score each catalog row with ``material_type_req`` = that row's ``material_type``,
    then keep the single best row per distinct ``material_type`` (multi-trade assembly).
    """
    rows = []
    for _, m in materials.iterrows():
        req = str(m.get("material_type", "unknown"))
        Xrow = pd.DataFrame(
            [
                {
                    "linear_ft_required": linear_ft,
                    "ceiling_height_ft": ceiling_ft,
                    "size_value": float(m.get("size_value", 0) or 0),
                    "unit_cost_usd": float(m.get("unit_cost_usd", 0) or 0),
                    "material_type_req": req,
                    "material_type": str(m.get("material_type", "unknown")),
                }
            ]
        )
        pr = model.predict_proba(Xrow)[0]
        p_match = float(pr[1]) if len(pr) > 1 else float(pr[0])
        unit_cost = float(m.get("unit_cost_usd", 0) or 0)
        unit = str(m.get("unit", "lf")).lower()
        if unit in ("lf", "linear_ft", "ft"):
            est = linear_ft * unit_cost
        elif unit in ("sqft", "sf"):
            est = linear_ft * ceiling_ft * unit_cost
        else:
            est = unit_cost
        rows.append(
            {
                "material_id": m["material_id"],
                "material_type": m["material_type"],
                "p_match": p_match,
                "unit": m.get("unit", ""),
                "unit_cost_usd": unit_cost,
                "rough_line_usd": round(est, 2),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (
        df.sort_values("p_match", ascending=False)
        .groupby("material_type", sort=False)
        .head(1)
        .sort_values("p_match", ascending=False)
        .reset_index(drop=True)
    )


def run_single_file(
    raw: bytes,
    filename: str,
    *,
    ceiling_ft: float,
    catalog: pd.DataFrame | None = None,
    dpi: int = DEFAULT_RASTER_DPI,
    user_roi: tuple[int, int, int, int] | None = None,
    skip_classifier_gate: bool = True,
) -> dict[str, Any]:
    """One blueprint file: ingest, infer scale, walls, quantities, material ranking."""
    ensure_src_path()
    from blueprint_estimator.ingest_raster import raster_ingest_bytes
    from blueprint_estimator.ingest_vector import vector_dxf_ingest_bytes, vector_json_ingest_bytes
    from blueprint_estimator.scale_inference import (
        infer_scale_raster_from_image_bgr,
        infer_scale_raster_from_pdf,
        infer_scale_vector_dxf_bytes,
        infer_scale_vector_json,
        scale_summary,
    )
    from blueprint_estimator.scale_qty import total_linear_feet_segments, wall_area_sheetboard_ft2
    from blueprint_estimator.wall_graph_cv import image_to_wall_segments, total_segment_length

    suffix = Path(filename).suffix.lower()
    size_mb = len(raw) / (1024 * 1024)
    if size_mb > MAX_PDF_MB:
        return {
            "ok": False,
            "filename": filename,
            "error": (
                f"File is {size_mb:.0f} MB. Hosted limit is {MAX_PDF_MB} MB to avoid OOM. "
                "Split the construction set into individual sheets (one floor plan per file) and re-upload."
            ),
        }
    try:
        if suffix == ".json":
            ingest = vector_json_ingest_bytes(raw, filename)
        elif suffix == ".dxf":
            ingest = vector_dxf_ingest_bytes(raw, filename)
        elif suffix in (".pdf", ".png", ".jpg", ".jpeg"):
            ingest = raster_ingest_bytes(raw, filename, dpi=int(dpi))
        else:
            return {"ok": False, "error": f"Unsupported type: {suffix}", "filename": filename}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e), "filename": filename}

    segments = list(ingest.segments_draw)
    graph = None
    preview_png = None
    overlay_png = None
    inf = None

    if ingest.image_bgr is not None:
        if suffix == ".pdf":
            inf = infer_scale_raster_from_pdf(raw, dpi=int(dpi))
        else:
            inf = infer_scale_raster_from_image_bgr(ingest.image_bgr, dpi=int(dpi))
        from blueprint_estimator.wall_detector import detect_walls
        from blueprint_estimator.vision_llm_filter import classify_page, crop_to_bbox
        from blueprint_estimator.sheet_classifier import should_run_takeoff
        from blueprint_estimator.building_isolator import isolate_building

        # Pre-flight is a HINT, not a gate. Default behavior runs detection on
        # every page; the user picks which pages count toward the bid in the UI.
        # Pass skip_classifier_gate=False to re-enable the old "auto-skip"
        # behavior (kept for testing).
        _bmask, _binfo = isolate_building(ingest.image_bgr)
        _bbox = _binfo.get("best_bbox")
        _bscore = float(_binfo.get("best_score", 0.0))
        _take_ok, _take_meta = should_run_takeoff(filename, ingest.image_bgr,
                                                    bbox=_bbox, score=_bscore,
                                                    candidates=_binfo.get("candidates"))
        # Apply user-supplied ROI crop BEFORE detection — this is the right
        # source of truth for "which region is the floor plan."
        if user_roi is not None:
            ux, uy, uw, uh = user_roi
            H_full, W_full = ingest.image_bgr.shape[:2]
            ux = max(0, min(W_full - 1, int(ux)))
            uy = max(0, min(H_full - 1, int(uy)))
            uw = max(1, min(W_full - ux, int(uw)))
            uh = max(1, min(H_full - uy, int(uh)))
            ingest_image_for_detect = ingest.image_bgr[uy : uy + uh, ux : ux + uw].copy()
        else:
            ingest_image_for_detect = ingest.image_bgr

        if (not skip_classifier_gate) and (not _take_ok) and user_roi is None:
            wall_info = {
                "raw_segments": 0, "wall_segments": 0, "mean_confidence": 0.0,
                "method": "skipped_non_floor_plan", "stucco_linear_ft": 0.0,
                "stucco_segments": 0, "sheet_classifier": _take_meta,
            }
            segments, graph = [], None
            linear_ft = 0.0
            stucco_linear_ft = 0.0
            preview_png = png_bytes_bgr(ingest.image_bgr)
            overlay_png = png_bytes_bgr(ingest.image_bgr)
            hires_overlay_png = b""
            hires_overlay_pdf = b""
            wall_info["building_isolator"] = _binfo
            return {
                "ok": True, "error": None, "filename": filename,
                "linear_ft": 0.0, "wall_area": 0.0,
                "graph_nodes": 0, "graph_edges": 0,
                "ranked_html": "", "top_material_id": None,
                "top_p_match": None, "top_rough_usd": None,
                "preview_png": preview_png, "overlay_png": overlay_png,
                "ingest_source": ingest.source, "segment_count": 0,
                "scale_method": "", "scale_confidence": 0.0,
                "scale_notes": _take_meta.get("skip_reason", "Not a floor plan"),
                "scale_summary": "⚠ Skipped — not a floor plan: " + _take_meta.get("skip_reason", "filename / visual check failed"),
                "wall_info": wall_info, "stucco_linear_ft": 0.0,
                "stucco_wall_yards_sq": 0.0,
                "hires_overlay_png": b"", "hires_overlay_pdf": b"",
            }

        decision = classify_page(ingest_image_for_detect)
        det_image = ingest_image_for_detect
        if decision.page_kind == "skip":
            # Vision LLM says this isn't a plan/elevation worth running takeoff on.
            wall_info = {
                "raw_segments": 0, "wall_segments": 0, "mean_confidence": 0.0,
                "method": "skipped_by_vision_llm", "page_kind": decision.page_kind,
                "stucco_linear_ft": 0.0, "stucco_segments": 0,
            }
            segments, graph = [], None
            linear_ft = 0.0
            stucco_linear_ft = 0.0
            overlay_bgr = ingest.image_bgr
            preview_png = png_bytes_bgr(ingest.image_bgr)
            overlay_png = png_bytes_bgr(ingest.image_bgr)
        else:
            if decision.bbox != (0, 0, ingest.image_bgr.shape[1], ingest.image_bgr.shape[0]):
                det_image = crop_to_bbox(ingest.image_bgr, decision.bbox)
            segments, graph, wall_info = detect_walls(
                det_image, use_text_mask=True, scale_config=inf.scale_config
            )
            wall_info["page_kind"] = decision.page_kind
            wall_info["building_bbox"] = list(decision.bbox)
        linear_ft = total_linear_feet_segments(segments, inf.scale_config)
        # If stucco classifier is loaded, also compute stucco-only linear feet
        stucco_segments = [s for s in segments if s.meta.get("stucco_probability", 0.0) >= 0.5]
        stucco_linear_ft = total_linear_feet_segments(stucco_segments, inf.scale_config) if stucco_segments else 0.0
        wall_info["stucco_linear_ft"] = float(stucco_linear_ft)
        wall_info["stucco_segments"] = len(stucco_segments)
        # Render preview/overlay against the SAME image we ran detection on
        # (the user crop if supplied, otherwise the full page).
        overlay_bgr = draw_segments_overlay(det_image, segments)
        preview_png = png_bytes_bgr(det_image)
        overlay_png = png_bytes_bgr(overlay_bgr)
        hires_overlay_png = high_res_overlay_png(det_image, segments, target_long_side=1920)
        try:
            hires_overlay_pdf = overlay_pdf_bytes(det_image, segments, target_long_side=2400)
        except Exception:
            hires_overlay_pdf = b""
        wall_info["classifier_hint"] = {
            "suggested_run": _take_ok,
            "reason": _take_meta.get("skip_reason") or "looks like a floor plan",
            "iso_score": _bscore,
        }
        wall_info["user_roi"] = list(user_roi) if user_roi is not None else None
    else:
        if suffix == ".dxf":
            inf = infer_scale_vector_dxf_bytes(raw, filename)
        else:
            inf = infer_scale_vector_json()
        vf = inf.vector_feet_per_unit or (1.0 / 12.0)
        linear_ft = total_segment_length(segments) * float(vf)
        wall_info = {"raw_segments": len(segments), "wall_segments": len(segments), "mean_confidence": 1.0, "text_masked": False}
        hires_overlay_png = b""
        hires_overlay_pdf = b""

    wall_area = wall_area_sheetboard_ft2(linear_ft, float(ceiling_ft))
    smry = scale_summary(inf, int(dpi)) if inf else ""

    if catalog is not None and not catalog.empty:
        model, _ = load_trained_rf()
        ranked = rank_materials_multi_trade(
            model, catalog, linear_ft, float(ceiling_ft)
        )
        if ranked.empty:
            ranked_html = ""
            top_material_id = None
            top_p_match = None
            top_rough_usd = None
        else:
            top = ranked.iloc[0]
            ranked_html = ranked.to_html(
                classes="data", index=False, float_format=lambda x: f"{x:.4f}"
            )
            top_material_id = str(top["material_id"])
            top_p_match = float(top["p_match"])
            top_rough_usd = float(top["rough_line_usd"])
    else:
        ranked_html = ""
        top_material_id = None
        top_p_match = None
        top_rough_usd = None

    return {
        "ok": True,
        "error": None,
        "filename": filename,
        "linear_ft": linear_ft,
        "wall_area": wall_area,
        "graph_nodes": len(graph.nodes) if graph else None,
        "graph_edges": len(graph.edges) if graph else None,
        "ranked_html": ranked_html,
        "top_material_id": top_material_id,
        "top_p_match": top_p_match,
        "top_rough_usd": top_rough_usd,
        "preview_png": preview_png,
        "overlay_png": overlay_png,
        "ingest_source": ingest.source,
        "segment_count": len(segments),
        "scale_method": inf.method if inf else "",
        "scale_confidence": inf.confidence if inf else 0.0,
        "scale_notes": inf.notes if inf else "",
        "scale_summary": smry,
        "wall_info": wall_info,
        "stucco_linear_ft": wall_info.get("stucco_linear_ft", 0.0),
        "stucco_wall_yards_sq": (wall_info.get("stucco_linear_ft", 0.0) * float(ceiling_ft) / 9.0),
        "hires_overlay_png": hires_overlay_png,
        "hires_overlay_pdf": hires_overlay_pdf,
    }


def load_materials_from_upload(
    data: bytes | None,
    filename: str | None,
    *,
    default_material_type: str = "misc",
) -> tuple[pd.DataFrame | None, str | None]:
    """
    Parse materials catalog from upload bytes. Returns (materials_df, error_message).
    If data is None, returns (None, None).

    For DOCX/PDF without tables, each line becomes a row; ``default_material_type`` is used as ``material_type``.
    """
    if not data:
        return None, None
    name = (filename or "materials.csv").strip() or "materials.csv"
    ensure_src_path()
    from blueprint_estimator.materials_ingest import parse_catalog_bytes

    materials, err = parse_catalog_bytes(
        name, data, default_material_type=default_material_type
    )
    if err:
        return None, err
    return materials, None


def build_estimation_excel_bytes(project: dict[str, Any]) -> bytes:
    """Two-sheet workbook: Summary + Estimates (one row per material_type when multi-trade)."""
    buf = io.BytesIO()
    recs = project.get("ranked_records") or []
    trade_ids = ", ".join(str(r.get("material_id", "")) for r in recs) if recs else ""
    summary = pd.DataFrame(
        [
            {"field": "ranking_mode", "value": project.get("ranking_mode", "multi_trade")},
            {"field": "trades_represented", "value": len(recs)},
            {"field": "material_ids_by_trade", "value": trade_ids},
            {"field": "total_linear_ft", "value": project.get("total_linear_ft", "")},
            {"field": "total_wall_area_sqft", "value": project.get("total_wall_area", "")},
            {"field": "drawing_count", "value": project.get("file_count", "")},
            {"field": "top_material_id_highest_p", "value": project.get("top_material_id", "")},
            {"field": "top_p_match", "value": project.get("top_p_match", "")},
            {"field": "top_rough_usd", "value": project.get("top_rough_usd", "")},
        ]
    )
    est = pd.DataFrame(project.get("ranked_records") or [])
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        if not est.empty:
            est.to_excel(writer, sheet_name="Estimates", index=False)
        else:
            pd.DataFrame({"message": ["No estimates"]}).to_excel(writer, sheet_name="Estimates", index=False)
    return buf.getvalue()


def run_project(
    files: list[tuple[str, bytes]],
    *,
    ceiling_ft: float,
    materials_bytes: bytes | None = None,
    materials_filename: str | None = None,
    materials_csv_bytes: bytes | None = None,
    dpi: int = DEFAULT_RASTER_DPI,
) -> dict[str, Any]:
    """
    Process many files (paths may include subfolders). Returns per-file results and project roll-up.

    Material ranking runs only when a catalog is uploaded (``materials_bytes`` / ``materials_csv_bytes``).
    Uses multi-trade ranking: one recommended SKU per distinct ``material_type`` in the catalog.

    For backward compatibility, ``materials_csv_bytes`` is still accepted and treated as CSV.
    """
    entries = normalize_uploads(files)
    if not entries:
        return {"ok": False, "error": "No supported files after expanding ZIPs.", "files": [], "project": None}

    raw_mat = materials_bytes if materials_bytes is not None else materials_csv_bytes
    mat_name = materials_filename
    if raw_mat is not None and mat_name is None and materials_csv_bytes is not None:
        mat_name = "materials.csv"
    catalog: pd.DataFrame | None = None
    if raw_mat:
        materials, mat_err = load_materials_from_upload(
            raw_mat, mat_name, default_material_type=DEFAULT_LINE_MATERIAL_TYPE
        )
        if mat_err:
            return {"ok": False, "error": mat_err, "files": [], "project": None}
        catalog = materials

    file_rows: list[dict[str, Any]] = []
    total_linear = 0.0
    total_area = 0.0

    for name, raw in entries:
        one = run_single_file(
            raw,
            name,
            ceiling_ft=ceiling_ft,
            catalog=catalog,
            dpi=dpi,
        )
        file_rows.append(one)
        if one.get("ok"):
            total_linear += float(one["linear_ft"])
            total_area += float(one["wall_area"])

    if catalog is None or catalog.empty:
        return {
            "ok": True,
            "error": None,
            "files": file_rows,
            "project": {
                "total_linear_ft": total_linear,
                "total_wall_area": total_area,
                "file_count": len(entries),
                "materials_enabled": False,
                "ranking_mode": "none",
                "ranked_html": "",
                "ranked_records": [],
                "top_material_id": None,
                "top_p_match": None,
                "top_rough_usd": None,
            },
        }

    model, _ = load_trained_rf()
    ranked_proj = rank_materials_multi_trade(
        model, catalog, total_linear, float(ceiling_ft)
    )
    if ranked_proj.empty:
        top_material_id = None
        top_p_match = None
        top_rough_usd = None
    else:
        top = ranked_proj.iloc[0]
        top_material_id = str(top["material_id"])
        top_p_match = float(top["p_match"])
        top_rough_usd = float(top["rough_line_usd"])
    ranked_records = ranked_proj.to_dict(orient="records")

    return {
        "ok": True,
        "error": None,
        "files": file_rows,
        "project": {
            "total_linear_ft": total_linear,
            "total_wall_area": total_area,
            "file_count": len(entries),
            "materials_enabled": True,
            "ranking_mode": "one_row_per_material_type",
            "ranked_html": ranked_proj.to_html(
                classes="data", index=False, float_format=lambda x: f"{x:.4f}"
            ),
            "ranked_records": ranked_records,
            "top_material_id": top_material_id,
            "top_p_match": top_p_match,
            "top_rough_usd": top_rough_usd,
        },
    }
