"""Infer drawing scale per file from PDF text, OCR on title blocks, or vector defaults."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from blueprint_estimator.scale_qty import ScaleConfig

if TYPE_CHECKING:
    import numpy as np

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None  # type: ignore

# Default rasterization DPI (must match ingest used for wall CV)
DEFAULT_RASTER_DPI = 150

# When nothing matches: 1/4" = 1' style (4 ft per drawing inch) at DEFAULT_RASTER_DPI
DEFAULT_DRAWING_FEET_PER_INCH = 4.0


@dataclass(frozen=True)
class ScaleInferenceResult:
    scale_config: ScaleConfig
    confidence: float
    method: str
    notes: str
    # For vector JSON/DXF: multiply geometric segment length (drawing units) by this to get feet.
    vector_feet_per_unit: float | None = None


def _feet_per_drawing_inch_from_arch_fraction(num: int, den: int, feet_right: float) -> float:
    """e.g. 1/4\" on left (inches on paper), 1' on right -> feet per one drawing inch."""
    if den <= 0 or num <= 0:
        return DEFAULT_DRAWING_FEET_PER_INCH
    paper_inches = num / float(den)
    if paper_inches <= 0:
        return DEFAULT_DRAWING_FEET_PER_INCH
    return float(feet_right) / paper_inches


def _parse_text_for_scale(text: str) -> tuple[float | None, str]:
    """Return (drawing_feet_per_drawing_inch, matched_note) or (None, '')."""
    t = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    t = re.sub(r"\s+", " ", t)

    # 1/4" = 1' or 1/4" = 1'-0"
    m = re.search(
        r"(\d+)\s*/\s*(\d+)\s*[\"']?\s*=\s*(\d+)\s*[\u2019']?\s*(?:-\s*0\s*[\"']?)?",
        t,
        re.IGNORECASE,
    )
    if m:
        num, den, ft = int(m.group(1)), int(m.group(2)), float(m.group(3))
        fpi = _feet_per_drawing_inch_from_arch_fraction(num, den, ft)
        return fpi, f'arch fraction {num}/{den}" = {ft}ft'

    # SCALE 1 : 48  (common US: 1:48 => 1/4"=1')
    m = re.search(r"SCALE\s*[:\s]*1\s*:\s*(\d+)", t, re.IGNORECASE)
    if m:
        r = int(m.group(1))
        if r == 48:
            return 4.0, "SCALE 1:48 (assumed 1/4\"=1')"
        if r > 0:
            # crude: 1:96 -> 1/8"=1' => 8 ft/in
            fpi = r / 12.0
            return float(fpi), f"SCALE 1:{r} (ft per drawing inch ~{fpi:.2f})"

    # 1" = 20' style
    m = re.search(r'1\s*["\u201d]\s*=\s*(\d+)\s*[\u2019\']', t, re.IGNORECASE)
    if m:
        ft = float(m.group(1))
        return ft, f'1" = {ft}ft'

    return None, ""


def _pdf_text_first_pages(data: bytes, max_pages: int = 2) -> str:
    if fitz is None:
        return ""
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        parts = []
        for i in range(min(max_pages, len(doc))):
            parts.append(doc.load_page(i).get_text("text"))
        return "\n".join(parts)
    finally:
        doc.close()


def _raster_pdf_page0_bgr(data: bytes, dpi: int) -> "np.ndarray | None":
    if fitz is None:
        return None
    from blueprint_estimator.ingest_raster import pdf_bytes_to_bgr_first_page

    try:
        return pdf_bytes_to_bgr_first_page(data, dpi=dpi)
    except Exception:  # noqa: BLE001
        return None


def _ocr_title_bands(bgr: "np.ndarray") -> str:
    try:
        import cv2
        import numpy as np
        import pytesseract
    except ImportError:
        return ""

    h, w = bgr.shape[:2]
    bands = []
    for y0, y1 in ((0, int(h * 0.12)), (int(h * 0.82), h)):
        roi = bgr[y0:y1, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        thr = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 10
        )
        try:
            txt = pytesseract.image_to_string(thr, config="--psm 6")
        except Exception:  # noqa: BLE001
            txt = ""
        bands.append(txt)
    return "\n".join(bands)


def infer_scale_raster_from_pdf(
    pdf_bytes: bytes,
    dpi: int = DEFAULT_RASTER_DPI,
) -> ScaleInferenceResult:
    """Extract scale from PDF text; fall back to OCR on title bands; then default."""
    text = _pdf_text_first_pages(pdf_bytes)
    fpi, note = _parse_text_for_scale(text)
    if fpi is not None and fpi > 0:
        return ScaleInferenceResult(
            scale_config=ScaleConfig(
                dpi=float(dpi),
                drawing_feet_per_drawing_inch=float(fpi),
            ),
            confidence=0.85,
            method="pdf_text",
            notes=note or "regex on PDF text",
        )

    bgr = _raster_pdf_page0_bgr(pdf_bytes, dpi=dpi)
    if bgr is not None:
        ocr_txt = _ocr_title_bands(bgr)
        fpi2, note2 = _parse_text_for_scale(ocr_txt)
        if fpi2 is not None and fpi2 > 0:
            return ScaleInferenceResult(
                scale_config=ScaleConfig(
                    dpi=float(dpi),
                    drawing_feet_per_drawing_inch=float(fpi2),
                ),
                confidence=0.55,
                method="ocr",
                notes=note2 or "OCR title band",
            )

    return ScaleInferenceResult(
        scale_config=ScaleConfig(
            dpi=float(dpi),
            drawing_feet_per_drawing_inch=DEFAULT_DRAWING_FEET_PER_INCH,
        ),
        confidence=0.25,
        method="default",
        notes="No scale in PDF text or OCR; using 1/4\"=1' equivalent (4 ft per drawing inch)",
    )


def infer_scale_raster_from_image_bgr(
    bgr: "np.ndarray",
    dpi: int = DEFAULT_RASTER_DPI,
) -> ScaleInferenceResult:
    """PNG/JPG: OCR only, then default."""
    ocr_txt = _ocr_title_bands(bgr)
    fpi, note = _parse_text_for_scale(ocr_txt)
    if fpi is not None and fpi > 0:
        return ScaleInferenceResult(
            scale_config=ScaleConfig(
                dpi=float(dpi),
                drawing_feet_per_drawing_inch=float(fpi),
            ),
            confidence=0.5,
            method="ocr",
            notes=note or "OCR on image",
        )
    return ScaleInferenceResult(
        scale_config=ScaleConfig(
            dpi=float(dpi),
            drawing_feet_per_drawing_inch=DEFAULT_DRAWING_FEET_PER_INCH,
        ),
        confidence=0.25,
        method="default",
        notes="No scale in OCR; using default 4 ft per drawing inch at stated DPI",
    )


def infer_scale_vector_dxf_bytes(data: bytes, filename: str) -> ScaleInferenceResult:
    """DXF: map common INSUNITS to feet per drawing unit (length in file units -> feet)."""
    import tempfile
    from pathlib import Path

    import ezdxf

    suffix = Path(filename).suffix or ".dxf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        doc = ezdxf.readfile(tmp_path)
        units_code = int(doc.header.get("$INSUNITS", 6))
        # Common DXF $INSUNITS: 4=mm, 5=cm, 6=in, 8=ft (see AutoCAD docs)
        if units_code == 8:
            feet_per_du = 1.0
            note = "DXF INSUNITS=feet"
        elif units_code == 6:
            feet_per_du = 1.0 / 12.0
            note = "DXF INSUNITS=inches (1 drawing unit = 1/12 ft)"
        elif units_code == 4:
            feet_per_du = 0.00328084
            note = "DXF INSUNITS=mm"
        elif units_code == 5:
            feet_per_du = 0.0328084
            note = "DXF INSUNITS=cm"
        else:
            feet_per_du = 1.0 / 12.0
            note = f"DXF INSUNITS={units_code}; fallback treat as inches"

        return ScaleInferenceResult(
            scale_config=ScaleConfig(feet_per_pixel=1.0),
            confidence=0.6,
            method="vector_units",
            notes=note,
            vector_feet_per_unit=feet_per_du,
        )
    except Exception as e:  # noqa: BLE001
        return ScaleInferenceResult(
            scale_config=ScaleConfig(feet_per_pixel=1.0),
            confidence=0.3,
            method="default",
            notes=f"DXF header read failed ({e}); assuming inch-like units",
            vector_feet_per_unit=1.0 / 12.0,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def infer_scale_vector_json() -> ScaleInferenceResult:
    """JSON segments: assume coordinates in inches (common for exports)."""
    return ScaleInferenceResult(
        scale_config=ScaleConfig(feet_per_pixel=1.0),
        confidence=0.35,
        method="vector_default",
        notes="JSON: assume drawing units are inches (1 unit = 1/12 ft); adjust in source if needed",
        vector_feet_per_unit=1.0 / 12.0,
    )


def scale_summary(result: ScaleInferenceResult, dpi: int) -> str:
    if result.vector_feet_per_unit is not None:
        return f"{result.method}: {result.vector_feet_per_unit:.6f} ft/drawing-unit — {result.notes[:100]}"
    try:
        fpp = result.scale_config.resolved_feet_per_pixel()
        return f"{result.method}: {fpp:.6f} ft/px @ {dpi} DPI — {result.notes[:80]}"
    except Exception:  # noqa: BLE001
        return f"{result.method}: {result.notes}"
