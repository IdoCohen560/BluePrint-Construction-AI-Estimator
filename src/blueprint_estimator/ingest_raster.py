from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    fitz = None  # type: ignore

from blueprint_estimator.schemas import IngestResult, Segment


def load_image_bgr(path: str | Path) -> np.ndarray:
    p = Path(path)
    data = np.frombuffer(p.read_bytes(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image: {p}")
    return img


def pdf_to_bgr_first_page(path: str | Path, dpi: int = 150) -> np.ndarray:
    """Rasterize first PDF page to BGR using PyMuPDF."""
    if fitz is None:
        raise RuntimeError("PyMuPDF (pymupdf) is required for PDF ingestion.")
    doc = fitz.open(path)
    try:
        page = doc.load_page(0)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        # pixmap is RGB
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return bgr
    finally:
        doc.close()


def raster_ingest(path: str | Path, dpi: int = 150) -> IngestResult:
    """
    Load a raster image (.png/.jpg) or PDF (first page).
    Segments are not filled here — run wall_graph_cv on image_bgr.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        img = pdf_to_bgr_first_page(p, dpi=dpi)
        source = "raster_pdf"
    else:
        img = load_image_bgr(p)
        source = "raster_image"
    return IngestResult(
        source=source,
        image_bgr=img,
        segments_draw=[],
        meta={"path": str(p.resolve()), "dpi": dpi},
    )
