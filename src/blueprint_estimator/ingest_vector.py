from __future__ import annotations

import json
from pathlib import Path

from blueprint_estimator.schemas import IngestResult, Segment


def segments_from_json(path: str | Path) -> list[Segment]:
    """
    Load segments from a simple JSON file:
    {"segments": [{"x1":0,"y1":0,"x2":10,"y2":0}, ...]}
    Coordinates are drawing units (same as CAD export scale).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: list[Segment] = []
    for row in data.get("segments", []):
        out.append(
            Segment(
                float(row["x1"]),
                float(row["y1"]),
                float(row["x2"]),
                float(row["y2"]),
                meta={"source": "vector_json"},
            )
        )
    return out


def vector_json_ingest(path: str | Path) -> IngestResult:
    segs = segments_from_json(path)
    return IngestResult(
        source="vector_json",
        image_bgr=None,
        segments_draw=segs,
        meta={"path": str(Path(path).resolve())},
    )


def dxf_lines_to_segments(path: str | Path, layer_filter: set[str] | None = None) -> list[Segment]:
    """
    Minimal DXF LINE / LWPOLYLINE ingestion via ezdxf.
    layer_filter: if set, only include entities on these layer names (lower case compared).
    """
    import ezdxf

    doc = ezdxf.readfile(path)
    msp = doc.modelspace()
    segs: list[Segment] = []
    lf = {x.lower() for x in layer_filter} if layer_filter else None

    for e in msp:
        if lf is not None and e.dxf.layer.lower() not in lf:
            continue
        dxftype = e.dxftype()
        if dxftype == "LINE":
            s = e.dxf.start
            t = e.dxf.end
            segs.append(
                Segment(
                    float(s.x),
                    float(s.y),
                    float(t.x),
                    float(t.y),
                    meta={"source": "dxf", "layer": e.dxf.layer},
                )
            )
        elif dxftype == "LWPOLYLINE":
            pts = list(e.get_points("xy"))
            for i in range(len(pts) - 1):
                x0, y0 = float(pts[i][0]), float(pts[i][1])
                x1, y1 = float(pts[i + 1][0]), float(pts[i + 1][1])
                segs.append(
                    Segment(x0, y0, x1, y1, meta={"source": "dxf", "layer": e.dxf.layer})
                )
    return segs


def vector_dxf_ingest(path: str | Path, layer_filter: set[str] | None = None) -> IngestResult:
    segs = dxf_lines_to_segments(path, layer_filter=layer_filter)
    return IngestResult(
        source="vector_dxf",
        image_bgr=None,
        segments_draw=segs,
        meta={"path": str(Path(path).resolve())},
    )
