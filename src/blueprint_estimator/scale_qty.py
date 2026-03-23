"""Scale (drawing units to real feet) and quantity aggregation."""

from __future__ import annotations

from dataclasses import dataclass

from blueprint_estimator.schemas import Segment, WallGraph


@dataclass(frozen=True)
class ScaleConfig:
    """
    How to convert pixel lengths to real feet.

    - `feet_per_pixel`: direct calibration (e.g. from a known reference length on the image).
    - `drawing_feet_per_drawing_inch` + `dpi`: architectural scale, e.g. 1/4\" = 1'-0\"
      means one inch on the sheet equals 4 feet in the field => use 4.0.
    """

    feet_per_pixel: float | None = None
    dpi: float | None = None
    drawing_feet_per_drawing_inch: float | None = None

    def resolved_feet_per_pixel(self) -> float:
        if self.feet_per_pixel is not None and self.feet_per_pixel > 0:
            return self.feet_per_pixel
        if self.dpi and self.drawing_feet_per_drawing_inch:
            # 1 drawing inch = dpi pixels; 1 drawing inch = drawing_feet_per_drawing_inch real feet
            return float(self.drawing_feet_per_drawing_inch) / float(self.dpi)
        raise ValueError(
            "Set feet_per_pixel, or both dpi and drawing_feet_per_drawing_inch (e.g. 4 for 1/4\"=1')."
        )


def segment_length_feet(seg: Segment, scale: ScaleConfig) -> float:
    fpp = scale.resolved_feet_per_pixel()
    return seg.length_px() * fpp


def total_linear_feet_segments(segments: list[Segment], scale: ScaleConfig) -> float:
    fpp = scale.resolved_feet_per_pixel()
    return sum(s.length_px() * fpp for s in segments)


def total_linear_feet_graph(graph: WallGraph, scale: ScaleConfig) -> float:
    fpp = scale.resolved_feet_per_pixel()
    total = 0.0
    for _a, _b, m in graph.edges:
        total += float(m.get("length", 0.0)) * fpp
    return total


def wall_area_sheetboard_ft2(linear_feet: float, ceiling_height_ft: float = 8.0) -> float:
    """Naive one-sided wall area for drywall-style takeoff."""
    return max(0.0, linear_feet * ceiling_height_ft)
