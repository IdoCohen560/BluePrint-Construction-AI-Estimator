from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Segment:
    """A wall segment in image or drawing coordinates (start -> end)."""

    x1: float
    y1: float
    x2: float
    y2: float
    meta: dict[str, Any] = field(default_factory=dict)

    def length_px(self) -> float:
        import math

        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)


@dataclass
class WallGraph:
    """Planar graph: nodes are snapped point keys; edges store segment length in same units as coordinates."""

    nodes: set[tuple[float, float]] = field(default_factory=set)
    edges: list[tuple[tuple[float, float], tuple[float, float], dict[str, Any]]] = field(
        default_factory=list
    )

    def total_edge_length(self) -> float:
        total = 0.0
        for _a, _b, m in self.edges:
            total += float(m.get("length", 0.0))
        return total


@dataclass
class IngestResult:
    """Unified output from any blueprint ingestion path."""

    source: str
    image_bgr: Any | None  # numpy array or None for pure vector
    segments_draw: list[Segment]
    meta: dict[str, Any] = field(default_factory=dict)
