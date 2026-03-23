"""Deterministic cost roll-up from matched materials and quantities."""

from __future__ import annotations

import pandas as pd


def estimate_line_cost(
    materials: pd.DataFrame,
    material_id: str,
    quantity: float,
    unit_column: str = "unit",
    cost_column: str = "unit_cost_usd",
    size_column: str = "size_value",
) -> float:
    row = materials[materials["material_id"] == material_id]
    if row.empty:
        return 0.0
    r = row.iloc[0]
    unit = str(r.get(unit_column, "each")).lower()
    cost = float(r.get(cost_column, 0.0))
    if unit in ("lf", "linear_ft", "ft"):
        return quantity * cost
    if unit in ("sqft", "sf"):
        return quantity * cost
    if unit in ("each", "ea", "sheet"):
        return max(1.0, quantity) * cost
    return quantity * cost


def rollup_estimate(
    materials: pd.DataFrame,
    selections: list[tuple[str, float, str]],
) -> tuple[float, pd.DataFrame]:
    """
    selections: list of (material_id, numeric_quantity, quantity_kind)
    Returns total USD and a detail table.
    """
    rows = []
    total = 0.0
    for mid, qty, kind in selections:
        row = materials[materials["material_id"] == mid]
        if row.empty:
            continue
        r = row.iloc[0]
        unit_cost = float(r.get("unit_cost_usd", 0.0))
        line = qty * unit_cost if kind != "count" else max(1, qty) * unit_cost
        total += line
        rows.append(
            {
                "material_id": mid,
                "material_type": r.get("material_type"),
                "quantity": qty,
                "kind": kind,
                "unit_cost_usd": unit_cost,
                "line_total_usd": line,
            }
        )
    return total, pd.DataFrame(rows)
