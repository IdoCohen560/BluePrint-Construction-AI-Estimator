"""Feature engineering for material match classification."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _one_hot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_materials_table(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def build_pair_features(
    materials: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Join labels with material rows; target column `match` (0/1).
    Returns (X, y, joined_df) where joined_df includes pair_id for diagnostics.
    """
    df = labels.merge(materials, on="material_id", how="left")
    y = df["match"].astype(int)
    feature_cols = [
        "linear_ft_required",
        "ceiling_height_ft",
        "size_value",
        "unit_cost_usd",
        "material_type_req",
        "material_type",
    ]
    X = df[feature_cols].copy()
    X["size_value"] = X["size_value"].fillna(0.0)
    X["unit_cost_usd"] = X["unit_cost_usd"].fillna(0.0)
    X["material_type"] = X["material_type"].fillna("unknown")
    X["material_type_req"] = X["material_type_req"].fillna("unknown")
    return X, y, df


def make_preprocessor(categorical: list[str]) -> ColumnTransformer:
    numeric = [
        "linear_ft_required",
        "ceiling_height_ft",
        "size_value",
        "unit_cost_usd",
    ]
    transformers = [
        ("num", StandardScaler(), numeric),
        ("cat", _one_hot(), categorical),
    ]
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
