"""Engineered patch features + Random Forest for wall vs background (tabular sklearn)."""

from __future__ import annotations

import random
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from blueprint_estimator.schemas import Segment


def patch_features(gray: np.ndarray, cx: int, cy: int, half: int = 16) -> np.ndarray:
    """Scalar features from a square patch: mean, std, edge density, gradient energy."""
    h, w = gray.shape[:2]
    x0, x1 = max(0, cx - half), min(w, cx + half)
    y0, y1 = max(0, cy - half), min(h, cy + half)
    patch = gray[y0:y1, x0:x1].astype(np.float64)
    if patch.size == 0:
        return np.zeros(8, dtype=np.float64)
    gx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    edges = cv2.Canny(patch.astype(np.uint8), 50, 150)
    return np.array(
        [
            float(patch.mean()),
            float(patch.std()),
            float(mag.mean()),
            float(mag.std()),
            float(edges.mean()),
            float((patch < 200).mean()),  # dark ink fraction
            float(patch.shape[0] * patch.shape[1]),
            float(np.percentile(patch, 90) - np.percentile(patch, 10)),
        ],
        dtype=np.float64,
    )


def build_synthetic_patch_dataset(
    image_bgr: np.ndarray,
    positive_segments: list[Segment],
    n_neg: int = 200,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Positive samples at segment midpoints; negative at random interior points."""
    rng = random.Random(seed)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    xs: list[np.ndarray] = []
    ys: list[int] = []

    for s in positive_segments:
        mx = int((s.x1 + s.x2) / 2)
        my = int((s.y1 + s.y2) / 2)
        xs.append(patch_features(gray, mx, my))
        ys.append(1)

    for _ in range(n_neg):
        cx, cy = rng.randint(20, w - 20), rng.randint(20, h - 20)
        xs.append(patch_features(gray, cx, cy))
        ys.append(0)

    return np.stack(xs, axis=0), np.array(ys, dtype=np.int64)


def train_wall_patch_rf(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 100,
    random_state: int = 42,
) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=12,
        min_samples_leaf=2,
        random_state=random_state,
        class_weight="balanced",
    )
    clf.fit(X, y)
    return clf


def score_segments_for_wall(
    clf: RandomForestClassifier,
    image_bgr: np.ndarray,
    segments: list[Segment],
) -> list[float]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    out: list[float] = []
    for s in segments:
        mx = int((s.x1 + s.x2) / 2)
        my = int((s.y1 + s.y2) / 2)
        feat = patch_features(gray, mx, my).reshape(1, -1)
        pr = clf.predict_proba(feat)[0]
        proba = float(pr[1]) if len(pr) > 1 else float(pr[0])
        out.append(float(proba))
    return out
