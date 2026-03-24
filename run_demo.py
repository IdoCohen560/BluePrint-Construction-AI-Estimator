#!/usr/bin/env python3
"""
Run the full pipeline in the terminal (no Jupyter browser).
Usage (from repo root):  python run_demo.py
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "src"))

    import pandas as pd
    from blueprint_estimator.cost import rollup_estimate
    from blueprint_estimator.features import build_pair_features, load_materials_table
    from blueprint_estimator.scale_qty import ScaleConfig, total_linear_feet_segments, wall_area_sheetboard_ft2
    from blueprint_estimator.synthetic import synthetic_ingest, total_gt_length_px
    from blueprint_estimator.train_trees import train_decision_tree, train_random_forest
    from blueprint_estimator.wall_graph_cv import image_to_wall_segments, total_segment_length
    from blueprint_estimator.wall_ml import build_synthetic_patch_dataset, train_wall_patch_rf

    print("=== 1. Synthetic ingest ===")
    syn = synthetic_ingest()
    print(f"  source={syn.source}  image shape={syn.image_bgr.shape}")

    print("\n=== 2. Wall graph (Hough) + RF patches ===")
    segs, graph = image_to_wall_segments(syn.image_bgr, merge=True)
    gt_px = total_gt_length_px(syn.segments_draw)
    det_px = total_segment_length(segs)
    print(f"  GT total length (px): {gt_px:.1f}")
    print(f"  Detected segments: {len(segs)}  sum length (px): {det_px:.1f}")
    Xp, yp = build_synthetic_patch_dataset(syn.image_bgr, syn.segments_draw)
    clf = train_wall_patch_rf(Xp, yp)
    print(f"  Wall-patch RF trained on {len(Xp)} samples (accuracy on train set: {clf.score(Xp, yp):.3f})")

    print("\n=== 3. Scale / quantities ===")
    feet_per_px = 4.0 / 150.0
    scale = ScaleConfig(feet_per_pixel=feet_per_px)
    lf = total_linear_feet_segments(syn.segments_draw, scale)
    print(f"  feet_per_pixel (example): {feet_per_px}")
    print(f"  Total linear feet (GT segments): {lf:.2f}")
    print(f"  Rough wall area @ 8' ceiling: {wall_area_sheetboard_ft2(lf, 8.0):.1f} sq ft")

    print("\n=== 4. Materials: Decision Tree vs Random Forest ===")
    materials = load_materials_table(ROOT / "data" / "materials" / "materials_sample.csv")
    labels = pd.read_csv(ROOT / "data" / "labels" / "labels_sample.csv")
    X, y, _joined = build_pair_features(materials, labels)
    cat = ["material_type_req", "material_type"]
    dt = train_decision_tree(X, y, cat)
    rf = train_random_forest(X, y, cat)
    print(f"  Decision Tree  F1={dt['f1']:.3f}  best={dt['best_params']}")
    print(f"  Random Forest  F1={rf['f1']:.3f}  best={rf['best_params']}")

    print("\n=== 5. Toy cost rollup ===")
    total, detail = rollup_estimate(
        materials,
        [(materials.iloc[0]["material_id"], lf, "lf"), (materials.iloc[3]["material_id"], 20, "count")],
    )
    print(f"  Example total USD: {total:.2f}")
    print(detail.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
