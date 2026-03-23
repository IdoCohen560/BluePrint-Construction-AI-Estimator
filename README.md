# BluePrint Construction AI Estimator

**CSUN — Introduction to Machine Learning (Fall 2026)**  
Supervised learning pipeline: extract wall geometry from blueprint images (classical CV + optional tree-based patch classifier), convert lengths to real units using scale, then train **Decision Tree** and **Random Forest** models on tabular material data for match/no-match prediction and deterministic cost roll-up.

**Repository:** [https://github.com/IdoCohen560/BluePrint-Construction-AI-Estimator](https://github.com/IdoCohen560/BluePrint-Construction-AI-Estimator)

## Setup

Requires **Python 3.10+**.

```bash
git clone https://github.com/IdoCohen560/BluePrint-Construction-AI-Estimator.git
cd BluePrint-Construction-AI-Estimator
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -e .
pip install -r requirements.txt
```

## Run notebooks

From the repository root (after activating the virtual environment):

```bash
jupyter notebook notebooks
```

Run in order:

1. `01_ingest_synthetic.ipynb` — synthetic floor plan + optional raster/vector paths  
2. `02_wall_graph.ipynb` — Hough line detection, wall graph, optional RF patch model  
3. `03_scale_quantities.ipynb` — feet per pixel / architectural scale, linear feet  
4. `04_materials_trees.ipynb` — Decision Tree vs Random Forest, metrics, cost rollup  
5. `05_future_algorithms.ipynb` — placeholder for KNN / neural net comparison (Phase 2)

**Notebook policy:** notebooks are committed **without heavy execution outputs** so diffs stay small; re-run locally for full results.

## Assumptions

- Default **ceiling height** for rough wall area (e.g. drywall) is **8 ft** unless overridden in data or code.  
- **Scale** is supplied as `feet_per_pixel` or as `dpi` + `drawing_feet_per_drawing_inch` (e.g. `4.0` for common `1/4" = 1'-0"`).  
- Material **unit costs** come from the CSV; the models predict **match**, not price.

## Data

See [`data/README.md`](data/README.md) and [`data/labels/LABELING_GUIDELINES.md`](data/labels/LABELING_GUIDELINES.md). Add your own blueprints under `data/blueprints/` if desired; large proprietary files are often kept local (see `.gitignore`).

## License

MIT — see [`LICENSE`](LICENSE).
