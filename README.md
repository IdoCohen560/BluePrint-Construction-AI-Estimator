<div align="center">

# BluePrint Construction AI Estimator

**Turn a floor-plan drawing into a material and cost estimate** — extract wall geometry from blueprint images with classical computer vision, convert pixel lengths to real feet using the drawing's scale, then predict material match and roll up cost with tree-based ML.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)

</div>

---

A supervised-learning pipeline that estimates construction materials and cost directly from blueprint drawings. It ingests raster (PNG/JPG/PDF) or vector (DXF) plans, detects walls with Hough-line CV (plus an optional tree-based patch classifier), infers the drawing scale per file — from PDF text, OCR on title-block crops, or documented defaults — converts wall runs to linear feet, then trains **Decision Tree** and **Random Forest** models on tabular material data to predict match/no-match and produce a deterministic cost roll-up.

Built for **CSUN — Introduction to Machine Learning (Fall 2026)** by Ido Cohen & Sannia Jean. The project is organized both as a teaching notebook series (five ordered notebooks) and as two runnable web apps for batch drawing uploads.

## How it works

The pipeline runs in five stages, mirrored by the notebooks in [`notebooks/`](notebooks/):

1. **Ingest** — load a synthetic or real floor plan (raster or vector) → `01_ingest_synthetic.ipynb`
2. **Wall graph** — Hough line detection builds a wall graph; optional Random Forest patch model → `02_wall_graph.ipynb`
3. **Scale & quantities** — feet-per-pixel / architectural scale, derive linear feet → `03_scale_quantities.ipynb`
4. **Materials & trees** — Decision Tree vs Random Forest, metrics, cost roll-up → `04_materials_trees.ipynb`
5. **Future work** — KNN / neural-net comparison placeholder (Phase 2) → `05_future_algorithms.ipynb`

Scale is **inferred per file** (PDF text regex → OCR on title-block crops → vector units → documented defaults), with a confidence level shown in the results. The models predict material **match**, not price; unit costs come from a materials CSV and the roll-up is deterministic.

## Setup

Requires **Python 3.10+**.

```bash
git clone https://github.com/IdoCohen560/BluePrint-Construction-AI-Estimator.git
cd BluePrint-Construction-AI-Estimator
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
pip install -r requirements.txt
```

## Quick check (no browser)

```bash
python run_demo.py
```

Prints ingest, wall detection, scale, tree-model metrics, and a toy cost roll-up in the terminal.

## Web apps

Both apps only read the files you upload (bytes in the request) — they do **not** browse folders on disk.

**Streamlit** — multi-select + ZIP upload:

```bash
pip install streamlit
python -m streamlit run app.py
```

Open **http://127.0.0.1:8501** (use `127.0.0.1`, not `localhost`, if you hit connection errors). The bundled [`.streamlit/config.toml`](.streamlit/config.toml) binds the server to `127.0.0.1:8501`. Choose many drawings at once (PDF, PNG, JPG, JSON, DXF) or upload `.zip` archives; set ceiling height, material category, and an optional materials CSV in the sidebar.

**Flask** — folder + multi-file + ZIP upload:

```bash
pip install flask
python web_flask.py
```

Open **http://127.0.0.1:8765** and use **Pick a folder** (Chromium / Edge / Safari send each file with its relative path) — useful because Streamlit's uploader cannot take a whole directory. Both apps share the same batch API (`web_core.run_project`). On Windows, `START_STREAMLIT.cmd` / `START_FLASK.cmd` keep a terminal window open for you.

### Optional: Tesseract OCR

OCR reads scale from title blocks on scanned drawings. Install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and put it on your `PATH`. If it's missing, the app skips the OCR branch and falls back to PDF text or defaults, with lower confidence shown in the UI.

## Notebooks

```bash
pip install notebook
jupyter notebook notebooks
```

Run them in order (1 → 5). Notebooks are committed **without heavy execution outputs** so diffs stay small — re-run locally for full results.

## Assumptions

- Default ceiling height for rough wall area (e.g. drywall) is **8 ft** unless overridden.
- Scale in the web apps is inferred per file; notebooks and `run_demo.py` may set `ScaleConfig` explicitly for teaching.
- Material unit costs come from the CSV; the models predict match, not price.

## Tech stack

| Layer | Tools |
|-------|-------|
| ML | scikit-learn (Decision Tree, Random Forest) |
| Computer vision | OpenCV (Hough lines), shapely |
| Ingest | PyMuPDF, pypdf, pdfplumber (PDF), ezdxf (DXF), pytesseract (OCR) |
| Web | Streamlit, Flask |
| Data / notebooks | NumPy, pandas, matplotlib, seaborn, Jupyter |

## Project structure

```
BluePrint-Construction-AI-Estimator/
├── src/blueprint_estimator/   # Ingest, wall detection, scale, trees, cost
├── notebooks/                 # 01–05 ordered ML pipeline
├── app.py                     # Streamlit web app
├── web_flask.py / web_core.py # Flask app + shared batch API
├── run_demo.py                # Terminal end-to-end demo
├── scripts/                   # Training, calibration, validation
├── data/                      # Blueprints, labels, materials CSV
└── pyproject.toml
```

## License

MIT — see [`LICENSE`](LICENSE).
