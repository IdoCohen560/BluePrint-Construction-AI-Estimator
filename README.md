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

## Web app (browser — multi-file uploads, per-file scale)

The server **only reads files you upload** (bytes in the request). It does **not** browse folders on disk.

### Streamlit (multi-select + ZIP)

From the repository root:

```bash
pip install streamlit
python -m streamlit run app.py
```

Open this URL in your browser (use **127.0.0.1**, not `localhost`, if you see connection errors):

**http://127.0.0.1:8501**

The repo includes [`.streamlit/config.toml`](.streamlit/config.toml) so the server binds to `127.0.0.1` on port `8501`.

- **Multi-select:** choose many drawings at once (PDF, PNG, JPG, JSON, DXF).
- **ZIP:** upload one or more `.zip` files; supported extensions inside are expanded and processed (same-name duplicates are skipped).
- **Scale:** inferred **per file** from PDF text (regex), OCR on title-block crops for scans, or documented defaults (see confidence in the results table). There is **no manual scale** sidebar.
- **Sidebar:** ceiling height, material category, optional materials CSV (`material_id`, `material_type`, `size_value`, `unit_cost_usd`).

**Folder upload (whole directory tree):** `st.file_uploader` does not expose HTML5 `webkitdirectory`. Use the **Flask** app below and use **Pick a folder** (Chromium / Edge / Safari send each file with a relative path like `MyProject/sheets/A1.pdf`). Alternatively, zip the folder and upload the ZIP in Streamlit.

### Flask (folder + multi + ZIP)

```bash
pip install flask
python web_flask.py
```

Open **http://127.0.0.1:8765**. Use **Pick a folder**, **multiple files**, and/or **ZIP**; processing matches Streamlit’s batch API (`web_core.run_project`).

### Tesseract OCR (optional, for scale on scanned drawings)

`pytesseract` is listed in `requirements.txt`. On **Windows**, install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) and ensure `tesseract.exe` is on your **PATH** (or set `TESSDATA_PREFIX` if needed). If Tesseract is missing, the app **skips** the OCR branch and falls back to PDF text or defaults, with lower confidence shown in the UI.

**If the page does not load (browser error -102 / connection refused):**

1. Leave the terminal **open** — closing it stops the server. You must see Streamlit’s “You can now view your Streamlit app” message before opening the URL.
2. Double-click **`START_STREAMLIT.cmd`** in the project folder (keeps a window open and starts Streamlit).
3. Try **http://127.0.0.1:8501** and **http://localhost:8501** after the server says it is running.
4. **Flask fallback (often works when Streamlit cannot connect):**
   ```bash
   pip install flask
   python web_flask.py
   ```
   Or double-click **`START_FLASK.cmd`**, then open **http://127.0.0.1:8765**.
5. If port 8501 is busy: `python -m streamlit run app.py --server.port 8502` → open **http://127.0.0.1:8502**.
6. Pause VPN / corporate tools briefly; some block loopback.
7. Allow **Python** through Windows Firewall if prompted.

## Run without Jupyter (quick check)

From the repository root:

```bash
python run_demo.py
```

This prints ingest, wall detection, scale, tree-model metrics, and a toy cost rollup in the terminal (no browser).

## Run notebooks

From the repository root (after activating the virtual environment):

```bash
pip install notebook
jupyter notebook notebooks
```

Copy the **`http://127.0.0.1:8888/tree?token=...`** URL from the terminal into your browser (the full URL including `token=` is required). If the page does not load, check that nothing else is using port 8888, or try:

```bash
jupyter notebook notebooks --port=8889
```

Run in order:

1. `01_ingest_synthetic.ipynb` — synthetic floor plan + optional raster/vector paths  
2. `02_wall_graph.ipynb` — Hough line detection, wall graph, optional RF patch model  
3. `03_scale_quantities.ipynb` — feet per pixel / architectural scale, linear feet  
4. `04_materials_trees.ipynb` — Decision Tree vs Random Forest, metrics, cost rollup  
5. `05_future_algorithms.ipynb` — placeholder for KNN / neural net comparison (Phase 2)

**Notebook policy:** notebooks are committed **without heavy execution outputs** so diffs stay small; re-run locally for full results.

## Assumptions

- Default **ceiling height** for rough wall area (e.g. drywall) is **8 ft** unless overridden in the web sidebar or data.  
- **Scale** in the web apps is **inferred per file** (PDF text / OCR / vector units / defaults). Notebooks and `run_demo.py` may still set `ScaleConfig` explicitly for teaching.  
- Material **unit costs** come from the CSV; the models predict **match**, not price.

## Data

See [`data/README.md`](data/README.md) and [`data/labels/LABELING_GUIDELINES.md`](data/labels/LABELING_GUIDELINES.md). Add your own blueprints under `data/blueprints/` if desired; large proprietary files are often kept local (see `.gitignore`).

## License

MIT — see [`LICENSE`](LICENSE).
