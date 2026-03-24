"""
Flask UI: folder upload (webkitdirectory), multi-file, ZIP — same batch API as Streamlit.

    python web_flask.py

Open: http://127.0.0.1:8765
"""

from __future__ import annotations

import time
import uuid
from html import escape

from flask import Flask, Response, render_template_string, request

import web_core

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 80 * 1024 * 1024  # 80 MB

# token -> (created_unix, xlsx bytes); popped on successful download
_EXPORT_CACHE: dict[str, tuple[float, bytes]] = {}
_EXPORT_TTL_SEC = 3600


def _ranked_records_html(records: list[dict]) -> str:
    if not records:
        return "<p class='hint'>No estimation rows.</p>"
    keys = list(records[0].keys())
    head = "".join(f"<th>{escape(str(k))}</th>" for k in keys)
    body_parts: list[str] = []
    for row in records:
        cells = []
        for k in keys:
            v = row.get(k, "")
            if isinstance(v, bool):
                cells.append(f"<td>{escape(str(v))}</td>")
            elif isinstance(v, (int, float)):
                cells.append(f"<td>{v:.4f}</td>" if isinstance(v, float) else f"<td>{v}</td>")
            else:
                cells.append(f"<td>{escape(str(v))}</td>")
        body_parts.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table class='data'><tr>{head}</tr>{''.join(body_parts)}</table>"

PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Blueprint Material Estimator</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 1200px; margin: 1rem auto; padding: 0 1rem; }
    label { display: block; margin-top: 0.75rem; font-weight: 600; }
    input, select { margin-top: 0.25rem; max-width: 100%; }
    table.data { border-collapse: collapse; width: 100%; margin-top: 1rem; font-size: 0.88rem; }
    table.data th, table.data td { border: 1px solid #ccc; padding: 0.35rem 0.5rem; text-align: left; }
    table.data th { background: #f0f0f0; }
    .err { color: #b00020; background: #ffeaea; padding: 1rem; border-radius: 6px; }
    .ok { background: #e8f5e9; padding: 1rem; border-radius: 6px; margin-top: 1rem; }
    img { max-width: 100%; height: auto; border: 1px solid #ddd; }
    button { margin-top: 1rem; padding: 0.5rem 1.2rem; font-size: 1rem; cursor: pointer; }
    .hint { font-size: 0.9rem; color: #444; margin-top: 0.25rem; }
    h2 { margin-top: 1.5rem; }
  </style>
</head>
<body>
  <h1>Blueprint wall takeoff and material assistant</h1>
  <p>Scale is <strong>inferred per file</strong> (PDF text, OCR on scans, or defaults). Only uploaded file bytes are read.</p>

  {% if error %}
  <div class="err"><strong>Error:</strong> {{ error }}</div>
  {% endif %}

  <form method="post" action="/analyze" enctype="multipart/form-data">
    <h2>1) Pick a folder (all files in the tree)</h2>
    <p class="hint">Chromium / Edge / Safari: choose a folder; every file is sent like <code>MyProject/sheets/A1.pdf</code>.</p>
    <input type="file" name="folder_files" multiple webkitdirectory />

    <h2>2) Or pick multiple files (no folder)</h2>
    <input type="file" name="flat_files" multiple accept=".pdf,.png,.jpg,.jpeg,.json,.dxf"/>

    <h2>3) Or add ZIP archive(s)</h2>
    <input type="file" name="zips" multiple accept=".zip"/>

    <label>Ceiling height (ft)</label>
    <input type="number" name="ceiling_ft" value="8" step="0.5" min="6"/>

    <label>Materials catalog (CSV, DOCX, or PDF) — required for material suggestions</label>
    <input type="file" name="materials" accept=".csv,.docx,.pdf"/>

    <div><button type="submit">Run analysis on all</button></div>
  </form>

  {% if result and result.ok %}
  <hr/>
  <h2>Per-file</h2>
  {{ result.per_file_table|safe }}

  <h2>Project totals</h2>
  <p>Total linear feet: <strong>{{ result.proj_linear | round(2) }}</strong> &nbsp;|&nbsp;
     Wall area (rough): <strong>{{ result.proj_area | round(1) }} sq ft</strong></p>
  {% if result.materials_enabled %}
  <h3>Project material estimation (one SKU per trade)</h3>
  {{ result.project_ranked_table|safe }}
  <p><a href="/download_estimation/{{ result.download_token }}">Download estimation as Excel</a></p>
  <div class="ok">
    {% if result.proj_top_id %}
    Highest-confidence pick: <strong>{{ result.proj_top_id }}</strong> — P≈{{ result.proj_top_p | round(3) }},
    rough ≈ ${{ result.proj_top_usd | round(2) }}
    {% else %}
    No ranked rows (check catalog).
    {% endif %}
  </div>
  {% else %}
  <p class="hint">Upload a materials catalog on the form to see suggested SKUs per trade and Excel download.</p>
  {% endif %}
  {% endif %}
</body>
</html>
"""


def _collect_uploads() -> list[tuple[str, bytes]]:
    items: list[tuple[str, bytes]] = []
    for key in ("folder_files", "flat_files"):
        for f in request.files.getlist(key):
            if f and f.filename:
                items.append((f.filename.replace("\\", "/"), f.read()))
    for z in request.files.getlist("zips"):
        if z and z.filename:
            items.append((z.filename.replace("\\", "/"), z.read()))
    return items


@app.route("/")
def index():
    return render_template_string(PAGE, error=None, result=None)


@app.route("/analyze", methods=["POST"])
def analyze():
    error = None
    result = None
    try:
        items = _collect_uploads()
        if not items:
            error = "Add at least one file: folder pick, multi-file, or ZIP."
        else:
            ceiling_ft = float(request.form.get("ceiling_ft") or 8)
            mf = request.files.get("materials")
            mat_bytes = mf.read() if mf and mf.filename else None
            mat_name = mf.filename if mf and mf.filename else None

            res = web_core.run_project(
                items,
                ceiling_ft=ceiling_ft,
                materials_bytes=mat_bytes,
                materials_filename=mat_name,
                dpi=web_core.DEFAULT_RASTER_DPI,
            )
            if not res.get("ok"):
                error = res.get("error", "Failed")
            else:
                rows = [
                    "<table class='data'><tr><th>File</th><th>Status</th><th>Linear ft</th><th>Wall area</th>"
                    "<th>Scale confidence</th><th>Scale</th></tr>"
                ]
                for fr in res["files"]:
                    if not fr.get("ok"):
                        rows.append(
                            "<tr><td>{}</td><td>error</td><td colspan='4'>{}</td></tr>".format(
                                escape(str(fr.get("filename", ""))),
                                escape(str(fr.get("error", ""))),
                            )
                        )
                        continue
                    rows.append(
                        "<tr><td>{}</td><td>ok</td><td>{:.2f}</td><td>{:.1f}</td><td>{:.2f}</td><td>{}</td></tr>".format(
                            escape(str(fr["filename"])),
                            fr["linear_ft"],
                            fr["wall_area"],
                            fr.get("scale_confidence", 0),
                            escape((fr.get("scale_summary", "") or "")[:120]),
                        )
                    )
                rows.append("</table>")
                proj = res["project"]
                materials_enabled = bool(proj.get("materials_enabled"))
                ranked = proj.get("ranked_records") or []
                dl_token = None
                if materials_enabled:
                    dl_token = str(uuid.uuid4())
                    xlsx = web_core.build_estimation_excel_bytes(proj)
                    _EXPORT_CACHE[dl_token] = (time.time(), xlsx)
                result = {
                    "ok": True,
                    "per_file_table": "\n".join(rows),
                    "proj_linear": proj["total_linear_ft"],
                    "proj_area": proj["total_wall_area"],
                    "materials_enabled": materials_enabled,
                    "project_ranked_table": _ranked_records_html(ranked),
                    "download_token": dl_token,
                    "proj_top_id": proj.get("top_material_id"),
                    "proj_top_p": proj.get("top_p_match"),
                    "proj_top_usd": proj.get("top_rough_usd"),
                }
    except Exception as e:  # noqa: BLE001
        error = str(e)

    return render_template_string(PAGE, error=error, result=result)


@app.route("/download_estimation/<token>")
def download_estimation(token: str):
    item = _EXPORT_CACHE.pop(token, None)
    if not item:
        return Response("Invalid or expired link.", status=404, mimetype="text/plain")
    ts, data = item
    if time.time() - ts > _EXPORT_TTL_SEC:
        return Response("Download link expired.", status=404, mimetype="text/plain")
    return Response(
        data,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=material_estimation.xlsx"},
    )


def main() -> None:
    app.run(host="0.0.0.0", port=8765, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
