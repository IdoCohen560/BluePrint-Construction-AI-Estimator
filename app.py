"""
Streamlit UI: multi-file + ZIP uploads, per-file automatic scale (see web_core + scale_inference).

    python -m streamlit run app.py

For native **folder** (all files in a tree), use `python web_flask.py` — browsers expose folder contents only there reliably.
"""

from __future__ import annotations

import io
from pathlib import Path

import streamlit as st

import web_core

ROOT = Path(__file__).resolve().parent


def main() -> None:
    st.set_page_config(page_title="Blueprint Material Estimator", layout="wide")
    st.title("Blueprint wall takeoff and material assistant")
    st.caption(
        "**Step 1:** Upload your materials catalog. **Step 2:** Add drawings or ZIPs. "
        "Scale is **inferred per file**. For **folder upload**, use the Flask app: `python web_flask.py`."
    )

    st.subheader("Step 1 — Materials catalog")
    st.caption(
        "Choose a **CSV**, **Word (.docx)**, or **PDF** first. "
        "Use a **table** with a **type** column for multiple trades; line-only lists use one trade (`misc`) until you add a priced table."
    )
    materials_file = st.file_uploader(
        "Materials catalog (required before drawings)",
        type=["csv", "docx", "pdf"],
        label_visibility="visible",
        help="Spreadsheet-style table (CSV / Word table / PDF table), or a DOCX/PDF with one material per line.",
    )

    if not materials_file:
        st.info(
            "Upload a **materials catalog** above. After a file is selected, **drawings** and **ZIP** uploaders will appear."
        )
        return

    st.subheader("Step 2 — Drawings and archives")
    st.caption("Add at least one blueprint file or a ZIP of drawings.")

    drawings = st.file_uploader(
        "Drawings (multi-select)",
        type=["pdf", "png", "jpg", "jpeg", "json", "dxf"],
        accept_multiple_files=True,
        help="Select many files at once (e.g. Ctrl+A in a folder dialog where supported).",
    )
    zips = st.file_uploader(
        "ZIP archives (optional)",
        type=["zip"],
        accept_multiple_files=True,
        help="Each ZIP is expanded; supported extensions inside are processed.",
    )

    with st.sidebar:
        st.header("Project options")
        ceiling_ft = st.number_input("Ceiling height (ft)", min_value=6.0, value=8.0)
        st.divider()
        st.markdown(
            "**Folder upload:** use [Flask UI](http://127.0.0.1:8765) (`python web_flask.py`) "
            "with *Pick a folder*."
        )

    run = st.button(
        "Run analysis on all files",
        type="primary",
        disabled=(not drawings and not zips),
    )

    if not drawings and not zips:
        st.info("Upload at least one drawing file or ZIP.")
        return

    files: list[tuple[str, bytes]] = []
    if drawings:
        for f in drawings:
            files.append((f.name, f.getvalue()))
    if zips:
        for z in zips:
            files.append((z.name, z.getvalue()))

    mat_bytes = materials_file.getvalue() if materials_file else None
    mat_name = materials_file.name if materials_file else None

    if not run:
        st.warning("Click **Run analysis on all files** to process.")
        st.caption(f"Queued: **{len(files)}** upload item(s) (ZIPs expand to multiple drawings).")
        return

    with st.spinner("Processing…"):
        res = web_core.run_project(
            files,
            ceiling_ft=float(ceiling_ft),
            materials_bytes=mat_bytes,
            materials_filename=mat_name,
            dpi=web_core.DEFAULT_RASTER_DPI,
        )

    if not res.get("ok"):
        st.error(res.get("error", "Unknown error"))
        return

    if materials_file is not None:
        st.info(
            "Using your uploaded catalog with multi-trade ranking: **one suggested SKU per material type** "
            "(walls are treated as using several trades)."
        )

    st.subheader("Per-file results")
    rows = []
    for fr in res["files"]:
        if not fr.get("ok"):
            rows.append(
                {
                    "file": fr.get("filename", "?"),
                    "status": "error",
                    "detail": fr.get("error", ""),
                    "linear_ft": None,
                    "wall_area": None,
                    "scale": "",
                    "conf": None,
                }
            )
            continue
        rows.append(
            {
                "file": fr["filename"],
                "status": "ok",
                "detail": "",
                "linear_ft": round(fr["linear_ft"], 2),
                "wall_area": round(fr["wall_area"], 1),
                "scale": fr.get("scale_summary", ""),
                "conf": round(fr.get("scale_confidence", 0), 2),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    import base64
    import streamlit.components.v1 as components

    for fr in res["files"]:
        if not fr.get("ok"):
            continue
        with st.expander(fr["filename"]):
            st.write(fr.get("scale_notes", ""))
            c1, c2 = st.columns(2)
            if fr.get("preview_png"):
                c1.image(io.BytesIO(fr["preview_png"]), caption="Input", use_container_width=True)
            if fr.get("overlay_png"):
                c2.image(io.BytesIO(fr["overlay_png"]), caption="Walls detected (preview)", use_container_width=True)

            # High-res overlay + downloads + zoom viewer
            if fr.get("source_image_bytes") and fr.get("segments_payload") is not None:
                # Lazy: regenerate hi-res from the cached source image
                pass

            hires_png = fr.get("hires_overlay_png")
            hires_pdf = fr.get("hires_overlay_pdf")
            if hires_png:
                st.markdown("**Zoomable wall overlay (drag / pinch / scroll)**")
                b64 = base64.b64encode(hires_png).decode("ascii")
                # OpenSeadragon-free panzoom via inline HTML/JS — no extra deps
                components.html(
                    f"""
                    <div id='wrap' style='position:relative;width:100%;height:600px;border:1px solid #ddd;
                        overflow:hidden;background:#fafafa;cursor:grab;touch-action:none;'>
                      <img id='img' src='data:image/png;base64,{b64}' draggable='false'
                           style='position:absolute;left:0;top:0;transform-origin:0 0;user-select:none;
                                  max-width:none;max-height:none;'/>
                    </div>
                    <div style='font-size:12px;color:#666;margin-top:4px;'>
                      Scroll = zoom · Drag = pan · Double-click = reset
                    </div>
                    <script>
                    (function() {{
                      const wrap = document.getElementById('wrap');
                      const img = document.getElementById('img');
                      let scale = 1, ox = 0, oy = 0, dragging = false, sx = 0, sy = 0;
                      function fit() {{
                        const r = wrap.getBoundingClientRect();
                        scale = Math.min(r.width / img.naturalWidth, r.height / img.naturalHeight);
                        ox = (r.width - img.naturalWidth * scale) / 2;
                        oy = (r.height - img.naturalHeight * scale) / 2;
                        apply();
                      }}
                      function apply() {{
                        img.style.transform = `translate(${{ox}}px,${{oy}}px) scale(${{scale}})`;
                      }}
                      img.onload = fit;
                      if (img.complete) fit();
                      wrap.addEventListener('wheel', e => {{
                        e.preventDefault();
                        const r = wrap.getBoundingClientRect();
                        const cx = e.clientX - r.left, cy = e.clientY - r.top;
                        const f = e.deltaY < 0 ? 1.15 : 1/1.15;
                        const ns = Math.max(0.1, Math.min(20, scale * f));
                        ox = cx - (cx - ox) * (ns / scale);
                        oy = cy - (cy - oy) * (ns / scale);
                        scale = ns; apply();
                      }}, {{ passive: false }});
                      wrap.addEventListener('mousedown', e => {{
                        dragging = true; sx = e.clientX - ox; sy = e.clientY - oy;
                        wrap.style.cursor = 'grabbing';
                      }});
                      window.addEventListener('mousemove', e => {{
                        if (!dragging) return;
                        ox = e.clientX - sx; oy = e.clientY - sy; apply();
                      }});
                      window.addEventListener('mouseup', () => {{
                        dragging = false; wrap.style.cursor = 'grab';
                      }});
                      wrap.addEventListener('dblclick', fit);
                    }})();
                    </script>
                    """,
                    height=650,
                )

                d1, d2 = st.columns(2)
                d1.download_button(
                    "⬇ Download high-res PNG (1920px)",
                    data=hires_png,
                    file_name=f"{fr['filename'].rsplit('.',1)[0]}_walls.png",
                    mime="image/png",
                )
                if hires_pdf:
                    d2.download_button(
                        "⬇ Download PDF (Acrobat)",
                        data=hires_pdf,
                        file_name=f"{fr['filename'].rsplit('.',1)[0]}_walls.pdf",
                        mime="application/pdf",
                    )

            if fr.get("ranked_html"):
                st.markdown("**Suggested materials (this file, one per trade)**", unsafe_allow_html=True)
                st.markdown(fr["ranked_html"], unsafe_allow_html=True)
            else:
                st.caption("No per-file ranking rows for this sheet.")

    proj = res["project"]
    st.subheader("Project totals")
    st.metric("Total linear feet", f"{proj['total_linear_ft']:,.2f} ft")
    st.metric("Total rough wall area (one side)", f"{proj['total_wall_area']:,.1f} sq ft")

    # Stucco-only takeoff (from trained classifier)
    stucco_lf = sum(fr.get("stucco_linear_ft", 0.0) for fr in res["files"] if fr.get("ok"))
    stucco_yds = stucco_lf * float(ceiling_ft) / 9.0
    st.subheader("Stucco bid takeoff (AI)")
    sc1, sc2 = st.columns(2)
    sc1.metric("Stucco linear feet", f"{stucco_lf:,.1f} ft")
    sc2.metric("Stucco wall yards (sq yd)", f"{stucco_yds:,.1f} sy")
    project_name = st.text_input("Project name for bid", value="My Project")
    if st.button("Generate populated bid spreadsheet"):
        try:
            from blueprint_estimator.bid_writer import fill_bid_template
            import tempfile, os
            tmp_out = os.path.join(tempfile.gettempdir(), "bid_filled.xlsx")
            out_path = fill_bid_template(project_name, wall_yards=stucco_yds, ceiling_yards=0.0,
                                         out_path=tmp_out)
            with open(out_path, "rb") as f:
                st.download_button(
                    label=f"Download bid for {project_name}",
                    data=f.read(),
                    file_name=f"bid_{project_name.replace(' ', '_')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        except FileNotFoundError as e:
            st.error(f"Bid template missing: {e}")

    st.subheader("Project material estimation")
    if proj.get("materials_enabled"):
        st.dataframe(
            proj.get("ranked_records") or [],
            use_container_width=True,
            hide_index=True,
        )
        xlsx_bytes = web_core.build_estimation_excel_bytes(proj)
        st.download_button(
            label="Download estimation as Excel",
            data=xlsx_bytes,
            file_name="material_estimation.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        n_trades = len(proj.get("ranked_records") or [])
        top_id = proj.get("top_material_id")
        if top_id is not None and proj.get("top_p_match") is not None:
            st.success(
                f"**{n_trades} trade(s)** in assembly suggestion. "
                f"Highest-confidence pick: **{top_id}** "
                f"(P≈{proj['top_p_match']:.2f}, rough ≈ ${proj['top_rough_usd']:.2f})."
            )
        else:
            st.success(f"Catalog loaded; **{n_trades}** trade row(s) in the table above.")
    else:
        st.warning("Material ranking was disabled for this run (unexpected). Re-upload your catalog and run again.")



if __name__ == "__main__":
    main()
