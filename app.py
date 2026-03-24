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

    for fr in res["files"]:
        if not fr.get("ok"):
            continue
        with st.expander(fr["filename"]):
            st.write(fr.get("scale_notes", ""))
            c1, c2 = st.columns(2)
            if fr.get("preview_png"):
                c1.image(io.BytesIO(fr["preview_png"]), caption="Input", use_container_width=True)
            if fr.get("overlay_png"):
                c2.image(io.BytesIO(fr["overlay_png"]), caption="Walls detected", use_container_width=True)
            if fr.get("ranked_html"):
                st.markdown("**Suggested materials (this file, one per trade)**", unsafe_allow_html=True)
                st.markdown(fr["ranked_html"], unsafe_allow_html=True)
            else:
                st.caption("No per-file ranking rows for this sheet.")

    proj = res["project"]
    st.subheader("Project totals")
    st.metric("Total linear feet", f"{proj['total_linear_ft']:,.2f} ft")
    st.metric("Total rough wall area (one side)", f"{proj['total_wall_area']:,.1f} sq ft")

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
