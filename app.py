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
    st.caption(
        "Wall detection runs on every uploaded sheet. Use the **'Include in bid'** "
        "checkbox to pick which pages count. The classifier hint (✓ likely floor plan / "
        "⚠ may not be a floor plan) is just a suggestion — you choose."
    )
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

    # Collect per-file include flags so totals reflect user picks.
    if "include_in_bid" not in st.session_state:
        st.session_state["include_in_bid"] = {}

    for fr in res["files"]:
        if not fr.get("ok"):
            continue
        wall_info = fr.get("wall_info", {})
        method = wall_info.get("method", "")
        was_skipped = method == "skipped_non_floor_plan"

        if was_skipped:
            with st.expander(f"⏭️ {fr['filename']}  —  skipped (not a floor plan)", expanded=False):
                st.warning(fr.get("scale_summary", "Skipped — not a floor plan"))
                st.caption(
                    "If this is actually a floor plan and you want it processed, "
                    "re-upload it after renaming so the filename contains "
                    "'FLOOR PLAN' (e.g. `MyBuilding_FloorPlan.pdf`). "
                    "The skip rules use filename keywords like ELEVATION, RCP, "
                    "SCHEDULE, ACCESSIB, FLOOR AREA, SLAB, etc."
                )
                st.session_state["include_in_bid"][fr["filename"]] = False
            continue

        hint = wall_info.get("classifier_hint", {})
        suggested = hint.get("suggested_run", True)
        hint_label = ("✓ likely floor plan" if suggested
                       else f"⚠ may not be a floor plan: {hint.get('reason','')}")
        default = st.session_state["include_in_bid"].get(fr["filename"], suggested)
        with st.expander(f"{fr['filename']}  —  {hint_label}", expanded=suggested):
            include = st.checkbox(
                "✅ Include this page in bid total",
                value=default,
                key=f"inc_{fr['filename']}",
                help="The classifier suggestion is just a hint. You decide.",
            )
            st.session_state["include_in_bid"][fr["filename"]] = include
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
                st.caption(
                    "Open the downloaded PNG or PDF in your viewer to zoom in — "
                    "browsers handle that better than the in-page widget."
                )

            if fr.get("ranked_html"):
                st.markdown("**Suggested materials (this file, one per trade)**", unsafe_allow_html=True)
                st.markdown(fr["ranked_html"], unsafe_allow_html=True)
            else:
                st.caption("No per-file ranking rows for this sheet.")

    proj = res["project"]
    st.subheader("Project totals")
    # Only sum files the user has marked as 'include in bid'
    included = {n for n, v in st.session_state.get("include_in_bid", {}).items() if v}
    incl_files = [fr for fr in res["files"] if fr.get("ok") and fr["filename"] in included]
    incl_lf = sum(fr.get("linear_ft", 0.0) for fr in incl_files)
    incl_wa = sum(fr.get("wall_area", 0.0) for fr in incl_files)
    st.metric(f"Total linear feet ({len(incl_files)} sheet(s) included)", f"{incl_lf:,.2f} ft")
    st.metric("Total rough wall area (one side)", f"{incl_wa:,.1f} sq ft")

    # Stucco-only takeoff (from trained classifier), respecting user picks
    stucco_lf = sum(fr.get("stucco_linear_ft", 0.0) for fr in incl_files)
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
