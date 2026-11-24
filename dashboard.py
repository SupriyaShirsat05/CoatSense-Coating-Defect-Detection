"""
CoatSense dashboard.   Run with:  streamlit run dashboard.py
"""
import glob
import os
import tempfile

import streamlit as st

import pipeline
from pipeline import ROOT, analyze_detections, draw_findings, load_detector, rust_overlay
from service_life import apply_thickness_override, remaining_life
from audit_record import (clear_all_inspections, get_all_inspections,
                          get_inspections_by_date_range, save_inspection)

st.set_page_config(page_title="CoatSense", layout="wide")
st.title("CoatSense - Coating Defect Inspection")
st.caption("AI-assisted detection, severity estimation and ISO 4628 grading for corrosion, cracking, flaking and blistering")


@st.cache_resource
def get_detector(model_path):
    return load_detector(model_path)


model_files = sorted(glob.glob(os.path.join(ROOT, "models", "*.pt")))

tab_inspect, tab_history = st.tabs(["New Inspection", "Inspection History"])

with tab_inspect:
    if not model_files:
        st.error("No model found. Copy your trained weights (best.pt) into the 'models' folder.")
        st.stop()

    names = [os.path.basename(f) for f in model_files]
    default_idx = names.index("best_safenet_120ep.pt") if "best_safenet_120ep.pt" in names else 0

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("1. Upload an image")
        uploaded = st.file_uploader("Coating surface photo", type=["jpg", "jpeg", "png"])
        st.subheader("2. Settings")
        model_name = st.selectbox("Model", names, index=default_idx)
        conf = st.slider("Detection confidence threshold", 0.05, 0.9, 0.15, 0.05)
        min_sat = st.slider("Rust colour strictness (min saturation)", 20, 150, 50, 5,
                            help="Raise this if pale beige/cream surfaces are being counted as rust.")
        show_mask = st.checkbox("Show which pixels are counted as rust")
        asset_id = st.text_input("Asset ID", value="Unspecified")
        location_tag = st.text_input("Location tag", value="Unspecified")
        with st.expander("Optional: thickness reading (enables a remaining-life estimate)"):
            t_orig = st.number_input("Original thickness (mm)", min_value=0.0, value=0.0, step=0.5)
            t_meas = st.number_input("Measured thickness now (mm)", min_value=0.0, value=0.0, step=0.1)
            t_min = st.number_input("Minimum required thickness (mm)", min_value=0.0, value=0.0, step=0.1)
            age = st.number_input("Years in service", min_value=0.0, value=0.0, step=0.5)
        run = st.button("Run Inspection", type="primary", disabled=uploaded is None)

    with col_right:
        st.subheader("Result")
        if uploaded is not None and not run:
            st.image(uploaded, caption="Uploaded image", use_container_width=True)

        if run and uploaded is not None:
            with st.spinner("Detecting and grading..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name
                try:
                    detector = get_detector(os.path.join(ROOT, "models", model_name))
                    result = analyze_detections(tmp_path, detector.detect(tmp_path, conf), min_sat)
                    annotated = draw_findings(tmp_path, result["findings"])
                    mask_img = rust_overlay(tmp_path, result["findings"], min_sat) if show_mask else None
                finally:
                    os.unlink(tmp_path)
                result["image"] = uploaded.name
                life = None
                if min(t_orig, t_meas, t_min, age) > 0:
                    life = remaining_life(t_orig, t_meas, t_min, age)
                    result["reuse"] = apply_thickness_override(result["reuse"], life)
                saved = save_inspection(result, asset_id=asset_id, location_tag=location_tag, life=life)

            st.image(annotated, caption="Detections", use_container_width=True)
            if mask_img is not None:
                st.image(mask_img, caption="Cyan = pixels counted as rust", use_container_width=True)
            st.subheader(f"Overall: {result['overall_grade']}")
            st.write(f"Recommended action: **{result['overall_condition']}**")

            reuse = result["reuse"]
            st.subheader(f"Reuse assessment: {reuse['status']}")
            st.write(reuse["action"])
            if reuse["basis"]:
                st.caption("Based on: " + "; ".join(reuse["basis"]))
            if reuse["provisional"]:
                st.warning("Provisional: some findings are low-confidence.")
            if life is None:
                st.info("Remaining life in years needs a thickness reading, because a photo cannot show how much "
                        "metal has been lost. Fill in the optional thickness fields to get an estimate.")
            elif not life["valid"]:
                st.error(life["error"])
            elif life["status"] == "NO_MEASURABLE_LOSS":
                st.success("No measurable thickness loss so far, so no finite remaining life can be calculated.")
            else:
                m1, m2 = st.columns(2)
                m1.metric("Corrosion rate (mm/yr)", f"{life['corrosion_rate_mm_per_year']}")
                m2.metric("Estimated remaining life (yr)", f"{life['remaining_years']}")
                if life["reinspect_years"] is not None:
                    st.caption(f"Suggested re-inspection within {life['reinspect_years']} years. "
                               "Assumes the average rate so far continues; local pitting is not captured.")
            st.caption(reuse["caveat"])
            if result["needs_review"]:
                st.warning("Some findings are low-confidence - have an inspector verify them before acting.")

            if not result["findings"]:
                st.warning("No defects detected. Try lowering the confidence threshold.")
            for f in result["findings"]:
                st.markdown(f"**{f['class']}** ({f['standard']}) - {f['n_detections']} detection(s), "
                            f"max confidence {f['confidence_max']}")
                if f["needs_review"]:
                    st.caption("Low confidence - treat this grade as unverified.")
                c1, c2 = st.columns(2)
                if f["metric"] == "area_pct":
                    c1.metric("Affected area", f"{f['measured_value']}%")
                elif f["defect_type"] == "cracking":
                    c1.metric("Crack regions", f"{int(f['measured_value'])}")
                else:
                    c1.metric("Blisters", f"{int(f['measured_value'])}")
                c2.metric("Grade", f["grade"])
                st.info(f"**{f['label']}** - {f['recommendation']}")
                st.caption(f"Method: {f['method']}")
                st.divider()
            st.success(f"Saved as inspection record `{saved['inspection_id']}`")

with tab_history:
    st.subheader("Past Inspections")
    c1, c2, _ = st.columns([1, 2, 1])
    with c1:
        mode = st.selectbox("Filter by", ["All time", "Last N days", "Custom date range"])

    records = []
    if mode == "All time":
        records = get_all_inspections()
    elif mode == "Last N days":
        with c2:
            days = st.number_input("Days back", min_value=1, max_value=365, value=7, step=1)
        records = get_inspections_by_date_range(days_back=int(days))
    else:
        with c2:
            rng = st.date_input("Select range", value=[])
        if len(rng) == 2:
            records = get_inspections_by_date_range(start_date=str(rng[0]), end_date=str(rng[1]))
        else:
            st.info("Pick a start and end date to see results.")

    st.caption(f"Showing {len(records)} inspection(s)")

    with st.expander("Danger zone: clear inspection history"):
        st.warning("This permanently deletes ALL saved inspection records.")
        ok = st.checkbox("I understand this is permanent")
        if st.button("Clear all history", disabled=not ok):
            st.success(f"Deleted {clear_all_inspections()} record(s). Refresh the page.")

    st.divider()
    if not records:
        st.write("No inspections found for this filter.")
    for rec in records:
        with st.expander(f"{rec['inspection_id']} - {rec['asset_id']} - {rec['overall_grade']}"):
            st.write(f"**Timestamp:** {rec['timestamp']}")
            st.write(f"**Location:** {rec['location_tag']}")
            st.write(f"**Image:** {rec['image_path']}")
            if rec.get("reuse"):
                st.write(f"**Reuse assessment:** {rec['reuse']['status']}")
            if rec.get("life") and rec["life"].get("valid"):
                st.write(f"**Remaining life estimate:** {rec['life'].get('remaining_years')} years "
                         f"(rate {rec['life']['corrosion_rate_mm_per_year']} mm/yr)")
            st.json(rec["detections"])
