"""
Step 8: Interactive Streamlit Web Dashboard
Explainable Multimodal ECG Intelligence for Early Cardiac Risk Assessment
Features:
- Dual-mode input: 12-lead paper ECG image upload (or held-out test sample) + all 19 clinical variables
- Proper training-fitted StandardScaler/OneHotEncoder preprocessing via models/clinical_preprocessor.joblib
- 3 Inference Modes: ECG-Only (ResNet-18), Multimodal Fusion, and Clinical-Only (Random Forest)
- Validation-fitted temperature calibration loaded from models/calibration_config.json
- Dynamic interactive visualizations:
  * Interactive Diagnostic Confidence Gauge & Risk Severity Meter
  * Interactive Multi-Class Calibrated Probability Distribution Chart with rich tooltips
  * Dynamic Electrophysiological Waveform Simulation (Lead II rhythm strip responding to sliders)
  * Interactive Clinical Biomarker Radar / Physiological Vitals Profile vs healthy reference bounds
  * Interactive Grad-CAM with real-time opacity blend slider
  * Interactive Explainable AI (SHAP / Gradient Saliency) Horizontal Bar Chart
  * Exportable Clinical Diagnostic Test Report (JSON & Markdown)
- Real-time updates as input parameters change
- Transparent architectural benchmark summary, limitations, and research-only disclaimers
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from train_ecg_model import CLASS_NAMES
from inference import (
    CLINICAL_FEATURE_DEFAULTS,
    get_test_sample_choices,
    load_runtime_models,
    predict_clinical,
    predict_ecg,
    predict_multimodal,
    validate_and_load_image,
)
from dashboard_visualizations import (
    CLASS_COLORS,
    CLASS_DESCRIPTIONS,
    CLASS_SEVERITY,
    CLINICAL_REFERENCE_BOUNDS,
    build_clinical_parameters_table,
    create_biomarker_radar_chart,
    create_clinical_alerts,
    create_diagnostic_gauge,
    create_ecg_waveform_simulation,
    create_interactive_attribution_chart,
    create_probability_bar_chart,
    generate_exportable_report,
    generate_markdown_report,
)

# Page Configuration
st.set_page_config(
    page_title="ECG Intelligence Dashboard",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 0.9rem;
        color: #92400E;
        margin-top: 15px;
    }
    .alert-card {
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.88rem;
    }
    .alert-critical {
        background-color: #FEE2E2;
        border-left: 4px solid #EF4444;
        color: #991B1B;
    }
    .alert-warning {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        color: #92400E;
    }
    .alert-info {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        color: #1E40AF;
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.82rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_models():
    return load_runtime_models()


def _render_gradcam_section(image: Image.Image, heatmap: np.ndarray, pred_class: str) -> None:
    """Render interactive Grad-CAM spatial attribution with opacity blend control."""
    st.markdown("#### 🔍 Visual Attribution: Grad-CAM Saliency Overlay")
    st.caption(
        "Salient visual regions on the 12-lead paper ECG driving the predicted logit "
        "(indicates mathematical convolutional attribution, not biological causality)."
    )

    blend_col1, blend_col2 = st.columns([1.5, 1])
    with blend_col1:
        blend_alpha = st.slider(
            "Heatmap Blend Opacity",
            min_value=0.10,
            max_value=0.90,
            value=0.45,
            step=0.05,
            help="Adjust blend transparency to inspect the underlying ECG lead grid and waveforms beneath the attribution heat regions.",
        )
    with blend_col2:
        st.caption(f"Showing **{blend_alpha*100:.0f}% Heatmap** + **{(1.0-blend_alpha)*100:.0f}% Raw ECG**")

    resized_orig = image.resize((224, 224), Image.Resampling.LANCZOS)
    orig_np = np.array(resized_orig) / 255.0
    heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize(
        (224, 224), resample=Image.Resampling.BILINEAR
    )
    heatmap_np = np.array(heatmap_resized) / 255.0
    colored_heatmap = plt.colormaps["jet"](heatmap_np)[:, :, :3]
    overlay = np.clip((1.0 - blend_alpha) * orig_np + blend_alpha * colored_heatmap, 0.0, 1.0)

    vcol1, vcol2 = st.columns(2)
    with vcol1:
        st.image(image, caption="Input 12-Lead Paper ECG Scan", use_container_width=True)
    with vcol2:
        st.image(
            overlay,
            caption=f"Grad-CAM Heatmap Overlay ({pred_class.replace('_', ' ')})",
            use_container_width=True,
        )


def main():
    st.markdown(
        '<div class="main-header">🫀 Explainable Multimodal ECG Intelligence</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Interactive Clinical Diagnostic Dashboard & Real-Time Test Report Intelligence</div>',
        unsafe_allow_html=True,
    )

    models = get_models()
    device = models["device"]
    cal_cfg = models.get("calibration", {})

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        mode = st.radio(
            "Inference Mode",
            ["Multimodal Fusion", "ECG Vision Only", "Clinical Tabular Only"],
            index=0,
        )

        apply_calibration = st.checkbox("Apply Validation-Fitted Calibration", value=True)
        ecg_cal = cal_cfg.get("ecg", {})
        mm_cal = cal_cfg.get("multimodal", {})
        if mode == "ECG Vision Only":
            active_temp = float(ecg_cal.get("recommended_temperature", 1.0)) if apply_calibration else 1.0
        elif mode == "Multimodal Fusion":
            active_temp = float(mm_cal.get("recommended_temperature", 1.0)) if apply_calibration else 1.0
        else:
            active_temp = 1.0

        st.caption(f"Active Softmax Temperature: `T = {active_temp:.4f}`")

        st.markdown("---")
        st.markdown("### 📋 System Architecture")
        st.write(f"**Compute Device**: `{device}`")
        st.write("**ECG Vision**: `ResNet-18 (2D-CNN, 224x224)`")
        st.write("**Clinical Model**: `Random Forest (19 features)`")
        st.write("**Multimodal Network**: `Late-Fusion Net (96-d)`")
        st.write(
            f"**Fitted Calibration**: ECG `T={ecg_cal.get('recommended_temperature', 1.0):.2f}` "
            f"| MM `T={mm_cal.get('recommended_temperature', 1.0):.2f}`"
        )

        st.markdown(
            '<div class="disclaimer-box">⚠️ <b>Research Use Only</b><br>'
            "Experimental prototype trained on digitized 12-lead ECG scans and synthetic tabular variables. "
            "Not a certified medical device; must not be used for clinical diagnosis or emergency triage.</div>",
            unsafe_allow_html=True,
        )

    # Main layout columns
    col_input, col_results = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("1. Patient & Diagnostic Input")

        image_source = None
        image_label: Optional[str] = None
        if mode in ["ECG Vision Only", "Multimodal Fusion"]:
            st.markdown("**Upload 12-Lead Paper ECG Scan** (JPEG/PNG)")
            uploaded_file = st.file_uploader("Select ECG scan file", type=["jpg", "jpeg", "png"])

            test_choices = get_test_sample_choices()
            if uploaded_file is not None:
                image_source = uploaded_file
                image_label = f"Uploaded Scan: {uploaded_file.name}"
            elif test_choices:
                use_sample = st.checkbox("Use held-out test sample from repository", value=True)
                if use_sample:
                    selected_label = st.selectbox("Held-out Test ECG Sample", list(test_choices.keys()))
                    image_source = test_choices[selected_label]
                    image_label = f"Held-out Sample: {selected_label}"

        clinical_data: Dict[str, Any] = {}
        if mode in ["Multimodal Fusion", "Clinical Tabular Only"]:
            with st.expander("🩺 Clinical & Laboratory Parameters (12 variables)", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    age = st.slider("Age (years)", 30, 84, int(CLINICAL_FEATURE_DEFAULTS["Age"]))
                    gender = st.selectbox("Biological Gender", ["Male", "Female"])
                    hr = st.slider("Heart Rate (bpm)", 55, 139, int(CLINICAL_FEATURE_DEFAULTS["Heart_Rate"]))
                    sbp = st.slider("Systolic BP (mmHg)", 90, 179, int(CLINICAL_FEATURE_DEFAULTS["Systolic_BP"]))
                    dbp = st.slider("Diastolic BP (mmHg)", 60, 109, int(CLINICAL_FEATURE_DEFAULTS["Diastolic_BP"]))
                    chol = st.slider("Cholesterol (mg/dL)", 150, 319, int(CLINICAL_FEATURE_DEFAULTS["Cholesterol"]))
                with c2:
                    glucose = st.slider("Fasting Glucose (mg/dL)", 70, 199, int(CLINICAL_FEATURE_DEFAULTS["Glucose"]))
                    troponin = st.slider(
                        "Cardiac Troponin (ng/mL)",
                        0.001,
                        2.500,
                        float(CLINICAL_FEATURE_DEFAULTS["Troponin"]),
                        step=0.01,
                        help="Standard clinical upper normal threshold is 0.04 ng/mL. Levels >= 0.40 indicate critical myocardial necrosis.",
                    )
                    pr = st.slider("PR Interval (ms)", 120, 219, int(CLINICAL_FEATURE_DEFAULTS["PR_Interval"]))
                    qrs = st.slider("QRS Duration (ms)", 70, 159, int(CLINICAL_FEATURE_DEFAULTS["QRS_Duration"]))
                    qt = st.slider("QT Interval (ms)", 300, 499, int(CLINICAL_FEATURE_DEFAULTS["QT_Interval"]))
                    st_level = st.slider(
                        "ST Segment Level (mm)",
                        -0.20,
                        0.50,
                        float(CLINICAL_FEATURE_DEFAULTS["ST_Segment_Level"]),
                        step=0.01,
                        help="Normal is -0.05 to +0.10 mm. >= +0.10 mm indicates STEMI criteria elevation.",
                    )

            with st.expander("⚡ Additional Electrophysiological Waveform Measurements (7 parameters)", expanded=False):
                e1, e2 = st.columns(2)
                with e1:
                    p_wave = st.slider(
                        "P Wave Duration (ms)", 80, 119, int(CLINICAL_FEATURE_DEFAULTS["P_Wave_Duration"])
                    )
                    t_wave = st.slider(
                        "T Wave Duration (ms)", 120, 199, int(CLINICAL_FEATURE_DEFAULTS["T_Wave_Duration"])
                    )
                    rr_int = st.slider("RR Interval (ms)", 600, 1199, int(CLINICAL_FEATURE_DEFAULTS["RR_Interval"]))
                    hrv_sdnn = st.slider("HRV SDNN (ms)", 20, 119, int(CLINICAL_FEATURE_DEFAULTS["HRV_SDNN"]))
                with e2:
                    hrv_rmssd = st.slider("HRV RMSSD (ms)", 15, 99, int(CLINICAL_FEATURE_DEFAULTS["HRV_RMSSD"]))
                    r_amp = st.slider(
                        "R Peak Amplitude (mV)",
                        0.50,
                        3.00,
                        float(CLINICAL_FEATURE_DEFAULTS["R_Peak_Amplitude"]),
                        step=0.05,
                    )
                    s_amp = st.slider(
                        "S Wave Amplitude (mV)",
                        -2.50,
                        -0.10,
                        float(CLINICAL_FEATURE_DEFAULTS["S_Wave_Amplitude"]),
                        step=0.05,
                    )

            clinical_data = {
                "Age": age,
                "Heart_Rate": hr,
                "Systolic_BP": sbp,
                "Diastolic_BP": dbp,
                "Cholesterol": chol,
                "Glucose": glucose,
                "Troponin": troponin,
                "PR_Interval": pr,
                "QRS_Duration": qrs,
                "QT_Interval": qt,
                "P_Wave_Duration": p_wave,
                "T_Wave_Duration": t_wave,
                "ST_Segment_Level": st_level,
                "RR_Interval": rr_int,
                "HRV_SDNN": hrv_sdnn,
                "HRV_RMSSD": hrv_rmssd,
                "R_Peak_Amplitude": r_amp,
                "S_Wave_Amplitude": s_amp,
                "Gender": gender,
            }

    with col_results:
        st.subheader("2. Model Prediction & Diagnostic Test Report")

        if mode in ["ECG Vision Only", "Multimodal Fusion"] and image_source is None:
            st.info("👈 Please upload a 12-lead ECG scan or enable a held-out test sample to generate test results.")
            return

        image = None
        if image_source is not None:
            try:
                image = validate_and_load_image(image_source)
            except ValueError as err:
                st.error(f"Invalid ECG image input: {err}")
                return

        result = None
        try:
            if mode == "ECG Vision Only":
                if "ecg" not in models:
                    st.error("Missing models/ecg_model_best.pt checkpoint.")
                    return
                result = predict_ecg(
                    models["ecg"],
                    image,
                    device=device,
                    temperature=active_temp,
                    compute_gradcam=True,
                )
            elif mode == "Clinical Tabular Only":
                if "clinical" not in models or "preprocessor" not in models:
                    st.error("Missing models/clinical_model.joblib or models/clinical_preprocessor.joblib.")
                    return
                result = predict_clinical(
                    models["clinical"],
                    clinical_data,
                    preprocessor=models["preprocessor"],
                    compute_shap=True,
                )
            elif mode == "Multimodal Fusion":
                if "multimodal" not in models or "preprocessor" not in models:
                    st.error("Missing models/multimodal_model_best.pt or models/clinical_preprocessor.joblib.")
                    return
                result = predict_multimodal(
                    models["multimodal"],
                    image,
                    clinical_data,
                    preprocessor=models["preprocessor"],
                    device=device,
                    temperature=active_temp,
                    compute_xai=True,
                )
        except ValueError as err:
            st.error(f"Input validation failed: {err}")
            return

        if result is not None:
            probs = result["probs"]
            pred_class = result["predicted_class"]
            confidence = result["confidence"] * 100.0
            color = CLASS_COLORS[pred_class]
            severity_label, _ = CLASS_SEVERITY[pred_class]
            cal_badge = (
                f"Calibrated (T = {active_temp:.4f})"
                if (apply_calibration and active_temp != 1.0)
                else "Raw Softmax (T = 1.0)"
            )

            # Fallback clinical parameters if running ECG-only mode
            display_clinical = clinical_data if clinical_data else CLINICAL_FEATURE_DEFAULTS

            # Primary Diagnostic Classification Card
            st.markdown(
                f"""
                <div style="background-color: {color}15; border-left: 6px solid {color}; padding: 16px 20px; border-radius: 8px; margin-bottom: 14px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 0.88rem; font-weight: 700; color: {color}; text-transform: uppercase; letter-spacing: 0.5px;">
                            Primary Diagnostic Assessment
                        </span>
                        <span class="metric-badge" style="background-color: {color}25; color: {color};">
                            {severity_label}
                        </span>
                    </div>
                    <h2 style="color: {color}; margin: 6px 0 4px 0; font-size: 1.65rem;">{pred_class.replace('_', ' ')}</h2>
                    <p style="font-size: 1.05rem; margin: 0; color: #334155;">
                        Diagnostic Confidence: <b>{confidence:.1f}%</b> &nbsp;|&nbsp; 
                        <span style="font-size: 0.9rem; color: #64748B;">{cal_badge}</span>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Clinical Alerts & Flags
            alerts = create_clinical_alerts(display_clinical)
            if alerts and mode in ["Multimodal Fusion", "Clinical Tabular Only"]:
                critical_count = sum(1 for a in alerts if a["level"] == "critical")
                alert_header = (
                    f"🚨 **{critical_count} Critical Clinical Findings** & {len(alerts) - critical_count} Diagnostic Flags"
                    if critical_count > 0
                    else f"⚠️ **{len(alerts)} Diagnostic Clinical Flags**"
                )
                with st.expander(alert_header, expanded=(critical_count > 0)):
                    for alert in alerts:
                        css_class = f"alert-{alert['level']}"
                        st.markdown(
                            f"""
                            <div class="alert-card {css_class}">
                                <b>{alert['icon']} {alert['title']}</b><br>
                                {alert['message']}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            # Interactive Visualization Modules (Tabs)
            tab_overview, tab_waveform, tab_radar, tab_xai, tab_report = st.tabs([
                "📊 Risk & Probabilities",
                "🫀 Live ECG Waveform",
                "🕸️ Biomarker Radar",
                "🔍 Explainable AI",
                "📋 Diagnostic Report",
            ])

            # Tab 1: Risk & Probabilities
            with tab_overview:
                col_gauge, col_probs = st.columns([1, 1.25])
                with col_gauge:
                    gauge_fig = create_diagnostic_gauge(
                        predicted_class=pred_class,
                        confidence=confidence,
                        active_temp=active_temp,
                        is_calibrated=apply_calibration,
                    )
                    st.plotly_chart(gauge_fig, use_container_width=True)

                with col_probs:
                    prob_fig = create_probability_bar_chart(
                        probs=probs,
                        predicted_class=pred_class,
                    )
                    st.plotly_chart(prob_fig, use_container_width=True)

                st.markdown(
                    f"""
                    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 12px 16px; border-radius: 6px; font-size: 0.9rem;">
                        <b>Diagnostic Classification Context:</b> {CLASS_DESCRIPTIONS.get(pred_class, '')}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Tab 2: Live Electrophysiological Waveform Simulation
            with tab_waveform:
                st.caption(
                    "Dynamic Lead II rhythm strip generated directly from the input electrophysiological parameters. "
                    "Adjust HR, PR, QRS, QT, ST level, or wave amplitudes to observe morphological changes."
                )

                wf_fig = create_ecg_waveform_simulation(display_clinical)
                st.plotly_chart(wf_fig, use_container_width=True)

                # Metric Cards
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    hr_val = float(display_clinical.get("Heart_Rate", 75))
                    hr_status = "Normal" if 60 <= hr_val <= 100 else ("Tachycardia" if hr_val > 100 else "Bradycardia")
                    st.metric("Heart Rate", f"{hr_val:.0f} bpm", hr_status)
                with m2:
                    pr_val = float(display_clinical.get("PR_Interval", 160))
                    pr_status = "Normal" if 120 <= pr_val <= 200 else ("Prolonged (AV Block)" if pr_val > 200 else "Shortened")
                    st.metric("PR Interval", f"{pr_val:.0f} ms", pr_status)
                with m3:
                    qrs_val = float(display_clinical.get("QRS_Duration", 100))
                    qrs_status = "Normal" if qrs_val <= 120 else "Wide (Delay)"
                    st.metric("QRS Duration", f"{qrs_val:.0f} ms", qrs_status)
                with m4:
                    st_val = float(display_clinical.get("ST_Segment_Level", 0.0))
                    st_status = "STEMI Elevation" if st_val >= 0.10 else ("Depression" if st_val <= -0.05 else "Isoelectric")
                    st.metric("ST Segment", f"{st_val:+.2f} mm", st_status)

            # Tab 3: Biomarker Radar & Vitals
            with tab_radar:
                st.caption(
                    "Multi-axis polar biomarker profile comparing the patient's inputs against the healthy physiological reference zone "
                    "(green dashed corridor). Values extending outside indicate acute elevation or depression."
                )
                radar_fig = create_biomarker_radar_chart(display_clinical)
                st.plotly_chart(radar_fig, use_container_width=True)

                with st.expander("📋 Full Clinical Parameters Reference Table", expanded=False):
                    summary_df = build_clinical_parameters_table(display_clinical)
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)

            # Tab 4: Explainable AI
            with tab_xai:
                if result.get("heatmap") is not None and image is not None:
                    _render_gradcam_section(image, result["heatmap"], pred_class)

                if result.get("attributions") is not None:
                    st.markdown("---")
                    xai_title = "SHAP TreeExplainer" if mode == "Clinical Tabular Only" else "Late-Fusion Gradient Saliency"
                    attr_fig = create_interactive_attribution_chart(result["attributions"], xai_title)
                    st.plotly_chart(attr_fig, use_container_width=True)

            # Tab 5: Diagnostic Report & Export
            with tab_report:
                st.markdown("#### 📄 Patient Diagnostic Assessment Report")
                st.caption("Complete synthesized clinical test report reflecting the current input parameters and predictions.")

                report_json = generate_exportable_report(
                    result=result,
                    clinical_data=display_clinical,
                    mode=mode,
                    active_temp=active_temp,
                    apply_calibration=apply_calibration,
                    image_label=image_label,
                )
                report_md = generate_markdown_report(
                    result=result,
                    clinical_data=display_clinical,
                    mode=mode,
                    active_temp=active_temp,
                    apply_calibration=apply_calibration,
                    image_label=image_label,
                )

                dcol1, dcol2 = st.columns(2)
                with dcol1:
                    st.download_button(
                        label="📥 Download Diagnostic Report (JSON)",
                        data=json.dumps(report_json, indent=2),
                        file_name=f"ecg_diagnostic_report_{pred_class.lower()}.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                with dcol2:
                    st.download_button(
                        label="📥 Download Diagnostic Report (Markdown)",
                        data=report_md,
                        file_name=f"ecg_diagnostic_report_{pred_class.lower()}.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )

                st.markdown("---")
                st.markdown(report_md)

    with st.expander("📖 Model Benchmarks, Architecture & Dataset Limitations", expanded=False):
        st.markdown(
            """
            - **Held-Out Test Benchmark (`N = 137` ECG scans, zero MD5 hash leakage)**:
              - **Multimodal Late Fusion**: `87.59%` Accuracy | `0.8651` Macro-F1 | `0.9716` OVR ROC-AUC
              - **ECG-Only Vision (ResNet-18)**: `86.13%` Accuracy | `0.8516` Macro-F1 | `0.9672` OVR ROC-AUC
              - **Clinical-Only (Random Forest)**: `18.25%` Accuracy on matched test cohort (`24.96%` on full clinical validation set)
            - **Key Scientific Caveat**: Clinical CSV rows are synthetic uniform distributions with no ground-truth patient key linking them to the 928 physical ECG scans; visual ECG features provide nearly all diagnostic signal.
            """
        )


if __name__ == "__main__":
    main()
