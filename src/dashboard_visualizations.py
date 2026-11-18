"""
Interactive Visualization Engine for ECG Dashboard Test Reports
Provides dynamic Plotly figures, electrophysiological waveform synthesis,
biomarker radar profiling, and clinical alert generation that update
in real time as input parameters change.
"""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image

CLASS_NAMES = ["Normal", "Abnormal_Heartbeat", "Myocardial_Infarction", "Post_MI_History"]

CLASS_COLORS = {
    "Normal": "#10B981",              # Emerald Green
    "Abnormal_Heartbeat": "#3B82F6",  # Vibrant Blue
    "Myocardial_Infarction": "#EF4444",# Crimson Red
    "Post_MI_History": "#F59E0B",     # Amber Orange
}

CLASS_SEVERITY = {
    "Normal": ("LOW RISK", "#10B981"),
    "Abnormal_Heartbeat": ("MODERATE RISK (Arrhythmia)", "#3B82F6"),
    "Post_MI_History": ("HIGH RISK (Prior Infarction)", "#F59E0B"),
    "Myocardial_Infarction": ("CRITICAL RISK (Acute Ischemia)", "#EF4444"),
}

CLASS_DESCRIPTIONS = {
    "Normal": "Standard sinus rhythm with preserved electrophysiological conduction and no acute ischemic repolarization changes.",
    "Abnormal_Heartbeat": "Dysrhythmia or irregular rhythm alteration requiring close electrophysiological evaluation.",
    "Myocardial_Infarction": "Acute ischemic injury / transmural infarction characterized by diagnostic ST-segment elevation or Q-wave formation.",
    "Post_MI_History": "Persistent pathological Q-waves or T-wave inversion indicative of prior myocardial necrosis and fibrous scarring.",
}

# Standard cardiology reference bounds for the 19 clinical parameters
CLINICAL_REFERENCE_BOUNDS = {
    "Heart_Rate": {"name": "Heart Rate", "unit": "bpm", "norm_min": 60, "norm_max": 100, "plot_min": 40, "plot_max": 160},
    "Systolic_BP": {"name": "Systolic BP", "unit": "mmHg", "norm_min": 90, "norm_max": 120, "plot_min": 70, "plot_max": 200},
    "Diastolic_BP": {"name": "Diastolic BP", "unit": "mmHg", "norm_min": 60, "norm_max": 80, "plot_min": 40, "plot_max": 120},
    "Cholesterol": {"name": "Cholesterol", "unit": "mg/dL", "norm_min": 150, "norm_max": 200, "plot_min": 120, "plot_max": 340},
    "Glucose": {"name": "Fasting Glucose", "unit": "mg/dL", "norm_min": 70, "norm_max": 100, "plot_min": 60, "plot_max": 220},
    "Troponin": {"name": "Cardiac Troponin", "unit": "ng/mL", "norm_min": 0.00, "norm_max": 0.04, "plot_min": 0.00, "plot_max": 2.50},
    "PR_Interval": {"name": "PR Interval", "unit": "ms", "norm_min": 120, "norm_max": 200, "plot_min": 100, "plot_max": 240},
    "QRS_Duration": {"name": "QRS Duration", "unit": "ms", "norm_min": 80, "norm_max": 120, "plot_min": 60, "plot_max": 170},
    "QT_Interval": {"name": "QT Interval", "unit": "ms", "norm_min": 350, "norm_max": 440, "plot_min": 280, "plot_max": 520},
    "ST_Segment_Level": {"name": "ST Segment Level", "unit": "mm", "norm_min": -0.05, "norm_max": 0.10, "plot_min": -0.25, "plot_max": 0.55},
    "P_Wave_Duration": {"name": "P Wave Duration", "unit": "ms", "norm_min": 80, "norm_max": 110, "plot_min": 70, "plot_max": 130},
    "T_Wave_Duration": {"name": "T Wave Duration", "unit": "ms", "norm_min": 120, "norm_max": 180, "plot_min": 100, "plot_max": 220},
    "RR_Interval": {"name": "RR Interval", "unit": "ms", "norm_min": 600, "norm_max": 1000, "plot_min": 500, "plot_max": 1300},
    "HRV_SDNN": {"name": "HRV SDNN", "unit": "ms", "norm_min": 50, "norm_max": 100, "plot_min": 15, "plot_max": 130},
    "HRV_RMSSD": {"name": "HRV RMSSD", "unit": "ms", "norm_min": 25, "norm_max": 75, "plot_min": 10, "plot_max": 110},
    "R_Peak_Amplitude": {"name": "R Peak Amplitude", "unit": "mV", "norm_min": 0.80, "norm_max": 2.50, "plot_min": 0.40, "plot_max": 3.20},
    "S_Wave_Amplitude": {"name": "S Wave Amplitude", "unit": "mV", "norm_min": -2.00, "norm_max": -0.30, "plot_min": -2.70, "plot_max": 0.00},
}


def create_diagnostic_gauge(
    predicted_class: str,
    confidence: float,
    active_temp: float = 1.0,
    is_calibrated: bool = True,
) -> go.Figure:
    """Create an interactive medical diagnostic gauge showing confidence and risk level."""
    color = CLASS_COLORS.get(predicted_class, "#2563EB")
    severity_label, _ = CLASS_SEVERITY.get(predicted_class, ("UNKNOWN RISK", "#6B7280"))
    cal_text = f"T={active_temp:.2f}" if (is_calibrated and active_temp != 1.0) else "Raw (T=1.0)"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(confidence),
            number={"suffix": "%", "font": {"size": 36, "color": "#0F172A", "family": "Inter, sans-serif"}},
            title={
                "text": f"<b>{predicted_class.replace('_', ' ')}</b><br><span style='font-size:0.85em;color:{color}'>{severity_label}</span><br><span style='font-size:0.7em;color:#64748B'>{cal_text}</span>",
                "font": {"size": 15, "family": "Inter, sans-serif"},
            },
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94A3B8", "ticksuffix": "%"},
                "bar": {"color": color, "thickness": 0.3},
                "bgcolor": "#F1F5F9",
                "borderwidth": 1,
                "bordercolor": "#CBD5E1",
                "steps": [
                    {"range": [0, 40], "color": "#F8FAFC"},
                    {"range": [40, 70], "color": "#F1F5F9"},
                    {"range": [70, 90], "color": "#E2E8F0"},
                    {"range": [90, 100], "color": f"{color}30"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 4},
                    "thickness": 0.8,
                    "value": float(confidence),
                },
            },
        )
    )

    fig.update_layout(
        height=240,
        margin=dict(l=25, r=25, t=55, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif"},
    )
    return fig


def create_probability_bar_chart(
    probs: np.ndarray,
    predicted_class: str,
) -> go.Figure:
    """Create an interactive horizontal probability distribution chart with rich hover data."""
    labels = [c.replace("_", " ") for c in CLASS_NAMES]
    percentages = [float(p) * 100.0 for p in probs]
    colors = [CLASS_COLORS[c] for c in CLASS_NAMES]
    borders = ["#0F172A" if c == predicted_class else "#E2E8F0" for c in CLASS_NAMES]
    widths = [2.5 if c == predicted_class else 1.0 for c in CLASS_NAMES]

    hover_data = []
    for c, pct in zip(CLASS_NAMES, percentages):
        is_pred = "Primary Classification (Highest Confidence)" if c == predicted_class else "Alternative Differential"
        desc = CLASS_DESCRIPTIONS.get(c, "")
        hover_data.append((is_pred, desc))

    customdata = np.array(hover_data)

    fig = go.Figure(
        go.Bar(
            y=labels,
            x=percentages,
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(color=borders, width=widths),
                opacity=[0.95 if c == predicted_class else 0.75 for c in CLASS_NAMES],
            ),
            text=[f"<b>{pct:.1f}%</b>" for pct in percentages],
            textposition="outside",
            customdata=customdata,
            hovertemplate=(
                "<b>%{y}</b><br>"
                + "Probability: <b>%{x:.2f}%</b><br>"
                + "Status: %{customdata[0]}<br>"
                + "<span style='font-size:0.9em;color:#475569'>%{customdata[1]}</span>"
                + "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(text="<b>Diagnostic Class Probability Distribution</b>", font=dict(size=14)),
        xaxis=dict(
            title="Calibrated Diagnostic Probability (%)",
            range=[0, max(max(percentages) * 1.18, 100.0)],
            gridcolor="#E2E8F0",
            zerolinecolor="#CBD5E1",
        ),
        yaxis=dict(autorange="reversed"),
        height=240,
        margin=dict(l=10, r=25, t=35, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12),
    )
    return fig


def create_biomarker_radar_chart(clinical_data: Dict[str, Any]) -> go.Figure:
    """
    Create an interactive polar/spider radar chart comparing the patient's
    current clinical inputs against standard physiological reference intervals.
    """
    radar_keys = [
        "Heart_Rate",
        "Systolic_BP",
        "Diastolic_BP",
        "Cholesterol",
        "Glucose",
        "Troponin",
        "ST_Segment_Level",
        "HRV_SDNN",
    ]

    labels = []
    patient_scaled = []
    normal_scaled = []
    patient_raw_vals = []
    normal_ranges = []
    units = []
    statuses = []

    for key in radar_keys:
        meta = CLINICAL_REFERENCE_BOUNDS[key]
        raw_val = float(clinical_data.get(key, 0.0))
        norm_min = meta["norm_min"]
        norm_max = meta["norm_max"]
        p_min = meta["plot_min"]
        p_max = meta["plot_max"]

        # Scale linearly between 0 and 100 where [30, 70] corresponds to the normal interval
        # Value < 30 is depressed, 30-70 is normal, > 70 is elevated
        if raw_val <= norm_min:
            denom = max(norm_min - p_min, 1e-6)
            scaled = 30.0 + 20.0 * (raw_val - norm_min) / denom
        elif raw_val >= norm_max:
            denom = max(p_max - norm_max, 1e-6)
            scaled = 70.0 + 30.0 * (raw_val - norm_max) / denom
        else:
            denom = max(norm_max - norm_min, 1e-6)
            scaled = 30.0 + 40.0 * (raw_val - norm_min) / denom

        scaled = float(np.clip(scaled, 5.0, 105.0))

        # Status text
        if raw_val > norm_max:
            if key == "Troponin" and raw_val >= 0.40:
                status = "CRITICAL ELEVATION"
            elif key == "ST_Segment_Level" and raw_val >= 0.10:
                status = "SIGNIFICANT ST ELEVATION"
            else:
                status = "ELEVATED"
        elif raw_val < norm_min:
            status = "DEPRESSED / LOW"
        else:
            status = "NORMAL (Physiological)"

        labels.append(meta["name"])
        patient_scaled.append(scaled)
        normal_scaled.append(50.0)  # Mid-normal baseline ring
        patient_raw_vals.append(raw_val)
        normal_ranges.append(f"{norm_min} - {norm_max}")
        units.append(meta["unit"])
        statuses.append(status)

    # Close polar loop
    labels_closed = labels + [labels[0]]
    patient_scaled_closed = patient_scaled + [patient_scaled[0]]
    normal_scaled_closed = [50.0] * (len(labels) + 1)
    norm_upper_ring = [70.0] * (len(labels) + 1)
    norm_lower_ring = [30.0] * (len(labels) + 1)

    customdata = []
    for i in range(len(labels)):
        customdata.append([patient_raw_vals[i], units[i], normal_ranges[i], statuses[i]])
    customdata.append(customdata[0])

    fig = go.Figure()

    # Normal Reference Corridor (shaded band between 30% and 70%)
    fig.add_trace(
        go.Scatterpolar(
            r=norm_upper_ring,
            theta=labels_closed,
            mode="lines",
            line=dict(color="rgba(16, 185, 129, 0.4)", width=1, dash="dash"),
            name="Normal Upper Bound",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=norm_lower_ring,
            theta=labels_closed,
            mode="lines",
            fill="tonext",
            fillcolor="rgba(16, 185, 129, 0.12)",
            line=dict(color="rgba(16, 185, 129, 0.4)", width=1, dash="dash"),
            name="Normal Reference Zone",
            hoverinfo="skip",
        )
    )

    # Patient Profile
    fig.add_trace(
        go.Scatterpolar(
            r=patient_scaled_closed,
            theta=labels_closed,
            fill="toself",
            fillcolor="rgba(37, 99, 235, 0.22)",
            mode="lines+markers",
            line=dict(color="#2563EB", width=2.5),
            marker=dict(size=7, color="#1D4ED8"),
            name="Patient Clinical Profile",
            customdata=customdata,
            hovertemplate=(
                "<b>%{theta}</b><br>"
                + "Measured Value: <b>%{customdata[0]:.2f} %{customdata[1]}</b><br>"
                + "Normal Range: %{customdata[2]} %{customdata[1]}<br>"
                + "Status: <b>%{customdata[3]}</b>"
                + "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 110],
                tickvals=[30, 50, 70],
                ticktext=["Low", "Normal", "High"],
                tickfont=dict(size=9, color="#64748B"),
                gridcolor="#E2E8F0",
            ),
            angularaxis=dict(
                tickfont=dict(size=10.5, family="Inter, sans-serif"),
                gridcolor="#E2E8F0",
                linecolor="#CBD5E1",
            ),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
        height=350,
        margin=dict(l=35, r=35, t=30, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_ecg_waveform_simulation(clinical_data: Dict[str, Any]) -> go.Figure:
    """
    Generate an interactive 2-cycle Lead II electrophysiological waveform simulation
    derived directly from the patient's current interval & amplitude parameters.
    Dynamically reflects changes to HR, RR, PR, QRS, QT, ST level, R amp, and S amp.
    """
    hr = float(clinical_data.get("Heart_Rate", 75))
    rr = float(clinical_data.get("RR_Interval", 60000.0 / max(hr, 30.0)))
    pr = float(clinical_data.get("PR_Interval", 160))
    qrs = float(clinical_data.get("QRS_Duration", 100))
    qt = float(clinical_data.get("QT_Interval", 400))
    p_dur = float(clinical_data.get("P_Wave_Duration", 100))
    t_dur = float(clinical_data.get("T_Wave_Duration", 160))
    st_level = float(clinical_data.get("ST_Segment_Level", 0.0))  # in mm (1 mm = 0.1 mV)
    r_amp = float(clinical_data.get("R_Peak_Amplitude", 1.8))     # in mV
    s_amp = float(clinical_data.get("S_Wave_Amplitude", -1.2))    # in mV (negative)

    total_time_ms = 2.0 * rr
    time_pts = np.linspace(0, total_time_ms, int(max(total_time_ms * 1.5, 600)))
    voltage = np.zeros_like(time_pts)

    st_mv = st_level * 0.10  # 1 mm = 0.1 mV standard ECG voltage calibration

    for beat_idx in range(2):
        t0 = beat_idx * rr

        # P wave
        t_p_center = t0 + pr * 0.40
        p_sigma = max(p_dur / 5.0, 8.0)
        p_wave = 0.18 * np.exp(-0.5 * ((time_pts - t_p_center) / p_sigma) ** 2)

        # Q wave
        t_q_center = t0 + pr - 14.0
        q_sigma = 8.0
        q_wave = -0.15 * np.exp(-0.5 * ((time_pts - t_q_center) / q_sigma) ** 2)

        # R wave (sharp tall positive spike)
        t_r_center = t0 + pr
        r_sigma = max(qrs / 6.0, 7.0)
        r_wave = r_amp * np.exp(-0.5 * ((time_pts - t_r_center) / r_sigma) ** 2)

        # S wave (sharp negative deflection right after R)
        t_s_center = t0 + pr + 16.0
        s_sigma = max(qrs / 6.0, 8.0)
        s_wave = s_amp * np.exp(-0.5 * ((time_pts - t_s_center) / s_sigma) ** 2)

        # ST Segment (plateau starting after S wave up to T wave onset)
        t_st_start = t_s_center + 15.0
        t_st_end = t0 + qt - t_dur * 0.60
        st_mask = (time_pts >= t_st_start) & (time_pts <= t_st_end)
        st_wave = np.zeros_like(time_pts)
        st_wave[st_mask] = st_mv

        # T wave (repolarization)
        t_t_center = t0 + qt - t_dur * 0.35
        t_sigma = max(t_dur / 4.5, 18.0)
        t_amp = 0.28 + 0.45 * st_mv
        t_wave = t_amp * np.exp(-0.5 * ((time_pts - t_t_center) / t_sigma) ** 2)

        voltage += p_wave + q_wave + r_wave + s_wave + st_wave + t_wave

    fig = go.Figure()

    # Medical ECG Paper Background Grid
    fig.add_trace(
        go.Scatter(
            x=time_pts,
            y=voltage,
            mode="lines",
            line=dict(color="#B91C1C", width=2.2),
            name="Lead II Rhythm Strip",
            hovertemplate="Time: <b>%{x:.1f} ms</b><br>Potential: <b>%{y:.3f} mV</b><extra></extra>",
        )
    )

    # Highlight PR Interval of 1st beat
    fig.add_vrect(
        x0=pr * 0.2,
        x1=pr,
        fillcolor="rgba(16, 185, 129, 0.12)",
        layer="below",
        line_width=1,
        line_color="rgba(16, 185, 129, 0.4)",
        annotation_text="PR",
        annotation_position="top left",
        annotation_font_size=10,
    )

    # Highlight QRS Complex of 1st beat
    fig.add_vrect(
        x0=pr - 15,
        x1=pr + 25,
        fillcolor="rgba(59, 130, 246, 0.12)",
        layer="below",
        line_width=1,
        line_color="rgba(59, 130, 246, 0.4)",
        annotation_text="QRS",
        annotation_position="bottom right",
        annotation_font_size=10,
    )

    # Highlight ST Segment of 1st beat if elevation/depression exists
    if abs(st_level) >= 0.05:
        st_color = "rgba(239, 68, 68, 0.18)" if st_level > 0 else "rgba(245, 158, 11, 0.18)"
        border_c = "#EF4444" if st_level > 0 else "#F59E0B"
        fig.add_vrect(
            x0=pr + 25,
            x1=qt - t_dur * 0.5,
            fillcolor=st_color,
            layer="below",
            line_width=1.5,
            line_color=border_c,
            annotation_text=f"ST {st_level:+.2f}mm",
            annotation_position="top right",
            annotation_font_size=10,
            annotation_font_color=border_c,
        )

    # Layout styling mimicking calibrated ECG strip
    fig.update_layout(
        title=dict(
            text=f"<b>Interactive Electrophysiological Waveform (Lead II)</b> | HR={hr:.0f} bpm, PR={pr:.0f}ms, QRS={qrs:.0f}ms, ST={st_level:+.2f}mm",
            font=dict(size=13),
        ),
        xaxis=dict(
            title="Time (ms) — standard 25 mm/s (0.04s small box)",
            gridcolor="#FEE2E2",
            zerolinecolor="#FECACA",
            dtick=200,  # 200 ms per major ECG box
        ),
        yaxis=dict(
            title="Amplitude (mV) — 10 mm/mV",
            gridcolor="#FEE2E2",
            zerolinecolor="#DC2626",
            dtick=0.5,  # 0.5 mV per major ECG box
            range=[min(-1.8, s_amp * 1.3), max(2.5, r_amp * 1.25)],
        ),
        height=280,
        margin=dict(l=40, r=25, t=40, b=35),
        paper_bgcolor="#FFFDFD",
        plot_bgcolor="#FEF2F2",  # Light pink hue matching authentic ECG recording paper
        font=dict(family="Inter, monospace", size=11),
        showlegend=False,
    )
    return fig


def create_interactive_attribution_chart(attr_df: pd.DataFrame, mode_label: str) -> go.Figure:
    """Create an interactive horizontal bar chart for feature attributions with hover tooltips."""
    if attr_df is None or attr_df.empty:
        return go.Figure()

    top_df = attr_df.head(10).iloc[::-1].copy()
    kind_label = top_df["Kind"].iloc[0]

    colors = []
    directions = []
    for val in top_df["Attribution"]:
        if val >= 0:
            colors.append("#10B981")
            directions.append("Positive Contribution (Drives Predicted Class)")
        else:
            colors.append("#EF4444")
            directions.append("Negative Contribution (Opposes Predicted Class)")

    features = [f.replace("_", " ") for f in top_df["Feature"]]
    values = top_df["Attribution"].values
    scaled_vals = top_df["Scaled_Value"].values

    customdata = np.stack([scaled_vals, directions], axis=-1)

    fig = go.Figure(
        go.Bar(
            y=features,
            x=values,
            orientation="h",
            marker=dict(color=colors, line=dict(color="#1E293B", width=1.0), opacity=0.88),
            customdata=customdata,
            hovertemplate=(
                "<b>%{y}</b><br>"
                + f"{kind_label}: <b>%{{x:.4f}}</b><br>"
                + "Transformed Standardized Value: <b>%{customdata[0]:.3f}</b><br>"
                + "Influence: <b>%{customdata[1]}</b>"
                + "<extra></extra>"
            ),
        )
    )

    fig.add_vline(x=0.0, line_width=1.2, line_dash="dash", line_color="#64748B")

    fig.update_layout(
        title=dict(text=f"<b>Top 10 Clinical Feature Attributions ({mode_label})</b>", font=dict(size=13)),
        xaxis=dict(title=kind_label, gridcolor="#E2E8F0"),
        yaxis=dict(tickfont=dict(size=11)),
        height=320,
        margin=dict(l=10, r=25, t=35, b=35),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=11),
    )
    return fig


def create_clinical_alerts(clinical_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Scan all 19 patient clinical parameters against diagnostic thresholds to generate alerts."""
    alerts: List[Dict[str, str]] = []

    # 1. Troponin Alert
    troponin = float(clinical_data.get("Troponin", 0.0))
    if troponin >= 0.40:
        alerts.append({
            "level": "critical",
            "icon": "🚨",
            "title": f"Critical Troponin Elevation ({troponin:.2f} ng/mL)",
            "message": "Substantially exceeds reference upper limit (0.04 ng/mL). Indicates active myocardial necrosis / acute coronary syndrome.",
        })
    elif troponin > 0.04:
        alerts.append({
            "level": "warning",
            "icon": "⚠️",
            "title": f"Borderline Troponin Level ({troponin:.2f} ng/mL)",
            "message": "Above normal threshold (0.04 ng/mL). Clinical correlation with serial biomarkers recommended.",
        })

    # 2. ST Segment Alert
    st_level = float(clinical_data.get("ST_Segment_Level", 0.0))
    if st_level >= 0.10:
        alerts.append({
            "level": "critical",
            "icon": "🚨",
            "title": f"Significant ST-Segment Elevation ({st_level:+.2f} mm)",
            "message": "Meets electrophysiological criteria for acute myocardial injury / STEMI. Immediate emergency evaluation warranted.",
        })
    elif st_level <= -0.05:
        alerts.append({
            "level": "warning",
            "icon": "⚠️",
            "title": f"ST-Segment Depression ({st_level:+.2f} mm)",
            "message": "Subendocardial ischemia or reciprocal change indicative of regional coronary hypoperfusion.",
        })

    # 3. Blood Pressure Alert
    sbp = float(clinical_data.get("Systolic_BP", 120))
    dbp = float(clinical_data.get("Diastolic_BP", 80))
    if sbp >= 140 or dbp >= 90:
        alerts.append({
            "level": "warning",
            "icon": "⚠️",
            "title": f"Hypertensive Range ({sbp:.0f}/{dbp:.0f} mmHg)",
            "message": "Stage 2 Hypertension criteria met. Chronic elevated afterload exacerbates myocardial strain.",
        })

    # 4. Heart Rate Alert
    hr = float(clinical_data.get("Heart_Rate", 75))
    if hr >= 100:
        alerts.append({
            "level": "warning",
            "icon": "⚠️",
            "title": f"Sinus Tachycardia ({hr:.0f} bpm)",
            "message": "Resting heart rate >= 100 bpm accelerates cardiac workload and oxygen consumption.",
        })
    elif hr < 60:
        alerts.append({
            "level": "info",
            "icon": "ℹ️",
            "title": f"Sinus Bradycardia ({hr:.0f} bpm)",
            "message": "Resting heart rate < 60 bpm. Observe for chronotropic incompetence or nodal disease.",
        })

    # 5. QRS Duration Alert
    qrs = float(clinical_data.get("QRS_Duration", 100))
    if qrs > 120:
        alerts.append({
            "level": "warning",
            "icon": "⚠️",
            "title": f"Conduction Delay / Wide QRS ({qrs:.0f} ms)",
            "message": "QRS duration exceeds 120 ms, consistent with intraventricular conduction defect or bundle branch block.",
        })

    # 6. QT Interval Alert
    qt = float(clinical_data.get("QT_Interval", 400))
    if qt > 450:
        alerts.append({
            "level": "warning",
            "icon": "⚠️",
            "title": f"Prolonged QT Interval ({qt:.0f} ms)",
            "message": "Elevated risk of malignant ventricular arrhythmias and Torsades de Pointes.",
        })

    # 7. Glucose Alert
    glucose = float(clinical_data.get("Glucose", 90))
    if glucose >= 126:
        alerts.append({
            "level": "info",
            "icon": "ℹ️",
            "title": f"Elevated Fasting Glucose ({glucose:.0f} mg/dL)",
            "message": "Fasting blood sugar in diabetic threshold (>= 126 mg/dL), a recognized micro- and macro-vascular risk factor.",
        })

    return alerts


def build_clinical_parameters_table(clinical_data: Dict[str, Any]) -> pd.DataFrame:
    """Build a complete clinical tabular summary with normal reference intervals and status."""
    rows = []
    for key, val in clinical_data.items():
        if key == "Gender":
            rows.append({
                "Parameter": "Biological Gender",
                "Patient Value": str(val),
                "Reference Interval": "Male / Female",
                "Status": "Recorded",
            })
            continue

        meta = CLINICAL_REFERENCE_BOUNDS.get(key)
        if not meta:
            continue

        fval = float(val)
        n_min = meta["norm_min"]
        n_max = meta["norm_max"]
        unit = meta["unit"]

        if fval > n_max:
            if key in ["Troponin", "ST_Segment_Level"] and fval >= (0.40 if key == "Troponin" else 0.10):
                status = "CRITICAL HIGH 🚨"
            else:
                status = "ELEVATED ⚠️"
        elif fval < n_min:
            status = "LOW / DEPRESSED ⚠️"
        else:
            status = "NORMAL ✅"

        rows.append({
            "Parameter": meta["name"],
            "Patient Value": f"{fval:.2f} {unit}",
            "Reference Interval": f"{n_min} - {n_max} {unit}",
            "Status": status,
        })

    return pd.DataFrame(rows)


def generate_exportable_report(
    result: Dict[str, Any],
    clinical_data: Dict[str, Any],
    mode: str,
    active_temp: float,
    apply_calibration: bool,
    image_label: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble a serializable diagnostic test report JSON object."""
    probs_dict = {
        CLASS_NAMES[i]: round(float(result["probs"][i]) * 100.0, 2)
        for i in range(len(CLASS_NAMES))
    }
    alerts = create_clinical_alerts(clinical_data)

    report = {
        "report_metadata": {
            "title": "Explainable Multimodal ECG Diagnostic Test Report",
            "inference_mode": mode,
            "softmax_temperature": round(float(active_temp), 4),
            "is_calibrated": bool(apply_calibration and active_temp != 1.0),
            "input_ecg_source": image_label or "User Uploaded Scan",
        },
        "diagnostic_prediction": {
            "primary_classification": result["predicted_class"],
            "confidence_percent": round(result["confidence"] * 100.0, 2),
            "severity_level": CLASS_SEVERITY[result["predicted_class"]][0],
            "class_probabilities_percent": probs_dict,
        },
        "patient_parameters": clinical_data,
        "clinical_alerts_count": len(alerts),
        "clinical_alerts": alerts,
        "compliance_notice": (
            "Research-grade prototype. Demonstrates explainable multimodal machine learning on "
            "paper ECG scans and clinical variables. Not certified for medical diagnosis or triage."
        ),
    }
    return report


def generate_markdown_report(
    result: Dict[str, Any],
    clinical_data: Dict[str, Any],
    mode: str,
    active_temp: float,
    apply_calibration: bool,
    image_label: Optional[str] = None,
) -> str:
    """Generate a clean clinical diagnostic test report in Markdown format."""
    pred_class = result["predicted_class"].replace("_", " ")
    conf = result["confidence"] * 100.0
    severity = CLASS_SEVERITY[result["predicted_class"]][0]
    cal_info = f"Calibrated (T = {active_temp:.4f})" if (apply_calibration and active_temp != 1.0) else "Raw Softmax (T = 1.0)"
    alerts = create_clinical_alerts(clinical_data)

    lines = [
        "# CLINICAL DIAGNOSTIC TEST REPORT",
        "**System**: Explainable Multimodal ECG Intelligence System (Research Prototype)",
        f"**Inference Track**: `{mode}` | **Temperature Calibration**: `{cal_info}`",
        f"**Input ECG Scan**: `{image_label or 'Uploaded 12-Lead Paper ECG'}`",
        "",
        "---",
        "",
        "## 1. PRIMARY DIAGNOSTIC FINDINGS",
        f"- **Primary Diagnostic Classification**: **{pred_class}**",
        f"- **Assigned Clinical Risk Tier**: **{severity}**",
        f"- **Model Predictive Confidence**: **{conf:.2f}%**",
        "",
        "### Multiclass Probability Distribution",
        "| Pathology Class | Probability (%) | Status |",
        "| :--- | :--- | :--- |",
    ]

    for idx, cname in enumerate(CLASS_NAMES):
        p = float(result["probs"][idx]) * 100.0
        status = "**Primary Diagnosis**" if cname == result["predicted_class"] else "Differential"
        lines.append(f"| {cname.replace('_', ' ')} | {p:.2f}% | {status} |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. CLINICAL & ELECTROPHYSIOLOGICAL PARAMETERS",
        "| Parameter | Patient Value | Reference Interval | Clinical Status |",
        "| :--- | :--- | :--- | :--- |",
    ])

    for key, val in clinical_data.items():
        if key == "Gender":
            lines.append(f"| Biological Gender | {val} | Male / Female | Baseline |")
            continue
        meta = CLINICAL_REFERENCE_BOUNDS.get(key)
        if not meta:
            continue
        fval = float(val)
        unit = meta["unit"]
        n_min, n_max = meta["norm_min"], meta["norm_max"]
        if fval > n_max:
            status = "CRITICAL ELEVATION" if (key in ["Troponin", "ST_Segment_Level"] and fval >= (0.40 if key == "Troponin" else 0.10)) else "ELEVATED"
        elif fval < n_min:
            status = "DEPRESSED / LOW"
        else:
            status = "NORMAL"
        lines.append(f"| {meta['name']} | {fval:.2f} {unit} | {n_min} - {n_max} {unit} | {status} |")

    if alerts:
        lines.extend([
            "",
            "---",
            "",
            "## 3. CLINICAL ALERTS & DIAGNOSTIC FLAGS",
        ])
        for a in alerts:
            lines.append(f"- **{a['icon']} {a['title']}**: {a['message']}")

    lines.extend([
        "",
        "---",
        "",
        "## 4. REGULATORY & MEDICAL DISCLAIMER",
        "This report is generated by an experimental research system. It is not an FDA/CE-cleared diagnostic device "
        "and must not be used as the sole basis for clinical diagnosis, pharmacotherapy, or triage decisions.",
    ])

    return "\n".join(lines)

