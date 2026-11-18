"""
Automated Software Tests for Interactive Dashboard Visualizations
Verifies:
1. Plotly diagnostic gauge structure and styling
2. Interactive probability bar chart and hover templates
3. Polar biomarker radar chart and physiological corridor
4. Live Lead II ECG electrophysiological waveform synthesis
5. Interactive XAI feature attribution charts
6. Clinical alert detection against diagnostic thresholds
7. Exportable test report serialization (JSON & Markdown)
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inference import CLINICAL_FEATURE_DEFAULTS
from dashboard_visualizations import (
    CLASS_COLORS,
    CLASS_NAMES,
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


def test_diagnostic_gauge_creation():
    """Verify gauge indicator figure properties."""
    fig = create_diagnostic_gauge(
        predicted_class="Myocardial_Infarction",
        confidence=95.4,
        active_temp=0.4378,
        is_calibrated=True,
    )
    assert fig is not None
    assert len(fig.data) == 1
    indicator = fig.data[0]
    assert indicator.type == "indicator"
    assert indicator.value == 95.4
    assert indicator.gauge.axis.range == (0, 100)
    assert indicator.gauge.bar.color == CLASS_COLORS["Myocardial_Infarction"]


def test_probability_bar_chart_creation():
    """Verify interactive probability distribution bar chart."""
    probs = np.array([0.05, 0.10, 0.80, 0.05])
    fig = create_probability_bar_chart(probs, predicted_class="Myocardial_Infarction")
    assert fig is not None
    assert len(fig.data) == 1
    bar = fig.data[0]
    assert bar.type == "bar"
    assert len(bar.x) == 4
    assert np.isclose(bar.x[2], 80.0)
    assert "<b>%{y}</b>" in bar.hovertemplate


def test_biomarker_radar_chart():
    """Verify polar radar chart and physiological corridor."""
    fig = create_biomarker_radar_chart(CLINICAL_FEATURE_DEFAULTS)
    assert fig is not None
    # Expect 3 traces: Normal Upper, Normal Lower, Patient Profile
    assert len(fig.data) == 3
    for trace in fig.data:
        assert trace.type == "scatterpolar"
    patient_trace = fig.data[2]
    assert patient_trace.name == "Patient Clinical Profile"
    assert len(patient_trace.r) == 9  # 8 radar keys + 1 closed loop point
    assert patient_trace.r[0] == patient_trace.r[-1]  # Loop closure


def test_ecg_waveform_simulation_dynamic_response():
    """Verify synthetic 2-cycle Lead II ECG waveform reacts to parameter shifts."""
    base_data = dict(CLINICAL_FEATURE_DEFAULTS)
    base_data["ST_Segment_Level"] = 0.00
    base_data["Heart_Rate"] = 60
    base_data["RR_Interval"] = 1000.0

    fig_normal = create_ecg_waveform_simulation(base_data)
    assert fig_normal is not None
    trace_normal = fig_normal.data[0]
    y_normal = np.array(trace_normal.y)

    # Shift ST segment to STEMI elevation (+0.35 mm -> +0.035 mV plateau)
    stemi_data = dict(base_data)
    stemi_data["ST_Segment_Level"] = 0.35
    fig_stemi = create_ecg_waveform_simulation(stemi_data)
    trace_stemi = fig_stemi.data[0]
    y_stemi = np.array(trace_stemi.y)

    assert len(y_normal) == len(y_stemi)
    # The ST-elevated waveform must have higher mean/integral during repolarization
    assert np.mean(y_stemi) > np.mean(y_normal)


def test_interactive_attribution_chart():
    """Verify interactive horizontal attribution bar chart."""
    df = pd.DataFrame({
        "Feature": ["ST_Segment_Level", "Troponin", "Glucose"],
        "Scaled_Value": [1.5, 2.1, -0.4],
        "Attribution": [0.42, 0.35, -0.15],
        "Abs_Attribution": [0.42, 0.35, 0.15],
        "Kind": "SHAP Value",
    })
    fig = create_interactive_attribution_chart(df, "SHAP Baseline")
    assert fig is not None
    assert len(fig.data) == 1
    bar = fig.data[0]
    assert len(bar.y) == 3
    assert bar.orientation == "h"


def test_clinical_alerts_detection():
    """Verify clinical alerts trigger on critical cardiac thresholds."""
    critical_data = dict(CLINICAL_FEATURE_DEFAULTS)
    critical_data["Troponin"] = 1.25        # Critical elevation >= 0.40
    critical_data["ST_Segment_Level"] = 0.25# STEMI elevation >= 0.10
    critical_data["Systolic_BP"] = 165      # Stage 2 Hypertension >= 140
    critical_data["Heart_Rate"] = 120       # Tachycardia >= 100

    alerts = create_clinical_alerts(critical_data)
    assert len(alerts) >= 4
    titles = [a["title"] for a in alerts]
    assert any("Troponin" in t for t in titles)
    assert any("ST-Segment" in t for t in titles)
    assert any("Hypertensive" in t for t in titles)
    assert any("Tachycardia" in t for t in titles)


def test_report_export_serialization():
    """Verify exportable diagnostic test report generation in JSON and Markdown."""
    dummy_result = {
        "probs": np.array([0.02, 0.03, 0.92, 0.03]),
        "predicted_idx": 2,
        "predicted_class": "Myocardial_Infarction",
        "confidence": 0.92,
    }
    report_json = generate_exportable_report(
        result=dummy_result,
        clinical_data=CLINICAL_FEATURE_DEFAULTS,
        mode="Multimodal Fusion",
        active_temp=0.4378,
        apply_calibration=True,
        image_label="Held-out Sample: Myocardial Infarction — MI(1).jpg",
    )
    assert report_json["diagnostic_prediction"]["primary_classification"] == "Myocardial_Infarction"
    assert report_json["diagnostic_prediction"]["confidence_percent"] == 92.0
    assert "Troponin" in report_json["patient_parameters"]

    # Must be JSON-serializable
    serialized = json.dumps(report_json)
    assert len(serialized) > 0

    report_md = generate_markdown_report(
        result=dummy_result,
        clinical_data=CLINICAL_FEATURE_DEFAULTS,
        mode="Multimodal Fusion",
        active_temp=0.4378,
        apply_calibration=True,
    )
    assert "# CLINICAL DIAGNOSTIC TEST REPORT" in report_md
    assert "Myocardial Infarction" in report_md
    assert "CRITICAL RISK" in report_md

