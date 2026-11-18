# Explainable Multimodal ECG Intelligence for Early Cardiac Risk Assessment

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6%2Bcu124-EE4C2C.svg)](https://pytorch.org/)
[![Tests Passing](https://img.shields.io/badge/tests-16%2F16%20passed-brightgreen.svg)]()
[![Status](https://img.shields.io/badge/status-Training%20%26%20Evaluation%20Complete-success.svg)]()

A research-grade deep learning and multimodal intelligence system designed to classify 4-class cardiac pathology (`Normal`, `Abnormal Heartbeat`, `Myocardial Infarction`, and `Post-MI History`) from **12-lead paper ECG scans** and **clinical tabular biomarkers**, featuring dual-track explainability (**Grad-CAM** & **SHAP**).

---

## 🚀 Key Performance Highlights

- **ECG Vision Model (ResNet-18)**:
  - **Validation Macro-F1**: `0.9430` | **Validation Accuracy**: `95.10%` | **OVR ROC-AUC**: `0.9799`
  - **Myocardial Infarction Sensitivity**: `100.0%` (40/40 correctly identified) | **Specificity**: `99.03%`
  - **Normal Sensitivity**: `100.0%` (42/42 correctly identified)
- **Multimodal Late-Fusion (Held-out Test Cohort)**:
  - **Test Accuracy**: `87.59%` (vs. 86.13% ECG-Only and 18.25% Clinical-Only)
  - **Macro-F1**: `0.8651` | **OVR ROC-AUC**: `0.9716`
- **Zero-Leakage Guarantee**:
  - Cryptographic MD5 hash partitioning ensuring 0 duplicate scans span Train, Validation, or Test splits.
- **Explainable AI (XAI)**:
  - Real-time Grad-CAM visual heatmaps highlighting ischemic ST-segment regions and rhythm anomalies.
  - TreeExplainer SHAP game-theoretic feature importance on clinical factors.

---

## 📁 Repository Structure

```
ECG/
|-- configs/
|   |-- training_config.yaml           # Master training hyperparameters and paths
|   `-- preprocessing_config.json      # Preprocessing configurations
|-- data/
|   |-- processed/                     # Leakage-audited CSV splits and manifests
|   `-- raw/                           # Raw cardiovascular dataset
|-- models/
|   |-- ecg_model_best.pt              # Trained PyTorch ResNet-18 checkpoint (Val F1: 0.9430)
|   |-- clinical_model.joblib          # Trained Random Forest tabular classifier
|   |-- multimodal_model_best.pt       # Trained Multimodal Late-Fusion checkpoint (Val F1: 0.9370)
|   |-- clinical_preprocessor.joblib   # Fitted training-set feature transformers
|   `-- calibration_config.json        # Validation-fitted temperature scaling parameters
|-- reports/
|   |-- 01_training_readiness_audit.md # Cryptographic hash & split audit
|   |-- 02_ecg_input_and_architecture.md # Formal 2D-CNN & tabular specifications
|   |-- 03_ecg_training_curves.png     # Loss, accuracy, and macro-F1 curves
|   |-- 04_ecg_validation_report.md    # Vision model validation metrics & confusion matrix
|   |-- 04_ecg_confusion_matrix.png    # Vision model confusion matrix exhibit
|   |-- 05_clinical_model_report.md    # Tabular baselines benchmark report
|   |-- 06_model_comparison.md         # Multi-model head-to-head test benchmark
|   |-- 06_model_comparison.csv        # Fair test cohort comparison metrics
|   |-- 07_explainability_report.md    # Comprehensive Grad-CAM & SHAP findings
|   |-- 07_ecg_attribution_examples/   # Generated visual attribution overlays
|   |-- 07_shap_summary.png            # SHAP feature importance summary plot
|   |-- 08_calibration_report.md       # Temperature scaling & ECE reliability report
|   |-- 08_reliability_diagram.png     # Reliability diagram exhibit
|   |-- 09_software_tests.md           # Automated Pytest suite verification report
|   `-- FINAL_MODEL_TRAINING_REPORT.md # Master supervisor report
|-- src/
|   |-- audit_training_data.py         # Step 1: Pre-training readiness audit
|   |-- train_ecg_model.py             # Step 3: ECG Vision model training (Track A)
|   |-- train_clinical_model.py        # Step 4: Clinical tabular baselines (Track B)
|   |-- train_multimodal_model.py      # Step 5: Multimodal Late-Fusion training (Track C)
|   |-- evaluate_models.py             # Step 5: Fair test cohort benchmark comparison
|   |-- explain_ecg_model.py           # Step 6: Grad-CAM visual heatmap generator
|   |-- explain_clinical_model.py      # Step 6: SHAP tabular feature importance
|   |-- explain_multimodal_model.py    # Step 6: Joint multimodal XAI generator
|   |-- evaluate_calibration.py        # Step 7: Probability calibration & ECE
|   |-- config_utils.py                # YAML config & deterministic seeding helpers
|   |-- dashboard_visualizations.py    # Interactive Plotly visualizations & test report engine
|   `-- inference.py                   # Shared dashboard runtime & XAI helpers
|-- tests/
|   |-- test_pipeline.py               # Step 9: Pytest automated test suite
|   `-- test_dashboard_visualizations.py # Pytest suite for interactive visualization engine
|-- app.py                             # Step 8: Interactive Streamlit web application
|-- requirements.txt                   # Production dependencies
`-- README.md
```

---

## 💻 Quickstart & Reproduction

### 1. Environment Setup
```powershell
# Using uv (recommended)
uv venv --python 3.11 .venv
.\.venv\Scripts\activate
uv pip install -r requirements.txt
```

### 2. Run Entire Training & Evaluation Pipeline
```powershell
# Step 1: Audit training readiness & zero-leakage constraints
python src/audit_training_data.py

# Step 3: Train ECG Vision Model (Track A)
python src/train_ecg_model.py

# Step 4: Train Clinical Tabular Baselines (Track B)
python src/train_clinical_model.py

# Step 5: Train Multimodal Late Fusion & Benchmark (Track C)
python src/train_multimodal_model.py
python src/evaluate_models.py

# Step 6: Generate Explainability (Grad-CAM & SHAP)
python src/explain_ecg_model.py
python src/explain_clinical_model.py
python src/explain_multimodal_model.py

# Step 7: Evaluate Probability Calibration
python src/evaluate_calibration.py

# Step 9: Run Automated Software Tests (16/16 passing)
pytest tests/ -v
```

### 3. Launch Interactive Web Dashboard
```powershell
streamlit run app.py
```
*Features dynamic Plotly gauges, multi-class probability distributions with tooltips, live Lead II electrophysiological waveform synthesis responding to slider adjustments, physiological biomarker radar profiling, interactive Grad-CAM opacity blend control, and downloadable clinical diagnostic test reports (JSON & Markdown).*

---

## ⚖️ Ethical & Clinical Disclaimer
This system is an **experimental research prototype** designed to demonstrate deep learning and explainable AI techniques on digitized 12-lead ECG scans and tabular biomarkers. It is **NOT** a certified medical device and must **NOT** be used as a substitute for professional clinical diagnosis, emergency triage, or physician judgment. Heatmaps and SHAP values indicate mathematical model attribution, not physiological causality.
