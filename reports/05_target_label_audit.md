# Phase 5: Target Label Audit & Clinical Semantics Report

## 1. Ground Truth Target Definitions
Based on official documentation (`Multimodal_Cardiovascular_Risk_Dataset_Overview_Final.pdf`), the dataset supports **four distinct diagnostic categories**:
1. **Normal (Encoded: 0)**: 12-lead ECG traces and clinical parameters within normal physiological limits.
2. **Abnormal Heartbeat (Encoded: 1)**: Cardiac arrhythmia / irregular rhythm disturbances (e.g., atrial premature beats, ectopic contractions, tachycardia/bradycardia patterns).
3. **Myocardial Infarction (Encoded: 2)**: Acute myocardial infarction showing ST-elevation or pathological Q-wave ischemic injury patterns.
4. **Post-MI History (Encoded: 3)**: Chronic, post-infarction status showing established pathological changes or prior tissue necrosis.

## 2. Clinical Task Formulation: Diagnostic Classification vs. Future Risk Prediction
> [!IMPORTANT]
> **Scientific Distinction (Rule 6)**: The dataset does **not** contain longitudinal follow-up data, time-to-event indicators, or future hazard outcomes. It records *current diagnostic state* and *prior cardiac history*. Consequently, the scientific task is strictly **ECG-Condition & Diagnostic-Category Classification**, not future cardiac-event forecasting.

## 3. Multimodal Class Distribution Comparison

| Target Label | Encoded ID | CSV Records (N=4,327) | CSV % | Image Files (N=928) | Image % | Unique Scans (N=491) | Unique % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Normal` | `0` | 1,113 | 25.72% | 284 | 30.60% | 142 | 28.92% |
| `Abnormal_Heartbeat` | `1` | 1,092 | 25.24% | 233 | 25.11% | 233 | 47.45% |
| `Myocardial_Infarction` | `2` | 1,031 | 23.83% | 239 | 25.75% | 30 | 6.11% |
| `Post_MI_History` | `3` | 1,091 | 25.21% | 172 | 18.53% | 86 | 17.52% |
| **Total** | | **4,327** | **100.00%** | **928** | **100.00%** | **491** | **100.00%** |

## 4. Class Imbalance and Duplication Assessment
- **CSV Distribution**: Artificially balanced at ~25% per class (Normal: 25.7%, Abnormal: 25.2%, Post-MI: 25.2%, MI: 23.8%).
- **Image Collection Imbalance**: In unique physical images, substantial imbalance exists:
  - `Abnormal_Heartbeat`: 233 unique scans (47.45%)
  - `Normal`: 142 unique scans (28.92%)
  - `Post_MI_History`: 86 unique scans (17.52%)
  - `Myocardial_Infarction`: 30 unique scans (6.11%)
- The 239 total MI image files were created by duplicating 30 unique scans 8 times, which would lead to catastrophic data leakage if split naively.
