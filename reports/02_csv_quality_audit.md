# Phase 2: CSV Data Quality Audit Report

## 1. Overview and Schema
- **CSV Path**: `data\raw\cardiovascular_dataset.csv`
- **Record Count**: 4,327 rows
- **Column Count**: 21 columns
- **Exact Full-Row Duplicates**: 0
- **Feature-Only Duplicate Rows (excluding ECG_Image_Path)**: 0

## 2. Target Label Distribution

| Target Class | Count | Percentage |
| :--- | :--- | :--- |
| `Normal` | 1,113 | 25.72% |
| `Abnormal_Heartbeat` | 1,092 | 25.24% |
| `Post_MI_History` | 1,091 | 25.21% |
| `Myocardial_Infarction` | 1,031 | 23.83% |
| **Total** | **4,327** | **100.00%** |

## 3. Data Integrity and Missingness
- **Null values**: Zero missing/null cells detected across all 21 columns.
- **Whitespace / Encoding Issues**: No leading/trailing whitespace anomalies detected in string columns (`Gender`, `ECG_Image_Path`, `Target`).
- **Data Types**: 14 integer features, 4 float features, 3 string features.

## 4. Critical Biomedical & Statistical Audit Findings
### A. Synthetic / Uniform Distribution Anomaly
Statistical analysis reveals that numerical variables exhibit flat distributions across integer boundaries, indicating that the tabular CSV was synthetically generated within uniform ranges rather than sampled from authentic human patients:
- `Age`: spans integers [30, 84] with mean 57.5, uniform distribution.
- `Heart_Rate`: spans integers [55, 139] with mean 97.4.
- `Systolic_BP`: spans integers [90, 179] with mean 134.0.
- `Diastolic_BP`: spans integers [60, 109] with mean 84.4.
- `Cholesterol`: spans integers [150, 319] with mean 234.2.
- `Glucose`: spans integers [70, 199] with mean 134.8.

### B. Physiological Decoupling
In genuine physiological datasets, fundamental biological correlations are preserved. Here, critical physiological pairs show near-zero correlation:
- **Heart Rate vs. RR Interval Correlation**: `-0.0089` (Expected in human biology: strong negative correlation, r ≈ -0.90 to -0.98, because HR = 60,000 / RR_ms).
- **Systolic BP vs. Diastolic BP Correlation**: `-0.0223` (Expected in human biology: moderate to strong positive correlation, r ≈ +0.65 to +0.80).
- **HRV SDNN vs. HRV RMSSD Correlation**: `0.0231` (Expected in human biology: positive correlation, r > +0.70).
- **Biomarker Invariance Across Disease Targets**:
  - In acute Myocardial Infarction (MI), cardiac Troponin is markedly elevated. In this CSV, Troponin averages **1.26 ng/mL** in MI, **1.28 ng/mL** in Normal, **1.29 ng/mL** in Abnormal Heartbeat, and **1.25 ng/mL** in Post-MI. There is zero clinical differentiation.
  - ST-segment level averages **0.154 mm** in MI and **0.148 mm** in Normal (clinically indistinguishable).

### C. Target Leakage Assessment
Because clinical features were uniformly generated without class conditional variance, there is **no target leakage** from clinical variables. However, the `ECG_Image_Path` column (`ecg_1.jpg` ... `ecg_4327.jpg`) does not correspond to any actual files on disk.
