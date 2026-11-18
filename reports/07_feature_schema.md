# Phase 7: Clinical Feature Taxonomy & Schema Specification

## 1. Feature Categorization

| Feature Name | Clinical Aspect | Variable Type | Expected Range | Documented Biological Meaning |
| :--- | :--- | :--- | :--- | :--- |
| `Record_ID` | Identifier | String (Alphanumeric) | `REC_00001` - `REC_04327` | Primary unique record key |
| `Age` | Demographic | Continuous (Integer) | 30 - 84 years | Patient chronological age |
| `Gender` | Demographic | Binary Categorical | Female, Male | Patient biological sex |
| `Heart_Rate` | Hemodynamic | Continuous (Integer) | 55 - 139 bpm | Beats per minute |
| `Systolic_BP` | Hemodynamic | Continuous (Integer) | 90 - 179 mmHg | Systolic blood pressure |
| `Diastolic_BP` | Hemodynamic | Continuous (Integer) | 60 - 109 mmHg | Diastolic blood pressure |
| `Cholesterol` | Laboratory | Continuous (Integer) | 150 - 319 mg/dL | Serum cholesterol level |
| `Glucose` | Laboratory | Continuous (Integer) | 70 - 199 mg/dL | Blood glucose concentration |
| `Troponin` | Laboratory | Continuous (Float) | 0.001 - 2.500 ng/mL | Cardiac biomarker for myocardial necrosis |
| `PR_Interval` | ECG Metric | Continuous (Integer) | 120 - 219 ms | AV nodal conduction time |
| `QRS_Duration` | ECG Metric | Continuous (Integer) | 70 - 159 ms | Ventricular depolarization duration |
| `QT_Interval` | ECG Metric | Continuous (Integer) | 300 - 499 ms | Total ventricular electrical activity |
| `P_Wave_Duration` | ECG Metric | Continuous (Integer) | 80 - 119 ms | Atrial depolarization duration |
| `T_Wave_Duration` | ECG Metric | Continuous (Integer) | 120 - 199 ms | Ventricular repolarization duration |
| `ST_Segment_Level`| ECG Metric | Continuous (Float) | -0.200 - +0.500 mm | Elevation/depression vs isoelectric baseline |
| `RR_Interval` | ECG Metric | Continuous (Integer) | 600 - 1199 ms | Inter-beat interval duration |
| `HRV_SDNN` | ECG Metric | Continuous (Integer) | 20 - 119 ms | Standard deviation of NN intervals |
| `HRV_RMSSD` | ECG Metric | Continuous (Integer) | 15 - 99 ms | Root mean square of successive differences |
| `R_Peak_Amplitude`| ECG Metric | Continuous (Float) | 0.50 - 3.00 mV | Ventricular depolarisation voltage amplitude |
| `S_Wave_Amplitude`| ECG Metric | Continuous (Float) | -2.50 - -0.10 mV | Late ventricular depolarisation voltage |
| `ECG_Image_Path` | Metadata / Identifier | String | `ecg_1.jpg` - `ecg_4327.jpg` | Referenced image pointer (excluded from tabular inputs) |
| `Target` | Ground Truth | String | 4 classes | Diagnostic class label |
| `Target_Encoded` | Ground Truth | Categorical (Integer) | 0, 1, 2, 3 | Encoded target identifier |
