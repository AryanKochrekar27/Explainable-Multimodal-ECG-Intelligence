# FINAL DATA PREPROCESSING REPORT FOR PROJECT SUPERVISOR & GUIDE

**Project Title**: Explainable ECG Intelligence for Early Cardiac Risk Assessment  
**Stage**: Data Preprocessing, Biomedical Quality Audit, Zero-Leakage Partitioning, and Multimodal Linkage Verification  
**Date**: October 19, 2026  
**Final Preprocessing Decision**: **READY WITH DOCUMENTED LIMITATIONS (Unimodal Tabular & Vision Pipelines Ready; Joint Multimodal Paired Pipeline Blocked Pending Ground-Truth Key)**

---

## 1. Project Objective & Context

The goal of this research project is to develop an interpretable artificial intelligence system that combines electrocardiogram (ECG) data with clinical variables to classify cardiac pathology (Normal, Arrhythmia / Abnormal Heartbeat, Myocardial Infarction, and Post-MI History) and explain predictions to clinicians using attribution methods (Grad-CAM and SHAP).

Before model training, a thorough biomedical data preprocessing audit was mandated to ensure data integrity, prevent target leakage, audit physiological plausibility, and resolve the relationship between tabular CSV records and physical ECG image scans.

---

## 2. Dataset Source & Workspace Discovery

- **Source Archive**: The workspace contains the extracted *Multimodal Cardiovascular Dataset*, comprising:
  1. A primary tabular file: `initial_dataset/cardiovascular_dataset.csv` (preserved in `data/raw/cardiovascular_dataset.csv`).
  2. An image directory: `initial_dataset/ECG Images/ECG Images/` containing 928 scans organized in 4 diagnostic subfolders.
  3. Author documentation: `Multimodal_Cardiovascular_Risk_Dataset_Overview_Final.pdf`, survey spreadsheet, and project documentation.
- **Initial Counts Discovered**:
  - **CSV Rows**: 4,327 records across 21 columns.
  - **Physical Image Files**: 928 JPEG files (dimensions: 2213 × 1572 pixels).
  - **Diagnostic Classes**: 4 documented categories (`Normal`, `Abnormal_Heartbeat`, `Myocardial_Infarction`, `Post_MI_History`).

---

## 3. CSV Quality & Biomedical Plausibility Findings

An exhaustive statistical and clinical audit of `cardiovascular_dataset.csv` revealed critical insights:
1. **Zero Missingness**: 0 null or missing cells across all 4,327 rows and 21 columns.
2. **Synthetic Uniformity**: All 18 numerical features exhibit flat, uniform distributions bounded across integer/decimal limits (confirmed via Kolmogorov-Smirnov goodness-of-fit testing).
3. **Physiological Decoupling**:
   - **Heart Rate vs. RR Interval**: Correlation is **r = -0.0089** (In human cardiovascular physiology, HR and RR interval share an inverse relationship, $HR = 60,000 / RR_{ms}$, with r ≈ -0.95).
   - **Blood Pressure Decoupling**: Systolic and Diastolic BP correlation is **r = -0.0223** (normally r ≈ +0.70).
   - **Biomarker Invariance**: Cardiac Troponin averages **1.26 ng/mL** in acute Myocardial Infarction and **1.28 ng/mL** in Normal patients (clinically indistinguishable).
4. **Target Leakage**: Because features were synthetically generated without target-conditional divergence, there is zero target leakage in clinical features.

---

## 4. ECG Image Quality & Duplication Findings

All 928 image files were systematically audited using cryptographic MD5 hashing, decodability checks, and pixel analysis:
1. **Integrity & Decodability**: 100% of images (928/928) decode successfully without file corruption or blank frames.
2. **Image Uniformity**: Every scan is a high-resolution 12-lead ECG printout from an EDAN SE-3 / SEMIP machine measuring 2213 × 1572 pixels in 3-channel RGB.
3. **Exact Duplication Pattern**:
   - MD5 hash auditing revealed only **491 unique ECG scans** across the 928 files.
   - `myocardial_infarction_ecg_images`: Contains **30 unique scans** duplicated ~8 times each to total 239 files.
   - `normal_ecg_images`: Contains **142 unique scans** duplicated 2 times each to total 284 files.
   - `post_mi_history_ecg_images`: Contains **86 unique scans** duplicated 2 times each to total 172 files.
   - `abnormal_heartbeat_ecg_images`: Contains **233 unique scans** (0 duplicates).
   - Zero duplicate scans appear across different class folders.

---

## 5. CSV-to-Image Linkage Investigation (Scientific Resolution)

The investigation evaluated direct filename correspondence, metadata keys, and sequential pairing hypotheses:
- **Empirical Finding**: The CSV column `ECG_Image_Path` contains fictional pointers (`ecg_1.jpg` to `ecg_4327.jpg`), whereas actual image files are named `HB(1).jpg`, `MI(1).jpg`, etc.
- **Evidence Search**: No cross-reference table, lookup dictionary, or deterministic rule exists in the dataset documentation or literature.
- **Scientific Decision (Non-Negotiable Rules 1, 2, and 9)**:
  - We **refused to invent false pairings** by naive alphabetical or numerical sorting.
  - Doing so would pair the ECG scan of patient A with the synthetic clinical variables of record B, corrupting multimodal research validity.
  - All 4,327 CSV records and 928 image files are documented as unmatched in `reports/04_unmatched_csv_records.csv` and `reports/04_unmatched_images.csv`.
  - The multimodal paired dataset is documented as **BLOCKED — GROUND-TRUTH LINKAGE MISSING**.

---

## 6. Preprocessing Operations Completed

### 6.1 Clinical Tabular Pipeline
- **Identifiers**: Added explicit `Record_ID` (`REC_00001` - `REC_04327`).
- **Target Encoding**: Mapped classes to `Normal: 0`, `Abnormal_Heartbeat: 1`, `Myocardial_Infarction: 2`, `Post_MI_History: 3`.
- **Partitioning**: Stratified 70% Train (N=3,028), 15% Validation (N=649), 15% Test (N=650) using random seed 42.
- **Pipeline Fitting**: Fitted a `ColumnTransformer` with `StandardScaler` (continuous features) and `OneHotEncoder` (Gender) **strictly on the training set**.
- **Transformation**: Applied the fitted pipeline to Val and Test partitions without test-set leakage.
- **Artifacts Saved**: `X_train_clinical.csv`, `X_val_clinical.csv`, `X_test_clinical.csv`, `y_train.csv`, `y_val.csv`, `y_test.csv`, and `models/clinical_preprocessor.joblib`.

### 6.2 ECG Image Computer Vision Pipeline
- **Group-Aware Hash Partitioning**: Grouped images by cryptographic MD5 hash to guarantee that identical duplicated scans are assigned exclusively to a single partition, completely preventing data leakage across Train, Val, and Test splits.
- **Image Preprocessor Module (`src/preprocess_images.py`)**: Standardized Lanczos downsampling to 224 × 224 × 3 and [0, 1] float32 normalization.
- **Preservation of Grid & Traces**: Strictly prohibited destructive augmentations (horizontal/vertical flips, aggressive rotations) that corrupt cardiac conduction polarity.

---

## 7. Automated Quality & Leakage Checklist

| Verification Condition | Target Standard | Achieved Result | Audit Status |
| :--- | :--- | :--- | :--- |
| **Physical File Existence** | 100% on disk | 928 / 928 files verified | **PASS** |
| **Image Decodability** | 0 decode errors | 0 corruptions detected | **PASS** |
| **Target Label Integrity** | 100% mapped | 4,327 / 4,327 complete | **PASS** |
| **Preprocessor Leakage** | Fit on train only | Fitted strictly on X_train | **PASS** |
| **Feature Column Consistency**| Identical columns | 19 features identical across train, val, test | **PASS** |
| **Transformed Value Integrity**| 0 NaNs / 0 Infs | 0 NaNs, 0 Infs across all sets | **PASS** |
| **Record Disjointness** | 0 record overlap | Disjoint splits (Train: 3028, Val: 649, Test: 650) | **PASS** |
| **Duplicate Image Leakage** | 0 hash overlap | 0 duplicate image hashes cross splits | **PASS** |
| **Patient-Level Leakage** | No patient cross-split | Patient IDs omitted by dataset authors | **NOT VERIFIABLE** |
| **Multimodal Ground Linkage** | Verified correspondence| No author mapping key provided | **BLOCKED** |

---

## 8. Summary of Generated Project Deliverables

- **Processing Code**:
  - `src/audit_dataset.py` (Phase 1 Inventory)
  - `src/audit_csv.py` (Phase 2 CSV Quality Audit)
  - `src/audit_images.py` (Phase 3 Image Quality Audit & Contact Sheet)
  - `src/verify_mapping.py` (Phase 4 Linkage Verification)
  - `src/audit_target_labels.py` (Phase 5 Target Semantic Audit)
  - `src/preprocess_clinical.py` (Phase 6 & 7 Clinical Cleaning & Schema)
  - `src/preprocess_images.py` (Phase 8 Image Preprocessor Module)
  - `src/split_dataset.py` (Phase 9 Zero-Leakage Hash Partitioning)
  - `src/fit_clinical_pipeline.py` (Phase 10 Sklearn Pipeline Fitting)
  - `src/construct_multimodal_manifest.py` (Phase 11 Multimodal Audit)
  - `src/analyze_imbalance.py` (Phase 12 Class Imbalance & Loss Weights)
  - `src/validate_preprocessed_data.py` (Phase 13 Automated Checks)
- **Persisted Artifacts**:
  - `models/clinical_preprocessor.joblib`
  - `data/processed/clinical_clean.csv`
  - `data/processed/split_manifest.csv`
  - `data/processed/image_split_manifest.csv`
  - `data/processed/X_train_clinical.csv`, `X_val_clinical.csv`, `X_test_clinical.csv`
  - `data/processed/y_train.csv`, `y_val.csv`, `y_test.csv`
  - `data/processed/label_mapping.json`, `feature_columns.json`, `transformed_feature_names.json`
- **Audit Reports & Visualizations**:
  - `reports/01_dataset_inventory.md` & `.csv`
  - `reports/02_csv_quality_audit.md` & `.csv`
  - `reports/03_image_quality_audit.md` & `.csv`
  - `reports/03_image_contact_sheet.png`
  - `reports/04_image_record_mapping.md`
  - `reports/04_unmatched_csv_records.csv` & `04_unmatched_images.csv`
  - `reports/05_target_label_audit.md` & `05_class_distribution.csv`
  - `reports/06_clinical_cleaning_report.md`
  - `reports/07_feature_schema.md`
  - `reports/08_image_preprocessing.md` & `08_original_vs_processed.png`
  - `reports/09_split_and_leakage_audit.md`
  - `reports/10_final_dataset_validation.md`
  - `reports/11_class_imbalance.md`
  - `reports/12_final_quality_checks.md` & `quality_checks_summary.json`
  - `reports/FINAL_DATA_PREPROCESSING_REPORT.md`
- **Documentation & Configuration**:
  - `configs/preprocessing_config.json`
  - `requirements.txt`
  - `README_PREPROCESSING.md`

---

## 9. Final Readiness Decision & Academic Recommendation

**Decision**: **READY WITH DOCUMENTED LIMITATIONS**

### Path Forward for the Final-Year Project:
1. **ECG Vision Intelligence Stream**: Fully unblocked. The 491 unique, high-resolution 12-lead ECG scans are partitioned with zero duplicate leakage and are ready for deep learning training (e.g., ResNet-50 / EfficientNet / Vision Transformers) and explainability via Grad-CAM / Integrated Gradients.
2. **Clinical Tabular Intelligence Stream**: Fully unblocked. The 4,327 clinical records are preprocessed, scaled, and partitioned for baseline classification (e.g., XGBoost / MLP) and tabular SHAP interpretability, with documented notes regarding its synthetic distribution.
3. **Multimodal Fusion Stream**: Blocked from ad-hoc pairing until the dataset curators provide the exact ground-truth cross-reference key linking `ecg_X.jpg` to the physical ECG scans.
