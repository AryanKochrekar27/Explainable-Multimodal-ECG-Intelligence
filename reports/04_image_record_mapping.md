# Phase 4: Image-to-Record Mapping Verification Report

## 1. Executive Summary & Verification Decision
- **Total CSV Records**: 4,327
- **Total Physical ECG Images**: 928 (491 unique scans)
- **Direct Filename Matches**: 0 (0.0%)
- **Verified Ground-Truth Matches**: **0**
- **Unmatched CSV Records**: **4,327** (100%)
- **Unmatched ECG Images**: **928** (100%)
- **Mapping Verification Status**: **BLOCKED — NO GROUND-TRUTH LINKAGE**

## 2. Evidence-Based Investigation of Matching Hypotheses

### Hypothesis 1: Direct Filename Correspondence
- **Claim**: The CSV column `ECG_Image_Path` points directly to image files on disk.
- **Empirical Finding**: False. The CSV references filenames `ecg_1.jpg` through `ecg_4327.jpg`. Across the entire filesystem, zero files matching `ecg_*.jpg` exist. Physical image files follow the format `HB(1).jpg` to `HB(233).jpg`, `MI(1).jpg` to `MI(239).jpg`, `Normal(1).jpg` to `Normal(284).jpg`, and `PMI(1).jpg` to `PMI(172).jpg`.

### Hypothesis 2: Deterministic Transformation / Manifest Key
- **Claim**: A documented mapping table, metadata manifest, or deterministic conversion rule links CSV rows to image filenames.
- **Empirical Finding**: Refuted. Comprehensive audits of all project documentation (`Multimodal_Cardiovascular_Risk_Dataset_Overview_Final.pdf`, literature surveys, and underlying published literature) reveal that **no mapping dictionary, cross-reference table, or conversion algorithm exists**.

### Hypothesis 3: Sequential / Positional Alignment by Class
- **Claim**: Could records be paired based on class ordering or index sequence?
- **Empirical Finding**: **Scientifically Prohibited and Disproven**.
  - *Scientific Rule 2*: 'Never assume that filenames with similar numbers refer to the same patient or record.'
  - *Scientific Rule 1*: 'Never invent a relationship between a CSV record and an ECG image.'
  - *Numerical impossibility*: CSV contains 1,031 MI records, whereas disk contains only 30 unique MI images (replicated to 239). CSV contains 1,113 Normal records, whereas disk contains only 142 unique Normal images (replicated to 284). CSV contains 1,091 Post-MI records, whereas disk contains only 86 unique Post-MI images (replicated to 172). Arbitrary alignment would fabricate 4,327 false patient pairings and duplicate images across different synthetic records.
  - *Biological independence*: CSV numerical features exhibit a uniform synthetic distribution where Troponin and ST-segment levels do not correlate with cardiac pathology, whereas image files are authentic 12-lead ECG scans from the CPEIC hospital cohort.

## 3. Mandatory Scientific Action Taken
In strict adherence to Rule 9 (*'If a required relationship cannot be established reliably, stop the affected preprocessing stage, document the issue, and explain what evidence is required to resolve it'*):
1. **Zero multimodal pairings have been fabricated**.
2. All 4,327 CSV records and 928 image files are fully documented in `reports/04_unmatched_csv_records.csv` and `reports/04_unmatched_images.csv`.
3. Preprocessing of both modalities continues rigorously as independent unimodal pipelines (Clinical Tabular Pipeline and ECG Image Computer Vision Pipeline).

## 4. Evidence Required to Resolve Linkage
To enable true multimodal pairing in future work, the dataset author must provide:
1. A two-column master manifest linking each `ecg_X.jpg` reference to the specific physical scan (`HB(x).jpg`, `MI(x).jpg`, etc.).
2. An authentic clinical table measured directly from the cohort of patients whose ECGs were recorded at CPEIC.
