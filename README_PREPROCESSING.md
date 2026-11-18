# Reproducible Data Preprocessing Pipeline

This directory contains the complete, scientifically validated preprocessing pipeline for the project:
**“Explainable ECG Intelligence for Early Cardiac Risk Assessment.”**

## Project Architecture
```
ECG/
├── data/
│   ├── raw/                       # Preserved immutable original data
│   │   └── cardiovascular_dataset.csv
│   └── processed/                 # Partitioned and transformed artifacts
│       ├── clinical_clean.csv
│       ├── mapping_manifest.csv
│       ├── multimodal_manifest.csv
│       ├── split_manifest.csv
│       ├── image_split_manifest.csv
│       ├── X_train_clinical.csv
│       ├── X_val_clinical.csv
│       ├── X_test_clinical.csv
│       ├── y_train.csv
│       ├── y_val.csv
│       ├── y_test.csv
│       ├── label_mapping.json
│       ├── feature_columns.json
│       └── transformed_feature_names.json
├── models/
│   └── clinical_preprocessor.joblib   # Scikit-learn ColumnTransformer fitted strictly on train set
├── reports/
│   ├── 01_dataset_inventory.md / .csv
│   ├── 02_csv_quality_audit.md / .csv
│   ├── 03_image_quality_audit.md / .csv
│   ├── 03_image_contact_sheet.png
│   ├── 04_image_record_mapping.md
│   ├── 04_unmatched_csv_records.csv
│   ├── 04_unmatched_images.csv
│   ├── 05_target_label_audit.md
│   ├── 05_class_distribution.csv
│   ├── 06_clinical_cleaning_report.md
│   ├── 07_feature_schema.md
│   ├── 08_image_preprocessing.md
│   ├── 08_original_vs_processed.png
│   ├── 09_split_and_leakage_audit.md
│   ├── 10_final_dataset_validation.md
│   ├── 11_class_imbalance.md
│   ├── 12_final_quality_checks.md
│   ├── quality_checks_summary.json
│   └── FINAL_DATA_PREPROCESSING_REPORT.md
├── src/
│   ├── audit_dataset.py
│   ├── audit_csv.py
│   ├── audit_images.py
│   ├── verify_mapping.py
│   ├── audit_target_labels.py
│   ├── preprocess_clinical.py
│   ├── preprocess_images.py
│   ├── split_dataset.py
│   ├── fit_clinical_pipeline.py
│   ├── construct_multimodal_manifest.py
│   ├── analyze_imbalance.py
│   └── validate_preprocessed_data.py
├── configs/
│   └── preprocessing_config.json
└── requirements.txt
```

## How to Execute the Entire Pipeline End-to-End

To re-run all audit, cleaning, transformation, and validation stages from scratch:

```powershell
# Phase 1: Audit Dataset Inventory
python src/audit_dataset.py

# Phase 2: Audit CSV Quality & Physiological Correlation
python src/audit_csv.py

# Phase 3: Audit ECG Images & Detect Duplicates
python src/audit_images.py

# Phase 4: Verify CSV-to-Image Linkage
python src/verify_mapping.py

# Phase 5: Audit Target Labels & Mapping
python src/audit_target_labels.py

# Phase 6 & 7: Clean Clinical Features & Schema
python src/preprocess_clinical.py

# Phase 8: Verify ECG Image Preprocessing
python src/preprocess_images.py

# Phase 9: Partition Data with Zero-Leakage Hash Grouping
python src/split_dataset.py

# Phase 10: Fit Scikit-Learn Preprocessor on Training Set
python src/fit_clinical_pipeline.py

# Phase 11: Multimodal Manifest & Integrity Audit
python src/construct_multimodal_manifest.py

# Phase 12: Class Imbalance Analysis
python src/analyze_imbalance.py

# Phase 13: Final Automated Quality & Leakage Checks
python src/validate_preprocessed_data.py
```
