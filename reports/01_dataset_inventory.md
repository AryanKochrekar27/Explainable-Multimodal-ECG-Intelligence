# Phase 1: Dataset Inventory Report

## 1. Executive Summary
- **Total files inventoried**: 938
- **Total ECG Images found**: 930
- **Primary Tabular CSV**: `initial_dataset/cardiovascular_dataset.csv` (and copied to `data/raw/`)
- **Primary Documentation**: `Multimodal_Cardiovascular_Risk_Dataset_Overview_Final.pdf`, `IEEE_Explainable_ECG_Intelligence.docx`, `Explainable_ECG_Intelligence_Literature_Survey_50_Papers.xlsx`

## 2. ECG Image Collection Breakdown
The ECG image files reside in `initial_dataset/ECG Images/ECG Images/` across 4 class subdirectories:

| Subdirectory / Class Folder | File Count | Sample Filename | Original Dimensions |
| :--- | :--- | :--- | :--- |
| `abnormal_heartbeat_ecg_images` | 233 | `HB(1).jpg` | 2213 × 1572 |
| `myocardial_infarction_ecg_images` | 239 | `MI(1).jpg` | 2213 × 1572 |
| `normal_ecg_images` | 284 | `Normal(1).jpg` | 2213 × 1572 |
| `post_mi_history_ecg_images` | 172 | `PMI(1).jpg` | 2213 × 1572 |
| **Total** | **928** | | |

## 3. Discrepancy & Ambiguity Findings
1. **Record Count Mismatch**: The tabular CSV contains **4,327** records with filenames formatted as `ecg_1.jpg` through `ecg_4327.jpg`. However, the physical image directory contains only **928** image files with names formatted as `HB(x).jpg`, `MI(x).jpg`, `Normal(x).jpg`, and `PMI(x).jpg`.
2. **Duplicate Image Files**: As documented in the image audit, only **491** unique image hashes exist among the 928 image files. Specifically, `myocardial_infarction_ecg_images` contains 8 repeated copies of 30 distinct images; `normal_ecg_images` contains 2 repeated copies of 142 distinct images; `post_mi_history_ecg_images` contains 2 repeated copies of 86 distinct images.
3. **Archive Status**: The raw data was found extracted into `initial_dataset/` rather than an unextracted `.zip` file on disk.
