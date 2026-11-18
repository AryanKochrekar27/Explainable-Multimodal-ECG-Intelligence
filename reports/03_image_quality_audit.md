# Phase 3: ECG Image Quality & Integrity Audit Report

## 1. Summary Statistics
- **Total Image Files**: 928
- **Unique Image Content (MD5 Hashes)**: 491
- **Image Dimensions**: 2213 × 1572 pixels (uniform across all images)
- **Color Channels / Mode**: RGB (3 channels)
- **Decodable / Openable Files**: 928 / 928 (100% decodable)
- **Corrupted Files**: 0
- **Blank Images**: 0

## 2. Class Folder Breakdown & Exact Duplication

| Class Folder | Total Files | Unique Images | Duplicate Files | Duplication Pattern |
| :--- | :--- | :--- | :--- | :--- |
| `abnormal_heartbeat_ecg_images` | 233 | 233 | 0 | 233 unique images, 0 duplicates |
| `myocardial_infarction_ecg_images` | 239 | 30 | 209 | 30 unique images repeated ~8 times each (total 239 files) |
| `normal_ecg_images` | 284 | 142 | 142 | 142 unique images repeated 2 times each (total 284 files) |
| `post_mi_history_ecg_images` | 172 | 86 | 86 | 86 unique images repeated 2 times each (total 172 files) |
| **Total** | **928** | **491** | **437** | **491 distinct ECG scans** |

## 3. Visual & Header Inspection
- **Image Source**: Standardized 12-lead ECG printouts acquired from EDAN SE-3 / SEMIP series machines (matching the CPEIC Multan open ECG dataset).
- **Visual Quality**: Waveform traces are sharp and high resolution (2213 × 1572), complete with 12-lead grid lines, lead labels (I, II, III, aVR, aVL, aVF, V1-V6), calibration pulses, and rhythm strips.
- **Header Metadata**: Header text contains machine-generated report ID, age/gender headers, and diagnostic timestamps.
- **Representative Contact Sheet**: Saved to `03_image_contact_sheet.png`.
