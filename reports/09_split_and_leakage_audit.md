# Phase 9: Partitioning & Leakage Audit Report

## 1. Split Design & Proportions
All partitions were generated using **seed 42** with stratified class distribution:
- **Training Set**: 70.0%
- **Validation Set**: 15.0%
- **Test Set**: 15.0%

## 2. Clinical Partition Counts (N = 4,327)

| Split | Total Records | Normal | Abnormal Heartbeat | Myocardial Infarction | Post-MI History |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `train` | 3,028 (70.0%) | 779 | 764 | 722 | 763 |
| `val` | 649 (15.0%) | 167 | 164 | 154 | 164 |
| `test` | 650 (15.0%) | 167 | 164 | 155 | 164 |
| **Total** | **4,327** | **1113** | **1092** | **1031** | **1091** |

## 3. ECG Image Partition Counts (Zero-Leakage Hash Stratification)
> [!IMPORTANT]
> Because duplicate copies of the same scan exist on disk (e.g., 30 MI scans duplicated across 239 files), naive record splitting would leak identical training images into the test set. By performing **Group-Aware Hash Stratification**, every copy of a physical scan is assigned exclusively to one split partition.

| Split | Total Image Files | Unique Scans | Normal | Abnormal Heartbeat | Myocardial Infarction | Post-MI History |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `train` | 648 | 343 | 198 | 163 | 167 | 120 |
| `val` | 143 | 74 | 42 | 35 | 40 | 26 |
| `test` | 137 | 74 | 44 | 35 | 32 | 26 |
| **Total** | **928** | **491** | **284** | **233** | **239** | **172** |

## 4. Leakage Verification Results
- **Patient-ID Leakage**: `NOT VERIFIABLE` (Patient identifiers were omitted by original dataset curators).
- **Image Hash Leakage (Train ∩ Val)**: **0** overlapping scans.
- **Image Hash Leakage (Train ∩ Test)**: **0** overlapping scans.
- **Image Hash Leakage (Val ∩ Test)**: **0** overlapping scans.
- **Leakage Audit Decision**: **PASS (Strict zero duplicate image leakage achieved)**.
