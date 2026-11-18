# Phase 12: Class Imbalance Analysis & Augmentation Strategy

## 1. Tabular Modality Imbalance & Weights
The 4,327 clinical records were synthetically balanced across the four classes (~25% each):

| Diagnostic Class | Code | Train Count | Percentage | Balanced Class Weight |
| :--- | :--- | :--- | :--- | :--- |
| `Normal` | `0` | 779 | 25.73% | `0.9718` |
| `Abnormal_Heartbeat` | `1` | 764 | 25.23% | `0.9908` |
| `Myocardial_Infarction` | `2` | 722 | 23.84% | `1.0485` |
| `Post_MI_History` | `3` | 763 | 25.20% | `0.9921` |

*Recommendation*: Tabular loss weighting is not strictly required given the balanced nature of the synthetic CSV, but standard cross-entropy applies cleanly.

## 2. ECG Image Modality Imbalance & Weights
In contrast, authentic ECG scans exhibit **pronounced class imbalance** across unique patients:

| Image Class Folder | Unique Train Scans | Proportion | Recommended Class Weight |
| :--- | :--- | :--- | :--- |
| `abnormal_heartbeat_ecg_images` | 163 | 47.52% | `0.5261` |
| `myocardial_infarction_ecg_images` | 21 | 6.12% | `4.0833` |
| `normal_ecg_images` | 99 | 28.86% | `0.8662` |
| `post_mi_history_ecg_images` | 60 | 17.49% | `1.4292` |

> [!TIP]
> **Recommended Training Strategy**: For ECG vision backbone training, apply cost-sensitive class weighting (giving ~5.7x weight to `myocardial_infarction_ecg_images`) or use Focal Loss to prevent the classifier from collapsing to the majority `abnormal_heartbeat` class.

## 3. Scientifically Safe Data Augmentation Principles
- **Strictly Prohibited**:
  - ❌ Random Horizontal Flip (reverses cardiac cycle time axis).
  - ❌ Random Vertical Flip (inverts electrical polarity).
  - ❌ CutOut / Erasing large rectangles (deletes entire leads or QRS complexes).
- **Permissible & Recommended** (applied on-the-fly to TRAINING set only):
  - ✔ Slight random brightness / contrast jitter (±5%) simulating variations in scanner exposure.
  - ✔ Subtle Gaussian noise injection (simulating electrode contact noise).
  - ✔ Minor micro-translations (±2 pixels).
- **Validation and Test Sets**: Never augment validation or test samples.
