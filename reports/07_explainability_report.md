# Step 6: Explainable AI (XAI) Comprehensive Report

## 1. Executive Summary
To satisfy high clinical trust standards, our diagnostic framework integrates dual-modality explainability:
1. **Vision Explanations (Grad-CAM)**: Pixel-level spatial gradient attribution highlighting salient lead trajectories and ST-segment elevations on 12-lead paper ECG scans.
2. **Tabular Explanations (TreeExplainer SHAP)**: Game-theoretic Shapley attribution quantifying individual and global clinical feature impact.
3. **Joint Multimodal Attributions**: Combined exhibits displaying synchronized visual saliency alongside ranked clinical biomarker contributions.

## 2. ECG Visual Attribution Findings (Grad-CAM)
- **Myocardial Infarction**: High activation gradient focuses heavily on anterior leads (V1-V4) and inferior leads (II, III, aVF) corresponding to acute ST-segment changes.
- **Abnormal Heartbeat**: Attribution concentrates across the rhythm strip (Lead II) and variable R-peak intervals.
- **Normal**: Distributed, low-contrast baseline activation across normal QRS complexes.

## 3. Clinical Feature Importance Findings (SHAP)
- Top contributing clinical features in the Random Forest baseline include `ST_Segment_Level`, `Troponin`, `QRS_Duration`, `HRV_SDNN`, and `Heart_Rate`.
- However, due to the synthetic uniformity established in Phase 1, tabular attributions exhibit lower variance than biological clinical data.

## 4. Biomedical Disclaimer & Governance
> [!IMPORTANT]
> **Attribution vs. Causality**: Heatmaps and SHAP values indicate regions and features that statistically influence the neural network's logit outputs. They reflect mathematical model attribution and must NOT be interpreted as autonomous physiological causality or independent clinical diagnoses.
