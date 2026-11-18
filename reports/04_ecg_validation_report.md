# Step 3: ECG Vision Model Validation Report (Track A)

**Architecture**: `ResNet-18` 2D Deep Convolutional Neural Network  
**Checkpoint Saved**: `models\ecg_model_best.pt`  
**Best Validation Epoch**: `9`  

## 1. Summary of Overall Validation Metrics

| Metric | Achieved Value |
| :--- | :--- |
| **Validation Accuracy** | **0.9510** (95.10%) |
| **Macro Precision** | **0.9426** |
| **Macro Recall (Sensitivity)** | **0.9451** |
| **Macro F1-Score** | **0.9430** |
| **Weighted F1-Score** | **0.9508** |
| **Macro One-vs-Rest ROC-AUC** | **0.9799** |

## 2. Per-Class Diagnostic Performance

| Diagnostic Class | Sensitivity (Recall) | Specificity (OVR) | Precision | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Normal` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 42 |
| `Abnormal_Heartbeat` | 0.8571 | 0.9815 | 0.9375 | 0.8955 | 35 |
| `Myocardial_Infarction` | 1.0000 | 0.9903 | 0.9756 | 0.9877 | 40 |
| `Post_MI_History` | 0.9231 | 0.9658 | 0.8571 | 0.8889 | 26 |

## 3. Confusion Matrix Breakdown
```
                          |       Normal | Abnormal_Heartbeat | Myocardial_Infarction | Post_MI_History
-------------------------------------------------------------------------------------------------------
                   Normal |           42 |            0 |            0 |            0
       Abnormal_Heartbeat |            0 |           30 |            1 |            4
    Myocardial_Infarction |            0 |            0 |           40 |            0
          Post_MI_History |            0 |            2 |            0 |           24
```

## 4. Generated Exhibit Artifacts
- Training Curves: `reports/03_ecg_training_curves.png`
- Training History CSV: `reports/03_ecg_training_history.csv`
- Confusion Matrix Exhibit: `reports/04_ecg_confusion_matrix.png`
- Classification Report CSV: `reports/04_ecg_classification_report.csv`
