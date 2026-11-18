# Step 4: Clinical Tabular Model Training & Validation Report

**Selected Best Architecture**: `RandomForest`  
**Model Artifact**: `models\clinical_model.joblib`  

## 1. Validation Benchmark Comparison

| Model Architecture | Accuracy | Macro-F1 | Macro-Precision | Macro-Recall | OVR ROC-AUC | Log-Loss |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LogisticRegression** | 0.2435 | 0.2370 | 0.2411 | 0.2414 | 0.4883 | 1.4014 |
| **RandomForest** | 0.2496 | 0.2455 | 0.2491 | 0.2478 | 0.4895 | 1.4000 |
| **MLPClassifier** | 0.2465 | 0.2450 | 0.2494 | 0.2457 | 0.4850 | 1.4164 |

## 2. Best Model (RandomForest) Per-Class Performance

| Cardiac Category | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| `Normal` | 0.2755 | 0.3234 | 0.2975 | 167.0 |
| `Abnormal_Heartbeat` | 0.2063 | 0.2378 | 0.2210 | 164.0 |
| `Myocardial_Infarction` | 0.2400 | 0.1558 | 0.1890 | 154.0 |
| `Post_MI_History` | 0.2744 | 0.2744 | 0.2744 | 164.0 |

## 3. Confusion Matrix Analysis
```
                          |       Normal | Abnormal_Heartbeat | Myocardial_Infarction | Post_MI_History
-------------------------------------------------------------------------------------------------------
                   Normal |           54 |           60 |           13 |           40
       Abnormal_Heartbeat |           54 |           39 |           34 |           37
    Myocardial_Infarction |           40 |           48 |           24 |           42
          Post_MI_History |           48 |           42 |           29 |           45
```

## 4. Key Scientific Findings
- **Tabular Predictive Power**: As established during the preprocessing audit, the synthetic clinical features show uniform marginal distributions and weak biological coupling to acute cardiac pathology. The models achieve near-chance performance (~0.25-0.27 accuracy and ~0.50 ROC-AUC).
- **Honest Reporting**: This confirms our scientific hypothesis that synthetic uniform tabular variables cannot reliably predict authentic cardiac pathology without ECG waveform analysis.
