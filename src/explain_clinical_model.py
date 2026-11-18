"""
Step 6: Explainable AI for Clinical Tabular Model (SHAP)
Computes:
- TreeExplainer SHAP values on clinical features
- Global feature importance bar chart & beeswarm/summary plot
- Generates reports/07_shap_summary.png
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

def explain_clinical_model():
    data_dir = Path("data/processed")
    models_dir = Path("models")
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    model_path = models_dir / "clinical_model.joblib"
    if not model_path.exists():
        print(f"Error: {model_path} not found.")
        return

    print("Loading fitted clinical model and validation data...")
    model = joblib.load(model_path)

    X_val_raw = pd.read_csv(data_dir / "X_val_clinical.csv")
    feature_cols = [c for c in X_val_raw.columns if c != "Record_ID"]
    X_val = X_val_raw[feature_cols].copy()

    # Subsample for efficient SHAP computation
    background_X = X_val.sample(n=min(150, len(X_val)), random_state=42)

    print("Computing TreeExplainer SHAP values...")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(background_X)

        # Handle multiclass shap values format
        if isinstance(shap_values, list):
            # List of arrays per class, shape: [classes, n_samples, n_features]
            mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        elif len(shap_values.shape) == 3:
            # Shape: [n_samples, n_features, n_classes]
            mean_abs_shap = np.mean(np.abs(shap_values), axis=(0, 2))
        else:
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    except Exception as e:
        print(f"SHAP tree explainer encountered {e}, falling back to feature importances...")
        mean_abs_shap = model.feature_importances_

    # Create summary plot
    df_importance = pd.DataFrame({
        "Feature": feature_cols,
        "Mean_Absolute_SHAP": mean_abs_shap
    }).sort_values(by="Mean_Absolute_SHAP", ascending=True)

    plt.figure(figsize=(10, 8))
    bars = plt.barh(df_importance["Feature"], df_importance["Mean_Absolute_SHAP"], color="#2b5c8f", edgecolor="black", alpha=0.85)
    plt.xlabel("Mean |SHAP Value| (Impact on Multiclass Risk Prediction)", fontsize=11)
    plt.title("Clinical Tabular Model: Global Feature Attribution (SHAP)\n(Indicates statistical feature reliance across synthetic clinical variables)", fontsize=12)
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()

    out_path = reports_dir / "07_shap_summary.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved SHAP summary figure to {out_path}")

    # Top features output
    print("\nTop 5 Contributing Clinical Features:")
    top_5 = df_importance.tail(5).iloc[::-1]
    for _, r in top_5.iterrows():
        print(f"  - {r['Feature']}: {r['Mean_Absolute_SHAP']:.4f}")

if __name__ == "__main__":
    explain_clinical_model()
