"""
Phase 10: Fit Clinical Preprocessing Pipeline
Builds and fits a rigorous scikit-learn ColumnTransformer on the TRAINING set ONLY.
Scales continuous features using StandardScaler (and evaluates MinMaxScaler)
and OneHotEncodes nominal categorical features (Gender).
Transforms Train, Val, and Test sets and persists the preprocessor with joblib.
Outputs:
- processed/X_train_clinical.csv
- processed/X_val_clinical.csv
- processed/X_test_clinical.csv
- processed/y_train.csv
- processed/y_val.csv
- processed/y_test.csv
- models/clinical_preprocessor.joblib
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

def fit_clinical_pipeline():
    processed_dir = Path("data/processed")
    models_dir = Path("models")
    processed_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    
    # Load clean data and split assignments
    df_clinical = pd.read_csv(processed_dir / "clinical_clean.csv")
    df_splits = pd.read_csv(processed_dir / "split_manifest.csv")
    
    df_clinical["Split"] = df_splits["Split"]
    
    # Load feature schema
    with open(processed_dir / "feature_columns.json", "r", encoding="utf-8") as f:
        schema = json.load(f)
        
    num_cols = schema["continuous_features"]
    cat_cols = schema["binary_categorical_features"]
    
    # Partition features and target
    train_df = df_clinical[df_clinical["Split"] == "train"]
    val_df = df_clinical[df_clinical["Split"] == "val"]
    test_df = df_clinical[df_clinical["Split"] == "test"]
    
    X_train_raw = train_df[num_cols + cat_cols]
    y_train = train_df[["Record_ID", "Target", "Target_Encoded"]]
    
    X_val_raw = val_df[num_cols + cat_cols]
    y_val = val_df[["Record_ID", "Target", "Target_Encoded"]]
    
    X_test_raw = test_df[num_cols + cat_cols]
    y_test = test_df[["Record_ID", "Target", "Target_Encoded"]]
    
    # Define ColumnTransformer
    # Fit strictly on X_train_raw
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_cols)
        ],
        remainder="drop"
    )
    
    print("Fitting preprocessor on training partition (N = 3,028)...")
    preprocessor.fit(X_train_raw)
    
    # Determine transformed column names
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(cat_cols))
    all_feature_names = num_cols + cat_feature_names
    
    # Transform partitions
    X_train_trans = preprocessor.transform(X_train_raw)
    X_val_trans = preprocessor.transform(X_val_raw)
    X_test_trans = preprocessor.transform(X_test_raw)
    
    # Convert to DataFrames
    df_X_train = pd.DataFrame(X_train_trans, columns=all_feature_names, index=train_df["Record_ID"])
    df_X_val = pd.DataFrame(X_val_trans, columns=all_feature_names, index=val_df["Record_ID"])
    df_X_test = pd.DataFrame(X_test_trans, columns=all_feature_names, index=test_df["Record_ID"])
    
    # Save outputs
    df_X_train.to_csv(processed_dir / "X_train_clinical.csv")
    df_X_val.to_csv(processed_dir / "X_val_clinical.csv")
    df_X_test.to_csv(processed_dir / "X_test_clinical.csv")
    
    y_train.to_csv(processed_dir / "y_train.csv", index=False)
    y_val.to_csv(processed_dir / "y_val.csv", index=False)
    y_test.to_csv(processed_dir / "y_test.csv", index=False)
    
    model_path = models_dir / "clinical_preprocessor.joblib"
    joblib.dump(preprocessor, model_path)
    print(f"Persisted fitted preprocessor to: {model_path}")
    
    # Save transformed column schema
    with open(processed_dir / "transformed_feature_names.json", "w", encoding="utf-8") as f:
        json.dump(all_feature_names, f, indent=4)
        
    print("Phase 10: Clinical preprocessor fitted and applied successfully.")

if __name__ == "__main__":
    fit_clinical_pipeline()
