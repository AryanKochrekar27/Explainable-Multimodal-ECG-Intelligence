"""
Shared Inference, Preprocessing, Calibration, and XAI Helpers
Used by app.py (Streamlit dashboard) and tests/test_pipeline.py.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import shap
import torch
import torchvision.transforms as transforms

from train_ecg_model import ECGVisionResNet, CLASS_NAMES, CLASS_FOLDER_MAP
from train_multimodal_model import MultimodalFusionNet
from explain_ecg_model import GradCAM
from explain_multimodal_model import MultimodalGradCAM


RAW_CONTINUOUS_COLUMNS: List[str] = [
    "Age",
    "Heart_Rate",
    "Systolic_BP",
    "Diastolic_BP",
    "Cholesterol",
    "Glucose",
    "Troponin",
    "PR_Interval",
    "QRS_Duration",
    "QT_Interval",
    "P_Wave_Duration",
    "T_Wave_Duration",
    "ST_Segment_Level",
    "RR_Interval",
    "HRV_SDNN",
    "HRV_RMSSD",
    "R_Peak_Amplitude",
    "S_Wave_Amplitude",
]

RAW_CLINICAL_COLUMNS: List[str] = RAW_CONTINUOUS_COLUMNS + ["Gender"]
TRANSFORMED_CLINICAL_COLUMNS: List[str] = RAW_CONTINUOUS_COLUMNS + ["Gender_Male"]

# Observed dataset feature bounds and medians from data/processed/clinical_clean.csv
CLINICAL_FEATURE_DEFAULTS: Dict[str, Any] = {
    "Age": 57,
    "Heart_Rate": 98,
    "Systolic_BP": 134,
    "Diastolic_BP": 84,
    "Cholesterol": 234,
    "Glucose": 135,
    "Troponin": 1.29,
    "PR_Interval": 171,
    "QRS_Duration": 114,
    "QT_Interval": 399,
    "P_Wave_Duration": 100,
    "T_Wave_Duration": 159,
    "ST_Segment_Level": 0.16,
    "RR_Interval": 899,
    "HRV_SDNN": 70,
    "HRV_RMSSD": 56,
    "R_Peak_Amplitude": 1.73,
    "S_Wave_Amplitude": -1.28,
    "Gender": "Male",
}

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_calibration_config(path: Union[str, Path] = Path("models/calibration_config.json")) -> Dict[str, Any]:
    """Load persisted validation-fitted calibration parameters if present."""
    cal_path = Path(path)
    if not cal_path.exists():
        return {
            "ecg": {"fitted_temperature": 1.0, "recommended_temperature": 1.0, "use_temperature_scaling": False},
            "multimodal": {"fitted_temperature": 1.0, "recommended_temperature": 1.0, "use_temperature_scaling": False},
        }
    with open(cal_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_test_sample_choices(
    manifest_path: Union[str, Path] = Path("data/processed/image_split_manifest.csv"),
) -> Dict[str, Path]:
    """Return one verified held-out test-split ECG image path per diagnostic class."""
    path = Path(manifest_path)
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    test_df = df[df["Split"] == "test"]
    choices: Dict[str, Path] = {}
    folder_to_idx = CLASS_FOLDER_MAP
    ordered = sorted(folder_to_idx.items(), key=lambda kv: kv[1])
    for folder, cls_idx in ordered:
        sub = test_df[test_df["class_folder"] == folder]
        for _, row in sub.iterrows():
            img_path = Path(row["relative_path"])
            if img_path.exists():
                pretty_cls = CLASS_NAMES[cls_idx].replace("_", " ")
                choices[f"{pretty_cls} — {row['filename']}"] = img_path
                break
    return choices


def load_runtime_models(models_dir: Union[str, Path] = Path("models")) -> Dict[str, Any]:
    """Load trained checkpoints, fitted preprocessor, and calibration config."""
    m_dir = Path(models_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded: Dict[str, Any] = {"device": device}

    ecg_path = m_dir / "ecg_model_best.pt"
    if ecg_path.exists():
        ecg_model = ECGVisionResNet(pretrained=False).to(device)
        ckpt = torch.load(ecg_path, map_location=device, weights_only=False)
        ecg_model.load_state_dict(ckpt["model_state_dict"])
        ecg_model.eval()
        loaded["ecg"] = ecg_model

    clin_path = m_dir / "clinical_model.joblib"
    if clin_path.exists():
        loaded["clinical"] = joblib.load(clin_path)

    mm_path = m_dir / "multimodal_model_best.pt"
    if mm_path.exists():
        mm_model = MultimodalFusionNet(pretrained=False).to(device)
        ckpt = torch.load(mm_path, map_location=device, weights_only=False)
        mm_model.load_state_dict(ckpt["model_state_dict"])
        mm_model.eval()
        loaded["multimodal"] = mm_model

    prep_path = m_dir / "clinical_preprocessor.joblib"
    if prep_path.exists():
        loaded["preprocessor"] = joblib.load(prep_path)

    loaded["calibration"] = load_calibration_config(m_dir / "calibration_config.json")
    return loaded


def validate_and_load_image(source: Any) -> Image.Image:
    """Safely open and validate an uploaded or path-referenced ECG image."""
    try:
        img = Image.open(source)
        img.load()
        if img.width < 16 or img.height < 16:
            raise ValueError(f"Image dimensions too small ({img.width}x{img.height}); expected a 12-lead ECG scan.")
        return img.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("Uploaded file is not a valid readable image (JPEG/PNG).") from exc


def preprocess_ecg_image(image: Image.Image) -> torch.Tensor:
    """Convert an RGB PIL image into a normalized (1, 3, 224, 224) float32 tensor."""
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL.Image.Image instance.")
    rgb = image.convert("RGB")
    return IMAGE_TRANSFORM(rgb).unsqueeze(0)


def preprocess_clinical_input(clinical_data: Dict[str, Any], preprocessor: Any) -> pd.DataFrame:
    """
    Validate raw patient variables and apply the training-fitted ColumnTransformer.
    Returns a 1-row DataFrame with exact transformed feature names expected by models.
    """
    if preprocessor is None:
        raise ValueError("Clinical preprocessor is not loaded.")

    missing = [col for col in RAW_CLINICAL_COLUMNS if col not in clinical_data]
    if missing:
        raise ValueError(f"Missing required clinical variables: {missing}")

    row: Dict[str, Any] = {}
    for col in RAW_CONTINUOUS_COLUMNS:
        val = clinical_data[col]
        try:
            fval = float(val)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Clinical variable '{col}' must be numeric, got {val!r}.") from exc
        if not np.isfinite(fval):
            raise ValueError(f"Clinical variable '{col}' must be finite, got {val!r}.")
        row[col] = fval

    gender = str(clinical_data["Gender"]).strip()
    if gender not in {"Male", "Female"}:
        raise ValueError(f"Gender must be 'Male' or 'Female', got {clinical_data['Gender']!r}.")
    row["Gender"] = gender

    raw_df = pd.DataFrame([row], columns=RAW_CLINICAL_COLUMNS)
    transformed = preprocessor.transform(raw_df)
    if np.isnan(transformed).any() or np.isinf(transformed).any():
        raise ValueError("Transformed clinical vector contains non-finite values.")

    return pd.DataFrame(transformed, columns=TRANSFORMED_CLINICAL_COLUMNS)


def predict_ecg(
    ecg_model: ECGVisionResNet,
    image: Image.Image,
    device: torch.device,
    temperature: float = 1.0,
    compute_gradcam: bool = True,
) -> Dict[str, Any]:
    """Run ECG-only 2D-CNN inference and optional Grad-CAM attribution."""
    if temperature <= 0:
        raise ValueError(f"Temperature must be > 0, got {temperature}.")

    img_tensor = preprocess_ecg_image(image).to(device)
    ecg_model.eval()

    with torch.no_grad():
        logits = ecg_model(img_tensor) / float(temperature)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))

    heatmap: Optional[np.ndarray] = None
    if compute_gradcam:
        target_layer = ecg_model.features[7][-1].conv2
        grad_cam = GradCAM(ecg_model, target_layer)
        try:
            heatmap, _, _ = grad_cam.generate_heatmap(img_tensor, target_class=pred_idx)
        finally:
            grad_cam.remove_hooks()

    return {
        "probs": probs,
        "predicted_idx": pred_idx,
        "predicted_class": CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "heatmap": heatmap,
    }


def predict_clinical(
    clinical_model: Any,
    clinical_data: Dict[str, Any],
    preprocessor: Any,
    compute_shap: bool = True,
) -> Dict[str, Any]:
    """Run tabular clinical inference with proper scaling and per-sample SHAP attribution."""
    x_df = preprocess_clinical_input(clinical_data, preprocessor)
    probs = clinical_model.predict_proba(x_df)[0]
    pred_idx = int(np.argmax(probs))

    shap_df: Optional[pd.DataFrame] = None
    if compute_shap:
        try:
            explainer = shap.TreeExplainer(clinical_model)
            sv = explainer.shap_values(x_df)
            if isinstance(sv, list):
                impacts = np.asarray(sv[pred_idx])[0]
            elif getattr(sv, "ndim", 0) == 3:
                impacts = np.asarray(sv)[0, :, pred_idx]
            else:
                impacts = np.asarray(sv)[0]
            attr_kind = "SHAP Value (Log-Odds / Probability Impact)"
        except Exception:
            impacts = getattr(clinical_model, "feature_importances_", np.zeros(len(TRANSFORMED_CLINICAL_COLUMNS)))
            attr_kind = "Global Tree Feature Importance"

        shap_df = pd.DataFrame({
            "Feature": TRANSFORMED_CLINICAL_COLUMNS,
            "Scaled_Value": x_df.iloc[0].values,
            "Attribution": np.asarray(impacts, dtype=float),
            "Abs_Attribution": np.abs(np.asarray(impacts, dtype=float)),
            "Kind": attr_kind,
        }).sort_values("Abs_Attribution", ascending=False).reset_index(drop=True)

    return {
        "probs": probs,
        "predicted_idx": pred_idx,
        "predicted_class": CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "scaled_features": x_df,
        "attributions": shap_df,
    }


def predict_multimodal(
    mm_model: MultimodalFusionNet,
    image: Image.Image,
    clinical_data: Dict[str, Any],
    preprocessor: Any,
    device: torch.device,
    temperature: float = 1.0,
    compute_xai: bool = True,
) -> Dict[str, Any]:
    """Run multimodal late-fusion inference with joint Grad-CAM and clinical gradient attribution."""
    if temperature <= 0:
        raise ValueError(f"Temperature must be > 0, got {temperature}.")

    img_tensor = preprocess_ecg_image(image).to(device)
    x_df = preprocess_clinical_input(clinical_data, preprocessor)
    clin_tensor = torch.tensor(x_df.values, dtype=torch.float32, device=device)
    mm_model.eval()

    with torch.no_grad():
        logits = mm_model(img_tensor, clin_tensor) / float(temperature)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))

    heatmap: Optional[np.ndarray] = None
    attr_df: Optional[pd.DataFrame] = None
    if compute_xai:
        target_layer = mm_model.vision_features[7][-1].conv2
        mm_cam = MultimodalGradCAM(mm_model, target_layer)
        try:
            clin_grad_in = clin_tensor.clone().detach()
            heatmap, clin_grads, _, _ = mm_cam.generate(img_tensor, clin_grad_in, target_class=pred_idx)
        finally:
            mm_cam.remove_hooks()

        attr_df = pd.DataFrame({
            "Feature": TRANSFORMED_CLINICAL_COLUMNS,
            "Scaled_Value": x_df.iloc[0].values,
            "Attribution": np.asarray(clin_grads, dtype=float),
            "Abs_Attribution": np.abs(np.asarray(clin_grads, dtype=float)),
            "Kind": "Input Gradient Saliency |d(logit)/dx|",
        }).sort_values("Abs_Attribution", ascending=False).reset_index(drop=True)

    return {
        "probs": probs,
        "predicted_idx": pred_idx,
        "predicted_class": CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "heatmap": heatmap,
        "scaled_features": x_df,
        "attributions": attr_df,
    }

