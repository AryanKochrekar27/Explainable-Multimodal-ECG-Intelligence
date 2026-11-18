"""
Step 9: Automated Software Testing Suite
Rigorous Pytest suite verifying:
1. Label index-to-name mapping integrity
2. ECG Vision & Multimodal Late-Fusion checkpoint loading and parameter shapes
3. Softmax sum-to-one compliance
4. Clinical preprocessor alignment & Random Forest inference (zero feature-name mismatch)
5. Grad-CAM & Multimodal joint XAI generation on valid inputs
6. Input guards rejecting malformed images, channels, and clinical payloads
7. Calibration config and held-out test sample integrity
8. YAML training configuration loading
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config_utils import load_training_config
from train_ecg_model import ECGVisionResNet, CLASS_NAMES
from train_multimodal_model import MultimodalFusionNet
from explain_ecg_model import GradCAM
from inference import (
    CLINICAL_FEATURE_DEFAULTS,
    TRANSFORMED_CLINICAL_COLUMNS,
    get_test_sample_choices,
    load_calibration_config,
    predict_clinical,
    predict_ecg,
    predict_multimodal,
    preprocess_clinical_input,
    validate_and_load_image,
)


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def label_mapping():
    mapping_path = Path("data/processed/label_mapping.json")
    assert mapping_path.exists(), "label_mapping.json must exist"
    with open(mapping_path, encoding="utf-8") as f:
        return json.load(f)


def test_label_mapping_consistency(label_mapping):
    """Test 1: Verify label index-to-name mapping integrity."""
    expected = {
        "Normal": 0,
        "Abnormal_Heartbeat": 1,
        "Myocardial_Infarction": 2,
        "Post_MI_History": 3,
    }
    assert label_mapping == expected, f"Expected {expected}, got {label_mapping}"
    assert len(CLASS_NAMES) == 4
    for idx, name in enumerate(CLASS_NAMES):
        assert label_mapping[name] == idx


def test_ecg_checkpoint_integrity(device):
    """Test 2: Verify ECG Vision model checkpoint loading and parameter shapes."""
    ckpt_path = Path("models/ecg_model_best.pt")
    assert ckpt_path.exists(), "ecg_model_best.pt must exist"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    assert "model_state_dict" in ckpt
    assert "val_macro_f1" in ckpt
    assert ckpt["val_macro_f1"] > 0.70, "ECG model should achieve Macro-F1 > 0.70"

    model = ECGVisionResNet(pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (2, 4), f"Expected logits shape (2, 4), got {logits.shape}"


def test_multimodal_checkpoint_and_inference(device):
    """Test 3: Verify Multimodal Late-Fusion checkpoint loading and forward pass."""
    ckpt_path = Path("models/multimodal_model_best.pt")
    assert ckpt_path.exists(), "multimodal_model_best.pt must exist"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    assert "model_state_dict" in ckpt
    assert "val_macro_f1" in ckpt
    assert ckpt["val_macro_f1"] > 0.70, "Multimodal model should achieve Macro-F1 > 0.70"

    model = MultimodalFusionNet(pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy_img = torch.randn(2, 3, 224, 224).to(device)
    dummy_clin = torch.randn(2, 19).to(device)
    with torch.no_grad():
        logits = model(dummy_img, dummy_clin)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    assert logits.shape == (2, 4)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-3)


def test_softmax_probability_contract(device):
    """Test 4: Softmax output must sum to 1.0 (within epsilon=1e-3) and bounded in [0, 1]."""
    model = ECGVisionResNet(pretrained=False).to(device)
    ckpt = torch.load("models/ecg_model_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    with torch.no_grad():
        logits = model(dummy_input)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    assert np.all(probs >= 0.0), "Probabilities must be non-negative"
    assert np.all(probs <= 1.0), "Probabilities must be <= 1.0"
    sums = np.sum(probs, axis=1)
    assert np.allclose(sums, 1.0, atol=1e-3), f"Probabilities must sum to 1.0, got {sums}"


def test_clinical_model_checkpoint_and_inference():
    """Test 5: Verify clinical preprocessor alignment and Random Forest inference."""
    clin_path = Path("models/clinical_model.joblib")
    prep_path = Path("models/clinical_preprocessor.joblib")
    assert clin_path.exists(), "clinical_model.joblib must exist"
    assert prep_path.exists(), "clinical_preprocessor.joblib must exist"

    model = joblib.load(clin_path)
    preprocessor = joblib.load(prep_path)

    x_df = preprocess_clinical_input(CLINICAL_FEATURE_DEFAULTS, preprocessor)
    assert list(x_df.columns) == TRANSFORMED_CLINICAL_COLUMNS
    assert x_df.shape == (1, 19)
    # Age=57 is close to dataset median, so standardized value should be near 0 (not raw 57)
    assert abs(float(x_df["Age"].iloc[0])) < 3.0

    preds = model.predict(x_df)
    probs = model.predict_proba(x_df)

    assert preds.shape == (1,)
    assert probs.shape == (1, 4)
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-3)


def test_gradcam_generation(device):
    """Test 6: Verify Grad-CAM heatmap generation on valid input tensor."""
    model = ECGVisionResNet(pretrained=False).to(device)
    ckpt = torch.load("models/ecg_model_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    target_layer = model.features[7][-1].conv2
    grad_cam = GradCAM(model, target_layer)

    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    heatmap, conf, pred_class = grad_cam.generate_heatmap(dummy_input, target_class=2)
    grad_cam.remove_hooks()

    assert heatmap.shape == (7, 7), f"Expected heatmap feature map shape (7, 7), got {heatmap.shape}"
    assert np.min(heatmap) >= 0.0
    assert np.max(heatmap) <= 1.0
    assert 0 <= pred_class <= 3
    assert 0.0 <= conf <= 1.0


def test_input_guards_invalid_shapes(device):
    """Test 7: Verify error handling on invalid tensors, images, and clinical payloads."""
    model = ECGVisionResNet(pretrained=False).to(device)
    invalid_input = torch.randn(1, 1, 224, 224).to(device)
    with pytest.raises(Exception):
        model(invalid_input)

    preprocessor = joblib.load("models/clinical_preprocessor.joblib")
    bad_clin = dict(CLINICAL_FEATURE_DEFAULTS)
    bad_clin["Gender"] = "Unknown"
    with pytest.raises(ValueError, match="Gender"):
        preprocess_clinical_input(bad_clin, preprocessor)

    nan_clin = dict(CLINICAL_FEATURE_DEFAULTS)
    nan_clin["Troponin"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        preprocess_clinical_input(nan_clin, preprocessor)

    tiny_img = Image.new("RGB", (4, 4))
    from io import BytesIO
    buf = BytesIO()
    tiny_img.save(buf, format="PNG")
    buf.seek(0)
    with pytest.raises(ValueError, match="too small"):
        validate_and_load_image(buf)


def test_end_to_end_dashboard_inference_and_xai(device):
    """Test 8: Verify all three dashboard inference tracks + XAI on a held-out test sample."""
    test_choices = get_test_sample_choices()
    assert len(test_choices) == 4, f"Expected 4 held-out test class samples, got {len(test_choices)}"

    sample_path = next(iter(test_choices.values()))
    img = validate_and_load_image(sample_path)
    preprocessor = joblib.load("models/clinical_preprocessor.joblib")
    cal_cfg = load_calibration_config()

    ecg_model = ECGVisionResNet(pretrained=False).to(device)
    ecg_model.load_state_dict(
        torch.load("models/ecg_model_best.pt", map_location=device, weights_only=False)["model_state_dict"]
    )
    ecg_res = predict_ecg(
        ecg_model,
        img,
        device=device,
        temperature=float(cal_cfg["ecg"]["recommended_temperature"]),
        compute_gradcam=True,
    )
    assert ecg_res["heatmap"] is not None and ecg_res["heatmap"].shape == (7, 7)
    assert np.isclose(np.sum(ecg_res["probs"]), 1.0, atol=1e-3)

    clin_model = joblib.load("models/clinical_model.joblib")
    clin_res = predict_clinical(clin_model, CLINICAL_FEATURE_DEFAULTS, preprocessor, compute_shap=True)
    assert clin_res["attributions"] is not None and len(clin_res["attributions"]) == 19
    assert np.isclose(np.sum(clin_res["probs"]), 1.0, atol=1e-3)

    mm_model = MultimodalFusionNet(pretrained=False).to(device)
    mm_model.load_state_dict(
        torch.load("models/multimodal_model_best.pt", map_location=device, weights_only=False)["model_state_dict"]
    )
    mm_res = predict_multimodal(
        mm_model,
        img,
        CLINICAL_FEATURE_DEFAULTS,
        preprocessor=preprocessor,
        device=device,
        temperature=float(cal_cfg["multimodal"]["recommended_temperature"]),
        compute_xai=True,
    )
    assert mm_res["heatmap"] is not None and mm_res["heatmap"].shape == (7, 7)
    assert mm_res["attributions"] is not None and len(mm_res["attributions"]) == 19
    assert np.isclose(np.sum(mm_res["probs"]), 1.0, atol=1e-3)


def test_calibration_and_training_config():
    """Test 9: Verify YAML training config and persisted calibration config."""
    cfg = load_training_config()
    assert cfg["global"]["random_seed"] == 42
    assert "ecg_vision_model" in cfg and "multimodal_model" in cfg

    cal_path = Path("models/calibration_config.json")
    assert cal_path.exists(), "models/calibration_config.json must exist"
    cal = load_calibration_config(cal_path)
    assert "ecg" in cal and "multimodal" in cal
    assert cal["ecg"]["recommended_temperature"] > 0
    assert cal["multimodal"]["recommended_temperature"] > 0
