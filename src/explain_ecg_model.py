"""
Step 6: Explainable AI for ECG Vision Model (Grad-CAM)
Implements:
- Class Activation Mapping (Grad-CAM) hooked to ResNet-18 final residual block (layer4)
- Heatmap overlay generation on original 12-lead ECG images
- Visual exhibits with diagnostic annotations and scientific disclaimers
- Generates reports/07_ecg_attribution_examples/
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from train_ecg_model import ECGVisionResNet, CLASS_FOLDER_MAP, CLASS_NAMES

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_heatmap(self, input_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()

        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1)

        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()

        score = logits[0, target_class]
        score.backward()

        # Channel-wise mean of gradients (alphas)
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]

        # Weight the channels by corresponding gradients without mutating activations
        weighted = activations * pooled_gradients[:, None, None]

        heatmap = torch.mean(weighted, dim=0).squeeze()
        heatmap = F.relu(heatmap)

        # Normalize heatmap to [0, 1]
        if torch.max(heatmap) > 0:
            heatmap = heatmap / torch.max(heatmap)

        return heatmap.cpu().detach().numpy(), probs[0, target_class].item(), target_class

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()

def generate_ecg_explanations():
    data_dir = Path("data/processed")
    models_dir = Path("models")
    reports_dir = Path("reports")
    out_dir = reports_dir / "07_ecg_attribution_examples"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating Grad-CAM explanations using device: {device}")

    # Load Model
    model_path = models_dir / "ecg_model_best.pt"
    if not model_path.exists():
        print(f"Error: {model_path} does not exist.")
        return

    model = ECGVisionResNet(pretrained=False).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Hook last block of layer4 in ResNet-18
    # model.features has layers 0 to 7 (layer4 is features[7])
    target_layer = model.features[7][-1].conv2
    grad_cam = GradCAM(model, target_layer)

    # Load test manifest
    img_manifest = pd.read_csv(data_dir / "image_split_manifest.csv")
    test_imgs = img_manifest[img_manifest["Split"] == "test"]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Select representative samples per class
    sample_records = []
    for c_folder, c_idx in CLASS_FOLDER_MAP.items():
        subset = test_imgs[test_imgs["class_folder"] == c_folder]
        if not subset.empty:
            sample_records.append((c_idx, subset.iloc[0]))
            if len(subset) > 1:
                sample_records.append((c_idx, subset.iloc[1]))

    for c_idx, row in sample_records:
        img_path = Path(row["relative_path"])
        orig_img = Image.open(img_path).convert("RGB")
        resized_orig = orig_img.resize((224, 224))

        input_tensor = transform(orig_img).unsqueeze(0).to(device)

        heatmap, conf, pred_class = grad_cam.generate_heatmap(input_tensor)

        # Create overlay
        heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize((224, 224), resample=Image.Resampling.BILINEAR)
        heatmap_np = np.array(heatmap_resized) / 255.0
        color_map = plt.colormaps["jet"]
        colored_heatmap = color_map(heatmap_np)[:, :, :3]

        orig_np = np.array(resized_orig) / 255.0
        overlay = 0.55 * orig_np + 0.45 * colored_heatmap

        # Plot figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(orig_np)
        axes[0].set_title(f"Original 12-Lead ECG ({CLASS_NAMES[c_idx]})\nFile: {row['filename']}", fontsize=11)
        axes[0].axis("off")

        im = axes[1].imshow(overlay)
        axes[1].set_title(f"Grad-CAM Attribution Overlay\nPredicted: {CLASS_NAMES[pred_class]} (Confidence: {conf*100:.1f}%)", fontsize=11)
        axes[1].axis("off")

        fig.suptitle("Explainable AI: ECG Visual Feature Attribution\n(Disclaimer: Highlights salient pixel regions correlating with prediction; not causal medical diagnosis)", fontsize=10, style="italic")
        plt.tight_layout()

        safe_filename = f"gradcam_{CLASS_NAMES[c_idx]}_{Path(row['filename']).stem}.png"
        save_path = out_dir / safe_filename
        plt.savefig(save_path, dpi=250, bbox_inches="tight")
        plt.close()
        print(f"Saved Grad-CAM exhibit: {save_path}")

    grad_cam.remove_hooks()
    print("All Grad-CAM exhibits generated successfully.")

if __name__ == "__main__":
    generate_ecg_explanations()
