"""
Step 3: ECG-Only Vision Model Training & Validation (Track A)
Trains a 2D Computer Vision Deep Convolutional Model (ResNet-18) on 12-lead paper ECG scans.
Implements:
- Class-weighted categorical cross-entropy
- Strict group-aware split loading (zero duplicate leakage)
- Early stopping & ReduceLROnPlateau
- Validation metrics: Macro Accuracy, Precision, Recall, Macro-F1, per-class Sensitivity, Specificity (OVR), and ROC-AUC
- Generation of training curves, confusion matrix, history CSV, metrics JSON, and Markdown report
"""
import os
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config_utils import load_training_config, set_global_seed

CLASS_FOLDER_MAP = {
    "normal_ecg_images": 0,
    "abnormal_heartbeat_ecg_images": 1,
    "myocardial_infarction_ecg_images": 2,
    "post_mi_history_ecg_images": 3
}
CLASS_NAMES = ["Normal", "Abnormal_Heartbeat", "Myocardial_Infarction", "Post_MI_History"]

class ECGImageDataset(Dataset):
    def __init__(self, manifest_df, transform=None):
        self.manifest = manifest_df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        img_path = Path(row["relative_path"])
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found at {img_path}")
        
        image = Image.open(img_path).convert("RGB")
        label = CLASS_FOLDER_MAP[row["class_folder"]]

        if self.transform:
            image = self.transform(image)
        return image, label, str(img_path)

class ECGVisionResNet(nn.Module):
    def __init__(self, num_classes=4, embedding_dim=64, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        
        # Feature extractor up to adaptive pooling
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(512, embedding_dim),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, x, return_embedding=False):
        f = self.features(x)
        emb = self.projector(f)
        logits = self.classifier(emb)
        if return_embedding:
            return logits, emb
        return logits

def calculate_specificity_ovr(cm):
    """Calculates per-class specificity in One-vs-Rest fashion."""
    specificities = []
    total = np.sum(cm)
    for i in range(len(cm)):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = total - (tp + fn + fp)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(float(spec))
    return specificities

def train_and_validate_ecg_model(config_path="configs/training_config.yaml"):
    cfg = load_training_config(config_path)
    global_cfg = cfg.get("global", {})
    ecg_cfg = cfg.get("ecg_vision_model", {})
    paths_cfg = cfg.get("paths", {})
    sched_cfg = ecg_cfg.get("lr_scheduler", {})
    es_cfg = ecg_cfg.get("early_stopping", {})
    aug_cfg = ecg_cfg.get("augmentation", {})

    set_global_seed(int(global_cfg.get("random_seed", 42)))

    data_dir = Path(paths_cfg.get("data_dir", "data/processed"))
    reports_dir = Path(paths_cfg.get("reports_dir", "reports"))
    models_dir = Path(paths_cfg.get("models_dir", "models"))
    reports_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    # 1. Device configuration
    preferred_device = global_cfg.get("device", "cuda")
    use_cuda = preferred_device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if use_cuda else 'CPU'})")

    # 2. Load manifest
    manifest_path = Path(paths_cfg.get("image_manifest", data_dir / "image_split_manifest.csv"))
    df = pd.read_csv(manifest_path)
    train_df = df[df["Split"] == "train"]
    val_df = df[df["Split"] == "val"]
    print(f"Data Partitions: Train={len(train_df)} images, Val={len(val_df)} images (Test remains strictly held out)")

    # 3. Calculate class weights on unique train scans
    unique_train = train_df.drop_duplicates(subset=["md5_hash"])
    class_counts = [sum(unique_train["class_folder"] == k) for k in CLASS_FOLDER_MAP.keys()]
    total_unique = len(unique_train)
    weights = [total_unique / (4.0 * max(c, 1)) for c in class_counts]
    class_weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)
    print(f"Computed Cost-Sensitive Class Weights: {weights}")

    # 4. Transforms
    # Prohibit spatial flips; allow subtle contrast/brightness jitter
    res_h, res_w = ecg_cfg.get("input_resolution", [224, 224])
    bright = float(aug_cfg.get("color_jitter_brightness", 0.05))
    contrast = float(aug_cfg.get("color_jitter_contrast", 0.05))
    train_transform = transforms.Compose([
        transforms.Resize((res_h, res_w)),
        transforms.ColorJitter(brightness=bright, contrast=contrast),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((res_h, res_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ECGImageDataset(train_df, transform=train_transform)
    val_dataset = ECGImageDataset(val_df, transform=val_transform)

    batch_size = int(ecg_cfg.get("batch_size", 32))
    num_workers = int(ecg_cfg.get("num_workers", 0))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 5. Model, Loss, Optimizer, Scheduler
    model = ECGVisionResNet(num_classes=int(global_cfg.get("num_classes", 4)), embedding_dim=64).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor if ecg_cfg.get("use_class_weights", True) else None)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(ecg_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(ecg_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=str(sched_cfg.get("mode", "min")),
        factor=float(sched_cfg.get("factor", 0.5)),
        patience=int(sched_cfg.get("patience", 3)),
        min_lr=float(sched_cfg.get("min_lr", 1e-6)),
    )

    # 6. Training Loop with Early Stopping
    epochs = int(ecg_cfg.get("epochs", 25))
    best_val_f1 = -1.0
    patience = int(es_cfg.get("patience", 7))
    patience_counter = 0
    history = []
    best_model_path = Path(paths_cfg.get("ecg_model_checkpoint", models_dir / "ecg_model_best.pt"))

    print("\nStarting ECG Model Training...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_targets = []
        all_val_probs = []

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                probs = torch.softmax(outputs, dim=1)

                val_running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                all_val_preds.extend(predicted.cpu().numpy())
                all_val_targets.extend(labels.cpu().numpy())
                all_val_probs.extend(probs.cpu().numpy())

        epoch_val_loss = val_running_loss / val_total
        epoch_val_acc = val_correct / val_total
        epoch_val_f1 = f1_score(all_val_targets, all_val_preds, average="macro", zero_division=0)

        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 4),
            "train_acc": round(epoch_train_acc, 4),
            "val_loss": round(epoch_val_loss, 4),
            "val_acc": round(epoch_val_acc, 4),
            "val_macro_f1": round(epoch_val_f1, 4),
            "lr": current_lr
        })

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f} Macro-F1: {epoch_val_f1:.4f} | LR: {current_lr:.6f}")

        # Checkpoint best model on Macro-F1
        if epoch_val_f1 > best_val_f1:
            best_val_f1 = epoch_val_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": epoch_val_f1,
                "val_acc": epoch_val_acc,
                "class_weights": weights,
                "class_names": CLASS_NAMES
            }, best_model_path)
            print(f"  --> Saved new best checkpoint to {best_model_path} (Val Macro-F1: {best_val_f1:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    train_duration = time.time() - start_time
    print(f"Training completed in {train_duration:.1f}s. Best Val Macro-F1: {best_val_f1:.4f}")

    # 7. Final Evaluation on Validation Partition using Best Weights
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    val_preds, val_targets, val_probs = [], [], []
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            val_preds.extend(predicted.cpu().numpy())
            val_targets.extend(labels.numpy())
            val_probs.extend(probs.cpu().numpy())

    val_targets = np.array(val_targets)
    val_preds = np.array(val_preds)
    val_probs = np.array(val_probs)

    # Calculate rigorous metrics
    val_acc = float(accuracy_score(val_targets, val_preds))
    macro_prec = float(precision_score(val_targets, val_preds, average="macro", zero_division=0))
    macro_rec = float(recall_score(val_targets, val_preds, average="macro", zero_division=0))
    macro_f1 = float(f1_score(val_targets, val_preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(val_targets, val_preds, average="weighted", zero_division=0))

    try:
        val_auc_ovr = float(roc_auc_score(val_targets, val_probs, multi_class="ovr", average="macro"))
    except Exception:
        val_auc_ovr = 0.5

    cm = confusion_matrix(val_targets, val_preds)
    specificities = calculate_specificity_ovr(cm)
    rep_dict = classification_report(val_targets, val_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)

    # 8. Save artifacts
    # (a) Training history CSV
    df_history = pd.DataFrame(history)
    df_history.to_csv(reports_dir / "03_ecg_training_history.csv", index=False)

    # (b) Training curves plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(df_history["epoch"], df_history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(df_history["epoch"], df_history["val_loss"], label="Val Loss", marker="s")
    axes[0].set_title("ECG Model: Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(df_history["epoch"], df_history["train_acc"], label="Train Accuracy", marker="o")
    axes[1].plot(df_history["epoch"], df_history["val_acc"], label="Val Accuracy", marker="s")
    axes[1].plot(df_history["epoch"], df_history["val_macro_f1"], label="Val Macro-F1", marker="^", color="purple")
    axes[1].set_title("ECG Model: Accuracy & Macro-F1 Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(reports_dir / "03_ecg_training_curves.png", dpi=300)
    plt.close()

    # (c) Confusion matrix plot
    fig, ax = plt.subplots(figsize=(7, 6))
    cax = ax.matshow(cm, cmap="Blues")
    fig.colorbar(cax)
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black" if cm[i, j] < cm.max()/2 else "white", fontsize=12)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels([c.replace("_", "\n") for c in CLASS_NAMES], rotation=45, ha="left")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Ground Truth")
    ax.set_title("ECG Vision Model: Validation Confusion Matrix", pad=20)
    plt.tight_layout()
    plt.savefig(reports_dir / "04_ecg_confusion_matrix.png", dpi=300)
    plt.close()

    # (d) Classification report CSV
    df_rep = pd.DataFrame(rep_dict).transpose()
    df_rep.to_csv(reports_dir / "04_ecg_classification_report.csv")

    # (e) Validation metrics JSON
    metrics_json = {
        "architecture": "ResNet18_2D_CNN",
        "validation_samples": len(val_df),
        "best_epoch": checkpoint["epoch"],
        "accuracy": round(val_acc, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "roc_auc_ovr": round(val_auc_ovr, 4),
        "per_class_specificity_ovr": {CLASS_NAMES[i]: round(specificities[i], 4) for i in range(4)},
        "per_class_sensitivity": {CLASS_NAMES[i]: round(rep_dict[CLASS_NAMES[i]]["recall"], 4) for i in range(4)},
        "confusion_matrix": cm.tolist()
    }
    with open(reports_dir / "04_ecg_validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=4)

    # (f) Validation report Markdown
    with open(reports_dir / "04_ecg_validation_report.md", "w", encoding="utf-8") as f:
        f.write("# Step 3: ECG Vision Model Validation Report (Track A)\n\n")
        f.write(f"**Architecture**: `ResNet-18` 2D Deep Convolutional Neural Network  \n")
        f.write(f"**Checkpoint Saved**: `{best_model_path}`  \n")
        f.write(f"**Best Validation Epoch**: `{checkpoint['epoch']}`  \n\n")

        f.write("## 1. Summary of Overall Validation Metrics\n\n")
        f.write("| Metric | Achieved Value |\n")
        f.write("| :--- | :--- |\n")
        f.write(f"| **Validation Accuracy** | **{val_acc:.4f}** ({val_acc*100:.2f}%) |\n")
        f.write(f"| **Macro Precision** | **{macro_prec:.4f}** |\n")
        f.write(f"| **Macro Recall (Sensitivity)** | **{macro_rec:.4f}** |\n")
        f.write(f"| **Macro F1-Score** | **{macro_f1:.4f}** |\n")
        f.write(f"| **Weighted F1-Score** | **{weighted_f1:.4f}** |\n")
        f.write(f"| **Macro One-vs-Rest ROC-AUC** | **{val_auc_ovr:.4f}** |\n\n")

        f.write("## 2. Per-Class Diagnostic Performance\n\n")
        f.write("| Diagnostic Class | Sensitivity (Recall) | Specificity (OVR) | Precision | F1-Score | Support |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for i, c in enumerate(CLASS_NAMES):
            c_info = rep_dict[c]
            f.write(f"| `{c}` | {c_info['recall']:.4f} | {specificities[i]:.4f} | {c_info['precision']:.4f} | {c_info['f1-score']:.4f} | {int(c_info['support'])} |\n")

        f.write("\n## 3. Confusion Matrix Breakdown\n")
        f.write("```\n")
        header = f"{'':>25} | " + " | ".join([f"{c:>12}" for c in CLASS_NAMES])
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for idx, row in enumerate(cm):
            f.write(f"{CLASS_NAMES[idx]:>25} | " + " | ".join([f"{v:>12}" for v in row]) + "\n")
        f.write("```\n\n")

        f.write("## 4. Generated Exhibit Artifacts\n")
        f.write("- Training Curves: `reports/03_ecg_training_curves.png`\n")
        f.write("- Training History CSV: `reports/03_ecg_training_history.csv`\n")
        f.write("- Confusion Matrix Exhibit: `reports/04_ecg_confusion_matrix.png`\n")
        f.write("- Classification Report CSV: `reports/04_ecg_classification_report.csv`\n")

    print("ECG model training, validation, and exhibit generation completed successfully.")
    return metrics_json

if __name__ == "__main__":
    train_and_validate_ecg_model()
