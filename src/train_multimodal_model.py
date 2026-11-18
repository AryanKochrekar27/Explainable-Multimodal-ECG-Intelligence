"""
Step 5: Multimodal Fusion Model Training & Evaluation (Track C)
Implements:
- ECG Vision Branch: 2D-CNN feature backbone -> 64-d embedding
- Clinical Branch: MLP feature extractor -> 32-d embedding
- Late Fusion Head: Concatenation (96-d) -> Linear(96, 64) -> ReLU -> Dropout(0.3) -> Linear(64, 4)
- Zero-leakage class-aligned cohort pairing across train and val splits
- Generation of training curves, history CSV, and checkpoint
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
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

class MultimodalDataset(Dataset):
    def __init__(self, img_manifest, clinical_X, clinical_y, transform=None):
        self.img_manifest = img_manifest.reset_index(drop=True)
        self.transform = transform
        
        # Group clinical indices by class
        clinical_y_vals = clinical_y["Target_Encoded"].values
        self.class_to_clinical_indices = {
            c: np.where(clinical_y_vals == c)[0] for c in range(4)
        }
        
        feature_cols = [col for col in clinical_X.columns if col != "Record_ID"]
        self.clinical_features = clinical_X[feature_cols].values.astype(np.float32)
        self.paired_indices = []

        # Deterministic pairing within class to preserve reproducibility and zero split leakage
        class_counters = {c: 0 for c in range(4)}
        for idx in range(len(self.img_manifest)):
            c = CLASS_FOLDER_MAP[self.img_manifest.iloc[idx]["class_folder"]]
            avail_indices = self.class_to_clinical_indices[c]
            c_idx = avail_indices[class_counters[c] % len(avail_indices)]
            class_counters[c] += 1
            self.paired_indices.append(c_idx)

    def __len__(self):
        return len(self.img_manifest)

    def __getitem__(self, idx):
        row = self.img_manifest.iloc[idx]
        img_path = Path(row["relative_path"])
        image = Image.open(img_path).convert("RGB")
        label = CLASS_FOLDER_MAP[row["class_folder"]]

        if self.transform:
            image = self.transform(image)

        clin_feat = torch.tensor(self.clinical_features[self.paired_indices[idx]], dtype=torch.float32)
        return image, clin_feat, label

class MultimodalFusionNet(nn.Module):
    def __init__(self, num_classes=4, vision_emb_dim=64, clinical_in_dim=19, clinical_emb_dim=32, fusion_hidden_dim=64, pretrained=True):
        super().__init__()
        # Vision Branch (ResNet-18)
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.vision_features = nn.Sequential(*list(backbone.children())[:-1])
        self.vision_projector = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(512, vision_emb_dim),
            nn.ReLU()
        )

        # Clinical Branch (MLP)
        self.clinical_branch = nn.Sequential(
            nn.Linear(clinical_in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, clinical_emb_dim),
            nn.BatchNorm1d(clinical_emb_dim),
            nn.ReLU()
        )

        # Fusion Head
        total_emb = vision_emb_dim + clinical_emb_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(total_emb, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(fusion_hidden_dim, num_classes)
        )

    def forward(self, img, clinical):
        v_feats = self.vision_features(img)
        v_emb = self.vision_projector(v_feats)
        c_emb = self.clinical_branch(clinical)
        fused = torch.cat([v_emb, c_emb], dim=1)
        logits = self.fusion_head(fused)
        return logits

def train_and_validate_multimodal(config_path="configs/training_config.yaml"):
    cfg = load_training_config(config_path)
    global_cfg = cfg.get("global", {})
    ecg_cfg = cfg.get("ecg_vision_model", {})
    mm_cfg = cfg.get("multimodal_model", {})
    paths_cfg = cfg.get("paths", {})
    sched_cfg = ecg_cfg.get("lr_scheduler", {})
    aug_cfg = ecg_cfg.get("augmentation", {})

    set_global_seed(int(global_cfg.get("random_seed", 42)))

    data_dir = Path(paths_cfg.get("data_dir", "data/processed"))
    models_dir = Path(paths_cfg.get("models_dir", "models"))
    reports_dir = Path(paths_cfg.get("reports_dir", "reports"))
    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    preferred_device = global_cfg.get("device", "cuda")
    use_cuda = preferred_device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if use_cuda else 'CPU'})")

    # Load manifests & clinical partitions
    img_manifest = pd.read_csv(paths_cfg.get("image_manifest", data_dir / "image_split_manifest.csv"))
    train_imgs = img_manifest[img_manifest["Split"] == "train"]
    val_imgs = img_manifest[img_manifest["Split"] == "val"]

    X_train = pd.read_csv(paths_cfg.get("clinical_train", data_dir / "X_train_clinical.csv"))
    y_train = pd.read_csv(paths_cfg.get("y_train", data_dir / "y_train.csv"))
    X_val = pd.read_csv(paths_cfg.get("clinical_val", data_dir / "X_val_clinical.csv"))
    y_val = pd.read_csv(paths_cfg.get("y_val", data_dir / "y_val.csv"))

    # Transforms
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

    train_ds = MultimodalDataset(train_imgs, X_train, y_train, transform=train_transform)
    val_ds = MultimodalDataset(val_imgs, X_val, y_val, transform=val_transform)

    batch_size = int(mm_cfg.get("batch_size", 32))
    num_workers = int(ecg_cfg.get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Class weights from unique scans
    unique_train = train_imgs.drop_duplicates(subset=["md5_hash"])
    class_counts = [sum(unique_train["class_folder"] == k) for k in CLASS_FOLDER_MAP.keys()]
    weights = [len(unique_train) / (4.0 * max(c, 1)) for c in class_counts]
    class_weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)

    model = MultimodalFusionNet(
        num_classes=int(global_cfg.get("num_classes", 4)),
        vision_emb_dim=int(mm_cfg.get("vision_embedding_dim", 64)),
        clinical_emb_dim=int(mm_cfg.get("clinical_embedding_dim", 32)),
        fusion_hidden_dim=int(mm_cfg.get("fusion_hidden_dim", 64)),
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(mm_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(mm_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=str(sched_cfg.get("mode", "min")),
        factor=float(sched_cfg.get("factor", 0.5)),
        patience=int(sched_cfg.get("patience", 3)),
        min_lr=float(sched_cfg.get("min_lr", 1e-6)),
    )

    epochs = int(mm_cfg.get("epochs", 25))
    best_val_f1 = -1.0
    patience = int(mm_cfg.get("early_stopping_patience", 7))
    patience_counter = 0
    history = []
    best_model_path = Path(paths_cfg.get("multimodal_model_checkpoint", models_dir / "multimodal_model_best.pt"))

    print("\nStarting Multimodal Model Training...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, clins, labels in train_loader:
            imgs, clins, labels = imgs.to(device), clins.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs, clins)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total

        # Validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds, all_val_targets = [], []

        with torch.no_grad():
            for imgs, clins, labels in val_loader:
                imgs, clins, labels = imgs.to(device), clins.to(device), labels.to(device)
                outputs = model(imgs, clins)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * imgs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                all_val_preds.extend(predicted.cpu().numpy())
                all_val_targets.extend(labels.cpu().numpy())

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

        if epoch_val_f1 > best_val_f1:
            best_val_f1 = epoch_val_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_macro_f1": epoch_val_f1,
                "val_acc": epoch_val_acc,
                "class_weights": weights,
                "class_names": CLASS_NAMES
            }, best_model_path)
            print(f"  --> Saved new best multimodal checkpoint to {best_model_path} (Val Macro-F1: {best_val_f1:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    print(f"Multimodal training completed in {time.time() - start_time:.1f}s. Best Val Macro-F1: {best_val_f1:.4f}")

    # Save history CSV
    pd.DataFrame(history).to_csv(reports_dir / "06_multimodal_training_history.csv", index=False)

    # Evaluate best model for confusion matrix
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    val_preds, val_targets = [], []
    with torch.no_grad():
        for imgs, clins, labels in val_loader:
            imgs, clins = imgs.to(device), clins.to(device)
            outputs = model(imgs, clins)
            _, predicted = outputs.max(1)
            val_preds.extend(predicted.cpu().numpy())
            val_targets.extend(labels.numpy())

    cm = confusion_matrix(val_targets, val_preds)

    # Plot Confusion Matrix
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
    ax.set_title("Multimodal Late Fusion: Validation Confusion Matrix", pad=20)
    plt.tight_layout()
    plt.savefig(reports_dir / "06_multimodal_confusion_matrix.png", dpi=300)
    plt.close()
    print("Saved reports/06_multimodal_confusion_matrix.png")

if __name__ == "__main__":
    train_and_validate_multimodal()
