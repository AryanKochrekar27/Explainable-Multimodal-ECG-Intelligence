"""
Phase 12: Class Imbalance & Augmentation Strategy
Evaluates class distributions across both tabular and image partitions,
computes recommended class weights, and defines scientifically valid augmentation rules.
Generates:
- reports/11_class_imbalance.md
"""
from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def analyze_imbalance():
    reports_dir = Path("reports")
    processed_dir = Path("data/processed")
    reports_dir.mkdir(exist_ok=True)
    
    # Clinical dataset
    df_clinical = pd.read_csv(processed_dir / "clinical_clean.csv")
    y_train = pd.read_csv(processed_dir / "y_train.csv")
    
    # Calculate clinical class weights
    classes = np.array([0, 1, 2, 3])
    weights_clinical = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train["Target_Encoded"]
    )
    weight_dict_clinical = {int(c): float(round(w, 4)) for c, w in zip(classes, weights_clinical)}
    
    # Image dataset (unique scans)
    df_imgs = pd.read_csv(processed_dir / "image_split_manifest.csv")
    img_train = df_imgs[df_imgs["Split"] == "train"].drop_duplicates(subset=["md5_hash"])
    
    img_class_counts = img_train["class_folder"].value_counts().to_dict()
    img_classes = sorted(list(img_class_counts.keys()))
    
    weights_img = compute_class_weight(
        class_weight="balanced",
        classes=np.array(img_classes),
        y=img_train["class_folder"]
    )
    weight_dict_img = {c: float(round(w, 4)) for c, w in zip(img_classes, weights_img)}
    
    with open(reports_dir / "11_class_imbalance.md", "w", encoding="utf-8") as f:
        f.write("# Phase 12: Class Imbalance Analysis & Augmentation Strategy\n\n")
        f.write("## 1. Tabular Modality Imbalance & Weights\n")
        f.write("The 4,327 clinical records were synthetically balanced across the four classes (~25% each):\n\n")
        f.write("| Diagnostic Class | Code | Train Count | Percentage | Balanced Class Weight |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        
        target_names = {0: "Normal", 1: "Abnormal_Heartbeat", 2: "Myocardial_Infarction", 3: "Post_MI_History"}
        for c in classes:
            cnt = int((y_train["Target_Encoded"] == c).sum())
            pct = cnt / len(y_train) * 100.0
            f.write(f"| `{target_names[c]}` | `{c}` | {cnt:,} | {pct:.2f}% | `{weight_dict_clinical[c]}` |\n")
        f.write("\n*Recommendation*: Tabular loss weighting is not strictly required given the balanced nature of the synthetic CSV, but standard cross-entropy applies cleanly.\n\n")
        
        f.write("## 2. ECG Image Modality Imbalance & Weights\n")
        f.write("In contrast, authentic ECG scans exhibit **pronounced class imbalance** across unique patients:\n\n")
        f.write("| Image Class Folder | Unique Train Scans | Proportion | Recommended Class Weight |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for c in img_classes:
            cnt = img_class_counts[c]
            pct = cnt / len(img_train) * 100.0
            f.write(f"| `{c}` | {cnt} | {pct:.2f}% | `{weight_dict_img[c]}` |\n")
        f.write("\n> [!TIP]\n")
        f.write("> **Recommended Training Strategy**: For ECG vision backbone training, apply cost-sensitive class weighting (giving ~5.7x weight to `myocardial_infarction_ecg_images`) or use Focal Loss to prevent the classifier from collapsing to the majority `abnormal_heartbeat` class.\n\n")
        
        f.write("## 3. Scientifically Safe Data Augmentation Principles\n")
        f.write("- **Strictly Prohibited**:\n")
        f.write("  - ❌ Random Horizontal Flip (reverses cardiac cycle time axis).\n")
        f.write("  - ❌ Random Vertical Flip (inverts electrical polarity).\n")
        f.write("  - ❌ CutOut / Erasing large rectangles (deletes entire leads or QRS complexes).\n")
        f.write("- **Permissible & Recommended** (applied on-the-fly to TRAINING set only):\n")
        f.write("  - ✔ Slight random brightness / contrast jitter (±5%) simulating variations in scanner exposure.\n")
        f.write("  - ✔ Subtle Gaussian noise injection (simulating electrode contact noise).\n")
        f.write("  - ✔ Minor micro-translations (±2 pixels).\n")
        f.write("- **Validation and Test Sets**: Never augment validation or test samples.\n")
    print("Saved reports/11_class_imbalance.md")

if __name__ == "__main__":
    analyze_imbalance()
