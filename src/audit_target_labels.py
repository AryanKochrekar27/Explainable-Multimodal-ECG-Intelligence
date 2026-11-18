"""
Phase 5: Verify Target Labels and Dataset Meaning
Verifies the 4 target categories across documentation, CSV, and image folder taxonomy.
Generates:
- reports/05_target_label_audit.md
- reports/05_class_distribution.csv
- processed/label_mapping.json
"""
import json
from pathlib import Path
import pandas as pd

def audit_target_labels():
    reports_dir = Path("reports")
    processed_dir = Path("data/processed")
    reports_dir.mkdir(exist_ok=True)
    processed_dir.mkdir(exist_ok=True)
    
    # Load CSV
    csv_path = Path("data/raw/cardiovascular_dataset.csv")
    if not csv_path.exists():
        csv_path = Path("initial_dataset/cardiovascular_dataset.csv")
    df_csv = pd.read_csv(csv_path)
    
    # Load Image Audit
    img_audit_path = reports_dir / "03_image_quality_audit.csv"
    df_imgs = pd.read_csv(img_audit_path)
    
    # Target definition mapping
    label_mapping = {
        "Normal": 0,
        "Abnormal_Heartbeat": 1,
        "Myocardial_Infarction": 2,
        "Post_MI_History": 3
    }
    
    # Save label_mapping.json
    with open(processed_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=4)
    print("Saved data/processed/label_mapping.json")
    
    # CSV class distribution
    csv_counts = df_csv["Target"].value_counts().to_dict()
    total_csv = len(df_csv)
    
    # Image class distribution
    img_folder_to_target = {
        "normal_ecg_images": "Normal",
        "abnormal_heartbeat_ecg_images": "Abnormal_Heartbeat",
        "myocardial_infarction_ecg_images": "Myocardial_Infarction",
        "post_mi_history_ecg_images": "Post_MI_History"
    }
    df_imgs["target_label"] = df_imgs["class_folder"].map(img_folder_to_target)
    img_counts = df_imgs["target_label"].value_counts().to_dict()
    img_unique_counts = df_imgs.drop_duplicates(subset=["md5_hash"])["target_label"].value_counts().to_dict()
    total_imgs = len(df_imgs)
    total_unique_imgs = df_imgs["md5_hash"].nunique()
    
    # Combined distribution table
    dist_rows = []
    for lbl, code in label_mapping.items():
        c_count = csv_counts.get(lbl, 0)
        c_pct = (c_count / total_csv) * 100.0
        i_count = img_counts.get(lbl, 0)
        i_pct = (i_count / total_imgs) * 100.0
        u_count = img_unique_counts.get(lbl, 0)
        u_pct = (u_count / total_unique_imgs) * 100.0
        dist_rows.append({
            "target_label": lbl,
            "encoded_id": code,
            "csv_records": c_count,
            "csv_percentage": round(c_pct, 2),
            "image_files": i_count,
            "image_percentage": round(i_pct, 2),
            "unique_images": u_count,
            "unique_image_percentage": round(u_pct, 2)
        })
        
    df_dist = pd.DataFrame(dist_rows)
    df_dist.to_csv(reports_dir / "05_class_distribution.csv", index=False)
    print("Saved reports/05_class_distribution.csv")
    
    # Write Phase 5 Markdown Report
    with open(reports_dir / "05_target_label_audit.md", "w", encoding="utf-8") as f:
        f.write("# Phase 5: Target Label Audit & Clinical Semantics Report\n\n")
        f.write("## 1. Ground Truth Target Definitions\n")
        f.write("Based on official documentation (`Multimodal_Cardiovascular_Risk_Dataset_Overview_Final.pdf`), the dataset supports **four distinct diagnostic categories**:\n")
        f.write("1. **Normal (Encoded: 0)**: 12-lead ECG traces and clinical parameters within normal physiological limits.\n")
        f.write("2. **Abnormal Heartbeat (Encoded: 1)**: Cardiac arrhythmia / irregular rhythm disturbances (e.g., atrial premature beats, ectopic contractions, tachycardia/bradycardia patterns).\n")
        f.write("3. **Myocardial Infarction (Encoded: 2)**: Acute myocardial infarction showing ST-elevation or pathological Q-wave ischemic injury patterns.\n")
        f.write("4. **Post-MI History (Encoded: 3)**: Chronic, post-infarction status showing established pathological changes or prior tissue necrosis.\n\n")
        
        f.write("## 2. Clinical Task Formulation: Diagnostic Classification vs. Future Risk Prediction\n")
        f.write("> [!IMPORTANT]\n")
        f.write("> **Scientific Distinction (Rule 6)**: The dataset does **not** contain longitudinal follow-up data, time-to-event indicators, or future hazard outcomes. It records *current diagnostic state* and *prior cardiac history*. Consequently, the scientific task is strictly **ECG-Condition & Diagnostic-Category Classification**, not future cardiac-event forecasting.\n\n")
        
        f.write("## 3. Multimodal Class Distribution Comparison\n\n")
        f.write("| Target Label | Encoded ID | CSV Records (N=4,327) | CSV % | Image Files (N=928) | Image % | Unique Scans (N=491) | Unique % |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in dist_rows:
            f.write(f"| `{r['target_label']}` | `{r['encoded_id']}` | {r['csv_records']:,} | {r['csv_percentage']:.2f}% | {r['image_files']} | {r['image_percentage']:.2f}% | {r['unique_images']} | {r['unique_image_percentage']:.2f}% |\n")
        f.write(f"| **Total** | | **{total_csv:,}** | **100.00%** | **{total_imgs}** | **100.00%** | **{total_unique_imgs}** | **100.00%** |\n\n")
        
        f.write("## 4. Class Imbalance and Duplication Assessment\n")
        f.write("- **CSV Distribution**: Artificially balanced at ~25% per class (Normal: 25.7%, Abnormal: 25.2%, Post-MI: 25.2%, MI: 23.8%).\n")
        f.write("- **Image Collection Imbalance**: In unique physical images, substantial imbalance exists:\n")
        f.write("  - `Abnormal_Heartbeat`: 233 unique scans (47.45%)\n")
        f.write("  - `Normal`: 142 unique scans (28.92%)\n")
        f.write("  - `Post_MI_History`: 86 unique scans (17.52%)\n")
        f.write("  - `Myocardial_Infarction`: 30 unique scans (6.11%)\n")
        f.write("- The 239 total MI image files were created by duplicating 30 unique scans 8 times, which would lead to catastrophic data leakage if split naively.\n")
    print("Saved reports/05_target_label_audit.md")

if __name__ == "__main__":
    audit_target_labels()
