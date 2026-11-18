"""
Audit Dataset Inventory (Phase 1)
Discovers raw datasets, inspects file structures, extensions, sizes, and produces inventory report and CSV.
"""
import os
import hashlib
from pathlib import Path
import pandas as pd

def audit_inventory(workspace_dir: str = "."):
    workspace = Path(workspace_dir).resolve()
    records = []
    
    # Target areas to inspect: initial_dataset, data/raw, and root documents
    search_dirs = [workspace / "initial_dataset", workspace / "data" / "raw", workspace]
    
    seen_files = set()
    for s_dir in search_dirs:
        if not s_dir.exists():
            continue
        for root, dirs, files in os.walk(s_dir):
            # skip git, system dirs
            if any(part.startswith(".") for part in Path(root).parts):
                continue
            for f in files:
                f_path = Path(root) / f
                if f_path in seen_files:
                    continue
                seen_files.add(f_path)
                
                size = f_path.stat().st_size
                ext = f_path.suffix.lower()
                rel_path = f_path.relative_to(workspace)
                
                # Check category
                if ext in [".jpg", ".jpeg", ".png"]:
                    cat = "ECG Image"
                elif ext == ".csv":
                    cat = "Structured Tabular (CSV)"
                elif ext in [".pdf", ".docx", ".xlsx", ".txt", ".md"]:
                    cat = "Documentation / Reference"
                elif ext in [".py"]:
                    cat = "Code"
                elif ext in [".zip", ".tar", ".gz", ".7z"]:
                    cat = "Archive"
                else:
                    cat = "Other Asset"
                    
                records.append({
                    "relative_path": str(rel_path).replace("\\", "/"),
                    "filename": f,
                    "extension": ext,
                    "size_bytes": size,
                    "category": cat
                })
                
    df_inv = pd.DataFrame(records)
    reports_dir = workspace / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    csv_out = reports_dir / "01_dataset_inventory.csv"
    df_inv.to_csv(csv_out, index=False)
    print(f"Saved inventory CSV: {csv_out} with {len(df_inv)} files.")
    
    # Summarize image folder counts
    img_records = df_inv[df_inv["category"] == "ECG Image"]
    class_counts = {}
    for p in img_records["relative_path"]:
        parts = p.split("/")
        if len(parts) >= 4 and parts[1] == "ECG Images":
            cls = parts[3]
            class_counts[cls] = class_counts.get(cls, 0) + 1
            
    md_out = reports_dir / "01_dataset_inventory.md"
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("# Phase 1: Dataset Inventory Report\n\n")
        f.write("## 1. Executive Summary\n")
        f.write(f"- **Total files inventoried**: {len(df_inv)}\n")
        f.write(f"- **Total ECG Images found**: {len(img_records)}\n")
        f.write(f"- **Primary Tabular CSV**: `initial_dataset/cardiovascular_dataset.csv` (and copied to `data/raw/`)\n")
        f.write(f"- **Primary Documentation**: `Multimodal_Cardiovascular_Risk_Dataset_Overview_Final.pdf`, `IEEE_Explainable_ECG_Intelligence.docx`, `Explainable_ECG_Intelligence_Literature_Survey_50_Papers.xlsx`\n\n")
        
        f.write("## 2. ECG Image Collection Breakdown\n")
        f.write("The ECG image files reside in `initial_dataset/ECG Images/ECG Images/` across 4 class subdirectories:\n\n")
        f.write("| Subdirectory / Class Folder | File Count | Sample Filename | Original Dimensions |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        f.write(f"| `abnormal_heartbeat_ecg_images` | {class_counts.get('abnormal_heartbeat_ecg_images', 0)} | `HB(1).jpg` | 2213 × 1572 |\n")
        f.write(f"| `myocardial_infarction_ecg_images` | {class_counts.get('myocardial_infarction_ecg_images', 0)} | `MI(1).jpg` | 2213 × 1572 |\n")
        f.write(f"| `normal_ecg_images` | {class_counts.get('normal_ecg_images', 0)} | `Normal(1).jpg` | 2213 × 1572 |\n")
        f.write(f"| `post_mi_history_ecg_images` | {class_counts.get('post_mi_history_ecg_images', 0)} | `PMI(1).jpg` | 2213 × 1572 |\n")
        f.write(f"| **Total** | **{sum(class_counts.values())}** | | |\n\n")
        
        f.write("## 3. Discrepancy & Ambiguity Findings\n")
        f.write("1. **Record Count Mismatch**: The tabular CSV contains **4,327** records with filenames formatted as `ecg_1.jpg` through `ecg_4327.jpg`. However, the physical image directory contains only **928** image files with names formatted as `HB(x).jpg`, `MI(x).jpg`, `Normal(x).jpg`, and `PMI(x).jpg`.\n")
        f.write("2. **Duplicate Image Files**: As documented in the image audit, only **491** unique image hashes exist among the 928 image files. Specifically, `myocardial_infarction_ecg_images` contains 8 repeated copies of 30 distinct images; `normal_ecg_images` contains 2 repeated copies of 142 distinct images; `post_mi_history_ecg_images` contains 2 repeated copies of 86 distinct images.\n")
        f.write("3. **Archive Status**: The raw data was found extracted into `initial_dataset/` rather than an unextracted `.zip` file on disk.\n")
    print(f"Saved inventory Markdown report: {md_out}")

if __name__ == "__main__":
    audit_inventory()
