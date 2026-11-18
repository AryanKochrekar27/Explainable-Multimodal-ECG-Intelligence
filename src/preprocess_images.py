"""
Phase 8: ECG Image Preprocessing Pipeline
Provides a deterministic, reproducible image preprocessing module:
- Safe image loading & decoding validation
- Color space conversion (RGB / Grayscale)
- Resizing to standard model input resolutions (e.g. 224x224, 256x256)
- Normalization ([0, 1] scaling or ImageNet normalization)
- Saves sample comparison: reports/08_original_vs_processed.png
- Generates reports/08_image_preprocessing.md
"""
from pathlib import Path
from typing import Tuple, Union, Optional
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class ECGImagePreprocessor:
    """
    Standardized, reproducible ECG Image Preprocessor for deep learning vision backbones.
    Preserves diagnostic waveform features while resizing and normalizing.
    """
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        color_mode: str = "RGB",
        normalization: str = "zero_one"  # 'zero_one' or 'imagenet'
    ):
        self.target_size = target_size
        self.color_mode = color_mode.upper()
        self.normalization = normalization
        
        # ImageNet constants
        self.imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def load_and_preprocess(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Loads an ECG image, verifies decodability, converts mode, resizes, and normalizes.
        Returns a float32 numpy array.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"ECG image not found: {path}")
            
        with Image.open(path) as img:
            # Validate decoding
            img.verify()
            
        with Image.open(path) as img:
            # Convert color mode
            if self.color_mode == "RGB":
                img = img.convert("RGB")
            elif self.color_mode == "L":
                img = img.convert("L")
            else:
                raise ValueError(f"Unsupported color mode: {self.color_mode}")
                
            # Resize with Lanczos interpolation for sharp line preservation
            img_resized = img.resize(self.target_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy float32
            arr = np.array(img_resized, dtype=np.float32)
            
            # Normalization
            if self.normalization == "zero_one":
                arr = arr / 255.0
            elif self.normalization == "imagenet":
                arr = (arr / 255.0 - self.imagenet_mean) / self.imagenet_std
            elif self.normalization == "none":
                pass
            else:
                raise ValueError(f"Unknown normalization: {self.normalization}")
                
            return arr

def generate_comparison_report():
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    preprocessor = ECGImagePreprocessor(target_size=(224, 224), color_mode="RGB", normalization="zero_one")
    
    sample_classes = {
        "Normal": "initial_dataset/ECG Images/ECG Images/normal_ecg_images/Normal(1).jpg",
        "Abnormal Heartbeat": "initial_dataset/ECG Images/ECG Images/abnormal_heartbeat_ecg_images/HB(1).jpg",
        "Myocardial Infarction": "initial_dataset/ECG Images/ECG Images/myocardial_infarction_ecg_images/MI(1).jpg",
        "Post-MI History": "initial_dataset/ECG Images/ECG Images/post_mi_history_ecg_images/PMI(1).jpg"
    }
    
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    fig.suptitle("Original (2213x1572) vs. Preprocessed Model Input (224x224)", fontsize=16)
    
    for idx, (cls_name, img_path) in enumerate(sample_classes.items()):
        p = Path(img_path)
        with Image.open(p) as orig_im:
            axes[idx, 0].imshow(orig_im)
            axes[idx, 0].set_title(f"Original: {cls_name}\n({orig_im.size[0]}x{orig_im.size[1]})", fontsize=10)
            axes[idx, 0].axis("off")
            
        processed_arr = preprocessor.load_and_preprocess(p)
        axes[idx, 1].imshow(processed_arr)
        axes[idx, 1].set_title(f"Preprocessed: {cls_name}\n(224x224, Range: [{processed_arr.min():.2f}, {processed_arr.max():.2f}])", fontsize=10)
        axes[idx, 1].axis("off")
        
    plt.tight_layout()
    comparison_path = reports_dir / "08_original_vs_processed.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved original vs processed comparison to: {comparison_path}")
    
    # Generate Phase 8 Markdown report
    with open(reports_dir / "08_image_preprocessing.md", "w", encoding="utf-8") as f:
        f.write("# Phase 8: ECG Image Preprocessing Specification Report\n\n")
        f.write("## 1. Vision Pipeline Architecture\n")
        f.write("Electrocardiogram scans require specialized computer vision preprocessing to preserve diagnostic waveform geometry:\n")
        f.write("- **Decodability Check**: Strict two-phase PIL verification to prevent silent IO failures.\n")
        f.write("- **Color Space**: Maintained in 3-channel RGB to support transfer learning on convolutional / vision transformer architectures (e.g., EfficientNet, ResNet-50, ViT).\n")
        f.write("- **Interpolation Algorithm**: High-order Lanczos resampling (`Image.Resampling.LANCZOS`) is utilized to retain fine trace edges, sharp QRS complexes, and calibration ticks without introducing blur artifacts.\n")
        f.write("- **Target Resolution**: Default standardized resolution is set to **224 × 224 × 3** (configurable to 256×256 or 512×512 in `configs/preprocessing_config.json`).\n")
        f.write("- **Pixel Normalization**: Scaled strictly to [0.0, 1.0] float32 values (or standard ImageNet z-score normalization).\n\n")
        
        f.write("## 2. Prohibition of Destructive Augmentations\n")
        f.write("> [!CAUTION]\n")
        f.write("> **Non-Negotiable Rule**: In clinical ECG analysis, operations such as random horizontal/vertical flipping, aggressive random rotations, and spatial shear **MUST NOT** be applied. A horizontal flip reverses temporal cardiac conduction (P-wave appearing after T-wave), and a vertical flip turns positive R-waves into negative deflections, completely corrupting diagnostic validity.\n\n")
        
        f.write("## 3. Storage Efficiency\n")
        f.write("To prevent storing hundreds of megabytes of redundant downscaled image files, deterministic on-the-fly loading is implemented via `src/preprocess_images.py`, which is called directly by PyTorch/TensorFlow Dataset loaders.\n")
    print("Saved reports/08_image_preprocessing.md")

if __name__ == "__main__":
    generate_comparison_report()
