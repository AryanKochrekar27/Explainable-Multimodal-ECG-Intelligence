# Phase 8: ECG Image Preprocessing Specification Report

## 1. Vision Pipeline Architecture
Electrocardiogram scans require specialized computer vision preprocessing to preserve diagnostic waveform geometry:
- **Decodability Check**: Strict two-phase PIL verification to prevent silent IO failures.
- **Color Space**: Maintained in 3-channel RGB to support transfer learning on convolutional / vision transformer architectures (e.g., EfficientNet, ResNet-50, ViT).
- **Interpolation Algorithm**: High-order Lanczos resampling (`Image.Resampling.LANCZOS`) is utilized to retain fine trace edges, sharp QRS complexes, and calibration ticks without introducing blur artifacts.
- **Target Resolution**: Default standardized resolution is set to **224 × 224 × 3** (configurable to 256×256 or 512×512 in `configs/preprocessing_config.json`).
- **Pixel Normalization**: Scaled strictly to [0.0, 1.0] float32 values (or standard ImageNet z-score normalization).

## 2. Prohibition of Destructive Augmentations
> [!CAUTION]
> **Non-Negotiable Rule**: In clinical ECG analysis, operations such as random horizontal/vertical flipping, aggressive random rotations, and spatial shear **MUST NOT** be applied. A horizontal flip reverses temporal cardiac conduction (P-wave appearing after T-wave), and a vertical flip turns positive R-waves into negative deflections, completely corrupting diagnostic validity.

## 3. Storage Efficiency
To prevent storing hundreds of megabytes of redundant downscaled image files, deterministic on-the-fly loading is implemented via `src/preprocess_images.py`, which is called directly by PyTorch/TensorFlow Dataset loaders.
