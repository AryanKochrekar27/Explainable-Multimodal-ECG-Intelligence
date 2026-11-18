# Step 2: ECG Input Representation & Model Architecture Specification

**Project**: Explainable ECG Intelligence for Early Cardiac Risk Assessment  
**Stage**: Phase 1 & 2 Model Architecture Formalization  
**Date**: October 2026  
**Document Status**: Approved Engineering Specification  

---

## 1. Input Modality Formalization & Biomedical Scientific Rationale

### 1.1 ECG Image Modality (2D Computer Vision on Paper ECG Scans)
- **Input Nature**: Scanned, digitized 12-lead paper ECG recordings from an EDAN SE-3 / SEMIP machine.
- **Native Resolution**: 2213 × 1572 pixels, 3-channel RGB.
- **Standardized Model Input**: $224 \times 224 \times 3$ RGB tensors, Lanczos downsampled, pixel-scaled to $[0.0, 1.0]$.
- **Critical Scientific Constraint**: 
  These files are rasterized visual images depicting 12 rhythm strips printed on standard grid paper ($25\text{ mm/s}, 10\text{ mm/mV}$). They are **not** raw voltage-time vectors. Applying 1D digital signal processing (such as Butterworth bandpass filtering, Pan-Tompkins QRS detection, or 1D-CNNs) directly to image pixels is physically invalid. Therefore, the vision branch operates **strictly as a 2D Computer Vision deep convolutional network**.
- **Data Augmentation Guardrails**:
  - Prohibited: Horizontal flips (inverses cardiac conduction direction in time), Vertical flips (inverts electrical lead vectors), large bounding box cutouts (deletes leads).
  - Permissible: Subtle scanner exposure jitter ($\pm 5\%$), Gaussian sensor noise injection ($\sigma=0.01$).

### 1.2 Clinical Tabular Modality
- **Input Nature**: 19 structured clinical variables comprising demographics, hemodynamics, biomarkers, and automated electrocardiographic interval measurements.
- **Preprocessing Pipeline**:
  - Numerical features (18): Standardized via $Z = (X - \mu) / \sigma$ fitted strictly on $X_{\text{train}}$.
  - Categorical features (1: `Gender`): One-Hot Encoded.
- **Input Shape**: $\mathbf{x}_{\text{clinical}} \in \mathbb{R}^{19}$.

---

## 2. Vision Backbone Architecture (Track A: ECG-Only Model)

We employ a customized **ResNet-18** 2D convolutional neural network architecture:

```
Input Image [3, 224, 224]
  │
  ▼
Conv2d(3 -> 64, 7x7, stride=2, padding=3) + BatchNorm2d + ReLU
  │
  ▼
MaxPool2d(3x3, stride=2, padding=1) -> [64, 56, 56]
  │
  ▼
Layer 1: 2x BasicBlock (64 filters)  -> [64, 56, 56]
  │
  ▼
Layer 2: 2x BasicBlock (128 filters) -> [128, 28, 28]
  │
  ▼
Layer 3: 2x BasicBlock (256 filters) -> [256, 14, 14]
  │
  ▼
Layer 4: 2x BasicBlock (512 filters) -> [512, 7, 7]  <-- [Grad-CAM Target Layer]
  │
  ▼
AdaptiveAvgPool2d((1, 1)) -> [512]
  │
  ▼
Flatten -> Dropout(p=0.3)
  │
  ▼
Dense / Linear(512 -> 64) + ReLU + Dropout(p=0.2) -> [Vision Embedding: 64-d]
  │
  ▼
Linear(64 -> 4) -> Raw Logits
  │
  ▼
Softmax -> [p_Normal, p_Abnormal, p_MI, p_PostMI]
```

### 2.1 Loss Function & Class-Weighted Cross-Entropy
Given significant imbalance in unique scans (Myocardial Infarction has only 30 unique scans vs 233 Abnormal Heartbeat scans in the dataset), we configure cost-sensitive categorical cross-entropy:
$$\mathcal{L} = -\sum_{c=0}^{3} w_c \cdot y_c \cdot \log(\hat{y}_c)$$
where $w_c = \frac{N_{\text{total}}}{4 \times N_c}$ based on unique training patient scans:
- $w_{\text{Normal}} \approx 0.86$
- $w_{\text{Abnormal\_Heartbeat}} \approx 0.49$
- $w_{\text{Myocardial\_Infarction}} \approx 3.42$
- $w_{\text{Post\_MI\_History}} \approx 1.48$

### 2.2 Optimization & Learning Rate Scheduling
- **Optimizer**: AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $= 10^{-4}$).
- **Initial Learning Rate**: $3 \times 10^{-4}$.
- **LR Scheduler**: `ReduceLROnPlateau` monitoring validation loss (factor $= 0.5$, patience $= 3$ epochs, min LR $= 10^{-6}$).
- **Early Stopping**: Patience $= 7$ epochs on validation loss to prevent overfitting.

---

## 3. Tabular Baseline Architectures (Track B: Clinical-Only Model)

To rigorously assess tabular signal strength, three distinct inductive biases are benchmarked:
1. **Linear / Interpretable**: L2-regularized Multiclass Logistic Regression (`solver='lbfgs'`, $C=1.0$).
2. **Tree Ensemble**: Balanced Random Forest (100 estimators, max depth $= 10$, min samples split $= 5$).
3. **Neural MLP**: Tabular Multi-Layer Perceptron:
   $$\mathbf{h}_1 = \text{ReLU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) \in \mathbb{R}^{64}$$
   $$\mathbf{h}_2 = \text{Dropout}_{0.2}(\text{ReLU}(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2)) \in \mathbb{R}^{32}$$
   $$\hat{\mathbf{y}} = \text{Softmax}(\mathbf{W}_3 \mathbf{h}_2 + \mathbf{b}_3) \in \mathbb{R}^{4}$$

---

## 4. Multimodal Late-Fusion Architecture (Track C)

The multimodal fusion architecture maps both modalities into compact latent vectors before joint classification:

```
[ECG Image: 3x224x224]               [Clinical Vector: 19-d]
          │                                     │
          ▼                                     ▼
Vision Backbone (ResNet-18)            Clinical MLP Projector
          │                                     │
          ▼                                     ▼
 Vision Latent (64-d)                  Clinical Latent (32-d)
          │                                     │
          └──────────────────┬──────────────────┘
                             │
                             ▼
              Concatenation Layer (96-d)
                             │
                             ▼
                 Dense(64, ReLU) + LayerNorm
                             │
                             ▼
                       Dropout(0.3)
                             │
                             ▼
                      Dense(4, Softmax)
                             │
                             ▼
                   Joint Cardiac Risk
```

---

## 5. Summary of Architecture Specifications

| Model Track | Backbone / Method | Parameters | Input Dimensions | Primary Loss |
| :--- | :--- | :--- | :--- | :--- |
| **Track A: ECG Vision** | ResNet-18 + Projector | ~11.2M | $(3, 224, 224)$ | Weighted Cross-Entropy |
| **Track B: Clinical Tabular** | Random Forest / MLP | ~3.5K | $(19,)$ | Multi-LogLoss / Cross-Entropy |
| **Track C: Multimodal Fusion**| ResNet-18 + MLP + Late Fusion | ~11.25M | $(3, 224, 224) \times (19,)$ | Cross-Entropy |
