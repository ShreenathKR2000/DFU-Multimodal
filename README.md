# DFU-MMT: Diabetic Foot Ulcer Modeling (RGB, Thermal, Multimodal)

This repository implements and evaluates three DFU classifiers:

- RGB-only (ResNet50)
- Thermal-only (ViT-Base)
- Multimodal Fusion (ResNet50 + ViT-Base)

It includes training scripts, extended medical metrics, and Grad-CAM visualizations on the test sets in `Dataset/data/`. The current results indicate very strong single-modality performance and a limitation in the fusion model due to the lack of a truly paired dataset (explained below).

## Project Overview

Diabetic foot ulcers (DFUs) are a major complication of diabetes. Early detection (Healthy vs Ulcer classification) benefits from combining surface texture (RGB) and physiological signals (Thermal). This project trains and evaluates separate RGB-only and Thermal-only models and a late-fusion multimodal model.

Key scripts and notebooks:

- Training: `DFU_MMT/notebooks/train_rgb_only.py`, `DFU_MMT/notebooks/train_thermal_only.py`, `DFU_MMT/notebooks/train_multimodal_fusion.py`
- Evaluation: `DFU_MMT/notebooks/extended_metrics.py`
- Explainability: `DFU_MMT/notebooks/grad_cam_visualization.py`

## Dataset Structure

The organized dataset used for evaluation and visualizations follows:

```
Dataset/data/
  rgb/
    train|val|test/
      healthy/
      ulcer/
  thermal/
    train|val|test/
      healthy/
      ulcer/
```

Evaluation and Grad-CAM operate on the `test` split for both modalities:

- `Dataset/data/rgb/test` → RGB images (Healthy=0, Ulcer=1)
- `Dataset/data/thermal/test` → Thermal images (Healthy=0, Ulcer=1)

## Models

All models use a two-class output (`num_classes=2`) with CrossEntropyLoss and softmax inference. This matches training and fixes earlier evaluation mismatches.

- RGB-only (ResNet50)
  - Backbone: `torchvision.resnet50` (ImageNet weights)
  - Head: `nn.Sequential(Dropout(0.5), Linear(2048 → 2))`
  - Script: `train_rgb_only.py`

- Thermal-only (ViT-Base)
  - Backbone: `timm.create_model('vit_base_patch16_224')`
  - Head: `nn.Sequential(Dropout(0.5), Linear(768 → 2))`
  - Script: `train_thermal_only.py`

- Multimodal Fusion (Late fusion)
  - RGB branch: ResNet50 with `fc = Identity()` to output 2048-d features
  - Thermal branch: ViT-Base with features (768-d)
  - Fusion MLP: `Linear(2048+768 → 512) → ReLU → Dropout(0.7) → Linear(512 → 2)`
  - Script: `train_multimodal_fusion.py`

## Multimodal Pairing Limitation (Important)

The RGB and Thermal datasets come from different sources and are not patient-level paired. The current multimodal dataset is constructed by label-matching (Healthy ↔ Healthy, Ulcer ↔ Ulcer) and pseudo-pairing indices within each class. This means an RGB image and a Thermal image in a pair do not correspond to the same foot or patient.

Consequences observed:

- The fusion model learns to always predict the positive class (Ulcer), yielding specificity of 0% on the test set.
- This behavior is consistent with non-informative cross-modality pairing, where the model cannot learn coherent joint patterns.

Resolution path:

- Acquire or construct truly paired RGB+Thermal DFU data (same patient/foot, same visit).
- Alternatively, redesign fusion to be robust to unpaired data (e.g., contrastive pretraining, modality-specific heads with calibrated ensembling) — still inferior to real pairing.

## Results (Extended Metrics)

Extended test metrics are stored in:

- `DFU_MMT/logs/extended_metrics/rgb_only/results.pt`
- `DFU_MMT/logs/extended_metrics/thermal_only/results.pt`
- `DFU_MMT/logs/extended_metrics/multimodal/results.pt`

Curves and confusion matrices are saved as PNGs in the same directories.

Summary (from the saved metrics):

RGB-only (Test: 131 images → 36 Healthy, 95 Ulcer)

- Confusion: TN=35, FP=1, FN=1, TP=94
- Accuracy: 0.9847 | F1: 0.9895
- Sensitivity (Recall Ulcer): 0.9895 | Specificity (Healthy): 0.9722
- ROC-AUC: 0.9994 | PR-AUC: 0.9998

Thermal-only (Test: 276 images → 130 Healthy, 146 Ulcer)

- Confusion: TN=130, FP=0, FN=3, TP=143
- Accuracy: 0.9891 | F1: 0.9896
- Sensitivity: 0.9795 | Specificity: 1.0000
- ROC-AUC: 0.9997 | PR-AUC: 0.9997

Multimodal Fusion (Test: 276 pairs → 130 Healthy, 146 Ulcer)

- Confusion: TN=0, FP=130, FN=0, TP=146
- Accuracy: 0.5290 | F1: 0.6919
- Sensitivity: 1.0000 | Specificity: 0.0000
- ROC-AUC: 0.3876 | PR-AUC: 0.4653

Interpretation:

- Single-modality models (RGB-only, Thermal-only) perform near perfectly and match training-level performance.
- The fusion model collapses to predicting “Ulcer” for all samples — an expected outcome without true pairing.

## Grad-CAM Visualizations

The Grad-CAM script generates balanced visualizations (5 healthy + 5 ulcer per model) from the `test` sets:

- Script: `DFU_MMT/notebooks/grad_cam_visualization.py`
- Output directory: `DFU_MMT/logs/grad_cam_visualizations/`
  - Subfolders: `rgb_only/`, `thermal_only/`, `multimodal/`
  - Files: `healthy_00.png`, `ulcer_00.png`, etc.

The script uses softmax predictions and correct two-class heads for all models.

## How to Run

Activate the environment:

```bash
source /home/skr/CompVis/.venv/bin/activate
```

Train:

```bash
# RGB-only
python DFU_MMT/notebooks/train_rgb_only.py

# Thermal-only
python DFU_MMT/notebooks/train_thermal_only.py

# Multimodal fusion (note the pairing limitation)
python DFU_MMT/notebooks/train_multimodal_fusion.py
```

Evaluate extended metrics (uses test folders under `Dataset/data/`):

```bash
python DFU_MMT/notebooks/extended_metrics.py
```

Generate Grad-CAM visualizations (balanced 5+5 per model):

```bash
python DFU_MMT/notebooks/grad_cam_visualization.py
```

## Checkpoints & Outputs

Checkpoints

- RGB-only: `DFU_MMT/logs/checkpoints_rgb_only/best_model.pt`
- Thermal-only: `DFU_MMT/logs/checkpoints_thermal_only/best_model.pt`
- Multimodal fusion: `DFU_MMT/logs/checkpoints_multimodal/best_model.pt`

Extended Metrics

- `DFU_MMT/logs/extended_metrics/{rgb_only,thermal_only,multimodal}/results.pt`
- Curves: `roc_curve_*.png`, `pr_curve_*.png`
- Confusion: `confusion_matrix_*.png`

Grad-CAM Visualizations

- `DFU_MMT/logs/grad_cam_visualizations/{rgb_only,thermal_only,multimodal}/`

## Implementation Notes

- All evaluation scripts now use two-class outputs with CrossEntropyLoss and softmax predictions.
- Checkpoint loaders map training-time keys (`backbone.*`) to evaluation-time modules (`resnet.*` or `vit.*`) and load with `strict=False` for compatibility.
- Thermal images are converted to three-channel RGB for consistency with ViT input.

## Future Work

- Build or acquire a truly paired RGB+Thermal DFU dataset (same patient/visit) to enable effective fusion.
- Explore robust fusion strategies with unpaired data (contrastive multimodal pretraining, calibrated ensembling).
- Add patient-level metrics, calibration curves, and threshold analysis.
- Automate Grad-CAM batch export and reports across all splits.

## Acknowledgements

This work uses PyTorch, TorchVision, TIMM, scikit-learn, and common explainability tools (Grad-CAM). Data organization follows the `Dataset/data` structure prepared by the included scripts.
RGB Dataset (Kaggle: laithjj/diabetic-foot-ulcer-dfu)
Thermal Dataset (Kaggle: vuppalaadithyasairam/thermography-images-of-diabetic-foot)

