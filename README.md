# Automated Early Detection of Diabetic Foot Ulcers Using Multimodal Deep Learning

Main Takeaway
-------------
Develop a multimodal deep learning system that classifies and stratifies early-stage diabetic foot ulcers by integrating standard photographic images (RGB) with thermal infrared data to improve early diagnosis, risk assessment, and personalized treatment recommendations.

1. Project Overview
-------------------
Diabetic foot ulcers (DFUs) are a severe complication of diabetes, leading to infection, amputation, and high healthcare costs if not detected early. This project implements a multimodal CNN–Transformer fusion model that processes both RGB photographs and thermal infrared images of the foot to:

- Detect presence of ulcers at very early stages (Stage 0–I).
- Classify ulcer risk levels (low, moderate, high) based on tissue perfusion and temperature anomalies.
- Recommend personalized care interventions (e.g., offloading, topical treatments).

By combining surface texture features (RGB) with physiological cues (thermal), the system aims to outperform single-modality approaches in sensitivity and specificity.

2. Dataset and Tools
--------------------
2.1 Available / Used Datasets

- DFU RGB (Kaggle): ~493–512 RGB images depending on folder used (Original, Patches). Your local copy paths:
  - `DFU_RGB/Original Images`
  - `DFU_RGB/Patches/Abnormal` (ulcer)
  - `DFU_RGB/Patches/Normal` (healthy)

- DFU Thermal (ThermoDataBase): Train/Val split with `Control Group` (healthy) and `DM Group` (diabetic / ulcer). Your local copy paths:
  - `DFU_Thermal/ThermoDataBase/train/Control Group`
  - `DFU_Thermal/ThermoDataBase/train/DM Group`
  - `DFU_Thermal/ThermoDataBase/val/Control Group`
  - `DFU_Thermal/ThermoDataBase/val/DM Group`

- Supplementary: ISIC skin lesion dataset (optional for transfer learning).

Your recent processing output (from scripts) reports:
- RGB: 512 abnormal, 543 healthy (split -> Train: 358 ulcer / 380 healthy; Val: 77 / 81; Test: 77 / 82).
- Thermal (ThermoDataBase): Train control 720 / DM 724; val control 170 / DM 252. After splitting your processed counts: Train ~1,227, Val ~422, Test ~217.

2.2 Tools & Frameworks

- PyTorch, torchvision, timm
- OpenCV / Pillow for preprocessing
- Albumentations for strong augmentation (optional)
- scikit-learn for metrics and sampling
- Grad-CAM / SHAP for explainability

3. Methodology
--------------
3.1 Preprocessing & Standardization

- Images must be standardized to the same spatial resolution (recommended: 224×224) and channel format (3-channel RGB for both branches — convert/replicate thermal if single-channel).
- Use `scripts/analyze_image_sizes.py` to inspect distributions and `scripts/standardize_images.py` to produce `data/rgb_standardized/` and `data/thermal_standardized/`.
- Normalize with modality-specific statistics (ImageNet mean/std for RGB; custom mean/std for thermal e.g. mean=0.5 std=0.5 after scaling to [0,1]).

3.2 Model Architecture (high-level)

- RGB branch: pretrained CNN (EfficientNet-B0 or ResNet-50) → feature vector
- Thermal branch: lightweight CNN or modified ViT for small datasets → feature vector
- Fusion: early or late fusion (concatenate feature vectors, optional gated fusion) → classification head
- Outputs: binary ulcer presence + 3-class risk level (low/moderate/high)

3.3 Training

- Transfer learning: freeze most backbone layers initially, fine-tune later
- Losses: BCE for detection, categorical cross-entropy for risk. Weighted losses for imbalance
- Optimizer: AdamW with cosine LR schedule. Early stopping on validation sensitivity.

4. Pairing Strategy (Option 3: Hybrid Approach)
-----------------------------------------------
Important: The Kaggle RGB and thermal datasets are NOT pixel- or patient-level paired (they come from different sources). True pixel-aligned pairing is not possible without metadata. We therefore recommend label-level pairing and careful balancing for multimodal training.

Goals of pairing script/process:
- Create a `data/paired/` structure with `train/`, `val/`, `test/` subfolders, each containing `rgb/healthy`, `rgb/ulcer`, `thermal/healthy`, `thermal/ulcer` and a paired manifest (CSV) describing which RGB image is paired with which thermal image.
- Pair by label (healthy ↔ healthy, ulcer ↔ ulcer) using randomized, stratified sampling.

Strategies (choose one):

A) Label-matched random pairing (recommended)
- Shuffle each modality's images by label.
- For each RGB image in split X with label L, sample a thermal image of label L (without replacement) until one modality is exhausted.
- If counts differ: either (1) sample thermal images with replacement to match RGB count, (2) downsample the bigger modality, or (3) oversample the smaller modality using augmentations.

B) Stratified pairing by metadata (if available)
- If you have metadata like patient_id, device, or timestamp, pair samples by similar metadata (preferable). If not, fallback to (A).

C) Synchronized splitting then balanced pairing
- First split each modality into train/val/test with the same ratios (70/15/15 recommended), preserving label distributions.
- Then pair within each split using Strategy (A).

Why label-level pairing works
- Model learns complementary cues: RGB texture vs. thermal perfusion.
- The fusion module does feature-level combination rather than pixel-wise alignment.
- Common approach in literature when paired data isn't available.

5. Proposed File/Folder Outputs
------------------------------
data/
- paired/
  - train/
    - rgb/healthy/
    - rgb/ulcer/
    - thermal/healthy/
    - thermal/ulcer/
    - pairs_train.csv   # columns: rgb_path, thermal_path, label
  - val/
  - test/
- rgb/
- thermal/
- rgb_standardized/
- thermal_standardized/

6. Example pairing algorithm (label-matched)
--------------------------------------------
The README includes a sample script you can run or I can create for you as `scripts/pair_datasets.py`. The algorithm is:

1. Read file lists for each split and label from `data/rgb` and `data/thermal`.
2. For each split and label, shuffle lists.
3. If modality counts differ, choose one of: downsample (deterministic), oversample with replacement, or oversample with augmentations.
4. Pair by zipping lists and write `data/paired/<split>/...` copying or symlinking files; also write `pairs_<split>.csv` with (rgb, thermal, label).

Sample pairing code sketch (included below in README). If you'd like, I can create this as an executable `scripts/pair_datasets.py` now.

7. Commands & Quick Start
-------------------------
From project root (`~/DFU_MMT`):

```bash
# 1) Verify structure
python3 scripts/verify_structure.py

# 2) Prepare RGB / Thermal splits (if not done)
python3 scripts/prepare_datasets.py

# 3) Analyze sizes
python3 scripts/analyze_image_sizes.py

# 4) Standardize images to 224x224
python3 scripts/standardize_images.py

# 5) (Optional) Pair datasets (creates data/paired)
# If I create the script: python3 scripts/pair_datasets.py --rgb data/rgb_standardized --thermal data/thermal_standardized --out data/paired

# 6) Use data loaders / train
python3 notebooks/train_rgb_vit_fusion.py
```

8. Pairing script example (label-matched, simplified)
-----------------------------------------------------
```python
# Example: scripts/pair_datasets.py (sketch)
# - pairs by label within each split

import argparse
from pathlib import Path
import random
import csv
import shutil

def list_images(dirpath):
    return sorted([p for p in Path(dirpath).glob('**/*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}])

def ensure_dirs(base_out):
    for split in ['train','val','test']:
        for mod in ['rgb','thermal']:
            for lab in ['healthy','ulcer']:
                (base_out / split / mod / lab).mkdir(parents=True, exist_ok=True)

def pair_by_label(rgb_dir, th_dir, out_dir, split):
    rgb_h = list_images(rgb_dir / split / 'healthy')
    rgb_u = list_images(rgb_dir / split / 'ulcer')
    th_h = list_images(th_dir / split / 'healthy')
    th_u = list_images(th_dir / split / 'ulcer')

    pairs = []
    for rgb_list, th_list, label in [(rgb_h, th_h, 0),(rgb_u, th_u, 1)]:
        random.shuffle(rgb_list)
        random.shuffle(th_list)
        # handle differing lengths: sample with replacement from smaller
        if len(th_list) < len(rgb_list):
            th_list = th_list + list(random.choices(th_list, k=len(rgb_list)-len(th_list)))
        elif len(rgb_list) < len(th_list):
            rgb_list = rgb_list + list(random.choices(rgb_list, k=len(th_list)-len(rgb_list)))
        for r, t in zip(rgb_list, th_list):
            # copy or symlink
            r_out = out_dir / split / 'rgb' / ('healthy' if label==0 else 'ulcer') / r.name
            t_out = out_dir / split / 'thermal' / ('healthy' if label==0 else 'ulcer') / t.name
            shutil.copy2(r, r_out)
            shutil.copy2(t, t_out)
            pairs.append((str(r_out), str(t_out), label))
    # write CSV
    with open(out_dir / f'pairs_{split}.csv','w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rgb_path','thermal_path','label'])
        writer.writerows(pairs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb', required=True)
    parser.add_argument('--thermal', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    rgb_dir = Path(args.rgb)
    th_dir = Path(args.thermal)
    out_dir = Path(args.out)
    ensure_dirs(out_dir)
    for split in ['train','val','test']:
        pair_by_label(rgb_dir, th_dir, out_dir, split)
```

Notes about the sketch:
- It demonstrates label-level matching and handles differing counts by sampling with replacement.
- You can change sampling behavior to downsample the larger modality or use augmentation for upsampling.
- Use symlinks instead of copies to save disk if you prefer (`os.symlink`).

9. Image sizes & standardization (your questions answered)
---------------------------------------------------------
- Are the image sizes OK? No — standardization to 224×224 is required.
- Do both models require same image size for comparison? Yes — both branches should receive the same spatial size (224×224 recommended).
- Will mismatch occur during training if not standardized? Yes — DataLoader batching and fusion layers will fail.
- Do images need standardization? Yes — convert all to `224×224` and save to `data/*_standardized`.

10. Next steps I can do for you (pick any):
- Create `scripts/pair_datasets.py` (fully implemented) and run it here to produce `data/paired`.
- Create `scripts/pairing_report.md` summarizing exact counts after pairing and produce `data/dataset_info.txt`.
- Implement data loader changes to read `data/paired` (if you want a paired dataset class).

If you want me to create the pairing script now and run it on your local copies, reply: "Create and run pairing script" and I will:
1) create `scripts/pair_datasets.py`, 2) run `python3 scripts/pair_datasets.py --rgb data/rgb_standardized --thermal data/thermal_standardized --out data/paired`, 3) return a short report with counts and any warnings.

---

Project files of interest (already present):
- `scripts/prepare_datasets.py` — Splits and organizes RGB/Thermal into `data/rgb` and `data/thermal`.
- `scripts/verify_structure.py` — Checks folder layout and counts.
- `scripts/standardize_images.py` — Produces `data/*_standardized` (224×224).
- `notebooks/train_rgb_vit_fusion.py` — Example training script (uses paired dataset loader `DFUPairedDataset`).

Contact / Author
----------------
If you'd like I can now create and run the pairing script and produce `data/paired/` and the `pairs_*.csv` manifests. Reply with: "Please create and run pairing script" or say which behavior you prefer for unequal counts: `oversample`, `downsample`, or `augment`.
