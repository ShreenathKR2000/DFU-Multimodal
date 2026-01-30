#!/usr/bin/env python3
"""
COMPREHENSIVE DATASET PREPARATION SCRIPT
========================================

Organizes Kaggle DFU RGB and Thermal datasets into:
- Inconspicuous, numbered filenames (prevents model cheating on filenames)
- Train/Val/Test splits (70/15/15)
- Balanced class distribution
- Clean directory structure

Dataset 1: RGB Images (from laithjj/diabetic-foot-ulcer-dfu)
- Source: Patches/Healthy (normal images)
- Source: Patches/Abnormal (ulcer images)
- Source: test/ (additional test images)

Dataset 2: Thermal Images (from vuppalaadithyasairam/thermography)
- Source: train/Control Group (healthy)
- Source: train/DM Group (diabetic/ulcer)
- Source: val/Control Group (healthy)
- Source: val/DM Group (diabetic/ulcer)

Output: Clean dataset with structure:
data/
├── rgb/
│   ├── train/
│   │   ├── healthy/
│   │   └── ulcer/
│   ├── val/
│   ├── test/
├── thermal/
│   ├── train/
│   ├── val/
│   └── test/
└── dataset_manifest.json (mapping of new names to original files)
"""

import os
import shutil
import json
import hashlib
from pathlib import Path
from sklearn.model_selection import train_test_split
import random
from collections import defaultdict
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path.home() / "CompVis" / "Dataset"
RGB_SOURCE = BASE_DIR / "DFU_RGB"
THERMAL_SOURCE = BASE_DIR / "DFU_Thermal"

OUTPUT_DIR = BASE_DIR / "data"
RGB_OUT = OUTPUT_DIR / "rgb"
THERMAL_OUT = OUTPUT_DIR / "thermal"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Image extensions to look for
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.JPG', '.PNG'}

# ============================================================================
# UTILITIES
# ============================================================================

def count_files(directory):
    """Count image files in directory"""
    if not directory.exists():
        return 0
    return len([p for p in directory.rglob('*.*') if p.suffix.lower() in IMAGE_EXTS])

def create_anonymous_name(index):
    """Create numeric-only anonymous filename using global counter.

    Format: 000001.jpg (zero-padded 6 digits). Caller must ensure index uniqueness.
    """
    ext = '.jpg'
    return f"{index:06d}{ext}"

def copy_with_anonymous_name(src_path, dst_dir, index):
    """Copy file with anonymous numeric name and return mapping entry."""
    dst_name = create_anonymous_name(index)
    dst_path = dst_dir / dst_name

    try:
        shutil.copy2(src_path, dst_path)
        return {
            'original': str(src_path),
            'anonymous': dst_name,
            'success': True
        }
    except Exception as e:
        return {
            'original': str(src_path),
            'error': str(e),
            'success': False
        }


def compute_sha256(path, block_size=65536):
    h = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(block_size), b''):
                h.update(block)
        return h.hexdigest()
    except Exception:
        return None

# ============================================================================
# DATASET CREATION FUNCTIONS
# ============================================================================

def create_output_directories():
    """Create all output directories"""
    print("\n" + "="*70)
    print("CREATING OUTPUT DIRECTORY STRUCTURE")
    print("="*70)
    
    splits = ['train', 'val', 'test']
    classes = ['healthy', 'ulcer']
    
    # Create RGB directories
    for split in splits:
        for cls in classes:
            (RGB_OUT / split / cls).mkdir(parents=True, exist_ok=True)
    
    # Create Thermal directories
    for split in splits:
        for cls in classes:
            (THERMAL_OUT / split / cls).mkdir(parents=True, exist_ok=True)
    
    print("✓ Created directory structure")
    print(f"  RGB output: {RGB_OUT}")
    print(f"  Thermal output: {THERMAL_OUT}")



def process_rgb_dataset():
    """Process RGB dataset from raw source, deduplicate by SHA256 and create 70/15/15 splits.

    This function will:
    - Collect healthy and ulcer candidate image paths
    - Compute SHA256 for each image
    - Assign each unique hash to a single class (prefer ulcer if a hash appears in both)
    - Split unique hashes into train/val/test (by hash — prevents leakage)
    - Copy one representative file per hash to the output with numeric-only anonymous names
    - Build manifest entries mapping anonymous name -> list(originals), split, class, hash
    """
    print("\n" + "="*70)
    print("PROCESSING RGB DATASET (DEDUPED & NUMERIC NAMES)")
    print("="*70)

    patches_dir = RGB_SOURCE / "Patches"
    if not patches_dir.exists():
        print(f"❌ ERROR: {patches_dir} not found!")
        return {'healthy': 0, 'ulcer': 0, 'errors': 1, 'manifest': {}}

    # Collect candidate paths
    healthy_candidates = [patches_dir / 'Normal', patches_dir / 'Healthy']
    ulcer_candidates = [patches_dir / 'Abnormal', patches_dir / 'Ulcer']
    test_dir = RGB_SOURCE / 'TestSet'

    all_candidates = []

    for d in healthy_candidates:
        if d.exists():
            for p in d.rglob('*.*'):
                if p.suffix.lower() in IMAGE_EXTS:
                    all_candidates.append((p, 'healthy'))
            break

    for d in ulcer_candidates:
        if d.exists():
            for p in d.rglob('*.*'):
                if p.suffix.lower() in IMAGE_EXTS:
                    all_candidates.append((p, 'ulcer'))
            break

    if test_dir.exists():
        for p in test_dir.rglob('*.*'):
            if p.suffix.lower() in IMAGE_EXTS:
                all_candidates.append((p, 'ulcer'))

    if len(all_candidates) == 0:
        print("❌ ERROR: No RGB images found in source locations.")
        return {'healthy': 0, 'ulcer': 0, 'errors': 1, 'manifest': {}}

    print(f"\n📊 Found RGB candidate files: {len(all_candidates)}")

    # Build hash -> list(paths, class)
    hash_map = defaultdict(list)
    print("\n🔐 Computing SHA256 hashes for RGB images (this may take a while)...")
    for p, cls in tqdm(all_candidates, desc="Hashing RGB"):
        h = compute_sha256(p)
        if h is None:
            continue
        hash_map[h].append({'path': str(p.resolve()), 'class': cls})

    # Resolve class per-hash: if any original marked as 'ulcer', classify the hash as 'ulcer'
    healthy_hashes = []
    ulcer_hashes = []
    for h, items in sorted(hash_map.items()):
        classes = {it['class'] for it in items}
        if 'ulcer' in classes:
            ulcer_hashes.append(h)
        else:
            healthy_hashes.append(h)

    print(f"\n🔎 Unique RGB hashes: {len(hash_map)} (healthy: {len(healthy_hashes)}, ulcer: {len(ulcer_hashes)})")

    # Split hashes (train/val/test) — by-hash split to avoid leakage
    def split_hashes(hashes):
        train, temp = train_test_split(hashes, test_size=0.3, random_state=RANDOM_SEED)
        val, test = train_test_split(temp, test_size=0.5, random_state=RANDOM_SEED)
        return train, val, test

    h_train, h_val, h_test = split_hashes(healthy_hashes)
    u_train, u_val, u_test = split_hashes(ulcer_hashes)

    print(f"\n📋 Split by unique images (70/15/15):")
    print(f"  Healthy hashes: {len(h_train)} train, {len(h_val)} val, {len(h_test)} test")
    print(f"  Ulcer hashes: {len(u_train)} train, {len(u_val)} val, {len(u_test)} test")

    # Copy representative files (first original path for each hash) and build manifest
    manifest = {}
    errors = 0
    global_counter = 1

    def copy_for_hash(hash_list, split_name, cls_name, out_subdir):
        nonlocal global_counter, manifest, errors
        for h in hash_list:
            originals = [it['path'] for it in hash_map[h]]
            src = Path(originals[0])
            dst_dir = RGB_OUT / split_name / cls_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            res = copy_with_anonymous_name(src, dst_dir, global_counter)
            if res.get('success'):
                manifest[res['anonymous']] = {'originals': originals, 'split': split_name, 'class': cls_name, 'hash': h}
            else:
                errors += 1
            global_counter += 1

    print('\n🔄 Copying unique RGB images to output (numeric names)')
    copy_for_hash(h_train, 'train', 'healthy', RGB_OUT)
    copy_for_hash(h_val, 'val', 'healthy', RGB_OUT)
    copy_for_hash(h_test, 'test', 'healthy', RGB_OUT)
    copy_for_hash(u_train, 'train', 'ulcer', RGB_OUT)
    copy_for_hash(u_val, 'val', 'ulcer', RGB_OUT)
    copy_for_hash(u_test, 'test', 'ulcer', RGB_OUT)

    # Save per-modality dedupe report
    dedupe_report = {
        'total_candidate_files': len(all_candidates),
        'unique_hashes': len(hash_map),
        'duplicates_removed': sum(max(0, len(v)-1) for v in hash_map.values()),
    }
    try:
        with open(OUTPUT_DIR / 'rgb_dedupe_report.json', 'w') as f:
            json.dump(dedupe_report, f, indent=2)
    except Exception:
        pass

    stats = {
        'healthy': len(h_train) + len(h_val) + len(h_test),
        'ulcer': len(u_train) + len(u_val) + len(u_test),
        'errors': errors,
        'manifest': manifest
    }

    print(f"\n✓ RGB dataset processed — unique images: {stats['healthy'] + stats['ulcer']}")
    print(f"  Errors: {errors}")
    return stats

def process_thermal_dataset():
    """Process Thermal dataset similarly to RGB: dedupe by SHA256 and create splits."""
    print("\n" + "="*70)
    print("PROCESSING THERMAL DATASET (DEDUPED & NUMERIC NAMES)")
    print("="*70)

    thermo_db = THERMAL_SOURCE / "ThermoDataBase"
    if not thermo_db.exists():
        print(f"❌ ERROR: {thermo_db} not found!")
        return {'healthy': 0, 'ulcer': 0, 'errors': 1, 'manifest': {}}

    # Collect candidate images from train and val folders (Control Group -> healthy, DM Group -> ulcer)
    candidates = []
    for split_dir in ['train', 'val']:
        base = thermo_db / split_dir
        if not base.exists():
            continue
        ctrl = base / 'Control Group'
        dm = base / 'DM Group'
        if ctrl.exists():
            for p in ctrl.rglob('*.*'):
                if p.suffix.lower() in IMAGE_EXTS:
                    candidates.append((p, 'healthy'))
        if dm.exists():
            for p in dm.rglob('*.*'):
                if p.suffix.lower() in IMAGE_EXTS:
                    candidates.append((p, 'ulcer'))

    if len(candidates) == 0:
        print("❌ ERROR: No thermal images found in expected folders.")
        return {'healthy': 0, 'ulcer': 0, 'errors': 1, 'manifest': {}}

    print(f"\n📊 Found Thermal candidate files: {len(candidates)}")

    # Build hash map
    hash_map = defaultdict(list)
    print("\n🔐 Computing SHA256 hashes for Thermal images...")
    for p, cls in tqdm(candidates, desc="Hashing Thermal"):
        h = compute_sha256(p)
        if h is None:
            continue
        hash_map[h].append({'path': str(p.resolve()), 'class': cls})

    healthy_hashes = []
    ulcer_hashes = []
    for h, items in sorted(hash_map.items()):
        classes = {it['class'] for it in items}
        if 'ulcer' in classes:
            ulcer_hashes.append(h)
        else:
            healthy_hashes.append(h)

    print(f"\n🔎 Unique Thermal hashes: {len(hash_map)} (healthy: {len(healthy_hashes)}, ulcer: {len(ulcer_hashes)})")

    def split_hashes(hashes):
        train, temp = train_test_split(hashes, test_size=0.3, random_state=RANDOM_SEED)
        val, test = train_test_split(temp, test_size=0.5, random_state=RANDOM_SEED)
        return train, val, test

    h_train, h_val, h_test = split_hashes(healthy_hashes)
    u_train, u_val, u_test = split_hashes(ulcer_hashes)

    manifest = {}
    errors = 0
    global_counter = 1

    def copy_for_hash(hash_list, split_name, cls_name):
        nonlocal global_counter, manifest, errors
        for h in hash_list:
            originals = [it['path'] for it in hash_map[h]]
            src = Path(originals[0])
            dst_dir = THERMAL_OUT / split_name / cls_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            res = copy_with_anonymous_name(src, dst_dir, global_counter)
            if res.get('success'):
                manifest[res['anonymous']] = {'originals': originals, 'split': split_name, 'class': cls_name, 'hash': h}
            else:
                errors += 1
            global_counter += 1

    print('\n🔄 Copying unique Thermal images to output (numeric names)')
    copy_for_hash(h_train, 'train', 'healthy')
    copy_for_hash(h_val, 'val', 'healthy')
    copy_for_hash(h_test, 'test', 'healthy')
    copy_for_hash(u_train, 'train', 'ulcer')
    copy_for_hash(u_val, 'val', 'ulcer')
    copy_for_hash(u_test, 'test', 'ulcer')

    dedupe_report = {
        'total_candidate_files': len(candidates),
        'unique_hashes': len(hash_map),
        'duplicates_removed': sum(max(0, len(v)-1) for v in hash_map.values()),
    }
    try:
        with open(OUTPUT_DIR / 'thermal_dedupe_report.json', 'w') as f:
            json.dump(dedupe_report, f, indent=2)
    except Exception:
        pass

    stats = {
        'healthy': len(h_train) + len(h_val) + len(h_test),
        'ulcer': len(u_train) + len(u_val) + len(u_test),
        'errors': errors,
        'manifest': manifest
    }

    print(f"\n✓ Thermal dataset processed — unique images: {stats['healthy'] + stats['ulcer']}")
    print(f"  Errors: {errors}")
    return stats

def create_manifest_file(rgb_manifest, thermal_manifest):
    """Create JSON manifest for dataset mapping"""
    print("\n" + "="*70)
    print("CREATING DATASET MANIFEST")
    print("="*70)
    combined_manifest = {
        'created': str(Path(OUTPUT_DIR) / "dataset_manifest.json"),
        'description': 'Maps anonymous numeric filenames to original sources (list)',
        'rgb': rgb_manifest,
        'thermal': thermal_manifest,
        'notes': [
            'Filenames follow pattern: 000001.jpg (numeric-only, zero-padded 6 digits)',
            'One anonymous file corresponds to one unique image hash (SHA256)',
            'Original file paths are provided as a list under "originals" for each anonymous file',
            'Splits created by unique-image hashing to avoid leakage across train/val/test'
        ]
    }
    
    manifest_path = OUTPUT_DIR / 'dataset_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(combined_manifest, f, indent=2)
    
    print(f"✓ Manifest created: {manifest_path}")

def create_summary_file(rgb_stats, thermal_stats):
    """Create human-readable summary of dataset splits"""
    print("\n" + "="*70)
    print("CREATING DATASET SUMMARY")
    print("="*70)
    
    summary = f"""
DFU MULTIMODAL DATASET ORGANIZATION
====================================

Dataset organized with anonymous numeric filenames to prevent filename-based leakage.

RGB DATASET
-----------
Source: Kaggle (laithjj/diabetic-foot-ulcer-dfu)
Total unique images (post-dedupe): {rgb_stats['healthy'] + rgb_stats['ulcer']}
    Healthy: {rgb_stats['healthy']}
    Ulcer (with pathology): {rgb_stats['ulcer']}
Errors during processing: {rgb_stats['errors']}

Filenames: numeric-only (e.g., 000001.jpg)
Output location: {RGB_OUT}

THERMAL DATASET
---------------
Source: Kaggle (vuppalaadithyasairam/thermography-images-of-diabetic-foot)
Total unique images (post-dedupe): {thermal_stats['healthy'] + thermal_stats['ulcer']}
    Healthy (Control Group): {thermal_stats['healthy']}
    Diabetic (DM Group): {thermal_stats['ulcer']}
Errors during processing: {thermal_stats['errors']}

Filenames: numeric-only (e.g., 000001.jpg)
Output location: {THERMAL_OUT}

DIRECTORY STRUCTURE
-------------------
{OUTPUT_DIR}/
├── rgb/
│   ├── train/
│   │   ├── healthy/
│   │   └── ulcer/
│   ├── val/
│   │   ├── healthy/
│   │   └── ulcer/
│   └── test/
│       ├── healthy/
│       └── ulcer/
├── thermal/
│   ├── train/
│   │   ├── healthy/
│   │   └── ulcer/
│   ├── val/
│   │   ├── healthy/
│   │   └── ulcer/
│   └── test/
│       ├── healthy/
│       └── ulcer/
├── dataset_manifest.json
└── dataset_summary.txt (this file)

NAMING CONVENTION
-----------------
All images renamed numeric-only to prevent filename bias.

TOTAL DATASET SIZE
------------------
RGB: {rgb_stats['healthy'] + rgb_stats['ulcer']} images
Thermal: {thermal_stats['healthy'] + thermal_stats['ulcer']} images
Combined: {rgb_stats['healthy'] + rgb_stats['ulcer'] + thermal_stats['healthy'] + thermal_stats['ulcer']} images

Ready for training with updated dataloader!
"""
    
    summary_path = OUTPUT_DIR / 'dataset_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    print(f"✓ Summary created: {summary_path}")
    print(summary)

def main():
    print("="*70)
    print("CLEAN DATASET ORGANIZATION FOR MULTIMODAL DFU PROJECT")
    print("="*70)
    print("\nThis script will:")
    print("1. ✓ Organize RGB and Thermal images")
    print("2. ✓ Create 70/15/15 Train/Val/Test splits")
    print("3. ✓ Rename files to anonymous (prevent filename bias)")
    print("4. ✓ Balance class distribution")
    print("5. ✓ Generate manifest and summary files\n")
    
    # Start fresh: remove existing output directory if present
    if OUTPUT_DIR.exists():
        print(f"\nℹ️ Removing existing output directory to start fresh: {OUTPUT_DIR}")
        try:
            shutil.rmtree(OUTPUT_DIR)
        except Exception as e:
            print(f"Warning: failed to remove {OUTPUT_DIR}: {e}")

    # Create output directories
    create_output_directories()

    # Process RGB dataset
    rgb_stats = process_rgb_dataset()
    
    # Process Thermal dataset
    thermal_stats = process_thermal_dataset()
    
    # Create manifest
    create_manifest_file(rgb_stats['manifest'], thermal_stats['manifest'])
    
    # Create summary
    create_summary_file(rgb_stats, thermal_stats)
    
    print("\n" + "="*70)
    print("✅ DATASET ORGANIZATION COMPLETE!")
    print("="*70)
    print(f"\nNext steps:")
    print(f"1. Update dataloader.py to use new anonymous dataset")
    print(f"2. Verify dataset statistics")
    print(f"3. Begin training with clean, organized data")
    print(f"\nDataset location: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
