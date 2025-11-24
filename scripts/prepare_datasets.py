#!/usr/bin/env python3
"""
Dataset Preparation Script for Multimodal DFU Classification
Creates organized train/val/test splits for RGB and Thermal datasets
"""

import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import random
from collections import defaultdict

# Set random seed for reproducibility
random.seed(42)

# Base paths
BASE_DIR = Path.home() / "DFU_MMT"
RGB_SOURCE = BASE_DIR / "DFU_RGB"
THERMAL_SOURCE = BASE_DIR / "DFU_Thermal" / "ThermoDataBase"
OUTPUT_DIR = BASE_DIR / "data"

# Output directories
RGB_OUT = OUTPUT_DIR / "rgb"
THERMAL_OUT = OUTPUT_DIR / "thermal"
PAIRED_OUT = OUTPUT_DIR / "paired"

def count_files(directory):
    """Count image files in directory"""
    if not os.path.exists(directory):
        return 0
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    count = 0
    for root, dirs, files in os.walk(directory):
        count += sum(1 for f in files if Path(f).suffix.lower() in image_exts)
    return count

def create_directories():
    """Create output directory structure"""
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
    
    # Create Paired directories (for future use)
    for split in splits:
        for cls in classes:
            (PAIRED_OUT / split / 'rgb' / cls).mkdir(parents=True, exist_ok=True)
            (PAIRED_OUT / split / 'thermal' / cls).mkdir(parents=True, exist_ok=True)
    
    print("✓ Created directory structure")

def process_rgb_dataset():
    """Process RGB dataset into train/val/test splits"""
    print("\n=== Processing RGB Dataset ===")
    
    # Use Patches directory as it has clear labels
    patches_dir = RGB_SOURCE / "Patches"
    
    if not patches_dir.exists():
        print(f"ERROR: Patches directory not found at {patches_dir}")
        return
    
    # Read images from Abnormal and Healthy folders
    abnormal_dir = patches_dir / "Abnormal"
    healthy_dir = patches_dir / "Normal"
    
    # Alternative naming check
    if not abnormal_dir.exists():
        abnormal_dir = patches_dir / "Abnormal"
    if not healthy_dir.exists():
        healthy_dir = patches_dir / "Normal"
    
    print(f"Looking for images in:")
    print(f"  Abnormal: {abnormal_dir}")
    print(f"  Healthy: {healthy_dir}")
    
    # Collect image paths
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    
    abnormal_images = []
    healthy_images = []
    
    # Get abnormal (ulcer) images
    if abnormal_dir.exists():
        for img_path in abnormal_dir.rglob('*'):
            if img_path.suffix.lower() in image_exts:
                abnormal_images.append(img_path)
    
    # Get healthy images
    if healthy_dir.exists():
        for img_path in healthy_dir.rglob('*'):
            if img_path.suffix.lower() in image_exts:
                healthy_images.append(img_path)
    
    print(f"Found {len(abnormal_images)} abnormal images")
    print(f"Found {len(healthy_images)} healthy images")
    
    if len(abnormal_images) == 0 or len(healthy_images) == 0:
        print("ERROR: Could not find images. Checking directory structure...")
        print(f"\nContents of {patches_dir}:")
        for item in patches_dir.iterdir():
            print(f"  - {item.name}")
            if item.is_dir():
                for subitem in item.iterdir():
                    print(f"    - {subitem.name}")
        return
    
    # Split: 70% train, 15% val, 15% test
    def split_data(images, label):
        train, temp = train_test_split(images, test_size=0.3, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)
        return train, val, test
    
    ulcer_train, ulcer_val, ulcer_test = split_data(abnormal_images, 'ulcer')
    healthy_train, healthy_val, healthy_test = split_data(healthy_images, 'healthy')
    
    # Copy files
    def copy_images(image_list, dest_dir):
        count = 0
        for src_path in image_list:
            dst_path = dest_dir / f"{src_path.stem}_{count}{src_path.suffix}"
            shutil.copy2(src_path, dst_path)
            count += 1
        return count
    
    # Copy ulcer images
    ulcer_train_count = copy_images(ulcer_train, RGB_OUT / 'train' / 'ulcer')
    ulcer_val_count = copy_images(ulcer_val, RGB_OUT / 'val' / 'ulcer')
    ulcer_test_count = copy_images(ulcer_test, RGB_OUT / 'test' / 'ulcer')
    
    # Copy healthy images
    healthy_train_count = copy_images(healthy_train, RGB_OUT / 'train' / 'healthy')
    healthy_val_count = copy_images(healthy_val, RGB_OUT / 'val' / 'healthy')
    healthy_test_count = copy_images(healthy_test, RGB_OUT / 'test' / 'healthy')
    
    print(f"\nRGB Dataset Split:")
    print(f"  Train: {ulcer_train_count} ulcer, {healthy_train_count} healthy")
    print(f"  Val:   {ulcer_val_count} ulcer, {healthy_val_count} healthy")
    print(f"  Test:  {ulcer_test_count} ulcer, {healthy_test_count} healthy")
    print("✓ RGB dataset processed")

def process_thermal_dataset():
    """Process Thermal dataset"""
    print("\n=== Processing Thermal Dataset ===")
    
    train_dir = THERMAL_SOURCE / "train"
    val_dir = THERMAL_SOURCE / "val"
    
    if not train_dir.exists() or not val_dir.exists():
        print(f"ERROR: Thermal directories not found")
        return
    
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    
    # Process train split
    train_control = train_dir / "Control Group"
    train_dm = train_dir / "DM Group"
    
    val_control = val_dir / "Control Group"
    val_dm = val_dir / "DM Group"
    
    print(f"Train Control Group: {count_files(train_control)} images")
    print(f"Train DM Group: {count_files(train_dm)} images")
    print(f"Val Control Group: {count_files(val_control)} images")
    print(f"Val DM Group: {count_files(val_dm)} images")
    
    # Collect all images
    train_healthy = list(train_control.rglob('*')) if train_control.exists() else []
    train_ulcer = list(train_dm.rglob('*')) if train_dm.exists() else []
    val_healthy = list(val_control.rglob('*')) if val_control.exists() else []
    val_ulcer = list(val_dm.rglob('*')) if val_dm.exists() else []
    
    # Filter only image files
    train_healthy = [p for p in train_healthy if p.suffix.lower() in image_exts]
    train_ulcer = [p for p in train_ulcer if p.suffix.lower() in image_exts]
    val_healthy = [p for p in val_healthy if p.suffix.lower() in image_exts]
    val_ulcer = [p for p in val_ulcer if p.suffix.lower() in image_exts]
    
    # Create test split from train (take 15% of train for test)
    def create_test_split(images):
        random.shuffle(images)
        split_idx = int(len(images) * 0.85)
        return images[:split_idx], images[split_idx:]
    
    train_healthy_final, test_healthy = create_test_split(train_healthy)
    train_ulcer_final, test_ulcer = create_test_split(train_ulcer)
    
    # Copy files
    def copy_thermal_images(image_list, dest_dir):
        count = 0
        for src_path in image_list:
            if src_path.is_file():
                dst_path = dest_dir / f"thermal_{count}{src_path.suffix}"
                shutil.copy2(src_path, dst_path)
                count += 1
        return count
    
    # Copy to organized structure
    train_healthy_count = copy_thermal_images(train_healthy_final, THERMAL_OUT / 'train' / 'healthy')
    train_ulcer_count = copy_thermal_images(train_ulcer_final, THERMAL_OUT / 'train' / 'ulcer')
    
    val_healthy_count = copy_thermal_images(val_healthy, THERMAL_OUT / 'val' / 'healthy')
    val_ulcer_count = copy_thermal_images(val_ulcer, THERMAL_OUT / 'val' / 'ulcer')
    
    test_healthy_count = copy_thermal_images(test_healthy, THERMAL_OUT / 'test' / 'healthy')
    test_ulcer_count = copy_thermal_images(test_ulcer, THERMAL_OUT / 'test' / 'ulcer')
    
    print(f"\nThermal Dataset Split:")
    print(f"  Train: {train_ulcer_count} ulcer, {train_healthy_count} healthy")
    print(f"  Val:   {val_ulcer_count} ulcer, {val_healthy_count} healthy")
    print(f"  Test:  {test_ulcer_count} ulcer, {test_healthy_count} healthy")
    print("✓ Thermal dataset processed")

def create_metadata():
    """Create metadata files for easy reference"""
    print("\n=== Creating Metadata ===")
    
    splits = ['train', 'val', 'test']
    
    with open(OUTPUT_DIR / 'dataset_info.txt', 'w') as f:
        f.write("DFU Multimodal Dataset Information\n")
        f.write("="*50 + "\n\n")
        
        for split in splits:
            f.write(f"{split.upper()} Split:\n")
            f.write(f"  RGB Dataset:\n")
            rgb_healthy = count_files(RGB_OUT / split / 'healthy')
            rgb_ulcer = count_files(RGB_OUT / split / 'ulcer')
            f.write(f"    Healthy: {rgb_healthy}\n")
            f.write(f"    Ulcer:   {rgb_ulcer}\n")
            f.write(f"    Total:   {rgb_healthy + rgb_ulcer}\n\n")
            
            f.write(f"  Thermal Dataset:\n")
            thermal_healthy = count_files(THERMAL_OUT / split / 'healthy')
            thermal_ulcer = count_files(THERMAL_OUT / split / 'ulcer')
            f.write(f"    Healthy: {thermal_healthy}\n")
            f.write(f"    Ulcer:   {thermal_ulcer}\n")
            f.write(f"    Total:   {thermal_healthy + thermal_ulcer}\n\n")
    
    print("✓ Metadata created")

def main():
    print("="*60)
    print("DFU Multimodal Dataset Preparation")
    print("="*60)
    
    # Step 1: Create directories
    create_directories()
    
    # Step 2: Process RGB dataset
    process_rgb_dataset()
    
    # Step 3: Process Thermal dataset
    process_thermal_dataset()
    
    # Step 4: Create metadata
    create_metadata()
    
    print("\n" + "="*60)
    print("Dataset preparation complete!")
    print("="*60)
    print(f"\nOrganized datasets available at:")
    print(f"  RGB:     {RGB_OUT}")
    print(f"  Thermal: {THERMAL_OUT}")
    print(f"\nMetadata: {OUTPUT_DIR / 'dataset_info.txt'}")
    print("\nNote: RGB and Thermal images are NOT pixel-aligned pairs,")
    print("      but share classification labels for multimodal training.")

if __name__ == "__main__":
    main()