#!/usr/bin/env python3
"""
Dataset Structure Verification Script
Run this FIRST to verify your dataset structure before preparation
"""

import os
from pathlib import Path
from collections import defaultdict

# Base paths
BASE_DIR = Path.home() / "DFU_MMT"
RGB_SOURCE = BASE_DIR / "DFU_RGB"
THERMAL_SOURCE = BASE_DIR / "DFU_Thermal"

def count_images(directory):
    """Count image files recursively"""
    if not directory.exists():
        return 0
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.JPG', '.PNG'}
    count = 0
    for root, dirs, files in os.walk(directory):
        count += sum(1 for f in files if Path(f).suffix in image_exts)
    return count

def explore_directory(base_path, max_depth=3, current_depth=0):
    """Recursively explore directory structure"""
    if not base_path.exists():
        print(f"  ❌ NOT FOUND: {base_path}")
        return
    
    if current_depth >= max_depth:
        return
    
    indent = "  " * current_depth
    
    try:
        items = sorted(base_path.iterdir())
        for item in items:
            if item.is_dir():
                img_count = count_images(item)
                print(f"{indent}📁 {item.name}/ ({img_count} images)")
                if img_count > 0 and current_depth < max_depth - 1:
                    explore_directory(item, max_depth, current_depth + 1)
            else:
                if item.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}:
                    print(f"{indent}🖼️  {item.name}")
    except PermissionError:
        print(f"{indent}⚠️  Permission denied")

def main():
    print("="*70)
    print("DFU Dataset Structure Verification")
    print("="*70)
    
    print(f"\n📂 Base Directory: {BASE_DIR}")
    if not BASE_DIR.exists():
        print("❌ Base directory does not exist!")
        return
    
    # Check RGB Dataset
    print(f"\n{'='*70}")
    print("🔍 RGB DATASET STRUCTURE")
    print(f"{'='*70}")
    print(f"Location: {RGB_SOURCE}")
    
    if RGB_SOURCE.exists():
        total_rgb = count_images(RGB_SOURCE)
        print(f"Total RGB images: {total_rgb}\n")
        explore_directory(RGB_SOURCE, max_depth=4)
    else:
        print("❌ RGB dataset not found!")
    
    # Check Thermal Dataset
    print(f"\n{'='*70}")
    print("🔍 THERMAL DATASET STRUCTURE")
    print(f"{'='*70}")
    print(f"Location: {THERMAL_SOURCE}")
    
    if THERMAL_SOURCE.exists():
        total_thermal = count_images(THERMAL_SOURCE)
        print(f"Total Thermal images: {total_thermal}\n")
        explore_directory(THERMAL_SOURCE, max_depth=4)
    else:
        print("❌ Thermal dataset not found!")
    
    # Check for specific expected directories
    print(f"\n{'='*70}")
    print("📋 EXPECTED STRUCTURE VERIFICATION")
    print(f"{'='*70}")
    
    expected_rgb = {
        "Original Images": RGB_SOURCE / "Original Images",
        "Patches": RGB_SOURCE / "Patches",
        "Patches/Abnormal": RGB_SOURCE / "Patches" / "Abnormal",
        "Patches/Healthy skin": RGB_SOURCE / "Patches" / "Healthy skin",
        "TestSet": RGB_SOURCE / "TestSet",
    }
    
    print("\nRGB Dataset:")
    for name, path in expected_rgb.items():
        if path.exists():
            img_count = count_images(path)
            print(f"  ✅ {name}: {img_count} images")
        else:
            print(f"  ❌ {name}: NOT FOUND")
            # Try to find alternative names
            parent = path.parent
            if parent.exists():
                print(f"     Available in {parent.name}:")
                for item in parent.iterdir():
                    if item.is_dir():
                        print(f"       - {item.name}/ ({count_images(item)} images)")
    
    expected_thermal = {
        "ThermoDataBase": THERMAL_SOURCE / "ThermoDataBase",
        "train": THERMAL_SOURCE / "ThermoDataBase" / "train",
        "train/Control Group": THERMAL_SOURCE / "ThermoDataBase" / "train" / "Control Group",
        "train/DM Group": THERMAL_SOURCE / "ThermoDataBase" / "train" / "DM Group",
        "val": THERMAL_SOURCE / "ThermoDataBase" / "val",
        "val/Control Group": THERMAL_SOURCE / "ThermoDataBase" / "val" / "Control Group",
        "val/DM Group": THERMAL_SOURCE / "ThermoDataBase" / "val" / "DM Group",
    }
    
    print("\nThermal Dataset:")
    for name, path in expected_thermal.items():
        if path.exists():
            img_count = count_images(path)
            print(f"  ✅ {name}: {img_count} images")
        else:
            print(f"  ❌ {name}: NOT FOUND")
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 SUMMARY")
    print(f"{'='*70}")
    
    rgb_ready = (RGB_SOURCE / "Patches").exists()
    thermal_ready = (THERMAL_SOURCE / "ThermoDataBase" / "train").exists()
    
    print(f"RGB Dataset Ready: {'✅ YES' if rgb_ready else '❌ NO'}")
    print(f"Thermal Dataset Ready: {'✅ YES' if thermal_ready else '❌ NO'}")
    
    if rgb_ready and thermal_ready:
        print("\n✨ Both datasets are ready for processing!")
        print("   Run: python3 prepare_datasets.py")
    else:
        print("\n⚠️  Please fix the dataset structure before proceeding.")
        print("   Expected structure:")
        print("   ~/DFU_MMT/")
        print("   ├── DFU_RGB/")
        print("   │   ├── Original Images/")
        print("   │   ├── Patches/")
        print("   │   │   ├── Abnormal/")
        print("   │   │   └── Healthy skin/")
        print("   │   └── TestSet/")
        print("   └── DFU_Thermal/")
        print("       └── ThermoDataBase/")
        print("           ├── train/")
        print("           │   ├── Control Group/")
        print("           │   └── DM Group/")
        print("           └── val/")
        print("               ├── Control Group/")
        print("               └── DM Group/")

if __name__ == "__main__":
    main()
