#!/usr/bin/env python3
"""
Image Standardization Script
Resizes and pads all images to 224×224 while preserving aspect ratio
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

def standardize_images(input_dir, output_dir, target_size=224, verbose=True):
    """
    Standardize all images to target_size×target_size
    
    Method: Resize longest edge to target_size, then pad with black
    This preserves aspect ratio and avoids distortion
    """
    
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    
    # Find all images
    images = []
    for ext in image_exts:
        images.extend(input_dir.rglob(f'*{ext}'))
        images.extend(input_dir.rglob(f'*{ext.upper()}'))
    
    if not images:
        print(f"⚠️  No images found in {input_dir}")
        return False
    
    success_count = 0
    error_count = 0
    
    print(f"\n{'='*70}")
    print(f"STANDARDIZING {len(images)} IMAGES")
    print(f"{'='*70}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Target: {target_size}×{target_size} px")
    print(f"Method: Resize + Pad (preserves aspect ratio)\n")
    
    for img_path in tqdm(images, desc="Processing"):
        try:
            # Open image
            with Image.open(img_path) as img:
                # Convert to RGB if grayscale (for thermal)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Calculate padding to preserve aspect ratio
                width, height = img.size
                scale = target_size / max(width, height)
                
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                # Resize
                img_resized = img.resize((new_width, new_height), Image.BILINEAR)
                
                # Create new image with black background
                img_standardized = Image.new('RGB', (target_size, target_size), color=0)
                
                # Calculate padding
                pad_x = (target_size - new_width) // 2
                pad_y = (target_size - new_height) // 2
                
                # Paste resized image onto background
                img_standardized.paste(img_resized, (pad_x, pad_y))
                
                # Preserve directory structure
                rel_path = img_path.relative_to(input_dir)
                out_path = output_dir / rel_path
                out_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save
                img_standardized.save(out_path, quality=95)
                success_count += 1
                
        except Exception as e:
            if verbose:
                print(f"  ❌ Error processing {img_path.name}: {str(e)}")
            error_count += 1
    
    print(f"\n{'='*70}")
    print(f"STANDARDIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"✓ Successfully processed: {success_count} images")
    if error_count > 0:
        print(f"❌ Errors: {error_count} images")
    print(f"Output saved to: {output_dir}")
    
    return success_count > 0

def verify_standardization(directory, target_size=224):
    """Verify all images are correctly standardized"""
    
    print(f"\n{'='*70}")
    print(f"VERIFYING STANDARDIZATION")
    print(f"{'='*70}")
    print(f"Directory: {directory}\n")
    
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    
    images = []
    for ext in image_exts:
        images.extend(directory.rglob(f'*{ext}'))
        images.extend(directory.rglob(f'*{ext.upper()}'))
    
    if not images:
        print(f"❌ No images found in {directory}")
        return False
    
    all_correct = True
    size_distribution = {}
    
    for img_path in images:
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                size = f"{width}×{height}"
                
                if size not in size_distribution:
                    size_distribution[size] = 0
                size_distribution[size] += 1
                
                if width != target_size or height != target_size:
                    print(f"⚠️  {img_path.name}: {width}×{height} (expected {target_size}×{target_size})")
                    all_correct = False
                    
        except Exception as e:
            print(f"❌ Error reading {img_path.name}: {str(e)}")
            all_correct = False
    
    print(f"✓ Analyzed {len(images)} images\n")
    
    print(f"📊 Size Distribution:")
    for size, count in sorted(size_distribution.items()):
        if size == f"{target_size}×{target_size}":
            print(f"   ✓ {size}: {count} images")
        else:
            print(f"   ❌ {size}: {count} images")
    
    if all_correct and len(size_distribution) == 1:
        print(f"\n✅ ALL IMAGES CORRECTLY STANDARDIZED TO {target_size}×{target_size}!")
        return True
    else:
        print(f"\n⚠️  STANDARDIZATION ISSUES DETECTED")
        return False

def main():
    print("="*70)
    print("IMAGE STANDARDIZATION FOR MULTIMODAL DFU PROJECT")
    print("="*70)
    
    # Get base directory
    print("\nEnter path to DFU_MMT directory (or press Enter for ~/DFU_MMT):")
    user_input = input().strip()
    
    if user_input:
        base_dir = Path(user_input)
    else:
        base_dir = Path.home() / "DFU_MMT"
    
    if not base_dir.exists():
        print(f"❌ Directory not found: {base_dir}")
        return
    
    data_dir = base_dir / "data"
    rgb_dir = data_dir / "rgb"
    thermal_dir = data_dir / "thermal"
    
    # Check if data/rgb and data/thermal exist
    if not rgb_dir.exists() or not thermal_dir.exists():
        print(f"❌ Dataset directories not found!")
        print(f"   {rgb_dir.exists()}: {rgb_dir}")
        print(f"   {thermal_dir.exists()}: {thermal_dir}")
        return
    
    # Create output directories
    rgb_std_dir = data_dir / "rgb_standardized"
    thermal_std_dir = data_dir / "thermal_standardized"
    
    # Standardize RGB
    print("\n" + "="*70)
    print("STEP 1: STANDARDIZING RGB IMAGES")
    print("="*70)
    rgb_success = standardize_images(rgb_dir, rgb_std_dir, target_size=224)
    
    # Standardize Thermal
    print("\n" + "="*70)
    print("STEP 2: STANDARDIZING THERMAL IMAGES")
    print("="*70)
    thermal_success = standardize_images(thermal_dir, thermal_std_dir, target_size=224)
    
    # Verify
    print("\n" + "="*70)
    print("STEP 3: VERIFICATION")
    print("="*70)
    
    if rgb_success:
        rgb_verified = verify_standardization(rgb_std_dir)
    
    if thermal_success:
        thermal_verified = verify_standardization(thermal_std_dir)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if rgb_success and thermal_success:
        print(f"\n✅ STANDARDIZATION COMPLETE!")
        print(f"\nStandardized datasets ready for training:")
        print(f"  RGB:     {rgb_std_dir}")
        print(f"  Thermal: {thermal_std_dir}")
        print(f"\nNext steps:")
        print(f"  1. Create PyTorch DataLoaders")
        print(f"  2. Use rgb_std_dir and thermal_std_dir as data sources")
        print(f"  3. Apply normalization transforms in DataLoader")
        print(f"  4. Begin training")
    else:
        print(f"\n⚠️  STANDARDIZATION FAILED")
        print(f"  RGB:     {'✓' if rgb_success else '❌'}")
        print(f"  Thermal: {'✓' if thermal_success else '❌'}")

if __name__ == "__main__":
    main()