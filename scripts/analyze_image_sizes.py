#!/usr/bin/env python3
"""
Image Size Analysis and Standardization Checker
Analyzes image dimensions across RGB and Thermal datasets
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np
from collections import defaultdict

def analyze_image_sizes(directory, modality_name, max_samples=100):
    """Detailed analysis of image sizes in dataset"""
    print(f"\n{'='*75}")
    print(f"ANALYZING {modality_name.upper()} IMAGES")
    print(f"{'='*75}")
    
    if not directory.exists():
        print(f"⚠️  Directory not found: {directory}")
        return None
    
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.JPG', '.PNG'}
    
    # Statistics storage
    sizes_data = []
    corrupted = []
    by_split = defaultdict(lambda: defaultdict(list))
    
    print(f"Scanning: {directory}\n")
    
    # Walk through directory structure
    sample_count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if sample_count >= max_samples:
                break
                
            if Path(file).suffix not in image_exts:
                continue
            
            file_path = Path(root) / file
            
            # Determine split and class
            rel_path = file_path.relative_to(directory)
            parts = rel_path.parts
            
            split = parts[0] if len(parts) > 0 else "unknown"
            cls = parts[1] if len(parts) > 1 else "unknown"
            
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    mode = img.mode
                    file_size_kb = file_path.stat().st_size / 1024
                    
                    sizes_data.append({
                        'width': width,
                        'height': height,
                        'mode': mode,
                        'split': split,
                        'class': cls,
                        'file_size_kb': file_size_kb,
                        'path': str(file_path.relative_to(directory))
                    })
                    
                    by_split[split][cls].append((width, height))
                    sample_count += 1
                    
            except Exception as e:
                corrupted.append((str(file_path), str(e)))
    
    if not sizes_data:
        print(f"❌ No images found in {directory}")
        return None
    
    # Convert to numpy arrays
    widths = np.array([s['width'] for s in sizes_data])
    heights = np.array([s['height'] for s in sizes_data])
    file_sizes = np.array([s['file_size_kb'] for s in sizes_data])
    
    # Print statistics
    print(f"✓ Analyzed {len(sizes_data)} images\n")
    
    if corrupted:
        print(f"⚠️  Found {len(corrupted)} corrupted images:")
        for path, error in corrupted[:3]:
            print(f"   {path}: {error}")
        print()
    
    print(f"📊 WIDTH STATISTICS:")
    print(f"   Min:      {widths.min()} px")
    print(f"   Max:      {widths.max()} px")
    print(f"   Mean:     {widths.mean():.1f} px")
    print(f"   Median:   {np.median(widths):.1f} px")
    print(f"   Std Dev:  {widths.std():.1f} px")
    print(f"   Unique:   {len(np.unique(widths))} different widths")
    
    print(f"\n📊 HEIGHT STATISTICS:")
    print(f"   Min:      {heights.min()} px")
    print(f"   Max:      {heights.max()} px")
    print(f"   Mean:     {heights.mean():.1f} px")
    print(f"   Median:   {np.median(heights):.1f} px")
    print(f"   Std Dev:  {heights.std():.1f} px")
    print(f"   Unique:   {len(np.unique(heights))} different heights")
    
    print(f"\n💾 FILE SIZE STATISTICS:")
    print(f"   Min:      {file_sizes.min():.1f} KB")
    print(f"   Max:      {file_sizes.max():.1f} KB")
    print(f"   Mean:     {file_sizes.mean():.1f} KB")
    print(f"   Median:   {np.median(file_sizes):.1f} KB")
    
    # Aspect ratio analysis
    aspect_ratios = widths / heights
    print(f"\n📐 ASPECT RATIO STATISTICS:")
    print(f"   Min:      {aspect_ratios.min():.2f}")
    print(f"   Max:      {aspect_ratios.max():.2f}")
    print(f"   Mean:     {aspect_ratios.mean():.2f}")
    print(f"   Median:   {np.median(aspect_ratios):.2f}")
    
    # Most common sizes
    unique_sizes, counts = np.unique(np.column_stack([widths, heights]), axis=0, return_counts=True)
    top_sizes_idx = np.argsort(-counts)[:5]
    
    print(f"\n🔝 TOP 5 MOST COMMON IMAGE SIZES:")
    for i, idx in enumerate(top_sizes_idx, 1):
        w, h = unique_sizes[idx]
        cnt = counts[idx]
        percent = cnt / len(sizes_data) * 100
        print(f"   {i}. {int(w)}×{int(h)} px: {cnt} images ({percent:.1f}%)")
    
    # Color mode distribution
    modes = [s['mode'] for s in sizes_data]
    print(f"\n🎨 COLOR MODE DISTRIBUTION:")
    unique_modes = sorted(set(modes))
    for mode in unique_modes:
        cnt = modes.count(mode)
        percent = cnt / len(sizes_data) * 100
        print(f"   {mode}: {cnt} images ({percent:.1f}%)")
    
    # Size consistency by split
    print(f"\n📈 SIZE DISTRIBUTION BY SPLIT:")
    for split in sorted(by_split.keys()):
        split_images = by_split[split]
        total_split = sum(len(v) for v in split_images.values())
        unique_sizes_split = len(set().union(*([(w, h) for w, h in v] for v in split_images.values())))
        print(f"   {split.upper()}: {total_split} images, {unique_sizes_split} unique sizes")
        for cls in sorted(split_images.keys()):
            if split_images[cls]:
                w, h = split_images[cls][0]
                cnt = len(split_images[cls])
                unique_in_cls = len(set(split_images[cls]))
                print(f"      {cls}: {cnt} images, {unique_in_cls} unique sizes")
    
    # Consistency check
    print(f"\n✅ CONSISTENCY ANALYSIS:")
    if len(np.unique(widths)) == 1 and len(np.unique(heights)) == 1:
        print(f"   ✓ All images have SAME size: {widths[0]}×{heights[0]} px")
        consistency = 100
    else:
        consistency = (1 - widths.std() / widths.mean()) * 100
        if consistency > 95:
            print(f"   ⚠️  Images have VARIED sizes (std: {widths.std():.0f} px, {consistency:.1f}% consistent)")
        elif consistency > 80:
            print(f"   ⚠️  SIGNIFICANT size variation (consistency: {consistency:.1f}%)")
        else:
            print(f"   ❌ HIGH size variation (consistency: {consistency:.1f}%)")
    
    return {
        'widths': widths,
        'heights': heights,
        'modes': modes,
        'sizes_data': sizes_data,
        'aspect_ratios': aspect_ratios,
        'consistency': consistency,
        'file_sizes': file_sizes
    }

def main():
    print("="*75)
    print("IMAGE SIZE ANALYSIS FOR MULTIMODAL DFU PROJECT")
    print("="*75)
    
    # Get base directory from user
    print("\nEnter the path to your DFU_MMT directory (or press Enter for ~/DFU_MMT):")
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
    
    # Analyze datasets
    rgb_analysis = analyze_image_sizes(rgb_dir, "RGB")
    thermal_analysis = analyze_image_sizes(thermal_dir, "THERMAL")
    
    # Comparison
    if rgb_analysis and thermal_analysis:
        print(f"\n{'='*75}")
        print("RGB vs THERMAL COMPARISON")
        print(f"{'='*75}")
        
        rgb_w = rgb_analysis['widths']
        rgb_h = rgb_analysis['heights']
        thermal_w = thermal_analysis['widths']
        thermal_h = thermal_analysis['heights']
        
        print(f"\n📏 SIZE RANGES:")
        print(f"   RGB:     {rgb_w.min()}-{rgb_w.max()} × {rgb_h.min()}-{rgb_h.max()} px")
        print(f"   Thermal: {thermal_w.min()}-{thermal_w.max()} × {thermal_h.min()}-{thermal_h.max()} px")
        
        # Comparison metrics
        print(f"\n⚠️  COMPATIBILITY FOR TRAINING:")
        
        rgb_mean_aspect = rgb_analysis['aspect_ratios'].mean()
        thermal_mean_aspect = thermal_analysis['aspect_ratios'].mean()
        aspect_diff = abs(rgb_mean_aspect - thermal_mean_aspect)
        
        print(f"   RGB Mean Size:     {rgb_w.mean():.0f}×{rgb_h.mean():.0f} px")
        print(f"   Thermal Mean Size: {thermal_w.mean():.0f}×{thermal_h.mean():.0f} px")
        print(f"   RGB Mean Aspect Ratio:     {rgb_mean_aspect:.2f}")
        print(f"   Thermal Mean Aspect Ratio: {thermal_mean_aspect:.2f}")
        print(f"   Aspect Ratio Difference:   {aspect_diff:.2f} ({aspect_diff/max(rgb_mean_aspect, thermal_mean_aspect)*100:.1f}%)")
        
        # Recommendations
        print(f"\n{'='*75}")
        print("STANDARDIZATION RECOMMENDATIONS")
        print(f"{'='*75}")
        
        # Check if standardization is needed
        needs_standardization = (len(np.unique(rgb_w)) > 1 or len(np.unique(thermal_w)) > 1 or
                               len(np.unique(rgb_h)) > 1 or len(np.unique(thermal_h)) > 1)
        
        if needs_standardization:
            print(f"\n✅ STANDARDIZATION REQUIRED")
            print(f"\nRecommended standard size: 224×224 px")
            print(f"   - Good balance between model complexity and training speed")
            print(f"   - Compatible with EfficientNet and ViT pretrained models")
            print(f"   - Industry standard for medical imaging")
            
            print(f"\nAlternatives:")
            print(f"   - 256×256 px: Better detail preservation (slower training)")
            print(f"   - 384×384 px: Maximum detail (much slower, GPU intensive)")
            print(f"   - 192×192 px: Faster training (may lose fine details)")
            
            print(f"\nResizing Strategy:")
            print(f"   1. Preserve aspect ratio: Resize to fit within 224×224")
            print(f"   2. Pad with zeros: Add padding to reach 224×224 (recommended)")
            print(f"   3. Crop center: Take center 224×224 region")
        else:
            print(f"\n✓ All images already consistent size - No standardization needed")
        
        # Training considerations
        print(f"\n{'='*75}")
        print("TRAINING CONSIDERATIONS")
        print(f"{'='*75}")
        
        print(f"\n1. IMAGE SIZE MATCHING:")
        if rgb_w.mean() == thermal_w.mean() and rgb_h.mean() == thermal_h.mean():
            print(f"   ✓ RGB and Thermal have SAME mean size - Good!")
        else:
            print(f"   ⚠️  RGB and Thermal have DIFFERENT mean sizes - Standardization helps")
        
        print(f"\n2. DATA LOADER CONFIGURATION:")
        print(f"   - Use transform pipeline with:")
        print(f"     • Resize(224, 224)")
        print(f"     • Normalization (ImageNet mean/std for RGB)")
        print(f"     • Normalization (Thermal specific for thermal)")
        
        print(f"\n3. ASPECT RATIO PRESERVATION:")
        print(f"   RGB Consistency:     {rgb_analysis['consistency']:.1f}%")
        print(f"   Thermal Consistency: {thermal_analysis['consistency']:.1f}%")
        
        if rgb_analysis['consistency'] < 90 or thermal_analysis['consistency'] < 90:
            print(f"   ⚠️  High aspect ratio variation - Use padding method")
        else:
            print(f"   ✓ Reasonable consistency - Resizing acceptable")
        
        # Model recommendations
        print(f"\n4. MODEL ARCHITECTURE IMPLICATIONS:")
        print(f"   - Both branches MUST use SAME input size (224×224 recommended)")
        print(f"   - Use different normalization for RGB vs Thermal:")
        print(f"     • RGB: ImageNet normalization (mean: [0.485, 0.456, 0.406])")
        print(f"     • Thermal: Dataset-specific or [0.5, 0.5, 0.5]")
        print(f"   - This ensures fair comparison between branches")

if __name__ == "__main__":
    main()