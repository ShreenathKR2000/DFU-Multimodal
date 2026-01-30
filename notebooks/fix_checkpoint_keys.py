#!/usr/bin/env python3
"""
CHECKPOINT KEY MAPPING FIX
==========================

The trained models saved checkpoints with 'backbone' prefix,
but evaluation scripts expect 'resnet'/'vit' prefix.

This utility converts checkpoint keys automatically.
"""

import torch
from pathlib import Path

def fix_checkpoint_keys(checkpoint_path, output_path=None):
    """
    Convert checkpoint keys from 'backbone.*' to 'resnet.*' or 'vit.*'
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model_state_dict = checkpoint.get('model_state_dict', {})
    
    if not model_state_dict:
        print("⚠️  No 'model_state_dict' found in checkpoint")
        print(f"Available keys: {list(checkpoint.keys())}")
        return None
    
    # Get first key to determine architecture
    first_key = list(model_state_dict.keys())[0]
    print(f"First key in state dict: {first_key}")
    
    # Determine if this is RGB (resnet) or Thermal (vit)
    if 'backbone' in first_key:
        print("\n🔄 Converting 'backbone.*' → 'resnet.*' keys...")
        new_state_dict = {}
        for key, value in model_state_dict.items():
            new_key = key.replace('backbone.', 'resnet.')
            new_state_dict[new_key] = value
            if list(new_state_dict.keys()).index(new_key) < 3:
                print(f"  {key} → {new_key}")
    else:
        print(f"Keys already in correct format: {first_key}")
        new_state_dict = model_state_dict
    
    # Update checkpoint
    checkpoint['model_state_dict'] = new_state_dict
    
    # Save converted checkpoint
    if output_path is None:
        output_path = checkpoint_path
    
    torch.save(checkpoint, output_path)
    print(f"\n✅ Saved fixed checkpoint to: {output_path}")
    
    return checkpoint


def main():
    project_root = Path(__file__).resolve().parents[1]
    checkpoint_dir = project_root / "logs"
    
    checkpoints = [
        ("checkpoints_rgb_only", "RGB-Only (ResNet50)"),
        ("checkpoints_thermal_only", "Thermal-Only (ViT)"),
        ("checkpoints_multimodal", "Multimodal Fusion"),
    ]
    
    print("="*70)
    print("FIXING CHECKPOINT KEY MISMATCH")
    print("="*70)
    
    for checkpoint_name, model_name in checkpoints:
        checkpoint_path = checkpoint_dir / checkpoint_name / "best_model.pt"
        
        if checkpoint_path.exists():
            print(f"\n{model_name}")
            print("-" * 70)
            fix_checkpoint_keys(checkpoint_path)
        else:
            print(f"\n⚠️  Not found: {checkpoint_path}")
    
    print("\n" + "="*70)
    print("✅ All checkpoints fixed! Now try running extended_metrics.py again")
    print("="*70)


if __name__ == "__main__":
    main()
