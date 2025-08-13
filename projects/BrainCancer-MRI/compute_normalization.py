#!/usr/bin/env python3
"""
Comprehensive normalization statistics computation for Brain Cancer MRI dataset.

This script can:
1. Compute and display statistics (like compute_mean_std.py)
2. Pre-compute and save statistics (like precompute_normalization.py)
3. Verify existing pre-computed statistics

Since we use a fixed seed (42) for reproducible splits, we can pre-calculate
the mean and std values once and reuse them across all experiments.
"""

from torch.utils.data import DataLoader
import torch
import argparse
import yaml
import os
import json
import sys
from pathlib import Path

# Add the project directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

try:
    from data.dataset import load_datasets
    from utils.helpers import calculate_mean_std
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you're running the script from the correct directory")
    print("   Expected: src/projects/BrainCancer-MRI/")
    print(f"   Current: {os.getcwd()}")
    sys.exit(1)


def compute_normalization_stats(config_path, mode="display", output_file="normalization_stats.json"):
    """
    Compute normalization statistics for the fixed seed split.

    Args:
        config_path: Path to config file
        mode: "display", "save", or "verify"
        output_file: Output file to save statistics (for save mode)
    """

    # Resolve absolute path to config file
    if not os.path.isabs(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_path)

    print(f"📋 Loading config from: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("🔧 Configuration loaded successfully")
    print(f"📊 Dataset path: {config['dataset']['path']}")
    print(f"🖼️  Image size: {config['dataset']['img_size']}")
    print(f"🎲 Fixed seed: {config['dataset']['seed']}")

    # Load datasets without normalization (for statistics computation)
    print("\n📊 Loading datasets without normalization...")
    train_ds_raw, val_ds_raw, test_ds_raw = load_datasets(
        config, mean=None, std=None, grayscale=False
    )

    print(f"📈 Dataset sizes:")
    print(f"   Training: {len(train_ds_raw)} samples")
    print(f"   Validation: {len(val_ds_raw)} samples")
    print(f"   Test: {len(test_ds_raw)} samples")

    # Get batch size from config
    batch_size = config['dataset']['batch_size']
    num_workers = config['dataset']['num_workers']
    pin_memory = config['dataset']['pin_memory']

    print(f"\n🔧 Computing statistics with:")
    print(f"   Batch size: {batch_size}")
    print(f"   Num workers: {num_workers}")
    print(f"   Pin memory: {pin_memory}")

    # Compute mean/std on training set only
    print("\n📊 Computing mean/std on training set...")
    mean, std = calculate_mean_std(
        train_ds_raw,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # Convert to lists for JSON serialization
    mean_list = mean.tolist()
    std_list = std.tolist()

    print(f"\n✅ Statistics computed successfully!")
    print(f"📊 Training set mean: {mean_list}")
    print(f"📊 Training set std: {std_list}")

    # Mode-specific actions
    if mode == "display":
        print(f"\n📋 Mode: Display only")
        print(f"📊 Mean: {mean_list}")
        print(f"📊 Std: {std_list}")

    elif mode == "save":
        print(f"\n💾 Mode: Save to files")

        # Create statistics dictionary
        stats = {
            "mean": mean_list,
            "std": std_list,
            "config": {
                "dataset_path": config['dataset']['path'],
                "img_size": config['dataset']['img_size'],
                "seed": config['dataset']['seed'],
                "train_split": config['dataset']['train_split'],
                "val_split": config['dataset']['val_split'],
                "test_split": config['dataset']['test_split'],
                "batch_size": batch_size,
                "num_workers": num_workers,
                "pin_memory": pin_memory
            },
            "dataset_sizes": {
                "train": len(train_ds_raw),
                "val": len(val_ds_raw),
                "test": len(test_ds_raw)
            },
            "computation_info": {
                "description": "Pre-computed normalization statistics for fixed seed split",
                "usage": "These statistics can be reused across all experiments since we use a fixed seed",
                "warning": "Only valid for the exact same dataset and split configuration"
            }
        }

        # Save to JSON file
        output_path = os.path.join(script_dir, output_file)
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"💾 Statistics saved to: {output_path}")
        print(f"📋 File size: {os.path.getsize(output_path)} bytes")

        # Save as Python constants for easy import
        python_output_path = os.path.join(
            script_dir, "normalization_constants.py")
        with open(python_output_path, 'w') as f:
            f.write(f'''# Auto-generated normalization constants
# Generated from compute_normalization.py
# Fixed seed: {config['dataset']['seed']}

NORMALIZATION_MEAN = {mean_list}
NORMALIZATION_STD = {std_list}

# Dataset configuration
DATASET_CONFIG = {{
    "path": "{config['dataset']['path']}",
    "img_size": {config['dataset']['img_size']},
    "seed": {config['dataset']['seed']},
    "train_split": {config['dataset']['train_split']},
    "val_split": {config['dataset']['val_split']},
    "test_split": {config['dataset']['test_split']}
}}

# Dataset sizes
DATASET_SIZES = {{
    "train": {len(train_ds_raw)},
    "val": {len(val_ds_raw)},
    "test": {len(test_ds_raw)}
}}
''')

        print(f"🐍 Python constants saved to: {python_output_path}")

        print(f"\n🎯 Usage in training script:")
        print(
            f"   from normalization_constants import NORMALIZATION_MEAN, NORMALIZATION_STD")
        print(f"   train_ds, val_ds, _ = load_datasets(config, mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD)")

        print(f"\n✅ Pre-computation completed successfully!")
        print(f"📊 You can now use these statistics in all your experiments without recomputing them.")

    elif mode == "verify":
        print(f"\n🔍 Mode: Verify existing pre-computed statistics")

        # Try to import existing constants
        try:
            from normalization_constants import NORMALIZATION_MEAN, NORMALIZATION_STD
            print(f"✅ Found existing pre-computed statistics:")
            print(f"   Saved mean: {NORMALIZATION_MEAN}")
            print(f"   Saved std: {NORMALIZATION_STD}")
            print(f"   Computed mean: {mean_list}")
            print(f"   Computed std: {std_list}")

            # Check if they match
            if mean_list == NORMALIZATION_MEAN and std_list == NORMALIZATION_STD:
                print(f"✅ Statistics match! Pre-computed values are correct.")
            else:
                print(
                    f"⚠️  Statistics don't match! You may need to re-run pre-computation.")
                print(f"   This could happen if the dataset or configuration changed.")

        except ImportError:
            print(f"❌ No pre-computed statistics found.")
            print(f"💡 Run with --mode save to create pre-computed statistics.")


def main():
    parser = argparse.ArgumentParser(
        description='Compute normalization statistics for Brain Cancer MRI dataset'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file (default: config/config.yaml)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['display', 'save', 'verify'],
        default='display',
        help='Mode: display (print only), save (create files), verify (check existing)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='normalization_stats.json',
        help='Output file name for save mode (default: normalization_stats.json)'
    )

    args = parser.parse_args()

    try:
        compute_normalization_stats(args.config, args.mode, args.output)
    except Exception as e:
        print(f"❌ Error during computation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
