"""
One-Time Dataset Split Script
==============================
Run this ONCE on your full dataset to create fixed train/val/test splits.
Never run it again — just add new data to the train/ folder going forward.

Usage:
    python split_dataset.py --data_dir data/all/ --output_dir data/
    python split_dataset.py --data_dir data/all/ --output_dir data/ --val_ratio 0.1 --test_ratio 0.1
"""

import os
import shutil
import pickle
import numpy as np
from pathlib import Path
import argparse


def split_dataset(args):
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    train_dir = output_dir / 'train'
    val_dir   = output_dir / 'val'
    test_dir  = output_dir / 'test'

    # Safety check — don't accidentally re-split if folders already exist
    for d in [train_dir, val_dir, test_dir]:
        if d.exists() and any(d.iterdir()):
            print(f"ERROR: '{d}' already exists and is not empty.")
            print("This script is meant to be run once. If you really want to re-split, delete the folders manually first.")
            return

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Collect all pkl files
    all_files = sorted(data_dir.glob('*.pkl'))
    if not all_files:
        print(f"No .pkl files found in {data_dir}")
        return

    print(f"Found {len(all_files)} .pkl files")

    # Shuffle with a fixed seed so the split is reproducible
    rng = np.random.default_rng(seed=42)
    indices = np.arange(len(all_files))
    rng.shuffle(indices)
    all_files = [all_files[i] for i in indices]

    n       = len(all_files)
    n_val   = max(1, int(n * args.val_ratio))
    n_test  = max(1, int(n * args.test_ratio))
    n_train = n - n_val - n_test

    train_files = all_files[:n_train]
    val_files   = all_files[n_train:n_train + n_val]
    test_files  = all_files[n_train + n_val:]

    print(f"Splitting: {len(train_files)} train / {len(val_files)} val / {len(test_files)} test")

    for f in train_files:
        shutil.copy2(f, train_dir / f.name)
    for f in val_files:
        shutil.copy2(f, val_dir / f.name)
    for f in test_files:
        shutil.copy2(f, test_dir / f.name)

    # Save a manifest so you always know what went where
    manifest = {
        'train': [f.name for f in train_files],
        'val':   [f.name for f in val_files],
        'test':  [f.name for f in test_files],
    }
    manifest_path = output_dir / 'split_manifest.txt'
    with open(manifest_path, 'w') as mf:
        for split, files in manifest.items():
            mf.write(f"=== {split.upper()} ({len(files)} files) ===\n")
            for fname in files:
                mf.write(f"  {fname}\n")
            mf.write("\n")

    print(f"\nDone. Split manifest saved to '{manifest_path}'")
    print("Going forward, add new data ONLY to the train/ folder.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='One-time dataset split into train/val/test folders')
    parser.add_argument('--data_dir',   type=str, required=True, help='Folder containing all your .pkl files')
    parser.add_argument('--output_dir', type=str, required=True, help='Where to create train/, val/, test/ subfolders')
    parser.add_argument('--val_ratio',  type=float, default=0.1,  help='Fraction for validation')
    parser.add_argument('--test_ratio', type=float, default=0.1,  help='Fraction for test')
    args = parser.parse_args()
    split_dataset(args)