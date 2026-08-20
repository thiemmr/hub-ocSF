#!/usr/bin/env python3
"""Dataset preprocessing and environment check. Run: python preprocess.py --help"""
import argparse, os, shutil, sys
from pathlib import Path


def check_environment():
    """Verify Python packages and CUDA availability."""
    pkgs = {'torch': 'PyTorch', 'torchvision': 'Torchvision', 'PIL': 'Pillow',
            'numpy': 'NumPy', 'sklearn': 'scikit-learn', 'tqdm': 'tqdm'}
    for mod, name in pkgs.items():
        try:
            __import__(mod)
            print(f"[OK] {name}")
        except ImportError:
            print(f"[MISSING] {name} — pip install {name.lower()}")
    try:
        import torch
        print(f"[INFO] PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        pass


def organize_from_csv(csv_path, src_dir, out_dir, split=0.8, seed=42):
    """Organize images into ImageFolder structure from a CSV (cols: filename, label)."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(df, train_size=split, stratify=df['label'], random_state=seed)
    for split_name, split_df in [('train', train_df), ('val', val_df)]:
        for _, row in split_df.iterrows():
            dst = os.path.join(out_dir, split_name, str(row['label']), row['filename'])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(os.path.join(src_dir, row['filename']), dst)
    print(f"Done: {len(train_df)} train, {len(val_df)} val images")


def analyze_dataset(data_dir):
    """Print dataset statistics."""
    for split in ['train', 'val', 'test']:
        p = os.path.join(data_dir, split)
        if not os.path.isdir(p):
            continue
        classes = sorted(os.listdir(p))
        counts = {c: len(os.listdir(os.path.join(p, c))) for c in classes if os.path.isdir(os.path.join(p, c))}
        total = sum(counts.values())
        print(f"\n{split.upper()} ({total} images, {len(counts)} classes):")
        for c, n in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {n:5d}  {c}")


def create_template_structure(out_dir, class_names):
    """Create empty ImageFolder template."""
    for split in ['train', 'val']:
        for cls in class_names:
            os.makedirs(os.path.join(out_dir, split, cls), exist_ok=True)
    print(f"Created template at {out_dir}")


def main():
    p = argparse.ArgumentParser(description='Dataset preprocessing for image classifier')
    p.add_argument('--check', action='store_true', help='Verify environment')
    p.add_argument('--csv', default='', help='CSV with filename/label columns')
    p.add_argument('--src', default='./raw_images', help='Source image directory')
    p.add_argument('--out', default='./dataset', help='Output dataset directory')
    p.add_argument('--split', type=float, default=0.8, help='Train/val split ratio')
    p.add_argument('--analyze', action='store_true', help='Analyze existing dataset')
    p.add_argument('--template', nargs='*', help='Create empty template (list class names)')
    p.add_argument('--data_dir', default='./dataset', help='Dataset directory for analysis')
    args = p.parse_args()

    if args.check:
        check_environment()
    if args.csv:
        organize_from_csv(args.csv, args.src, args.out, args.split)
    if args.analyze:
        analyze_dataset(args.data_dir)
    if args.template is not None:
        create_template_structure(args.out, args.template if args.template else ['class_a', 'class_b'])
    if not any([args.check, bool(args.csv), args.analyze, args.template is not None]):
        check_environment()


if __name__ == '__main__':
    main()
