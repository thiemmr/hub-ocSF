---
name: pytorch-image-classifier-v2
description: This skill should be used when the user wants to train, fine-tune, or run inference with a PyTorch image classification model (especially ResNet18) on custom image datasets. Use this skill for tasks involving image classification, model training, data preprocessing for image datasets, model evaluation, and inference on new images. This is the optimized version (V2) using progressive disclosure - code lives in scripts/, detailed references in references/.
---

# PyTorch Image Classifier (V2 - Optimized)

## Overview

Build, train, and deploy image classifiers using PyTorch with pre-trained models (ResNet18 default). Uses progressive disclosure: scripts handle deterministic operations, references provide deep-dive docs on demand.

## Quick Start

### 1. Verify Environment

```bash
pip install torch torchvision pillow numpy scikit-learn tqdm
```

Run `scripts/preprocess.py --check` to verify GPU availability and package versions.

### 2. Prepare Dataset

Images must follow ImageFolder structure:
```
dataset/{train,val}/{class_name}/*.{jpg,png}
```

Use `scripts/preprocess.py` to organize raw images from a CSV mapping:
```bash
python scripts/preprocess.py --csv labels.csv --src ./raw_images --out ./dataset --split 0.8
```

### 3. Train

```bash
python scripts/train.py \
  --data_dir ./dataset \
  --arch resnet18 \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-3 \
  --save_dir ./checkpoints
```

Key flags: `--freeze_epochs 10` (phase 1), `--patience 10` (early stopping), `--amp` (mixed precision), `--scheduler cosine`.

### 4. Evaluate

```bash
python scripts/train.py --eval_only --model ./checkpoints/best_model.pth --data_dir ./dataset
```

### 5. Inference

```python
from scripts.inference import ImageClassifier
clf = ImageClassifier('checkpoints/best_model.pth', 'checkpoints/class_names.json')
pred, conf = clf.predict('test_image.jpg')
print(f"{pred}: {conf:.2%}")
```

Or batch inference from CLI:
```bash
python scripts/inference.py --model best_model.pth --images ./test_images/ --batch
```

## Workflow Decision Tree

```
User request
  ├─ "train"/"fine-tune" → Run scripts/train.py with appropriate flags
  ├─ "prepare dataset"   → Use scripts/preprocess.py or reference dataset structure
  ├─ "inference"/"classify" → Use scripts/inference.py (CLI or Python API)
  ├─ "evaluate"/"metrics" → Use scripts/train.py --eval_only
  ├─ "export"/"deploy"   → Use scripts/train.py --export {torchscript,onnx}
  └─ "troubleshoot"      → Load references/troubleshooting.md
```

## Available Models

| Architecture | Size | Use Case |
|---|---|---|
| resnet18 (default) | 11.7M | Best speed/accuracy balance |
| resnet34 | 21.8M | Slightly better accuracy |
| resnet50 | 25.6M | Higher accuracy needs |
| mobilenet_v2 | 3.5M | Mobile/edge deployment |
| efficientnet_b0 | 5.3M | Best efficiency |

Pass `--arch <name>` to `train.py`. For architecture-specific details, see `references/model_configs.md`.

## Scripts

### `scripts/train.py` — Complete training pipeline
- Two-phase training: freeze backbone → fine-tune all layers
- Mixed precision (AMP), gradient clipping, early stopping
- TensorBoard logging, checkpoint management
- Supports: `--arch`, `--data_dir`, `--epochs`, `--batch_size`, `--lr`, `--weight_decay`, `--freeze_epochs`, `--patience`, `--scheduler`, `--amp`, `--label_smoothing`, `--focal_loss`, `--eval_only`, `--export`
- **Execute directly** — no need to load into context. Run `python scripts/train.py --help` for full options.

### `scripts/inference.py` — Production-ready inference
- `ImageClassifier` class: `predict()`, `predict_top_k()`, `predict_batch()`
- CLI: `python scripts/inference.py --model <path> --images <dir> [--batch] [--top_k 5]`
- Loads TorchScript/state_dict, auto-detects class count

### `scripts/preprocess.py` — Data preparation
- `--check`: Verify environment (CUDA, package versions)
- `--csv/--src/--out/--split`: Organize dataset from CSV labels
- `--analyze`: Print dataset statistics (class distribution, image sizes)
- `--augment`: Test augmentation pipeline visually

## References

Load these files on demand when deeper context is needed:

- **`references/model_configs.md`**: Detailed architecture specs, hyperparameter search space, layer-by-layer config per model type
- **`references/troubleshooting.md`**: OOM errors, overfitting/underfitting, NaN loss, slow training, corrupted data

## Assets

- **`assets/dataset_template/`**: Empty ImageFolder directory structure for quick setup

Load-on-demand references keep the base prompt footprint minimal. Scripts execute without occupying context window.

## Export

```bash
python scripts/train.py --export torchscript --model best_model.pth --output model.pt
python scripts/train.py --export onnx --model best_model.pth --output model.onnx
```
