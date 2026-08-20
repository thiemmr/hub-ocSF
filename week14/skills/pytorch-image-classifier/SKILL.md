---
name: pytorch-image-classifier-v1
description: This skill should be used when the user wants to train, fine-tune, or run inference with a PyTorch image classification model (especially ResNet18) on custom image datasets. Use this skill for tasks involving image classification, model training, data preprocessing for image datasets, model evaluation with metrics like accuracy/precision/recall/F1-score, and inference on new images. This is the original verbose version (V1) with all content inlined.
---

# PyTorch Image Classifier (V1 - Original Verbose)

## Overview

This skill provides comprehensive guidance and code for building, training, and deploying PyTorch-based image classification models, with a focus on ResNet18 architecture. It supports end-to-end workflows from raw image data to trained model inference.

## Supported Image Classification Models

The skill supports the following pre-trained model architectures from torchvision.models:

1. **ResNet18** - Lightweight, 18 layers deep. Good balance of speed and accuracy. ~11.7M parameters.
2. **ResNet34** - 34 layers, slightly better accuracy than ResNet18. ~21.8M parameters.
3. **ResNet50** - 50 layers using bottleneck blocks. Much deeper, higher accuracy. ~25.6M parameters.
4. **MobileNetV2** - Ultra lightweight, designed for mobile devices. ~3.5M parameters.
5. **EfficientNet-B0** - State-of-the-art efficiency. ~5.3M parameters.
6. **VGG16** - Classic architecture, very large model. ~138M parameters.
7. **DenseNet121** - Densely connected convolutions. ~8M parameters.

When choosing a model, consider:
- ResNet18 is recommended as the default for a good speed/accuracy tradeoff
- MobileNetV2 is best for resource-constrained environments
- ResNet50 is recommended when accuracy is the top priority
- VGG16 is not recommended for most use cases due to its large size

## Prerequisites and Environment Setup

### Required Python Packages

Install the following packages before starting:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
# OR for CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# Additional dependencies:
pip install pillow numpy matplotlib scikit-learn tqdm tensorboard
```

If you are on a Mac with Apple Silicon (M1/M2/M3), use:

```bash
pip install torch torchvision
```

### GPU vs CPU Considerations

For training, a GPU is highly recommended. Training ResNet18 on ImageNet-1K (1.28M images) takes approximately:
- NVIDIA A100: ~4 hours
- NVIDIA RTX 3090: ~8 hours
- NVIDIA RTX 3060: ~15 hours
- CPU (32 cores): ~3-7 days (not recommended)

For small custom datasets (1000-5000 images), training on any GPU takes 5-30 minutes.

### Verifying Your Installation

Run this Python code to verify everything is set up correctly:

```python
import torch
import torchvision
import numpy as np
from PIL import Image
import sys

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
print(f"NumPy version: {np.__version__}")
print("Environment setup complete!")
```

## Dataset Organization

### Expected Directory Structure

Images should be organized in the following structure for use with `torchvision.datasets.ImageFolder`:

```
dataset/
├── train/
│   ├── class_a/
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   └── ...
│   ├── class_b/
│   │   ├── image_001.jpg
│   │   └── ...
│   └── class_c/
│       └── ...
├── val/
│   ├── class_a/
│   │   └── ...
│   ├── class_b/
│   │   └── ...
│   └── class_c/
│       └── ...
└── test/  (optional)
    ├── class_a/
    │   └── ...
    ├── class_b/
    │   └── ...
    └── class_c/
        └── ...
```

Each class gets its own subdirectory. The directory name becomes the class label. Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`.

### Generating a Dataset from Raw Images

If you have raw images and a CSV file mapping image paths to labels, use this script to reorganize them:

```python
import os
import shutil
import pandas as pd
from pathlib import Path

def organize_dataset_from_csv(csv_path, source_dir, output_dir, split_ratio=0.8):
    """
    Organize raw images into ImageFolder structure based on a CSV file.
    
    Args:
        csv_path: Path to CSV with columns 'filename' and 'label'
        source_dir: Directory containing raw images
        output_dir: Output directory for organized dataset
        split_ratio: Ratio of train/val split (default 0.8)
    """
    df = pd.read_csv(csv_path)
    
    # Create directory structure
    for split in ['train', 'val']:
        for label in df['label'].unique():
            os.makedirs(os.path.join(output_dir, split, label), exist_ok=True)
    
    # Split data
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df, train_size=split_ratio, 
                                         stratify=df['label'], random_state=42)
    
    # Copy files
    for split_name, split_df in [('train', train_df), ('val', val_df)]:
        for _, row in split_df.iterrows():
            src = os.path.join(source_dir, row['filename'])
            dst = os.path.join(output_dir, split_name, row['label'], row['filename'])
            shutil.copy2(src, dst)
    
    print(f"Dataset organized: {len(train_df)} train, {len(val_df)} val images")
    print(f"Classes: {list(df['label'].unique())}")
    return output_dir

# Example usage:
# organize_dataset_from_csv('labels.csv', './raw_images', './dataset')
```

### Data Augmentation

Data augmentation is crucial for preventing overfitting and improving model generalization. Here is the recommended augmentation pipeline:

```python
from torchvision import transforms

# Training augmentations (with strong augmentation)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(degrees=(-30, 30)),
    transforms.ColorJitter(
        brightness=0.2,   # Random brightness adjustment ±20%
        contrast=0.2,     # Random contrast adjustment ±20%
        saturation=0.2,   # Random saturation adjustment ±20%
        hue=0.1           # Random hue adjustment ±10%
    ),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),  # Random translation up to 10%
        scale=None
    ),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    ),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
])

# Validation augmentations (no augmentation, just resize + normalize)
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
```

### Dataset Loading

Load datasets using `torchvision.datasets.ImageFolder`:

```python
from torchvision import datasets
from torch.utils.data import DataLoader

# Load datasets
train_dataset = datasets.ImageFolder('dataset/train', transform=train_transform)
val_dataset = datasets.ImageFolder('dataset/val', transform=val_transform)

# Access class information
num_classes = len(train_dataset.classes)
class_names = train_dataset.classes
class_to_idx = train_dataset.class_to_idx

print(f"Number of classes: {num_classes}")
print(f"Class names: {class_names}")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# Create DataLoaders
BATCH_SIZE = 32  # Adjust based on GPU memory (16 for 4GB, 32 for 6GB, 64 for 8GB+, 128 for 12GB+)
NUM_WORKERS = 4   # Set to 0 on Windows, 4-8 on Linux

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True if torch.cuda.is_available() else False,
    drop_last=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True if torch.cuda.is_available() else False,
    drop_last=False
)
```

## Model Architecture

### Loading a Pre-trained ResNet18 Model

Here is how to create and configure a ResNet18 model for transfer learning:

```python
import torch
import torch.nn as nn
from torchvision import models

def create_resnet18_model(num_classes, pretrained=True, freeze_backbone=True):
    """
    Create a ResNet18 model for transfer learning.
    
    Args:
        num_classes (int): Number of output classes for the classifier
        pretrained (bool): Whether to use ImageNet pre-trained weights
        freeze_backbone (bool): Whether to freeze backbone layers initially
    
    Returns:
        model (nn.Module): Configured ResNet18 model
    """
    if pretrained:
        # Load pre-trained weights from ImageNet
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        # Initialize with random weights
        model = models.resnet18(weights=None)
    
    if freeze_backbone:
        # Freeze all layers initially for phase 1 training
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace the final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes)
    )
    
    # Weight initialization for the new layers
    for module in model.fc.modules():
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model on device: {device}")
    
    return model, device

# Usage example:
# model, device = create_resnet18_model(num_classes=10)
```

### Creating Other Model Architectures

For other model architectures, use these factory functions:

```python
def create_model(architecture, num_classes, pretrained=True):
    """Factory function for creating different model architectures."""
    
    architectures = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'mobilenet_v2': models.mobilenet_v2,
        'efficientnet_b0': models.efficientnet_b0,
        'densenet121': models.densenet121,
        'vgg16': models.vgg16,
    }
    
    if architecture not in architectures:
        raise ValueError(f"Unknown architecture: {architecture}. "
                         f"Available: {list(architectures.keys())}")
    
    weights = 'IMAGENET1K_V1' if pretrained else None
    model = architectures[architecture](weights=weights)
    
    # Handle different classifier structures
    if architecture.startswith('resnet') or architecture == 'googlenet':
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif architecture.startswith('mobilenet'):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif architecture.startswith('efficientnet'):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif architecture.startswith('densenet'):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif architecture.startswith('vgg'):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    
    return model
```

## Training Pipeline

### Complete Training Loop

Below is a comprehensive training function with all features:

```python
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
from datetime import datetime

class ImageClassifierTrainer:
    """
    A complete trainer class for image classification models.
    
    Features:
    - Two-phase training (freeze backbone -> unfreeze)
    - Early stopping with best model saving
    - TensorBoard logging
    - Comprehensive metrics (accuracy, precision, recall, F1)
    - Learning rate scheduling
    - Mixed precision training (AMP)
    - Gradient clipping
    - Confusion matrix generation
    """
    
    def __init__(
        self,
        model,
        device,
        train_loader,
        val_loader,
        num_classes,
        class_names=None,
        save_dir='./checkpoints',
        log_dir='./logs',
        experiment_name=None
    ):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]
        
        # Setup directories
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = experiment_name or f'exp_{timestamp}'
        self.save_dir = os.path.join(save_dir, exp_name)
        self.log_dir = os.path.join(log_dir, exp_name)
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Setup logging
        self.writer = SummaryWriter(self.log_dir)
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'val_precision': [], 'val_recall': [], 'val_f1': [],
            'lr': []
        }
        
        # Training state
        self.best_val_acc = 0.0
        self.best_model_wts = None
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
    def train_epoch(self, optimizer, criterion, epoch):
        """Run one training epoch."""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]', leave=False)
        for inputs, labels in pbar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
    
    @torch.no_grad()
    def validate_epoch(self, criterion, epoch):
        """Run one validation epoch and compute metrics."""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]', leave=False)
        for inputs, labels in pbar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        return epoch_loss, epoch_acc, precision, recall, f1, all_labels, all_preds
    
    def log_metrics(self, epoch, train_loss, train_acc, val_loss, val_acc, 
                    precision, recall, f1, lr):
        """Log metrics to TensorBoard and history."""
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['val_precision'].append(precision)
        self.history['val_recall'].append(recall)
        self.history['val_f1'].append(f1)
        self.history['lr'].append(lr)
        
        # TensorBoard logging
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Loss/val', val_loss, epoch)
        self.writer.add_scalar('Accuracy/train', train_acc, epoch)
        self.writer.add_scalar('Accuracy/val', val_acc, epoch)
        self.writer.add_scalar('Metrics/precision', precision, epoch)
        self.writer.add_scalar('Metrics/recall', recall, epoch)
        self.writer.add_scalar('Metrics/f1', f1, epoch)
        self.writer.add_scalar('LR', lr, epoch)
        
        # Print metrics
        print(f"Epoch {epoch:3d} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Prec: {precision:.4f} | Rec: {recall:.4f} | F1: {f1:.4f} | "
              f"LR: {lr:.6f}")
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history,
        }
        
        # Always save the latest checkpoint
        torch.save(checkpoint, os.path.join(self.save_dir, 'latest_checkpoint.pth'))
        
        # Save best model
        if is_best:
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best_model.pth'))
            # Also save as TorchScript for production deployment
            try:
                example_input = torch.randn(1, 3, 224, 224).to(self.device)
                traced_model = torch.jit.trace(self.model.cpu(), example_input.cpu())
                traced_model.save(os.path.join(self.save_dir, 'best_model_scripted.pt'))
                self.model.to(self.device)
            except Exception as e:
                print(f"Warning: Could not export TorchScript model: {e}")
    
    def plot_confusion_matrix(self, labels, preds, epoch):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(labels, preds)
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # Use class names if available
        tick_marks = np.arange(min(len(self.class_names), self.num_classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(self.class_names[:self.num_classes], rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(self.class_names[:self.num_classes], fontsize=8)
        
        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=6)
        
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title(f'Confusion Matrix - Epoch {epoch}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f'confusion_matrix_epoch_{epoch}.png'), dpi=150)
        plt.close()
        
        # Log to TensorBoard
        self.writer.add_figure('Confusion Matrix', fig, epoch)
    
    def plot_training_curves(self):
        """Plot and save training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss curves
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(self.history['train_acc'], label='Train Acc')
        axes[0, 1].plot(self.history['val_acc'], label='Val Acc')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1/Precision/Recall
        axes[1, 0].plot(self.history['val_precision'], label='Precision')
        axes[1, 0].plot(self.history['val_recall'], label='Recall')
        axes[1, 0].plot(self.history['val_f1'], label='F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Validation Metrics')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate
        axes[1, 1].plot(self.history['lr'])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_curves.png'), dpi=150)
        plt.close()
    
    def train(self, num_epochs=50, learning_rate=1e-3, weight_decay=1e-4,
              scheduler_type='cosine', early_stopping_patience=10, 
              phase1_epochs=10, phase2_epochs=40,
              criterion_type='cross_entropy'):
        """
        Execute full training pipeline with two-phase training.
        
        Phase 1: Train only the classifier head with frozen backbone
        Phase 2: Unfreeze all layers and fine-tune with lower LR
        
        Args:
            num_epochs: Total number of epochs to train
            learning_rate: Initial learning rate
            weight_decay: L2 regularization strength
            scheduler_type: 'cosine', 'plateau', or 'step'
            early_stopping_patience: Epochs to wait before early stopping
            phase1_epochs: Epochs for phase 1 (frozen backbone)
            phase2_epochs: Epochs for phase 2 (unfrozen backbone)
            criterion_type: Loss function type
        """
        print(f"\n{'='*60}")
        print(f"Starting Training Pipeline")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Total epochs: {num_epochs}")
        print(f"Phase 1 (frozen backbone): {phase1_epochs} epochs")
        print(f"Phase 2 (fine-tuning): {phase2_epochs} epochs")
        print(f"Initial LR: {learning_rate}")
        print(f"Scheduler: {scheduler_type}")
        print(f"Early stopping patience: {early_stopping_patience}")
        print(f"{'='*60}\n")
        
        # Loss function
        if criterion_type == 'cross_entropy':
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        elif criterion_type == 'focal':
            # Focal loss for imbalanced datasets
            criterion = self._focal_loss(alpha=0.25, gamma=2.0)
        else:
            raise ValueError(f"Unknown criterion: {criterion_type}")
        
        patience_counter = 0
        
        # ============ Phase 1: Train classifier head ============
        print("### Phase 1: Training classifier head (backbone frozen) ###")
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        if scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=phase1_epochs, eta_min=1e-6)
        elif scheduler_type == 'plateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        else:
            scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        
        for epoch in range(1, phase1_epochs + 1):
            train_loss, train_acc = self.train_epoch(optimizer, criterion, epoch)
            val_loss, val_acc, prec, rec, f1, val_labels, val_preds = self.validate_epoch(criterion, epoch)
            
            current_lr = optimizer.param_groups[0]['lr']
            self.log_metrics(epoch, train_loss, train_acc, val_loss, val_acc, prec, rec, f1, current_lr)
            
            if scheduler_type == 'plateau':
                scheduler.step(val_acc)
            else:
                scheduler.step()
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                self.save_checkpoint(epoch, is_best=True)
                patience_counter = 0
                print(f"  -> New best model! Val Acc: {val_acc:.4f}")
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break
        
        # ============ Phase 2: Fine-tune all layers ============
        if phase2_epochs > 0:
            print("\n### Phase 2: Fine-tuning (all layers unfrozen) ###")
            
            # Unfreeze all layers
            for param in self.model.parameters():
                param.requires_grad = True
            
            # Lower learning rate for fine-tuning
            fine_tune_lr = learning_rate * 0.1
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=fine_tune_lr,
                weight_decay=weight_decay
            )
            
            if scheduler_type == 'cosine':
                scheduler = CosineAnnealingLR(optimizer, T_max=phase2_epochs, eta_min=1e-7)
            else:
                scheduler = StepLR(optimizer, step_size=15, gamma=0.3)
            
            patience_counter = 0
            
            for epoch in range(phase1_epochs + 1, phase1_epochs + phase2_epochs + 1):
                train_loss, train_acc = self.train_epoch(optimizer, criterion, epoch)
                val_loss, val_acc, prec, rec, f1, val_labels, val_preds = self.validate_epoch(criterion, epoch)
                
                current_lr = optimizer.param_groups[0]['lr']
                self.log_metrics(epoch, train_loss, train_acc, val_loss, val_acc, prec, rec, f1, current_lr)
                
                if scheduler_type == 'plateau':
                    scheduler.step(val_acc)
                else:
                    scheduler.step()
                
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_model_wts = copy.deepcopy(self.model.state_dict())
                    self.save_checkpoint(epoch, is_best=True)
                    patience_counter = 0
                    print(f"  -> New best model! Val Acc: {val_acc:.4f}")
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    break
                
                # Plot confusion matrix every 10 epochs
                if epoch % 10 == 0:
                    self.plot_confusion_matrix(val_labels, val_preds, epoch)
        
        # ============ Finalize ============
        self.model.load_state_dict(self.best_model_wts)
        
        # Save final training curves
        self.plot_training_curves()
        
        # Save training history as JSON
        with open(os.path.join(self.save_dir, 'training_history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Close TensorBoard writer
        self.writer.close()
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Val Accuracy: {self.best_val_acc:.4f}")
        print(f"Model saved to: {self.save_dir}")
        print(f"Logs saved to: {self.log_dir}")
        print(f"Run: tensorboard --logdir {self.log_dir}")
        print(f"{'='*60}\n")
        
        return self.model, self.history
    
    @staticmethod
    def _focal_loss(alpha=0.25, gamma=2.0):
        """Focal Loss for handling class imbalance."""
        class FocalLoss(nn.Module):
            def __init__(self, alpha, gamma):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.ce = nn.CrossEntropyLoss(reduction='none')
            
            def forward(self, inputs, targets):
                ce_loss = self.ce(inputs, targets)
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                return focal_loss.mean()
        
        return FocalLoss(alpha=alpha, gamma=gamma)

# Complete usage example
def run_training_example():
    """
    Complete example of training an image classifier.
    Call this function to start a full training run.
    """
    # Configuration
    config = {
        'data_dir': './dataset',
        'model_arch': 'resnet18',
        'num_classes': 10,
        'batch_size': 32,
        'num_epochs': 50,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'scheduler_type': 'cosine',
        'early_stopping_patience': 10,
        'phase1_epochs': 10,
        'phase2_epochs': 40,
        'save_dir': './checkpoints',
        'log_dir': './logs',
        'num_workers': 4,
    }
    
    # Get transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load datasets
    train_dataset = datasets.ImageFolder(f'{config["data_dir"]}/train', transform=train_transform)
    val_dataset = datasets.ImageFolder(f'{config["data_dir"]}/val', transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                              shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                            shuffle=False, num_workers=config['num_workers'])
    
    # Create model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, config['num_classes']),
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create trainer and start training
    trainer = ImageClassifierTrainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=config['num_classes'],
        class_names=train_dataset.classes,
        save_dir=config['save_dir'],
        log_dir=config['log_dir'],
    )
    
    model, history = trainer.train(
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        scheduler_type=config['scheduler_type'],
        early_stopping_patience=config['early_stopping_patience'],
        phase1_epochs=config['phase1_epochs'],
        phase2_epochs=config['phase2_epochs'],
    )
    
    return model, trainer, history

# To run:
# model, trainer, history = run_training_example()
```

## Inference Pipeline

### Single Image Inference

```python
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import json

class ImageClassifier:
    """
    A production-ready inference class for image classification.
    
    Usage:
        classifier = ImageClassifier('best_model.pth', 'class_names.json')
        pred_class, confidence = classifier.predict('test_image.jpg')
        top5 = classifier.predict_top_k('test_image.jpg', k=5)
    """
    
    def __init__(self, model_path, class_names_path=None, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load class names
        if class_names_path:
            with open(class_names_path, 'r') as f:
                self.class_names = json.load(f)
        else:
            self.class_names = None
        
        # Define inference transform (deterministic, no augmentation)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def _load_model(self, model_path):
        """Load model from checkpoint or state dict."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Determine number of classes from state dict
        num_classes = state_dict['fc.weight'].shape[0] if 'fc.weight' in state_dict else 1000
        
        from torchvision import models
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
        model.load_state_dict(state_dict)
        
        return model
    
    @torch.no_grad()
    def predict(self, image_path_or_pil):
        """
        Predict the class of a single image.
        
        Args:
            image_path_or_pil: Path to image file or PIL Image object
        
        Returns:
            Tuple of (predicted_class: str or int, confidence: float)
        """
        # Load image if path is provided
        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil).convert('RGB')
        else:
            image = image_path_or_pil
        
        # Preprocess
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        output = self.model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, pred_idx = torch.max(probabilities, 1)
        
        pred_idx = pred_idx.item()
        confidence = confidence.item()
        
        if self.class_names and pred_idx < len(self.class_names):
            return self.class_names[pred_idx], confidence
        return pred_idx, confidence
    
    @torch.no_grad()
    def predict_top_k(self, image_path_or_pil, k=5):
        """Return top-k predictions with confidence scores."""
        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil).convert('RGB')
        else:
            image = image_path_or_pil
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        output = self.model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        
        topk_probs, topk_indices = torch.topk(probabilities, k)
        
        results = []
        for i in range(k):
            idx = topk_indices[0, i].item()
            prob = topk_probs[0, i].item()
            label = self.class_names[idx] if self.class_names and idx < len(self.class_names) else idx
            results.append((label, prob))
        
        return results
    
    @torch.no_grad()
    def predict_batch(self, image_paths, batch_size=32):
        """Predict multiple images in batches for efficiency."""
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []
            
            for path in batch_paths:
                image = Image.open(path).convert('RGB')
                tensor = self.transform(image)
                batch_tensors.append(tensor)
            
            batch = torch.stack(batch_tensors).to(self.device)
            outputs = self.model(batch)
            probs = F.softmax(outputs, dim=1)
            confidences, pred_indices = torch.max(probs, dim=1)
            
            for j, idx in enumerate(pred_indices.cpu().numpy()):
                label = self.class_names[idx] if self.class_names and idx < len(self.class_names) else int(idx)
                results.append((label, confidences[j].item()))
        
        return results


# Usage example:
# classifier = ImageClassifier('best_model.pth', 'class_names.json')
# pred, conf = classifier.predict('cat.jpg')
# print(f"Prediction: {pred} (confidence: {conf:.2%})")
```

## Model Evaluation

### Comprehensive Evaluation on Test Set

```python
@torch.no_grad()
def evaluate_model(model, test_loader, device, class_names=None):
    """
    Comprehensive evaluation of a trained model on a test set.
    
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    for inputs, labels in tqdm(test_loader, desc='Evaluating'):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        classification_report, confusion_matrix, roc_auc_score
    )
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    per_class_precision, per_class_recall, per_class_f1, _ = \
        precision_recall_fscore_support(all_labels, all_preds, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # ROC-AUC (one-vs-rest for multi-class)
    try:
        if all_probs.shape[1] == 2:
            roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
    except Exception:
        roc_auc = None
    
    # Classification report
    if class_names:
        report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    else:
        report = classification_report(all_labels, all_preds, zero_division=0)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist(),
        'per_class': {
            'precision': per_class_precision.tolist(),
            'recall': per_class_recall.tolist(),
            'f1': per_class_f1.tolist(),
        },
        'classification_report': report,
        'num_samples': len(all_labels),
    }
    
    # Print results
    print(f"\n{'='*50}")
    print("EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    if roc_auc:
        print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"\nClassification Report:\n{report}")
    print(f"{'='*50}\n")
    
    return results

# Usage:
# results = evaluate_model(model, test_loader, device, class_names)
```

## Hyperparameter Tuning

### Recommended Hyperparameter Search Space

| Hyperparameter | Range | Recommended Default |
|---------------|-------|-------------------|
| Learning Rate | 1e-5 to 1e-2 | 1e-3 (phase 1), 1e-4 (phase 2) |
| Batch Size | 8 to 256 | 32 (adjust based on GPU memory) |
| Weight Decay | 1e-5 to 1e-2 | 1e-4 |
| Dropout Rate | 0.1 to 0.7 | 0.5 (first), 0.3 (second) |
| Optimizer | AdamW, SGD, Adam | AdamW |
| Scheduler | Cosine, Plateau, Step | CosineAnnealingLR |
| Image Size | 128 to 512 | 224 |
| Phase 1 Epochs | 5 to 30 | 10 |
| Phase 2 Epochs | 10 to 100 | 40 |

### Quick Grid Search Script

```python
import itertools

def grid_search_hyperparams(train_loader, val_loader, num_classes, device):
    """Simple grid search over key hyperparameters."""
    param_grid = {
        'learning_rate': [1e-3, 1e-4, 5e-4],
        'batch_size': [16, 32],
        'weight_decay': [1e-4, 1e-5],
    }
    
    best_acc = 0
    best_params = None
    
    keys = list(param_grid.keys())
    for values in itertools.product(*param_grid.values()):
        params = dict(zip(keys, values))
        print(f"\nTesting: {params}")
        
        # Quick training with 5 epochs for evaluation
        model = create_model_for_search(num_classes, device)
        # ... training logic (abbreviated for this example) ...
        # val_acc = quick_train(model, train_loader, val_loader, params, epochs=5)
        
        # if val_acc > best_acc:
        #     best_acc = val_acc
        #     best_params = params
    
    print(f"\nBest params: {best_params}")
    print(f"Best val acc: {best_acc:.4f}")
    return best_params
```

## Troubleshooting Common Issues

### Issue 1: Out of Memory (OOM) Errors

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions** (try in order):
1. Reduce batch size: `BATCH_SIZE = 16` → `BATCH_SIZE = 8` → `BATCH_SIZE = 4`
2. Reduce image size: `224` → `192` → `160`
3. Use gradient accumulation:
```python
accumulation_steps = 4  # Effective batch size = BATCH_SIZE * accumulation_steps
for i, (inputs, labels) in enumerate(train_loader):
    outputs = model(inputs)
    loss = criterion(outputs, labels) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```
4. Use mixed precision training (AMP - already included in our trainer)
5. Clear GPU cache: `torch.cuda.empty_cache()`

### Issue 2: Overfitting

**Symptoms**: Training accuracy much higher than validation accuracy

**Solutions**:
1. Increase data augmentation (add more transforms)
2. Increase dropout rate (0.5 → 0.7)
3. Increase weight decay (1e-4 → 1e-3)
4. Reduce model capacity (use ResNet18 instead of ResNet50)
5. Add early stopping (already implemented)
6. Use label smoothing (already included)
7. Collect more training data

### Issue 3: Underfitting

**Symptoms**: Both training and validation accuracy are low

**Solutions**:
1. Train for more epochs
2. Increase learning rate
3. Reduce regularization (dropout, weight decay)
4. Use a larger model (ResNet50 instead of ResNet18)
5. Reduce data augmentation

### Issue 4: Slow Training

**Solutions**:
1. Increase num_workers: 0 → 4 → 8
2. Enable pin_memory=True in DataLoader
3. Use mixed precision (AMP)
4. Reduce image size
5. Use a lighter model (MobileNetV2)

### Issue 5: NaN Loss

**Symptoms**: Loss becomes NaN during training

**Solutions**:
1. Reduce learning rate (try 1/10 of current)
2. Add gradient clipping (already included)
3. Check for corrupted images in dataset
4. Normalize input data properly
5. Check for extreme class imbalance (use Focal Loss)

## Exporting the Model

### Export to TorchScript

```python
def export_torchscript(model, save_path, device):
    """Export model to TorchScript for production deployment."""
    model.eval()
    model.cpu()
    
    # Create example input
    example = torch.randn(1, 3, 224, 224)
    
    # Trace the model
    traced_model = torch.jit.trace(model, example)
    traced_model.save(save_path)
    
    # Verify the exported model
    loaded = torch.jit.load(save_path)
    original_output = model(example)
    traced_output = loaded(example)
    
    assert torch.allclose(original_output, traced_output, atol=1e-5), \
        "Exported model outputs differ from original!"
    
    print(f"Model exported successfully to: {save_path}")
    model.to(device)
    return save_path
```

### Export to ONNX

```python
def export_onnx(model, save_path, device):
    """Export model to ONNX format."""
    model.eval()
    model.cpu()
    
    example = torch.randn(1, 3, 224, 224)
    
    torch.onnx.export(
        model,
        example,
        save_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to ONNX: {save_path}")
    model.to(device)
```

## Full End-to-End Workflow

To train an image classifier from scratch, follow these steps in order:

### Step 1: Organize Data
Create the directory structure as described in the Dataset Organization section. Ensure images are split into train/val/test directories with class subdirectories.

### Step 2: Verify Environment
Run the environment verification code from the Prerequisites section.

### Step 3: Configure Parameters
Adjust the configuration dictionary in `run_training_example()` to match your dataset and requirements.

### Step 4: Train the Model
Call `run_training_example()` to start training. Monitor progress via:
- Console output (loss, accuracy per epoch)
- TensorBoard: `tensorboard --logdir ./logs`

### Step 5: Evaluate
Use the `evaluate_model()` function to get comprehensive metrics on the test set.

### Step 6: Export and Deploy
Export the best model to TorchScript or ONNX format for production use.

### Step 7: Run Inference
Use the `ImageClassifier` class to make predictions on new images.

## References

- PyTorch Documentation: https://pytorch.org/docs/stable/index.html
- Torchvision Models: https://pytorch.org/vision/stable/models.html
- ResNet Paper: https://arxiv.org/abs/1512.03385
- ImageNet Dataset: https://www.image-net.org/

## Resources

This skill contains all code and instructions inline for maximum accessibility. All utility functions are copy-paste ready and fully self-contained.
