#!/usr/bin/env python3
"""ResNet18 image classifier training pipeline. Run: python train.py --help"""

import argparse, copy, json, os, time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_transforms(img_size=224, aug_strength='medium'):
    """Return (train_transform, val_transform)."""
    aug_map = {
        'light': dict(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, degrees=15),
        'medium': dict(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, degrees=30),
        'strong': dict(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.15, degrees=45),
    }
    a = aug_map.get(aug_strength, aug_map['medium'])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(a['brightness'], a['contrast'], a['saturation'], a['hue']),
        transforms.RandomRotation(a['degrees']),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.143)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def create_model(arch, num_classes, pretrained=True):
    """Factory for torchvision models with replaced classifier."""
    model_map = {
        'resnet18': models.resnet18, 'resnet34': models.resnet34,
        'resnet50': models.resnet50, 'mobilenet_v2': models.mobilenet_v2,
        'efficientnet_b0': models.efficientnet_b0, 'densenet121': models.densenet121,
    }
    if arch not in model_map:
        raise ValueError(f"Unknown arch: {arch}. Choices: {list(model_map.keys())}")
    model = model_map[arch](weights='IMAGENET1K_V1' if pretrained else None)
    # Replace classifier for common architectures
    if arch.startswith('resnet') or arch == 'googlenet':
        in_feat = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_feat, 512),
                                 nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, num_classes))
    elif arch.startswith('mobilenet') or arch.startswith('efficientnet'):
        in_feat = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_feat, num_classes)
    elif arch.startswith('densenet'):
        in_feat = model.classifier.in_features
        model.classifier = nn.Linear(in_feat, num_classes)
    return model


class Trainer:
    def __init__(self, model, device, train_loader, val_loader, num_classes,
                 class_names=None, save_dir='./checkpoints', log_dir='./logs'):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dir = os.path.join(save_dir, f'exp_{ts}')
        self.log_dir = os.path.join(log_dir, f'exp_{ts}')
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        self.model, self.device = model, device
        self.train_loader, self.val_loader = train_loader, val_loader
        self.num_classes, self.class_names = num_classes, class_names or [f'c{i}' for i in range(num_classes)]
        self.writer = SummaryWriter(self.log_dir)
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
                        'val_precision': [], 'val_recall': [], 'val_f1': [], 'lr': []}
        self.best_val_acc = 0.0
        self.best_model_wts = None
        self.scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    def _step(self, loader, optimizer, criterion, training):
        if training:
            self.model.train()
        else:
            self.model.eval()
        total_loss, all_preds, all_labels = 0.0, [], []
        desc = '[Train]' if training else '[Val]'
        for inputs, labels in tqdm(loader, desc=desc, leave=False):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            if training:
                optimizer.zero_grad()
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)
                    self.scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
            else:
                with torch.no_grad():
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        return total_loss / len(loader.dataset), accuracy_score(all_labels, all_preds), all_labels, all_preds

    def train(self, epochs=50, lr=1e-3, wd=1e-4, freeze_epochs=10, patience=10,
              scheduler_type='cosine', label_smoothing=0.1):
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        # Phase 1: train head only
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.fc.parameters() if hasattr(self.model, 'fc') else self.model.classifier.parameters():
            p.requires_grad = True
        opt = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=wd)
        sched = CosineAnnealingLR(opt, T_max=freeze_epochs) if scheduler_type == 'cosine' else ReduceLROnPlateau(opt, patience=5, factor=0.5)
        print(f"Phase 1: training head ({freeze_epochs} epochs)")
        self._run_phase(freeze_epochs, opt, sched, criterion, scheduler_type, patience, start_epoch=1)
        # Phase 2: fine-tune all
        for p in self.model.parameters():
            p.requires_grad = True
        opt = optim.AdamW(self.model.parameters(), lr=lr * 0.1, weight_decay=wd)
        remaining = epochs - freeze_epochs
        sched = CosineAnnealingLR(opt, T_max=remaining) if scheduler_type == 'cosine' else ReduceLROnPlateau(opt, patience=5, factor=0.5)
        print(f"Phase 2: fine-tuning ({remaining} epochs)")
        self._run_phase(remaining, opt, sched, criterion, scheduler_type, patience, start_epoch=freeze_epochs + 1)
        # Finalize
        self.model.load_state_dict(self.best_model_wts)
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best_model.pth'))
        json.dump(self.class_names, open(os.path.join(self.save_dir, 'class_names.json'), 'w'))
        with open(os.path.join(self.save_dir, 'history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)
        self.writer.close()
        print(f"Done. Best val acc: {self.best_val_acc:.4f}. Saved to {self.save_dir}")
        return self.history

    def _run_phase(self, n_epochs, opt, sched, criterion, sched_type, patience, start_epoch):
        no_improve = 0
        for ep in range(start_epoch, start_epoch + n_epochs):
            train_loss, train_acc, _, _ = self._step(self.train_loader, opt, criterion, training=True)
            val_loss, val_acc, v_labels, v_preds = self._step(self.val_loader, opt, criterion, training=False)
            prec, rec, f1, _ = precision_recall_fscore_support(v_labels, v_preds, average='weighted', zero_division=0)
            lr_current = opt.param_groups[0]['lr']
            for k, v in [('train_loss', train_loss), ('train_acc', train_acc),
                         ('val_loss', val_loss), ('val_acc', val_acc),
                         ('val_precision', prec), ('val_recall', rec), ('val_f1', f1), ('lr', lr_current)]:
                self.history[k].append(v)
                self.writer.add_scalar(k.replace('_', '/'), v, ep)
            print(f"E{ep:3d} | TL:{train_loss:.4f} TA:{train_acc:.4f} | "
                  f"VL:{val_loss:.4f} VA:{val_acc:.4f} | F1:{f1:.4f} | LR:{lr_current:.6f}")
            if sched_type == 'plateau':
                sched.step(val_acc)
            else:
                sched.step()
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {ep}")
                    break

    def evaluate_only(self):
        """Run full evaluation on validation set."""
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []
        for inputs, labels in tqdm(self.val_loader, desc='Evaluating'):
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
        acc = accuracy_score(all_labels, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)
        print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'cm': cm.tolist()}


def main():
    parser = argparse.ArgumentParser(description='PyTorch Image Classifier Training')
    parser.add_argument('--data_dir', default='./dataset', help='Dataset root (must have train/val subdirs)')
    parser.add_argument('--arch', default='resnet18', help='Model architecture')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--freeze_epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--scheduler', default='cosine', choices=['cosine', 'plateau'])
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--amp', action='store_true', help='Enable AMP (auto if CUDA)')
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--save_dir', default='./checkpoints')
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--model', default='', help='Model path for eval/export')
    parser.add_argument('--export', default='', choices=['', 'torchscript', 'onnx'])
    parser.add_argument('--output', default='model.pt')
    parser.add_argument('--no_pretrained', action='store_true')
    parser.add_argument('--aug', default='medium', choices=['light', 'medium', 'strong'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_tf, val_tf = get_transforms(args.img_size, args.aug)

    train_ds = datasets.ImageFolder(os.path.join(args.data_dir, 'train'), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(args.data_dir, 'val'), transform=val_tf)
    num_classes = len(train_ds.classes)
    print(f"Classes ({num_classes}): {train_ds.classes}")

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))

    if args.eval_only or args.export:
        if not args.model:
            raise ValueError("--model is required for eval/export")
        model = create_model(args.arch, num_classes, pretrained=False)
        state = torch.load(args.model, map_location=device, weights_only=False)
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        model.load_state_dict(state, strict=False)
        model = model.to(device)

        if args.eval_only:
            trainer = Trainer(model, device, train_loader, val_loader, num_classes, train_ds.classes)
            return trainer.evaluate_only()

        if args.export == 'torchscript':
            model.eval().cpu()
            traced = torch.jit.trace(model, torch.randn(1, 3, args.img_size, args.img_size))
            traced.save(args.output)
            print(f"TorchScript exported to {args.output}")
        elif args.export == 'onnx':
            model.eval().cpu()
            torch.onnx.export(model, torch.randn(1, 3, args.img_size, args.img_size),
                              args.output, opset_version=12, input_names=['input'],
                              output_names=['output'], dynamic_axes={'input': {0: 'bs'}, 'output': {0: 'bs'}})
            print(f"ONNX exported to {args.output}")
    else:
        model = create_model(args.arch, num_classes, pretrained=not args.no_pretrained)
        model = model.to(device)
        trainer = Trainer(model, device, train_loader, val_loader, num_classes,
                          train_ds.classes, args.save_dir, args.log_dir)
        trainer.train(args.epochs, args.lr, args.weight_decay, args.freeze_epochs,
                      args.patience, args.scheduler, args.label_smoothing)


if __name__ == '__main__':
    main()
