# Model Architecture Reference

## Supported Architectures

| Architecture | Params | Default Input | Best For |
|---|---|---|---|
| resnet18 | 11.7M | 224×224 | Balanced speed/accuracy (default) |
| resnet34 | 21.8M | 224×224 | Slightly higher accuracy |
| resnet50 | 25.6M | 224×224 | High accuracy, 4× computation |
| mobilenet_v2 | 3.5M | 224×224 | Mobile/edge, 3× faster inference |
| efficientnet_b0 | 5.3M | 224×224 | Best accuracy-per-parameter |

## Classifier Head Design

All models get a two-stage classifier head:
```
Dropout(0.5) → Linear(in_features, 512) → ReLU → Dropout(0.3) → Linear(512, num_classes)
```

For ResNet variants, replaces `model.fc`. For MobileNet/EfficientNet, replaces `model.classifier[-1]`.

## Training Phases

### Phase 1: Head Training (--freeze_epochs)
- Backbone frozen (all pretrained weights locked)
- Only classifier head trained
- Higher LR (default: 1e-3)

### Phase 2: Fine-tuning (remaining epochs)
- All layers unfrozen
- Lower LR (default: 0.1 × phase 1 LR = 1e-4)
- Fine-tunes backbone for domain adaptation

## Recommended Hyperparameters by Dataset Size

| Dataset Size | Epochs | Batch Size | LR | Freeze Epochs | Augmentation |
|---|---|---|---|---|---|
| < 1K images | 30 | 16 | 5e-4 | 5 | strong |
| 1K-10K | 50 | 32 | 1e-3 | 10 | medium |
| 10K-100K | 50-100 | 64-128 | 1e-3 | 10 | light |
| > 100K | 100+ | 128+ | 1e-2 (SGD) | 5 | light |

## Image Size Trade-offs

| Size | Speed | Memory | Accuracy |
|---|---|---|---|
| 128 | 4× faster | 1/4 VRAM | -3-5% top-1 |
| 224 (default) | 1× | 1× | baseline |
| 384 | 2× slower | 3× VRAM | +1-2% top-1 |
| 512 | 4× slower | 6× VRAM | +2-3% top-1 |

## Learning Rate Schedulers

| Scheduler | Flag | Best For |
|---|---|---|
| CosineAnnealingLR | --scheduler cosine | Most cases (smooth decay) |
| ReduceLROnPlateau | --scheduler plateau | Unstable training |
| StepLR | Not included | Very long training runs |
