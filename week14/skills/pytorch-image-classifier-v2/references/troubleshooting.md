# Troubleshooting Guide

## OOM Errors

Issue: `RuntimeError: CUDA out of memory`

Solutions (in priority order):
1. Reduce `--batch_size` (32 → 16 → 8 → 4)
2. Reduce `--img_size` (224 → 192 → 160)
3. Use lower resolution model (resnet18 → mobilenet_v2)
4. Disable TensorBoard logging (modify train.py)

## Overfitting (train_acc >> val_acc)

- Increase `--aug strong`
- Reduce `--freeze_epochs` (less head training)
- Increase dropout in model (edit train.py)
- Reduce epochs, increase `--patience`

## Underfitting (both acc low)

- Train more epochs
- Increase `--lr` to 3e-3
- Use larger model: `--arch resnet50`
- Reduce augmentation: `--aug light`

## NaN Loss

1. Halve `--lr`
2. Check for corrupted images in dataset
3. Run `python preprocess.py --analyze --data_dir ./dataset`
4. If imbalance > 10:1, add `--label_smoothing 0.2`

## Slow Training

- Increase `--num_workers 8` (Linux) or keep `0` (Windows)
- Reduce `--img_size` to 160
- Use `--arch mobilenet_v2`
- Enable CUDA benchmarks: add `torch.backends.cudnn.benchmark = True` in train.py

## Multi-GPU Training

The current script uses single GPU. For multi-GPU, wrap model with:
```python
model = nn.DataParallel(model)  # Simple, single-machine
# or
model = nn.parallel.DistributedDataParallel(model)  # Multi-node
```

## Corrupted Images

Run analysis to detect:
```bash
python preprocess.py --analyze --data_dir ./dataset
```
This shows per-class counts. Manually check classes with unexpected counts.
