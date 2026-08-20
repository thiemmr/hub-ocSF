#!/usr/bin/env python3
"""Image classification inference. Run: python inference.py --help"""
import argparse, json, torch, torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image


class ImageClassifier:
    def __init__(self, model_path, class_names_path=None, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        # Load model
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        if hasattr(state, 'keys') and 'fc.weight' in state:
            nc = state['fc.weight'].shape[0]
        else:
            nc = 1000
        self.model = models.__dict__.get('resnet18', models.resnet18)(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, nc)
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device).eval()
        self.class_names = json.load(open(class_names_path)) if class_names_path else None

    @torch.no_grad()
    def predict(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        probs = F.softmax(self.model(tensor), dim=1)
        conf, idx = torch.max(probs, 1)
        idx, conf = idx.item(), conf.item()
        label = self.class_names[idx] if self.class_names and idx < len(self.class_names) else idx
        return label, conf

    @torch.no_grad()
    def predict_top_k(self, image, k=5):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        probs = F.softmax(self.model(tensor), dim=1)
        vals, ids = torch.topk(probs, k)
        return [(self.class_names[i.item()] if self.class_names and i < len(self.class_names) else i.item(),
                 v.item()) for i, v in zip(ids[0], vals[0])]

    @torch.no_grad()
    def predict_batch(self, paths, batch_size=32):
        results = []
        for i in range(0, len(paths), batch_size):
            batch = torch.stack([self.transform(Image.open(p).convert('RGB')) for p in paths[i:i + batch_size]])
            batch = batch.to(self.device)
            probs = F.softmax(self.model(batch), dim=1)
            confs, ids = torch.max(probs, dim=1)
            for j, idx in enumerate(ids.cpu().numpy()):
                label = self.class_names[idx] if self.class_names and idx < len(self.class_names) else int(idx)
                results.append((label, confs[j].item()))
        return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--class_names', default='')
    p.add_argument('--images', required=True, help='Image file or directory')
    p.add_argument('--batch', action='store_true')
    p.add_argument('--top_k', type=int, default=1)
    args = p.parse_args()

    clf = ImageClassifier(args.model, args.class_names or None)
    import os, glob
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    if os.path.isdir(args.images):
        paths = [f for ext in exts for f in glob.glob(os.path.join(args.images, ext))]
    else:
        paths = [args.images]

    if args.batch and len(paths) > 1:
        results = clf.predict_batch(paths)
    else:
        results = [clf.predict_top_k(p, args.top_k) if args.top_k > 1 else clf.predict(p) for p in paths]

    for path, res in zip(paths, results):
        if args.top_k > 1:
            top_str = ', '.join(f'{l}:{c:.2%}' for l, c in res)
            print(f"{os.path.basename(path)} -> {top_str}")
        else:
            print(f"{os.path.basename(path)} -> {res[0]}: {res[1]:.2%}")


if __name__ == '__main__':
    main()
