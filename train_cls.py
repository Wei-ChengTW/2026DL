"""3-class image classifier (Lung Opacity / Not Normal / Normal) used to gate the detector.

  python train_cls.py [fold]
Writes runs/cls_f{fold}/{best.pt, val_probs.csv, test_probs.csv}; p_opacity = P(Lung Opacity).
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
from torchvision.transforms import v2

CLASSES = ["Lung Opacity", "No Lung Opacity / Not Normal", "Normal"]
ROOT = Path("data/png512")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class XRay(Dataset):
    def __init__(self, paths, labels, train):
        self.paths, self.labels = paths, labels
        aug = [v2.RandomHorizontalFlip(), v2.RandomAffine(0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
               v2.ColorJitter(brightness=0.2, contrast=0.2)] if train else []
        self.tf = v2.Compose(aug + [v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = torch.from_numpy(cv2.imread(str(self.paths[i]), cv2.IMREAD_GRAYSCALE)).expand(3, -1, -1)
        return self.tf(img), self.labels[i]


@torch.no_grad()
def predict(model, loader):
    model.eval()
    out = []
    for x, _ in loader:
        x = x.cuda()
        with torch.autocast("cuda"):
            p = model(x).softmax(1) + model(x.flip(3)).softmax(1)  # hflip TTA
        out.append((p / 2).float().cpu())
    return torch.cat(out).numpy()


def main():
    fold = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    out_dir = Path(f"runs/cls_f{fold}")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "folds.csv")
    df["y"] = df.cls.map(CLASSES.index)
    df["path"] = [ROOT / "images" / f"{p}.png" for p in df.patientId]
    tr, va = df[df.fold != fold], df[df.fold == fold]

    dl = lambda d, train: DataLoader(XRay(list(d.path), list(d.y), train), batch_size=12, shuffle=train,
                                     num_workers=4, persistent_workers=True, drop_last=train)
    train_dl, val_dl = dl(tr, True), dl(va, False)

    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(CLASSES))
    model.cuda()
    epochs = 8
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-4, total_steps=epochs * len(train_dl), pct_start=0.1)
    scaler = torch.amp.GradScaler()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    best = 0
    for ep in range(epochs):
        model.train()
        for i, (x, y) in enumerate(train_dl):
            with torch.autocast("cuda"):
                loss = loss_fn(model(x.cuda()), y.cuda())
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            if i % 200 == 0:
                print(f"ep {ep} it {i}/{len(train_dl)} loss {loss.item():.4f}", flush=True)
        p = predict(model, val_dl)
        auc = roc_auc_score(va.y == 0, p[:, 0])
        print(f"ep {ep} val AUC(opacity) {auc:.4f} acc {(p.argmax(1) == va.y).mean():.4f}", flush=True)
        if auc > best:
            best = auc
            torch.save(model.state_dict(), out_dir / "best.pt")
            pd.DataFrame({"patientId": va.patientId, "p_opacity": p[:, 0]}).to_csv(out_dir / "val_probs.csv", index=False)

    model.load_state_dict(torch.load(out_dir / "best.pt", weights_only=True))
    test = sorted((ROOT / "test").glob("*.png"))
    p = predict(model, dl(pd.DataFrame({"path": test, "y": 0}), False))
    pd.DataFrame({"patientId": [t.stem for t in test], "p_opacity": p[:, 0]}).to_csv(out_dir / "test_probs.csv", index=False)
    print(f"best val AUC {best:.4f}")


if __name__ == "__main__":  # Windows dataloader workers re-import this file
    main()
