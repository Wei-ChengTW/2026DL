# [AI 協助] 本檔由 Claude（Anthropic，透過 Claude Code）協助撰寫，Leo 負責執行與驗證
# 「有沒有肺炎」分類器：ImageNet 預訓練 ResNet50，整張圖只輸出一個數字（有肺炎的機率）
# 用法: python classify.py --epochs 5 --out runs/cls
# 輸出: runs/cls/best.pth、val_probs.pkl、test_probs.pkl（{patientId: 機率}，給 combine.py 用）
import argparse
import os
import pickle
import random
import time

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms.functional as TF
from sklearn.metrics import roc_auc_score

from train_baseline import DATA, RSNADataset, load_boxes

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet 的平均與標準差，預訓練模型吃這個尺度


class ClsDataset(RSNADataset):
    """沿用偵測的 Dataset（讀圖、左右翻轉），只是答案從「框」換成「有沒有框」"""
    def __getitem__(self, i):
        img, target, pid = super().__getitem__(i)
        label = torch.tensor(float(len(target["boxes"]) > 0))
        return TF.normalize(img, MEAN, STD), label, pid


def build_model():
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)  # 原本輸出 1000 類，換成 1 個數字
    return model


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    probs = {}
    for images, _, pids in loader:
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images.to(device)).squeeze(1)
        for pid, p in zip(pids, torch.sigmoid(logits.float()).cpu().numpy()):
            probs[pid] = float(p)
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 張，測試用")
    ap.add_argument("--out", default="runs/cls")
    args = ap.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    boxes_by_id = load_boxes()
    split = pd.read_csv(f"{DATA}/split.csv")
    train_ids = split[split.split == "train"].patientId.tolist()
    val_ids = split[split.split == "val"].patientId.tolist()
    test_ids = [p[:-4] for p in os.listdir(f"{DATA}/test_512")]
    if args.limit:
        train_ids, val_ids, test_ids = train_ids[:args.limit], val_ids[:args.limit // 4], test_ids[:args.limit // 4]
    val_labels = np.array([pid in boxes_by_id for pid in val_ids])

    def loader(ids, img_dir, train):
        return torch.utils.data.DataLoader(ClsDataset(ids, boxes_by_id, img_dir, train=train),
                                           batch_size=args.batch, shuffle=train, num_workers=4)
    train_loader = loader(train_ids, f"{DATA}/train_512", True)
    val_loader = loader(val_ids, f"{DATA}/train_512", False)

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(train_loader))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    loss_fn = torch.nn.BCEWithLogitsLoss()  # 二元分類的標準 loss：猜錯又很有自信會被罰很重

    best = -1.0
    log = open(f"{args.out}/log.txt", "a", encoding="utf-8")
    for epoch in range(args.epochs):
        model.train()
        start, running = time.time(), 0.0
        for step, (images, labels, _) in enumerate(train_loader):
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                loss = loss_fn(model(images.to(device)).squeeze(1), labels.to(device))
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
            if step % 200 == 0:
                print(f"epoch {epoch+1} step {step}/{len(train_loader)} loss {loss.item():.3f}", flush=True)

        probs = predict_probs(model, val_loader, device)
        p = np.array([probs[pid] for pid in val_ids])
        # AUC：隨便抽一張有肺炎、一張沒有，模型給有肺炎那張比較高機率的比例（0.5 = 亂猜，1 = 完美）
        auc = roc_auc_score(val_labels, p) if 0 < val_labels.sum() < len(val_labels) else float("nan")
        acc = ((p >= 0.5) == val_labels).mean()
        msg = (f"epoch {epoch+1}/{args.epochs} train_loss {running/len(train_loader):.4f} "
               f"val_auc {auc:.4f} val_acc {acc:.4f} {(time.time()-start)/60:.1f} min")
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()
        if not auc <= best:  # nan（測試模式資料太少）也存
            best = auc
            torch.save({"model": model.state_dict(), "val_auc": float(auc)}, f"{args.out}/best.pth")

    model.load_state_dict(torch.load(f"{args.out}/best.pth", map_location=device)["model"])
    pickle.dump(predict_probs(model, val_loader, device), open(f"{args.out}/val_probs.pkl", "wb"))
    pickle.dump(predict_probs(model, loader(test_ids, f"{DATA}/test_512", False), device),
                open(f"{args.out}/test_probs.pkl", "wb"))
    print("wrote", f"{args.out}/val_probs.pkl", f"{args.out}/test_probs.pkl")


if __name__ == "__main__":
    main()
