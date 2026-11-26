"""Faster R-CNN baseline（COCO 預訓練）for RSNA Pneumonia Detection.

用法（在 project_rsna/ 底下、已啟用 .venv）：
    python train_frcnn.py --name frcnn_baseline --epochs 8
    python train_frcnn.py --name smoke --epochs 1 --limit 400        # 快速測試流程

輸出到 runs/<name>/：
    history.csv      每個 epoch 的 train loss、val 分數、最佳信心門檻
    best.pt          val 分數最高的權重
    submission.csv   用最佳權重 + 最佳門檻對測試集的預測，可直接上傳 Kaggle
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm.auto import tqdm

import rsna_utils as U

RUNS_DIR = Path(__file__).resolve().parent / "runs"
SCORE_THRESHOLDS = np.round(np.arange(0.30, 0.96, 0.05), 2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="frcnn_baseline")
    p.add_argument("--arch", choices=["v1", "v2"], default="v2", help="v2 較準但較慢")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--optim", choices=["adamw", "adam", "sgd"], default="adamw")
    p.add_argument("--neg-ratio", type=float, default=1.0,
                   help="每個 epoch 抽「陽性數 × neg-ratio」張沒肺炎的圖一起訓練")
    p.add_argument("--hflip", type=float, default=0.5)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help=">0 時只用少量資料，測試流程用")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_model(arch: str, size: int):
    builder = fasterrcnn_resnet50_fpn_v2 if arch == "v2" else fasterrcnn_resnet50_fpn
    # min_size / max_size 設成圖片本身大小，避免 torchvision 預設把圖放大到 800
    model = builder(weights="DEFAULT", min_size=size, max_size=size)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)  # 背景 + opacity
    return model


def build_optimizer(args, params):
    if args.optim == "sgd":
        return torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.wd)
    if args.optim == "adam":
        return torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)


def sample_epoch_ids(pos_ids, neg_ids, neg_ratio, rng):
    n_neg = min(len(neg_ids), int(len(pos_ids) * neg_ratio))
    ids = list(pos_ids) + list(rng.choice(neg_ids, n_neg, replace=False))
    rng.shuffle(ids)
    return ids


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device):
    model.train()
    total, n = 0.0, 0
    pbar = tqdm(loader, desc="train", leave=False)
    for imgs, tgts in pbar:
        imgs = [im.to(device, non_blocking=True) for im in imgs]
        tgts = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in tgts]
        with torch.autocast("cuda", dtype=torch.float16):
            losses = model(imgs, tgts)
            loss = sum(losses.values())
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total += loss.item()
        n += 1
        pbar.set_postfix(loss=f"{total / n:.4f}")
    return total / max(n, 1)


@torch.no_grad()
def predict(model, loader, device, scale):
    """回傳 patientId -> (boxes [x,y,w,h] 原圖 1024 尺度, scores)。"""
    model.eval()
    out = {}
    for imgs, tgts in tqdm(loader, desc="predict", leave=False):
        with torch.autocast("cuda", dtype=torch.float16):
            results = model([im.to(device, non_blocking=True) for im in imgs])
        for r, t in zip(results, tgts):
            b = r["boxes"].float().cpu().numpy() / scale
            b[:, 2:] -= b[:, :2]  # xyxy -> xywh
            out[t["patientId"]] = (b, r["scores"].float().cpu().numpy())
    return out


def filter_preds(preds, th):
    return {pid: (b[s >= th], s[s >= th]) for pid, (b, s) in preds.items()}


def sweep_thresholds(gt, preds):
    """在 val 上掃信心門檻，回傳 (最佳門檻, 最佳分數, 全部結果)。"""
    scores = {float(th): U.mean_score(gt, filter_preds(preds, th)) for th in SCORE_THRESHOLDS}
    best_th = max(scores, key=scores.get)
    return best_th, scores[best_th], scores


def make_loader(ds, bs, workers, shuffle):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=workers,
                      collate_fn=U.detection_collate, pin_memory=True,
                      persistent_workers=workers > 0)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda")
    out_dir = RUNS_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = U.load_labels()
    gt_boxes = U.boxes_by_patient(labels)
    split = pd.read_csv(U.PNG_DIR / "split.csv")
    train_split = split[split.split == "train"]
    val_ids = split[split.split == "val"].patientId.tolist()
    if args.limit:
        train_split = train_split.sample(args.limit, random_state=args.seed)
        val_ids = val_ids[: args.limit // 2]
    pos_ids = train_split[train_split.Target == 1].patientId.to_numpy()
    neg_ids = train_split[train_split.Target == 0].patientId.to_numpy()
    val_gt = {pid: gt_boxes[pid] for pid in val_ids}
    print(f"train pos={len(pos_ids)} neg={len(neg_ids)} | val={len(val_ids)}")

    scale = args.size / U.ORIG_SIZE
    val_ds = U.RSNADetectionDataset(val_ids, gt_boxes, U.PNG_DIR / "train", size=args.size)
    val_loader = make_loader(val_ds, args.bs, args.workers, shuffle=False)

    model = build_model(args.arch, args.size).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(args, params)
    steps_per_epoch = -(-int(len(pos_ids) * (1 + args.neg_ratio)) // args.bs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=steps_per_epoch * args.epochs + 10, pct_start=0.1)
    scaler = torch.amp.GradScaler()

    history, best_score = [], -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        ids = sample_epoch_ids(pos_ids, neg_ids, args.neg_ratio, rng)
        train_ds = U.RSNADetectionDataset(ids, gt_boxes, U.PNG_DIR / "train",
                                          size=args.size, hflip_prob=args.hflip)
        # 每個 epoch 重新抽負樣本，所以每個 epoch 建一次 loader；
        # num_workers>0 時不能用 persistent_workers 跨 epoch 重用
        train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=False, num_workers=args.workers,
                                  collate_fn=U.detection_collate, pin_memory=True)
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device)

        preds = predict(model, val_loader, device, scale)
        best_th, val_score, _ = sweep_thresholds(val_gt, preds)
        elapsed = time.time() - t0
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4), "val_score": round(val_score, 4),
                        "best_th": best_th, "lr": optimizer.param_groups[0]["lr"], "sec": round(elapsed)})
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        print(f"epoch {epoch}/{args.epochs}  loss {train_loss:.4f}  val {val_score:.4f} @th={best_th}  ({elapsed/60:.1f} min)")

        if val_score > best_score:
            best_score = val_score
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch,
                        "val_score": val_score, "best_th": best_th}, out_dir / "best.pt")

    # ---- 用最佳權重預測測試集 ----
    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    preds = predict(model, val_loader, device, scale)
    best_th, val_score, sweep = sweep_thresholds(val_gt, preds)
    pd.Series(sweep, name="val_score").rename_axis("score_threshold").to_csv(out_dir / "threshold_sweep.csv")
    print(f"best epoch {ckpt['epoch']}: val {val_score:.4f} @th={best_th}")

    test_ids = pd.read_csv(U.RAW_DIR / "stage_2_sample_submission.csv").patientId.tolist()
    if args.limit:
        test_ids = test_ids[: args.limit // 2]
    test_ds = U.RSNADetectionDataset(test_ids, None, U.PNG_DIR / "test", size=args.size)
    test_preds = filter_preds(predict(model, make_loader(test_ds, args.bs, args.workers, False), device, scale), best_th)
    sub = pd.DataFrame({
        "patientId": test_ids,
        "PredictionString": [U.to_prediction_string(*test_preds[pid]) for pid in test_ids],
    })
    sub.to_csv(out_dir / "submission.csv", index=False)
    n_with_box = (sub.PredictionString != "").sum()
    print(f"submission: {len(sub)} rows, {n_with_box} with boxes -> {out_dir / 'submission.csv'}")


if __name__ == "__main__":
    main()
