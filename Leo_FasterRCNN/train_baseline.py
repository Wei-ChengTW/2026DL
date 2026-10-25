# [AI 協助] 本檔由 Claude（Anthropic，透過 Claude Code）協助撰寫，Leo 負責執行與驗證
# RSNA Pneumonia baseline：拿 COCO 預訓練的 Faster R-CNN，微調成只偵測 1 類（肺部不透明區）
# 用法: python train_baseline.py --epochs 8          (正式)
#       python train_baseline.py --epochs 1 --limit 200   (先用 200 張確認程式能跑)
import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

DATA = "data"
SIZE = 512             # 前處理縮好的影像邊長
SCALE = SIZE / 1024    # 原始標註是 1024 尺度，要乘上這個比例
IOU_THRESHOLDS = np.arange(0.4, 0.76, 0.05)  # 比賽評分用的 IoU 門檻: 0.40, 0.45, ..., 0.75


# ---------------- 資料 ----------------
def load_boxes():
    """讀標註，回傳 {patientId: [[x1, y1, x2, y2], ...]}，座標已換成 512 尺度"""
    df = pd.read_csv(f"{DATA}/stage_2_train_labels.csv")
    df = df[df.Target == 1]
    boxes = {}
    for r in df.itertuples():
        x1, y1 = r.x * SCALE, r.y * SCALE
        boxes.setdefault(r.patientId, []).append([x1, y1, x1 + r.width * SCALE, y1 + r.height * SCALE])
    return boxes


class RSNADataset(torch.utils.data.Dataset):
    def __init__(self, ids, boxes_by_id, img_dir, train):
        self.ids = list(ids)
        self.boxes_by_id = boxes_by_id
        self.img_dir = img_dir
        self.train = train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        pid = self.ids[i]
        # 灰階複製成 3 個通道，因為預訓練模型是吃 RGB 影像
        img = Image.open(f"{self.img_dir}/{pid}.png").convert("RGB")
        # 沒肺炎的片子 boxes 是空的 (0, 4)，模型會把整張都當背景學
        boxes = torch.tensor(self.boxes_by_id.get(pid, []), dtype=torch.float32).reshape(-1, 4)

        # 資料擴增：一半機率左右翻轉，框的 x 座標也要跟著翻
        if self.train and random.random() < 0.5:
            img = TF.hflip(img)
            boxes[:, [0, 2]] = SIZE - boxes[:, [2, 0]]

        target = {"boxes": boxes, "labels": torch.ones(len(boxes), dtype=torch.int64)}  # 類別 1 = 肺炎
        return TF.to_tensor(img), target, pid


def collate(batch):
    # 每張圖的框數不同，不能疊成一個 tensor，所以用 tuple 包起來
    return tuple(zip(*batch))


# ---------------- 模型 ----------------
def build_model():
    # min_size/max_size 設成 512，不然 torchvision 會自動把圖放大到 800
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT, min_size=SIZE, max_size=SIZE)
    # COCO 有 91 類，換掉最後的分類頭，改成 2 類：0 = 背景、1 = 肺炎
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    return model


# ---------------- 評分（照比賽規則） ----------------
def image_score(pred_boxes, pred_scores, gt_boxes):
    """一張圖的分數：在每個 IoU 門檻下算 TP / (TP + FP + FN)，再平均"""
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return None  # 沒答案也沒預測：比賽規定這張不計分
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return 0.0   # 健康卻畫框，或有肺炎卻沒畫框：0 分
    order = np.argsort(-pred_scores)  # 信心高的框先配對
    ious = torchvision.ops.box_iou(torch.as_tensor(pred_boxes[order]), torch.as_tensor(gt_boxes)).numpy()
    precisions = []
    for t in IOU_THRESHOLDS:
        matched = set()
        tp = 0
        for p in range(len(pred_boxes)):
            # 找還沒被配對、且 IoU 超過門檻的答案框中重疊最多的那個
            best_g, best_iou = -1, t
            for g in range(len(gt_boxes)):
                if g not in matched and ious[p, g] > best_iou:
                    best_g, best_iou = g, ious[p, g]
            if best_g >= 0:
                matched.add(best_g)
                tp += 1
        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - tp
        precisions.append(tp / (tp + fp + fn))
    return float(np.mean(precisions))


def dataset_score(preds, boxes_by_id, threshold):
    """只保留信心 >= threshold 的框，算整個資料集的平均分數"""
    scores = []
    for pid, (b, s) in preds.items():
        keep = s >= threshold
        gt = np.array(boxes_by_id.get(pid, []), dtype=np.float32).reshape(-1, 4)
        sc = image_score(b[keep], s[keep], gt)
        if sc is not None:
            scores.append(sc)
    return float(np.mean(scores))


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = {}
    for images, _, pids in loader:
        images = [img.to(device) for img in images]
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            outputs = model(images)
        for pid, out in zip(pids, outputs):
            preds[pid] = (out["boxes"].float().cpu().numpy(), out["scores"].float().cpu().numpy())
    return preds


def best_threshold(preds, boxes_by_id):
    """在驗證集上試不同信心門檻，挑分數最高的"""
    results = {round(t, 2): dataset_score(preds, boxes_by_id, t) for t in np.arange(0.3, 0.96, 0.05)}
    t = max(results, key=results.get)
    return t, results[t], results


# ---------------- 主程式 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 張，測試用")
    ap.add_argument("--out", default="runs/baseline")
    args = ap.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    boxes_by_id = load_boxes()
    split = pd.read_csv(f"{DATA}/split.csv")
    train_ids = split[split.split == "train"].patientId.tolist()
    val_ids = split[split.split == "val"].patientId.tolist()
    if args.limit:
        train_ids, val_ids = train_ids[:args.limit], val_ids[:args.limit // 4]

    train_loader = torch.utils.data.DataLoader(
        RSNADataset(train_ids, boxes_by_id, f"{DATA}/train_512", train=True),
        batch_size=args.batch, shuffle=True, num_workers=4, collate_fn=collate, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(
        RSNADataset(val_ids, boxes_by_id, f"{DATA}/train_512", train=False),
        batch_size=16, shuffle=False, num_workers=4, collate_fn=collate)

    model = build_model().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    # 學習率：前 500 步從很小慢慢升上來（warmup，避免一開始梯度太大把預訓練權重弄壞），之後 cosine 慢慢降到 0
    total_steps = args.epochs * len(train_loader)
    warmup = min(500, total_steps // 5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (
        (s + 1) / warmup if s < warmup else 0.5 * (1 + np.cos(np.pi * (s - warmup) / max(1, total_steps - warmup)))))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")  # 混合精度，省顯示卡記憶體也比較快

    best = -1.0
    log = open(f"{args.out}/log.txt", "a", encoding="utf-8")
    for epoch in range(args.epochs):
        model.train()
        start, running = time.time(), 0.0
        for step, (images, targets, _) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                # 訓練模式下，Faster R-CNN 直接回傳 4 種 loss：RPN 的分類/框回歸、ROI head 的分類/框回歸
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
            if step % 200 == 0:
                print(f"epoch {epoch+1} step {step}/{len(train_loader)} loss {loss.item():.3f}", flush=True)

        preds = predict(model, val_loader, device)
        t, score, _ = best_threshold(preds, boxes_by_id)
        msg = (f"epoch {epoch+1}/{args.epochs} train_loss {running/len(train_loader):.4f} "
               f"val_score {score:.4f} (threshold {t}) {(time.time()-start)/60:.1f} min")
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()
        if score > best:
            best = score
            # 轉成 Python float：torch.load 預設只接受基本型別，numpy 的數字會被擋
            torch.save({"model": model.state_dict(), "threshold": float(t), "val_score": float(score)},
                       f"{args.out}/best.pth")

    # 用驗證分數最好的那個 epoch 預測測試集，產生提交檔
    ckpt = torch.load(f"{args.out}/best.pth", map_location=device)
    model.load_state_dict(ckpt["model"])
    test_ids = [f[:-4] for f in os.listdir(f"{DATA}/test_512")]
    test_loader = torch.utils.data.DataLoader(
        RSNADataset(test_ids, {}, f"{DATA}/test_512", train=False),
        batch_size=16, shuffle=False, num_workers=4, collate_fn=collate)
    test_preds = predict(model, test_loader, device)
    rows = []
    for pid in test_ids:
        b, s = test_preds[pid]
        keep = s >= ckpt["threshold"]
        parts = []
        for (x1, y1, x2, y2), sc in zip(b[keep] / SCALE, s[keep]):  # 換回 1024 尺度
            parts.append(f"{sc:.3f} {x1:.0f} {y1:.0f} {x2 - x1:.0f} {y2 - y1:.0f}")
        rows.append({"patientId": pid, "PredictionString": " ".join(parts)})
    pd.DataFrame(rows).to_csv(f"{args.out}/submission.csv", index=False)
    print(f"best val score {best:.4f}, threshold {ckpt['threshold']}, wrote {args.out}/submission.csv")


if __name__ == "__main__":
    main()
