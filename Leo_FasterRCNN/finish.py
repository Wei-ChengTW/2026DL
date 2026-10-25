# [AI 協助] 本檔由 Claude（Anthropic，透過 Claude Code）協助撰寫，Leo 負責執行與驗證
# 收尾：用 best.pth 預測測試集產生提交檔，並畫報告用的訓練曲線與預測框對照圖
import os
import re

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from train_baseline import DATA, SCALE, RSNADataset, build_model, collate, image_score, load_boxes, predict

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False
RUN = "runs/baseline"
OUT = "figures"
DPI = 200

if __name__ == "__main__":
    device = torch.device("cuda")
    ckpt = torch.load(f"{RUN}/best.pth", map_location=device)
    model = build_model().to(device)
    model.load_state_dict(ckpt["model"])
    thr = ckpt["threshold"]
    print(f"best val score {ckpt['val_score']:.4f}, threshold {thr}")

    # 1. 測試集 -> submission.csv（座標換回 1024 尺度）
    test_ids = [f[:-4] for f in os.listdir(f"{DATA}/test_512")]
    loader = torch.utils.data.DataLoader(RSNADataset(test_ids, {}, f"{DATA}/test_512", train=False),
                                         batch_size=16, num_workers=4, collate_fn=collate)
    preds = predict(model, loader, device)
    rows = []
    for pid in test_ids:
        b, s = preds[pid]
        keep = s >= thr
        parts = [f"{sc:.3f} {x1:.0f} {y1:.0f} {x2 - x1:.0f} {y2 - y1:.0f}"
                 for (x1, y1, x2, y2), sc in zip(b[keep] / SCALE, s[keep])]
        rows.append({"patientId": pid, "PredictionString": " ".join(parts)})
    sub = pd.DataFrame(rows)
    sub.to_csv(f"{RUN}/submission.csv", index=False)
    print(f"wrote submission: {len(sub)} rows, {(sub.PredictionString != '').sum()} 張有畫框")

    # 2. 訓練曲線（從 log.txt 讀）
    log = [re.search(r"epoch (\d+)/\d+ train_loss ([\d.]+) val_score ([\d.]+)", l) for l in open(f"{RUN}/log.txt", encoding="utf-8")]
    log = [(int(m[1]), float(m[2]), float(m[3])) for m in log if m]
    ep, loss, score = zip(*log)
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(ep, loss, "o-", color="#3B6EA8", label="訓練 loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("訓練 loss", color="#3B6EA8")
    ax1.set_xticks(ep)
    ax2 = ax1.twinx()
    ax2.plot(ep, score, "s-", color="#C8553D", label="驗證分數")
    ax2.set_ylabel("驗證分數（比賽評分方式）", color="#C8553D")
    for e, s_ in zip(ep, score):
        ax2.annotate(f"{s_:.4f}", (e, s_), textcoords="offset points", xytext=(0, 8), ha="center", color="#C8553D")
    plt.title("Faster R-CNN baseline 訓練過程")
    plt.tight_layout()
    plt.savefig(f"{OUT}/05_training_curve.png", dpi=DPI)
    plt.close()

    # 3. 預測框對照：驗證集挑「預測很準 / 漏掉 / 健康卻畫框」各 2 張
    boxes_by_id = load_boxes()
    split = pd.read_csv(f"{DATA}/split.csv")
    val_ids = split[split.split == "val"].patientId.tolist()[:600]
    vloader = torch.utils.data.DataLoader(RSNADataset(val_ids, boxes_by_id, f"{DATA}/train_512", train=False),
                                          batch_size=16, num_workers=4, collate_fn=collate)
    vpreds = predict(model, vloader, device)
    good, missed, false_pos = [], [], []
    for pid in val_ids:
        b, s = vpreds[pid]
        b = b[s >= thr]
        gt = np.array(boxes_by_id.get(pid, []), dtype=np.float32).reshape(-1, 4)
        if len(gt) and len(b):
            sc = image_score(b, s[s >= thr], gt)
            if sc >= 0.5:
                good.append((pid, sc))
        elif len(gt) and not len(b):
            missed.append((pid, 0.0))
        elif not len(gt) and len(b):
            false_pos.append((pid, 0.0))
    good.sort(key=lambda x: -x[1])
    picks = [("預測準確", p) for p in good[:2]] + [("漏掉肺炎", p) for p in missed[:2]] + \
            [("健康卻畫框", p) for p in false_pos[:2]]
    fig, axes = plt.subplots(2, 3, figsize=(13, 9.2))
    for ax, (kind, (pid, sc)) in zip(axes.T.flat, picks):
        ax.imshow(Image.open(f"{DATA}/train_512/{pid}.png"), cmap="gray")
        for x1, y1, x2, y2 in boxes_by_id.get(pid, []):
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#2ECC40", linewidth=2.5))
        b, s = vpreds[pid]
        for (x1, y1, x2, y2), c in zip(b[s >= thr], s[s >= thr]):
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=2, linestyle="--"))
            ax.text(x1, y1 - 4, f"{c:.2f}", color="red", fontsize=9)
        ax.set_title(kind + (f"（本張分數 {sc:.2f}）" if kind == "預測準確" else ""), fontsize=12)
        ax.axis("off")
    plt.suptitle("驗證集預測結果（綠框：醫師標註；紅色虛線：模型預測，數字為信心分數）", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{OUT}/06_predictions.png", dpi=DPI)
    plt.close()
    print("saved figures 05, 06")
