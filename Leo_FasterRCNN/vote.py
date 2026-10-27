# [AI 協助] 本檔由 Claude（Anthropic，透過 Claude Code）協助撰寫，Leo 負責執行與驗證
# 多個偵測模型「投票」合併：同一處病灶，每個模型拿出自己在那裡最高的框分數，取平均（沒畫框的模型算 0）
# 跟 ensemble.py 的 NMS 聯集差在：只有一個模型畫的框分數會被拉低，好幾個模型都畫的框才會留下來
# 用法: python vote.py 0.875 0.65 runs/v1aug2 runs/v2aug   (縮框比例、框門檻、要合併的模型們)
#   有驗證集預測的模型會先印出驗證分數；最後產生 runs/ens/vote_s{縮框}_t{門檻}.csv
import os
import sys

import numpy as np
import pandas as pd
import torch
import torchvision

from ensemble import NMS_IOU, run_preds
from shrink_boxes import shrink
from train_baseline import DATA, SCALE, best_threshold, load_boxes
from tta import SKIP


def vote(parts):
    n = len(parts)
    boxes = np.concatenate([p[0] for p in parts])
    scores = np.concatenate([p[1] for p in parts])
    src = np.concatenate([np.full(len(p[1]), i) for i, p in enumerate(parts)])  # 每個框是哪個模型畫的
    keep = scores >= SKIP
    boxes, scores, src = boxes[keep], scores[keep], src[keep]
    if len(boxes) == 0:
        return boxes.reshape(-1, 4), scores
    keep = torchvision.ops.nms(torch.as_tensor(boxes), torch.as_tensor(scores), NMS_IOU).numpy()
    same = torchvision.ops.box_iou(torch.as_tensor(boxes[keep]), torch.as_tensor(boxes)).numpy() > NMS_IOU
    new = [sum(scores[r & (src == i)].max(initial=0) for i in range(n)) / n for r in same]
    return boxes[keep], np.array(new, dtype=np.float32)


if __name__ == "__main__":
    factor, thr, runs = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3:]
    device = torch.device("cuda")
    split = pd.read_csv(f"{DATA}/split.csv")
    val_ids = split[split.split == "val"].patientId.tolist()
    test_ids = [p[:-4] for p in os.listdir(f"{DATA}/test_512")]
    preds = [run_preds(r, val_ids, test_ids, device) for r in runs]

    # 用 --full 訓練的模型看過驗證集，這裡印出來的分數會偏高，只能參考
    val = {pid: vote([v[pid] for v, _ in preds]) for pid in val_ids}
    t, score, _ = best_threshold(val, load_boxes())
    print(f"驗證分數 {score:.4f}（驗證集上最佳門檻 {t:.2f}，提交檔用 {thr}）", flush=True)

    rows = []
    for pid in test_ids:
        b, s = vote([te[pid] for _, te in preds])
        keep = s >= thr
        parts = [f"{sc:.3f} {x1:.0f} {y1:.0f} {x2 - x1:.0f} {y2 - y1:.0f}"
                 for (x1, y1, x2, y2), sc in zip(shrink(b[keep], factor) / SCALE, s[keep])]
        rows.append({"patientId": pid, "PredictionString": " ".join(parts)})
    os.makedirs("runs/ens", exist_ok=True)
    out = f"runs/ens/vote_s{factor}_t{thr}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)
