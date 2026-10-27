# [AI 協助] 本檔由 Claude（Anthropic，透過 Claude Code）協助撰寫，Leo 負責執行與驗證
# 兩個偵測模型的框合在一起（第 3 名的做法）：兩邊的框倒在一起，重疊的只留分數最高的那個（NMS）
# 驗證集上試過「重疊的框座標取平均」（tta.py 的 fuse），反而比單一模型差，所以改用 NMS
# 用法: python ensemble.py runs/full8 runs/retina 0.875
#   在驗證集上印出兩個模型各自的分數和合併後的分數，再用合併結果產生提交檔（框縮成 87.5%）
import os
import sys

import numpy as np
import pandas as pd
import torch
import torchvision

from shrink_boxes import cached_predict, shrink
from train_baseline import DATA, SCALE, best_threshold, build_model, load_boxes
from tta import SKIP

NMS_IOU = 0.3  # 兩個框 IoU 超過這個就當成同一處病灶，只留分數高的


def fuse(b1, s1, b2, s2):
    boxes, scores = np.concatenate([b1, b2]), np.concatenate([s1, s2])
    keep = scores >= SKIP
    boxes, scores = boxes[keep], scores[keep]
    keep = torchvision.ops.nms(torch.as_tensor(boxes), torch.as_tensor(scores), NMS_IOU).numpy()
    return boxes[keep].reshape(-1, 4), scores[keep]


def run_preds(run, val_ids, test_ids, device):
    ckpt = torch.load(f"{run}/best.pth", map_location=device)
    model = build_model(ckpt.get("arch", "frcnn")).to(device)  # 舊的 best.pth 沒存 arch，都是 Faster R-CNN
    model.load_state_dict(ckpt["model"])
    val = cached_predict(model, val_ids, f"{DATA}/train_512", {}, f"{run}/val_preds.pkl", device)
    test = cached_predict(model, test_ids, f"{DATA}/test_512", {}, f"{run}/test_preds.pkl", device)
    return val, test


if __name__ == "__main__":
    run_a, run_b, factor = sys.argv[1], sys.argv[2], float(sys.argv[3])
    device = torch.device("cuda")
    boxes_by_id = load_boxes()
    split = pd.read_csv(f"{DATA}/split.csv")
    val_ids = split[split.split == "val"].patientId.tolist()
    test_ids = [p[:-4] for p in os.listdir(f"{DATA}/test_512")]

    val_a, test_a = run_preds(run_a, val_ids, test_ids, device)
    val_b, test_b = run_preds(run_b, val_ids, test_ids, device)
    val_ab = {pid: fuse(*val_a[pid], *val_b[pid]) for pid in val_ids}

    for name, preds in [(run_a, val_a), (run_b, val_b), ("合併", val_ab)]:
        t, score, _ = best_threshold(preds, boxes_by_id)
        print(f"{name:14s} 驗證分數 {score:.4f}（框門檻 {t:.2f}）", flush=True)

    rows = []
    for pid in test_ids:
        b, s = fuse(*test_a[pid], *test_b[pid])
        keep = s >= t  # 迴圈最後一個是「合併」，t 就是合併後的最佳門檻
        parts = [f"{sc:.3f} {x1:.0f} {y1:.0f} {x2 - x1:.0f} {y2 - y1:.0f}"
                 for (x1, y1, x2, y2), sc in zip(shrink(b[keep], factor) / SCALE, s[keep])]
        rows.append({"patientId": pid, "PredictionString": " ".join(parts)})
    out = f"{run_b}/submission_ens_{os.path.basename(run_a)}_shrink{factor:.3f}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)
