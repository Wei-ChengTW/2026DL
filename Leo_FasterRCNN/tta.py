# [AI 協助] 本檔由 Claude（Anthropic，透過 Claude Code）協助撰寫，Leo 負責執行與驗證
# 翻轉測試（TTA）：原圖預測一次、左右翻轉再預測一次，框翻回來後用 Weighted Box Fusion 合併
# 用法: python tta.py runs/baseline              (只在驗證集上比較有無 TTA)
#       python tta.py runs/full8 --submit 0.875  (驗證集比較 + TTA 版提交檔，框縮成 87.5%)
import os
import sys

import numpy as np
import pandas as pd
import torch
import torchvision

from shrink_boxes import cached_predict, shrink
from train_baseline import DATA, SCALE, SIZE, RSNADataset, best_threshold, build_model, load_boxes

FUSE_IOU = 0.5  # 兩次預測的框重疊超過這個 IoU 就視為同一個框
SKIP = 0.3      # 信心低於這個的框先丟掉再合併，不然一堆 0.05 的雜框會把好框的平均分數拉低


class FlipDataset(RSNADataset):
    """跟 RSNADataset 一樣，只是影像左右翻轉（tensor 形狀是 C x H x W，翻最後一維就是左右翻）"""
    def __getitem__(self, i):
        img, target, pid = super().__getitem__(i)
        return img.flip(-1), target, pid


def unflip(boxes):
    """翻轉圖上的框翻回原圖座標：新 x1 = 寬 - 舊 x2、新 x2 = 寬 - 舊 x1"""
    out = boxes.copy()
    out[:, 0], out[:, 2] = SIZE - boxes[:, 2], SIZE - boxes[:, 0]
    return out


def fuse(b1, s1, b2, s2):
    """Weighted Box Fusion：重疊的框按分數加權平均；只出現在一次預測的框，分數打對折"""
    boxes, scores = np.concatenate([b1, b2]), np.concatenate([s1, s2])
    keep = scores >= SKIP
    boxes, scores = boxes[keep], scores[keep]
    clusters = []  # 每個元素是一群被視為同一個框的 index
    for i in np.argsort(-scores):
        for c in clusters:
            w = scores[c]
            center = (boxes[c] * w[:, None]).sum(0) / w.sum()
            iou = torchvision.ops.box_iou(torch.as_tensor(boxes[i:i + 1]), torch.as_tensor(center[None]))[0, 0]
            if iou > FUSE_IOU:
                c.append(i)
                break
        else:
            clusters.append([i])
    fb, fs = [], []
    for c in clusters:
        w = scores[c]
        fb.append((boxes[c] * w[:, None]).sum(0) / w.sum())
        fs.append(w.mean() * min(len(c), 2) / 2)
    return np.array(fb, dtype=np.float32).reshape(-1, 4), np.array(fs, dtype=np.float32)


def tta_preds(model, ids, img_dir, run, name, device):
    plain = cached_predict(model, ids, img_dir, {}, f"{run}/{name}_preds.pkl", device)
    # 翻轉版不能共用 cached_predict 的 Dataset，自己建 loader 再存檔
    path = f"{run}/{name}_preds_flip.pkl"
    if os.path.exists(path):
        flipped = pd.read_pickle(path)
    else:
        from train_baseline import collate, predict
        loader = torch.utils.data.DataLoader(FlipDataset(ids, {}, img_dir, train=False),
                                             batch_size=16, num_workers=4, collate_fn=collate)
        flipped = predict(model, loader, device)
        pd.to_pickle(flipped, path)
    return plain, {pid: fuse(*plain[pid], unflip(flipped[pid][0]), flipped[pid][1]) for pid in ids}


if __name__ == "__main__":
    run = sys.argv[1]
    device = torch.device("cuda")
    ckpt = torch.load(f"{run}/best.pth", map_location=device)
    model = build_model().to(device)
    model.load_state_dict(ckpt["model"])

    boxes_by_id = load_boxes()
    split = pd.read_csv(f"{DATA}/split.csv")
    val_ids = split[split.split == "val"].patientId.tolist()
    plain, fused = tta_preds(model, val_ids, f"{DATA}/train_512", run, "val", device)
    t0, s0, _ = best_threshold(plain, boxes_by_id)
    t1, s1, _ = best_threshold(fused, boxes_by_id)
    print(f"{run} 驗證集  不用 TTA：{s0:.4f}（門檻 {t0:.2f}）  用 TTA：{s1:.4f}（門檻 {t1:.2f}）", flush=True)
    with open(f"{run}/tta_result.txt", "w", encoding="utf-8") as f:
        f.write(f"plain {s0:.4f} thr {t0:.2f}\ntta {s1:.4f} thr {t1:.2f}\n")

    if "--submit" in sys.argv:
        factor = float(sys.argv[sys.argv.index("--submit") + 1])
        test_ids = [p[:-4] for p in os.listdir(f"{DATA}/test_512")]
        _, test_fused = tta_preds(model, test_ids, f"{DATA}/test_512", run, "test", device)
        rows = []
        for pid in test_ids:
            b, s = test_fused[pid]
            keep = s >= t1
            parts = [f"{sc:.3f} {x1:.0f} {y1:.0f} {x2 - x1:.0f} {y2 - y1:.0f}"
                     for (x1, y1, x2, y2), sc in zip(shrink(b[keep], factor) / SCALE, s[keep])]
            rows.append({"patientId": pid, "PredictionString": " ".join(parts)})
        out = f"{run}/submission_tta_shrink{factor:.3f}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print("wrote", out)
