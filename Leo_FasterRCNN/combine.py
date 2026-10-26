# [AI 協助] 本檔由 Claude（Anthropic，透過 Claude Code）協助撰寫，Leo 負責執行與驗證
# 偵測模型的框 + 分類器的「有肺炎機率」合在一起，在驗證集上比較合併方式，用最好的產生提交檔
# 用法: python combine.py runs/full8 runs/cls 0.875
#   gate：分類器機率 < g 的片子整張不畫框（g 從 0.1 試到 0.9）
#   geo ：每個框的分數改成 sqrt(框分數 x 分類器機率)，兩邊都有把握的框才留得下來
import pickle
import sys

import numpy as np
import pandas as pd
import torch

from shrink_boxes import cached_predict, shrink
from train_baseline import DATA, SCALE, best_threshold, build_model, load_boxes


def merge(det, probs, mode, g=0.0):
    out = {}
    for pid, p in probs.items():
        b, s = det[pid]
        if mode == "none":
            out[pid] = (b, s)
        elif mode == "gate":
            out[pid] = (b, s) if p >= g else (b[:0], s[:0])
        else:  # geo
            out[pid] = (b, np.sqrt(s * p))
    return out


if __name__ == "__main__":
    det_run, cls_run, factor = sys.argv[1], sys.argv[2], float(sys.argv[3])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    model.load_state_dict(torch.load(f"{det_run}/best.pth", map_location=device)["model"])

    boxes_by_id = load_boxes()
    val_probs = pickle.load(open(f"{cls_run}/val_probs.pkl", "rb"))
    test_probs = pickle.load(open(f"{cls_run}/test_probs.pkl", "rb"))
    det_val = cached_predict(model, list(val_probs), f"{DATA}/train_512", {}, f"{det_run}/val_preds.pkl", device)
    det_test = cached_predict(model, list(test_probs), f"{DATA}/test_512", {}, f"{det_run}/test_preds.pkl", device)

    # 每種合併方式都在驗證集上重挑信心門檻，記下 (分數, 方式, g, 門檻)
    results = []
    for mode, g in [("none", 0.0), ("geo", 0.0)] + [("gate", round(g, 1)) for g in np.arange(0.1, 0.95, 0.1)]:
        t, score, _ = best_threshold(merge(det_val, val_probs, mode, g), boxes_by_id)
        results.append((score, mode, g, t))
        print(f"{mode:4s} g={g:.1f}  驗證分數 {score:.4f}（框門檻 {t:.2f}）", flush=True)
    pd.DataFrame(results, columns=["val_score", "mode", "g", "box_threshold"]).to_csv(
        f"{cls_run}/combine_results.csv", index=False)

    score, mode, g, t = max(results)
    print(f"最好：{mode} g={g:.1f} 驗證分數 {score:.4f}（框門檻 {t:.2f}）")
    rows = []
    for pid, (b, s) in merge(det_test, test_probs, mode, g).items():
        keep = s >= t
        parts = [f"{sc:.3f} {x1:.0f} {y1:.0f} {x2 - x1:.0f} {y2 - y1:.0f}"
                 for (x1, y1, x2, y2), sc in zip(shrink(b[keep], factor) / SCALE, s[keep])]
        rows.append({"patientId": pid, "PredictionString": " ".join(parts)})
    out = f"{cls_run}/submission_{mode}_shrink{factor:.3f}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)
