# [AI 協助] 本檔由 Claude（Anthropic，透過 Claude Code）協助撰寫，Leo 負責執行與驗證
# 用某次訓練的 best.pth 預測測試集，依指定的縮框比例各產生一份提交檔
# 用法: python make_submission.py runs/full8 1.0 0.875
import os
import sys

import pandas as pd
import torch

from shrink_boxes import cached_predict, shrink
from train_baseline import DATA, SCALE, build_model

if __name__ == "__main__":
    run, factors = sys.argv[1], [float(f) for f in sys.argv[2:]]
    device = torch.device("cuda")
    ckpt = torch.load(f"{run}/best.pth", map_location=device)
    model = build_model().to(device)
    model.load_state_dict(ckpt["model"])
    thr = ckpt["threshold"]
    print(f"{run}: best val score {ckpt['val_score']:.4f}, threshold {thr}")

    test_ids = [p[:-4] for p in os.listdir(f"{DATA}/test_512")]
    preds = cached_predict(model, test_ids, f"{DATA}/test_512", {}, f"{run}/test_preds.pkl", device)
    for f in factors:
        rows = []
        for pid in test_ids:
            b, s = preds[pid]
            keep = s >= thr
            parts = [f"{sc:.3f} {x1:.0f} {y1:.0f} {x2 - x1:.0f} {y2 - y1:.0f}"
                     for (x1, y1, x2, y2), sc in zip(shrink(b[keep], f) / SCALE, s[keep])]
            rows.append({"patientId": pid, "PredictionString": " ".join(parts)})
        path = f"{run}/submission_shrink{f:.3f}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        print("wrote", path)
