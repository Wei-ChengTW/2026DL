# [AI 協助] 本檔由 Claude（Anthropic，透過 Claude Code）協助撰寫，Leo 負責執行與驗證
# python sub3.py 縮框 第一框門檻 第二框門檻 pkl...  （座標融合）
import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, "."); sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
from fuse2 import vote2, topk
from train_baseline import SCALE
from shrink_boxes import shrink
f, t1, t2, files = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4:]
T = [pickle.load(open(p, "rb")) for p in files]
rows = []
for pid in T[0]:
    b, s = topk(*vote2([x[pid] for x in T]), 2); keep = s >= t2; keep[:1] = s[:1] >= t1
    rows.append({"patientId": pid, "PredictionString": " ".join(
        f"{sc_:.3f} {x1:.0f} {y1:.0f} {x2-x1:.0f} {y2-y1:.0f}" for (x1, y1, x2, y2), sc_ in zip(shrink(b[keep], f) / SCALE, s[keep]))})
out = f"runs/ens/tt_{f}_{t1}_{t2}.csv"; pd.DataFrame(rows).to_csv(out, index=False); print(out)
