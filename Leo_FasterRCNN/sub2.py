# [AI 協助] 本檔由 Claude（Anthropic，透過 Claude Code）協助撰寫，Leo 負責執行與驗證
# python sub2.py 寬縮 高縮 門檻 名稱 pkl...  （座標融合、每張最多 2 框）
import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, "."); sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
from fuse2 import vote2, topk
from train_baseline import SCALE
fx, fy, t, name, files = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5:]
def shrink2(b):
    if len(b) == 0: return b
    cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    w, h = (b[:, 2] - b[:, 0]) * fx, (b[:, 3] - b[:, 1]) * fy
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
T = [pickle.load(open(p, "rb")) for p in files]
rows = []
for pid in T[0]:
    b, s = topk(*vote2([x[pid] for x in T]), 2); keep = s >= t
    rows.append({"patientId": pid, "PredictionString": " ".join(
        f"{sc_:.3f} {x1:.0f} {y1:.0f} {x2-x1:.0f} {y2-y1:.0f}" for (x1, y1, x2, y2), sc_ in zip(shrink2(b[keep]) / SCALE, s[keep]))})
out = f"runs/ens/w_{name}_{fx}_{fy}_t{t}.csv"; pd.DataFrame(rows).to_csv(out, index=False); print(out)
