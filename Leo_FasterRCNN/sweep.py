# [AI 協助] 本檔由 Claude（Anthropic，透過 Claude Code）協助撰寫，Leo 負責執行與驗證
# 用法：python sweep.py 縮框比例 門檻   → 產生 runs/ens/sweep_s{比例}_t{門檻}.csv
import os, sys, pickle, numpy as np, pandas as pd, torch, torchvision
sys.path.insert(0, ".")
from train_baseline import SCALE
from shrink_boxes import shrink
exec(open(os.path.join(os.path.dirname(__file__), "vote_fn.py")).read())
f, t = float(sys.argv[1]), float(sys.argv[2])
T = [pickle.load(open(f"runs/{r}/test_preds.pkl", "rb")) for r in ["v1aug2", "v2aug"]]
rows = []
for pid in T[0]:
    b, s = vote([x[pid] for x in T]); keep = s >= t
    rows.append({"patientId": pid, "PredictionString": " ".join(
        f"{sc_:.3f} {x1:.0f} {y1:.0f} {x2-x1:.0f} {y2-y1:.0f}" for (x1, y1, x2, y2), sc_ in zip(shrink(b[keep], f) / SCALE, s[keep]))})
out = f"runs/ens/sweep_s{f}_t{t}.csv"; pd.DataFrame(rows).to_csv(out, index=False); print(out)
