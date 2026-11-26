# [AI 協助] 本檔由 Claude（Anthropic，透過 Claude Code）協助撰寫，Leo 負責執行與驗證
# python pp3.py 縮框 門檻 K 名稱 pkl...   (座標融合投票)
import sys, pickle, pandas as pd
sys.path.insert(0, "."); sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
from pp import vote, topk
from train_baseline import SCALE
from shrink_boxes import shrink
f, t, K, name, files = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5:]
T = [pickle.load(open(p, "rb")) for p in files]
rows = []
for pid in T[0]:
    b, s = topk(*vote([x[pid] for x in T], 1), K); keep = s >= t
    rows.append({"patientId": pid, "PredictionString": " ".join(
        f"{sc_:.3f} {x1:.0f} {y1:.0f} {x2-x1:.0f} {y2-y1:.0f}" for (x1, y1, x2, y2), sc_ in zip(shrink(b[keep], f) / SCALE, s[keep]))})
out = f"runs/ens/pp3_{name}_s{f}_t{t}_k{K}.csv"; pd.DataFrame(rows).to_csv(out, index=False); print(out)
