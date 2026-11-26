# [AI 協助] 本檔由 Claude（Anthropic，透過 Claude Code）協助撰寫，Leo 負責執行與驗證
# 後處理實驗：座標融合 (fuse) / 每張最多 K 框 ；python pp.py val  或  python pp.py test 縮框 門檻 fuse K
import sys, pickle, numpy as np, pandas as pd, torch, torchvision
sys.path.insert(0, ".")
from train_baseline import SCALE, best_threshold, load_boxes
from shrink_boxes import shrink
def vote(parts, fuse=0, iou=0.3, skip=0.3):
    n = len(parts)
    b = np.concatenate([p[0] for p in parts]); s = np.concatenate([p[1] for p in parts])
    src = np.concatenate([np.full(len(p[1]), i) for i, p in enumerate(parts)])
    k = s >= skip; b, s, src = b[k], s[k], src[k]
    if len(b) == 0: return b.reshape(-1, 4), s
    keep = torchvision.ops.nms(torch.as_tensor(b), torch.as_tensor(s), iou).numpy()
    m = torchvision.ops.box_iou(torch.as_tensor(b[keep]), torch.as_tensor(b)).numpy() > iou
    nb, ns = [], []
    for j, r in enumerate(m):
        best = [np.flatnonzero(r & (src == i))[np.argmax(s[r & (src == i)])] for i in range(n) if (r & (src == i)).any()]
        ns.append(sum(s[best]) / n)
        nb.append((b[best] * s[best, None]).sum(0) / s[best].sum() if fuse else b[keep[j]])
    return np.array(nb, dtype=np.float32), np.array(ns, dtype=np.float32)
def topk(b, s, K):
    o = np.argsort(-s)[:K] if K else np.argsort(-s); return b[o], s[o]
mode = sys.argv[1] if __name__ == "__main__" else None
if mode == "val":
    import pandas as pd
    V = [pickle.load(open(f"runs/{r}/val_preds.pkl", "rb")) for r in ["v1aug2", "v2aug"]]
    bb = load_boxes()
    for fuse in [0, 1]:
        P = {pid: vote([v[pid] for v in V], fuse) for pid in V[0]}
        for K in [0, 1, 2, 3]:
            Q = {pid: topk(*P[pid], K) for pid in P}
            t, sc, _ = best_threshold(Q, bb); print(f"fuse {fuse} K {K}: {sc:.4f} thr {t:.2f}", flush=True)
elif mode == "test":
    f, t, fuse, K = float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    T = [pickle.load(open(f"runs/{r}/test_preds.pkl", "rb")) for r in ["v1aug2", "v2aug"]]
    rows = []
    for pid in T[0]:
        b, s = topk(*vote([x[pid] for x in T], fuse), K); keep = s >= t
        rows.append({"patientId": pid, "PredictionString": " ".join(
            f"{sc_:.3f} {x1:.0f} {y1:.0f} {x2-x1:.0f} {y2-y1:.0f}" for (x1, y1, x2, y2), sc_ in zip(shrink(b[keep], f) / SCALE, s[keep]))})
    out = f"runs/ens/pp_s{f}_t{t}_f{fuse}_k{K}.csv"; pd.DataFrame(rows).to_csv(out, index=False); print(out)
