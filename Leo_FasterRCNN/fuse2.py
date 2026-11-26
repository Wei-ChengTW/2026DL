# [AI 協助] 本檔由 Claude（Anthropic，透過 Claude Code）協助撰寫，Leo 負責執行與驗證
# 座標融合的變化：mode best=每份最佳框加權（目前）、all=群內所有框加權、pow=最佳框以分數^p 加權
import numpy as np, torch, torchvision
def vote2(parts, mode="best", p=1.0, iou=0.3, skip=0.3):
    n = len(parts)
    b = np.concatenate([q[0] for q in parts]); s = np.concatenate([q[1] for q in parts])
    src = np.concatenate([np.full(len(q[1]), i) for i, q in enumerate(parts)])
    k = s >= skip; b, s, src = b[k], s[k], src[k]
    if len(b) == 0: return b.reshape(-1, 4), s
    keep = torchvision.ops.nms(torch.as_tensor(b), torch.as_tensor(s), iou).numpy()
    m = torchvision.ops.box_iou(torch.as_tensor(b[keep]), torch.as_tensor(b)).numpy() > iou
    nb, ns = [], []
    for r in m:
        best = [np.flatnonzero(r & (src == i))[np.argmax(s[r & (src == i)])] for i in range(n) if (r & (src == i)).any()]
        ns.append(s[best].sum() / n)
        idx = np.flatnonzero(r) if mode == "all" else np.array(best)
        w = s[idx] ** p
        nb.append((b[idx] * w[:, None]).sum(0) / w.sum())
    return np.array(nb, dtype=np.float32), np.array(ns, dtype=np.float32)
def topk(b, s, K):
    o = np.argsort(-s)[:K] if K else np.argsort(-s); return b[o], s[o]
if __name__ == "__main__":
    import sys, pickle
    sys.path.insert(0, ".")
    from train_baseline import best_threshold, load_boxes
    bb = load_boxes()
    fs = ["v1aug2/val_preds", "v1aug2/val_preds_flip", "v2aug/val_preds", "v2aug/val_preds_flip"]
    V = [pickle.load(open(f"runs/{x}.pkl", "rb")) for x in fs]
    for mode, p in [("best", 1), ("all", 1), ("best", 0), ("best", 2), ("best", 4), ("all", 2)]:
        P = {pid: topk(*vote2([v[pid] for v in V], mode, p), 2) for pid in V[0]}
        t, sc, _ = best_threshold(P, bb); print(f"{mode} p{p}: {sc:.4f} thr {t:.2f}", flush=True)
