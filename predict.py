"""Predict with a YOLO detector (optionally gated by classifier probs), tune on val, write submission.

  python predict.py <weights> val --fold 0 [--cls runs/cls_f0/val_probs.csv]
  python predict.py <weights> test --conf C [--shrink 0.875] [--cls runs/cls_f0/test_probs.csv --t-cls T]
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO

from metric import score

SIZE = int(os.environ.get("SIZE", 512))


def predict(weights, paths, iou, size=SIZE):
    """patientId -> [(conf, x, y, w, h)] in original 1024px coords, conf >= 0.01."""
    model, out = YOLO(weights), {}
    for i in range(0, len(paths), 32):  # a list source is run as one batch, so chunk it
        for r in model.predict(paths[i:i + 32], imgsz=size, conf=0.01, iou=iou, verbose=False):
            boxes = r.boxes.xyxy.cpu().numpy() * (1024 / size)
            out[Path(r.path).stem] = [(float(c), x1, y1, x2 - x1, y2 - y1)
                                      for c, (x1, y1, x2, y2) in zip(r.boxes.conf.cpu().numpy(), boxes)]
    return out


def apply(preds, conf, shrink, probs=None, t_cls=0.0):
    """Keep boxes >= conf (top-1 at least, if the classifier says positive); scale w,h around the center."""
    res = {}
    for pid, boxes in preds.items():
        if probs is not None and probs[pid] < t_cls:
            res[pid] = []
            continue
        keep = [b for b in boxes if b[0] >= conf]
        if probs is not None and not keep and boxes:
            keep = [max(boxes)]
        res[pid] = [(c, x + w * (1 - shrink) / 2, y + h * (1 - shrink) / 2, w * shrink, h * shrink)
                    for c, x, y, w, h in keep]
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("mode", choices=["val", "test"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--cls", help="classifier probs csv (patientId, p_opacity)")
    ap.add_argument("--t-cls", type=float, default=0.5)
    ap.add_argument("--conf", type=float)
    ap.add_argument("--shrink", type=float, default=0.875)
    ap.add_argument("--iou", type=float, default=0.2, help="NMS IoU; 0.2 beat ultralytics' 0.7 on val")
    a = ap.parse_args()
    root = Path(f"data/png{SIZE}")
    probs = pd.read_csv(a.cls).set_index("patientId").p_opacity.to_dict() if a.cls else None

    if a.mode == "val":
        paths = (root / f"val_f{a.fold}.txt").read_text().splitlines()
        cache = Path(a.weights).with_name(f"val_f{a.fold}_preds_iou{a.iou}.pkl")  # our own local cache
        if not cache.exists():
            pd.to_pickle(predict(a.weights, paths, a.iou), cache)
        preds = pd.read_pickle(cache)
        labels = pd.read_csv("data/rsna/stage_2_train_labels.csv")
        gt = {Path(p).stem: [] for p in paths}
        for r in labels.dropna().itertuples():
            if r.patientId in gt:
                gt[r.patientId].append((r.x, r.y, r.width, r.height))
        t_cls_grid = np.arange(0.2, 0.75, 0.05) if probs else [0.0]
        grid = pd.DataFrame([(t, c, s, score(gt, apply(preds, c, s, probs, t)))
                             for t in t_cls_grid for c in np.arange(0.02, 0.32, 0.02) for s in (1.0, 0.95, 0.9, 0.875)],
                            columns=["t_cls", "conf", "shrink", "score"])
        best = grid.loc[grid.score.idxmax()]
        print(grid[grid.t_cls == best.t_cls].pivot(index="conf", columns="shrink", values="score").round(4))
        print("best:", best.round(4).to_dict())
    else:
        cache = Path(a.weights).with_name(f"test_preds_iou{a.iou}.pkl")
        if not cache.exists():
            pd.to_pickle(predict(a.weights, sorted(map(str, (root / "test").glob("*.png"))), a.iou), cache)
        preds = apply(pd.read_pickle(cache), a.conf, a.shrink, probs, a.t_cls)
        sub = pd.DataFrame({"patientId": list(preds),
                            "PredictionString": [" ".join(f"{c:.3f} {x:.0f} {y:.0f} {w:.0f} {h:.0f}" for c, x, y, w, h in b)
                                                 for b in preds.values()]})
        sub.to_csv("submission.csv", index=False)
        print(f"submission.csv: {len(sub)} rows, {(sub.PredictionString != '').sum()} with boxes")
