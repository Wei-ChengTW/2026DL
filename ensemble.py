"""5-fold ensemble on test: average classifier probs, WBF the detectors' boxes, write a submission.

  python ensemble.py --conf 0.06 --shrink 0.8 --t-cls 0.15 [--wbf-iou 0.4] [--out submission.csv]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from ensemble_boxes import weighted_boxes_fusion

from predict import apply, predict

FOLDS = range(5)


def wbf(fold_preds, iou_thr):
    """Fuse per-fold {pid: [(conf, x, y, w, h)]} (1024px) into one dict of the same shape."""
    out = {}
    for pid in fold_preds[0]:
        boxes, scores, labels = [], [], []
        for p in fold_preds:
            b = np.array([(x / 1024, y / 1024, (x + w) / 1024, (y + h) / 1024) for _, x, y, w, h in p[pid]]).reshape(-1, 4)
            boxes.append(b.clip(0, 1))
            scores.append([c for c, *_ in p[pid]])
            labels.append([0] * len(p[pid]))
        b, s, _ = weighted_boxes_fusion(boxes, scores, labels, iou_thr=iou_thr, skip_box_thr=0.01)
        out[pid] = [(float(c), x1 * 1024, y1 * 1024, (x2 - x1) * 1024, (y2 - y1) * 1024) for c, (x1, y1, x2, y2) in zip(s, b)]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.06)
    ap.add_argument("--shrink", type=float, default=0.8)
    ap.add_argument("--t-cls", type=float, default=0.15)
    ap.add_argument("--wbf-iou", type=float, default=0.4)
    ap.add_argument("--dets", nargs="+", default=["yolo11m"], help="detector run suffixes to fuse, e.g. yolo11m yolo11l")
    ap.add_argument("--out", default="submission.csv")
    a = ap.parse_args()

    fold_preds = []
    for f, d in [(f, d) for f in FOLDS for d in a.dets]:
        size = int(d.split("_")[1]) if "_" in d else 512  # run suffix, e.g. yolo11m_768
        cache = Path(f"runs/det_f{f}_{d}/test_preds_iou0.2.pkl")  # same cache predict.py writes
        if not cache.exists():
            paths = sorted(map(str, Path(f"data/png{size}/test").glob("*.png")))
            pd.to_pickle(predict(cache.with_name("weights") / "best.pt", paths, 0.2, size), cache)
        fold_preds.append(pd.read_pickle(cache))
    probs = pd.concat([pd.read_csv(f"runs/cls_f{f}/test_probs.csv").set_index("patientId").p_opacity for f in FOLDS],
                      axis=1).mean(axis=1).to_dict()

    preds = apply(wbf(fold_preds, a.wbf_iou), a.conf, a.shrink, probs, a.t_cls)
    sub = pd.DataFrame({"patientId": list(preds),
                        "PredictionString": [" ".join(f"{c:.3f} {x:.0f} {y:.0f} {w:.0f} {h:.0f}" for c, x, y, w, h in b)
                                             for b in preds.values()]})
    sub.to_csv(a.out, index=False)
    print(f"{a.out}: {len(sub)} rows, {(sub.PredictionString != '').sum()} with boxes")
