"""RSNA competition metric. Boxes are (x, y, w, h); preds are (conf, x, y, w, h)."""
import numpy as np

THRESHOLDS = np.arange(0.40, 0.76, 0.05)


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


def image_score(gts, preds):
    """None if the image has no GT and no preds (Kaggle skips it)."""
    if not gts and not preds:
        return None
    if not gts or not preds:
        return 0.0
    preds = sorted(preds, key=lambda p: -p[0])
    scores = []
    for t in THRESHOLDS:
        used, tp = set(), 0
        for p in preds:
            best, best_j = t, None
            for j, g in enumerate(gts):
                if j not in used and (v := iou(p[1:], g)) > best:
                    best, best_j = v, j
            if best_j is not None:
                used.add(best_j)
                tp += 1
        scores.append(tp / (len(preds) + len(gts) - tp))
    return float(np.mean(scores))


def score(gt, pred):
    """gt, pred: dict patientId -> list of boxes. Mean over non-skipped images."""
    s = [image_score(gt.get(k, []), pred.get(k, [])) for k in set(gt) | set(pred)]
    s = [v for v in s if v is not None]
    return float(np.mean(s)) if s else 0.0


if __name__ == "__main__":
    g = [100, 100, 200, 200]
    assert image_score([g], [(0.9, *g)]) == 1.0
    assert image_score([], []) is None
    assert image_score([], [(0.9, *g)]) == 0.0
    assert image_score([g], []) == 0.0
    # shifted box: IoU = 150*200 / (2*40000 - 30000) = 0.6 -> hits t=0.40..0.55 (4 of 8)
    assert image_score([g], [(0.9, 150, 100, 200, 200)]) == 0.5
    # extra FP halves the score at every threshold
    assert image_score([g], [(0.9, *g), (0.5, 500, 500, 50, 50)]) == 0.5
    assert score({"a": [g], "b": []}, {"a": [(1, *g)], "b": []}) == 1.0
    print("metric ok")
