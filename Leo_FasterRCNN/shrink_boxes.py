# [AI 協助] 本檔由 Claude（Anthropic，透過 Claude Code）協助撰寫，Leo 負責執行與驗證
# 縮框實驗（參考第 1 名縮成 87.5%、第 3 名縮成 83% 的做法）：
# 在驗證集上試不同的縮放比例，每個比例都重新挑信心門檻，選分數最高的組合去產生提交檔
# 第 1 名：https://github.com/i-pan/kaggle-rsna18（src/infer/SimpleSubmission.py）
# 第 3 名：https://github.com/pmcheng/rsna-pneumonia（predict.py、util.py）
import os
import pickle

import numpy as np
import pandas as pd
import torch

from train_baseline import DATA, SCALE, RSNADataset, best_threshold, build_model, collate, load_boxes, predict

RUN = "runs/baseline"
FACTORS = np.round(np.arange(0.80, 1.001, 0.025), 3)  # 0.800, 0.825, ..., 1.000


def shrink(boxes, f):
    """以框的中心點為準，寬和高都乘上 f（boxes 是 [x1, y1, x2, y2]）"""
    if len(boxes) == 0:
        return boxes
    cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
    w, h = (boxes[:, 2] - boxes[:, 0]) * f, (boxes[:, 3] - boxes[:, 1]) * f
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def cached_predict(model, ids, img_dir, boxes_by_id, path, device):
    """預測結果存成檔案，之後的實驗（TTA 等）可以直接重用，不用再跑一次模型"""
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    loader = torch.utils.data.DataLoader(RSNADataset(ids, boxes_by_id, img_dir, train=False),
                                         batch_size=16, num_workers=4, collate_fn=collate)
    preds = predict(model, loader, device)
    pickle.dump(preds, open(path, "wb"))
    return preds


if __name__ == "__main__":
    device = torch.device("cuda")
    ckpt = torch.load(f"{RUN}/best.pth", map_location=device)
    model = build_model().to(device)
    model.load_state_dict(ckpt["model"])

    boxes_by_id = load_boxes()
    split = pd.read_csv(f"{DATA}/split.csv")
    val_ids = split[split.split == "val"].patientId.tolist()
    val_preds = cached_predict(model, val_ids, f"{DATA}/train_512", boxes_by_id, f"{RUN}/val_preds.pkl", device)

    # 每個縮放比例都重新挑一次信心門檻
    rows = []
    for f in FACTORS:
        shrunk = {pid: (shrink(b, f), s) for pid, (b, s) in val_preds.items()}
        t, score, _ = best_threshold(shrunk, boxes_by_id)
        rows.append({"factor": f, "threshold": t, "val_score": score})
        print(f"縮放 {f:.3f}  最佳門檻 {t:.2f}  驗證分數 {score:.4f}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(f"{RUN}/shrink_results.csv", index=False)
    best = table.loc[table.val_score.idxmax()]
    f, thr = float(best.factor), float(best.threshold)
    print(f"最佳：縮放 {f:.3f}、門檻 {thr:.2f}、驗證分數 {best.val_score:.4f}（不縮框是 {table.val_score.iloc[-1]:.4f}）")

    # 用最佳組合預測測試集
    test_ids = [p[:-4] for p in os.listdir(f"{DATA}/test_512")]
    test_preds = cached_predict(model, test_ids, f"{DATA}/test_512", {}, f"{RUN}/test_preds.pkl", device)
    out = []
    for pid in test_ids:
        b, s = test_preds[pid]
        keep = s >= thr
        parts = [f"{sc:.3f} {x1:.0f} {y1:.0f} {x2 - x1:.0f} {y2 - y1:.0f}"
                 for (x1, y1, x2, y2), sc in zip(shrink(b[keep], f) / SCALE, s[keep])]
        out.append({"patientId": pid, "PredictionString": " ".join(parts)})
    sub = pd.DataFrame(out)
    sub.to_csv(f"{RUN}/submission_shrink.csv", index=False)
    print(f"wrote {RUN}/submission_shrink.csv：{(sub.PredictionString != '').sum()} 張有畫框")
