# [AI 協助] 本檔由 Claude（Anthropic，透過 Claude Code）協助撰寫，Leo 負責執行與驗證
# 進度報告用圖：08 三種類別各一張、09 驗證集預測的對／漏／誤（橘框 = 醫師標註，青色虛線 = 模型預測）
import pickle, pandas as pd
from PIL import Image, ImageDraw

DATA, THR = "data", 0.65
lab = pd.read_csv(f"{DATA}/stage_2_train_labels.csv")
gt = {p: g[["x", "y", "width", "height"]].values / 2 for p, g in lab[lab.Target == 1].groupby("patientId")}  # 1024 -> 512
V = pickle.load(open("runs/v1aug2/val_preds.pkl", "rb"))
ORANGE, CYAN, S = (255, 138, 61), (79, 209, 197), 1.5  # 512 -> 768

def dashed(d, x1, y1, x2, y2):
    for a, b, c, e in [(x1, y1, x2, y1), (x1, y2, x2, y2), (x1, y1, x1, y2), (x2, y1, x2, y2)]:
        n = int(max(abs(c - a), abs(e - b)) // 24) + 1
        for i in range(n):
            t0, t1 = i / n, min(i / n + 0.5 / n, 1)
            d.line([a + (c - a) * t0, b + (e - b) * t0, a + (c - a) * t1, b + (e - b) * t1], fill=CYAN, width=5)

def tile(pid, show_pred):
    im = Image.open(f"{DATA}/train_512/{pid}.png").convert("RGB").resize((768, 768), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    for x, y, w, h in gt.get(pid, []):
        d.rectangle([x * S, y * S, (x + w) * S, (y + h) * S], outline=ORANGE, width=6)
    if show_pred:
        b, s = V[pid]
        for x1, y1, x2, y2 in b[s >= THR]:
            dashed(d, x1 * S, y1 * S, x2 * S, y2 * S)
    return im

def row(pids, show_pred, out):
    c = Image.new("RGB", (768 * 3 + 48, 768), (13, 27, 42))
    for i, p in enumerate(pids):
        c.paste(tile(p, show_pred), (i * (768 + 24), 0))
    c.resize((c.width * 2 // 3, c.height * 2 // 3), Image.LANCZOS).save(out, quality=85)
    print("wrote", out)

row(["01027bc3-dc40-4165-a6c3-d6be2cb7ca34", "011d6f51-b732-4d45-a84d-90477325ef2d", "ba5b22cb-77dc-4ee7-b18f-8bbed5112712"], False, "figures/08_classes.jpg")
row(["ed55f8dc-45a7-4416-ae94-847bb41a35c3", "19720da5-a262-4c83-9ae9-fe6f7f5cc5fb", "e28db7bd-964c-4be0-a724-4a8a641d440b"], True, "figures/09_errors.jpg")
