# [AI 協助] 本檔由 Claude（Anthropic，透過 Claude Code）協助撰寫，Leo 負責執行與驗證
# 多尺度 TTA：同一個模型在推論時把圖縮放成不同大小（torchvision 會自動把框換回 512 座標）
# python ms_tta.py runs/v1aug2 runs/v2aug
import os, sys, pickle, torch, pandas as pd
sys.path.insert(0, ".")
from train_baseline import DATA, RSNADataset, build_model, collate, predict
from tta import FlipDataset, unflip
if __name__ == "__main__":
    dev = torch.device("cuda")
    split = pd.read_csv(f"{DATA}/split.csv"); val = split[split.split == "val"].patientId.tolist()
    test = [p[:-4] for p in os.listdir(f"{DATA}/test_512")]
    for run in sys.argv[1:]:
        ck = torch.load(f"{run}/best.pth", map_location=dev); m = build_model(ck.get("arch", "frcnn")).to(dev); m.load_state_dict(ck["model"])
        for sc in [448, 576, 640]:
            m.transform.min_size, m.transform.max_size = (sc,), sc
            for flip in [0, 1]:
                for name, ids, d in [("val", val, "train_512"), ("test", test, "test_512")]:
                    path = f"{run}/{name}_preds_s{sc}{'_flip' if flip else ''}.pkl"
                    if os.path.exists(path): continue
                    DS = FlipDataset if flip else RSNADataset
                    loader = torch.utils.data.DataLoader(DS(ids, {}, f"{DATA}/{d}", train=False), batch_size=16, num_workers=4, collate_fn=collate)
                    p = predict(m, loader, dev)
                    if flip: p = {k: (unflip(b), s) for k, (b, s) in p.items()}
                    pickle.dump(p, open(path, "wb")); print("wrote", path, flush=True)
