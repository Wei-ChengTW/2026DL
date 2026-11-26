# [AI 協助] 本檔由 Claude（Anthropic，透過 Claude Code）協助撰寫，Leo 負責執行與驗證
import os, sys, pickle, torch, pandas as pd
sys.path.insert(0, ".")
from train_baseline import DATA, build_model, collate, predict
from tta import FlipDataset, unflip
run = sys.argv[1]; dev = torch.device("cuda")
split = pd.read_csv(f"{DATA}/split.csv"); val = split[split.split == "val"].patientId.tolist()
test = [p[:-4] for p in os.listdir(f"{DATA}/test_512")]
ck = torch.load(f"{run}/best.pth", map_location=dev); m = build_model(ck.get("arch", "frcnn")).to(dev); m.load_state_dict(ck["model"])
for name, ids, d in [("val", val, "train_512"), ("test", test, "test_512")]:
    path = f"{run}/{name}_preds_flip.pkl"
    if os.path.exists(path): continue
    loader = torch.utils.data.DataLoader(FlipDataset(ids, {}, f"{DATA}/{d}", train=False), batch_size=16, num_workers=4, collate_fn=collate)
    p = predict(m, loader, dev)
    pickle.dump({k: (unflip(b), s) for k, (b, s) in p.items()}, open(path, "wb"))  # 存的時候就翻回原圖座標
    print("wrote", path, flush=True)
