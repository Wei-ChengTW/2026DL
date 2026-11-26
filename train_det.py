"""YOLO detector on SIZE px PNGs (env, default 512). Usage: [SIZE=768] python train_det.py [fold] [model]"""
import os
import sys
from pathlib import Path

from ultralytics import YOLO

if __name__ == "__main__":  # Windows dataloader workers re-import this file
    fold = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    model = sys.argv[2] if len(sys.argv) > 2 else "yolo11m.pt"
    size = int(os.environ.get("SIZE", 512))
    root = Path(f"data/png{size}").resolve()

    yaml = root / f"det_f{fold}.yaml"
    yaml.write_text(f"train: {root / f'train_f{fold}.txt'}\nval: {root / f'val_f{fold}.txt'}\nnames:\n  0: opacity\n")

    YOLO(model).train(
        data=str(yaml), imgsz=size, epochs=20,  # fold 0 ran 30 but plateaued around epoch 15-19
        batch=16 if size == 512 else 8, workers=4,  # 8GB VRAM
        project=str(Path("runs").resolve()), name=f"det_f{fold}_{Path(model).stem}" + ("" if size == 512 else f"_{size}"), exist_ok=True,
        # X-ray: no color jitter / rotation / vertical flip; mosaic breaks the left/right-lung layout
        hsv_h=0, hsv_s=0, hsv_v=0.2, degrees=0, flipud=0, fliplr=0.5,
        translate=0.1, scale=0.15, mosaic=0, mixup=0,
        cos_lr=True, patience=10, seed=42,
    )
