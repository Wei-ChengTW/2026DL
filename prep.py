"""DICOM -> SIZE px (env, default 512) CLAHE PNG + YOLO labels + patient-level stratified folds.

Output (data/png{SIZE}/):
  images/<pid>.png, test/<pid>.png
  labels/<pid>.txt        YOLO format (cls cx cy w h, normalized); empty file = no pneumonia
  folds.csv               patientId, cls (3-class), Target, fold
  train_f{k}.txt / val_f{k}.txt   image path lists for ultralytics
"""
import os
from multiprocessing import Pool
from pathlib import Path

import cv2
import pandas as pd
import pydicom
from sklearn.model_selection import StratifiedKFold

SRC = Path("data/rsna")
SIZE = int(os.environ.get("SIZE", 512))
DST = Path(f"data/png{SIZE}")
N_FOLDS = 5
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def convert(args):
    src, dst = args
    if dst.exists():
        return
    img = pydicom.dcmread(src).pixel_array  # 1024x1024 uint8
    img = cv2.resize(clahe.apply(img), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst), img)


def main():
    for d in ("images", "labels", "test"):
        (DST / d).mkdir(parents=True, exist_ok=True)

    jobs = [(p, DST / "images" / f"{p.stem}.png") for p in (SRC / "stage_2_train_images").glob("*.dcm")]
    jobs += [(p, DST / "test" / f"{p.stem}.png") for p in (SRC / "stage_2_test_images").glob("*.dcm")]
    with Pool() as pool:
        for i, _ in enumerate(pool.imap_unordered(convert, jobs, chunksize=64)):
            if i % 2000 == 0:
                print(f"{i}/{len(jobs)}")

    labels = pd.read_csv(SRC / "stage_2_train_labels.csv")
    for pid, g in labels.groupby("patientId"):
        g = g.dropna()  # source images are 1024x1024
        lines = [f"0 {(r.x + r.width / 2) / 1024:.6f} {(r.y + r.height / 2) / 1024:.6f} "
                 f"{r.width / 1024:.6f} {r.height / 1024:.6f}" for r in g.itertuples()]
        (DST / "labels" / f"{pid}.txt").write_text("\n".join(lines))

    cls = pd.read_csv(SRC / "stage_2_detailed_class_info.csv").drop_duplicates("patientId")
    df = cls.merge(labels.groupby("patientId").Target.max().reset_index(), on="patientId")
    df = df.rename(columns={"class": "cls"})
    df["fold"] = -1
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=42)
    for k, (_, va) in enumerate(skf.split(df, df.cls)):
        df.loc[df.index[va], "fold"] = k
    df.to_csv(DST / "folds.csv", index=False)

    root = DST.resolve() / "images"
    for k in range(N_FOLDS):
        for name, sel in (("train", df.fold != k), ("val", df.fold == k)):
            (DST / f"{name}_f{k}.txt").write_text("\n".join(str(root / f"{p}.png") for p in df.patientId[sel]))
    print(df.groupby(["fold", "cls"]).size().unstack())


if __name__ == "__main__":
    main()
