# [AI 協助] 本檔由 Claude（Anthropic，透過 Claude Code）協助撰寫，Leo 負責執行與驗證
# 前處理：DICOM (1024x1024) 縮成 512x512 PNG，並依三種類別等比例切 train / val
import os
from multiprocessing import Pool

import pandas as pd
import pydicom
from PIL import Image
from sklearn.model_selection import train_test_split

DATA = "data"
SIZE = 512


def convert(args):
    src, dst = args
    if os.path.exists(dst):
        return
    img = Image.fromarray(pydicom.dcmread(src).pixel_array)
    img.resize((SIZE, SIZE), Image.BILINEAR).save(dst)


if __name__ == "__main__":
    jobs = []
    for split in ["train", "test"]:
        src_dir = f"{DATA}/stage_2_{split}_images"
        dst_dir = f"{DATA}/{split}_{SIZE}"
        os.makedirs(dst_dir, exist_ok=True)
        for f in os.listdir(src_dir):
            jobs.append((f"{src_dir}/{f}", f"{dst_dir}/{f[:-4]}.png"))

    with Pool(max(1, os.cpu_count() - 2)) as pool:  # 留兩個核心給 HW2 的訓練
        for i, _ in enumerate(pool.imap_unordered(convert, jobs, chunksize=64)):
            if i % 5000 == 0:
                print(f"{i}/{len(jobs)}", flush=True)
    print(f"converted {len(jobs)} images")

    # 每位病人一筆，依類別分層切 80/20
    classes = pd.read_csv(f"{DATA}/stage_2_detailed_class_info.csv").drop_duplicates("patientId")
    train_ids, val_ids = train_test_split(
        classes.patientId, test_size=0.2, stratify=classes["class"], random_state=42)
    split = pd.DataFrame({"patientId": pd.concat([train_ids, val_ids]),
                          "split": ["train"] * len(train_ids) + ["val"] * len(val_ids)})
    split = split.merge(classes, on="patientId")
    split.to_csv(f"{DATA}/split.csv", index=False)
    print(split.groupby(["split", "class"]).size().unstack().to_string())
