# [AI 協助] 本檔由 Claude（Anthropic，透過 Claude Code）協助撰寫，Leo 負責執行與驗證
# 產生報告用的「過程圖」：資料分佈、框統計、範例影像、前處理前後對照
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
import pydicom
from PIL import Image

plt.rcParams["font.family"] = "Microsoft JhengHei"  # 中文字型，避免顯示成方框
plt.rcParams["axes.unicode_minus"] = False

DATA = "data"
OUT = "figures"
DPI = 200
os.makedirs(OUT, exist_ok=True)

labels = pd.read_csv(f"{DATA}/stage_2_train_labels.csv")
classes = pd.read_csv(f"{DATA}/stage_2_detailed_class_info.csv").drop_duplicates("patientId")
boxes = labels[labels.Target == 1]
n_boxes = boxes.groupby("patientId").size()
CLASS_ZH = {"Normal": "正常", "No Lung Opacity / Not Normal": "不正常但非肺炎", "Lung Opacity": "肺炎（有框）"}

# 1. 類別分佈
counts = classes["class"].value_counts()[list(CLASS_ZH)]
fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar([CLASS_ZH[c] for c in counts.index], counts.values, color=["#4C9F70", "#E0A030", "#C8553D"])
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 150, f"{v:,}\n({v / counts.sum():.1%})", ha="center", fontsize=11)
ax.set_ylabel("病人數")
ax.set_ylim(0, counts.max() * 1.2)
ax.set_title(f"訓練資料類別分佈（共 {counts.sum():,} 位病人）")
plt.tight_layout()
plt.savefig(f"{OUT}/01_class_distribution.png", dpi=DPI)
plt.close()

# 2. 框的統計
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
vc = n_boxes.value_counts().sort_index()
axes[0].bar(vc.index.astype(str), vc.values, color="#C8553D")
for x, v in zip(vc.index.astype(str), vc.values):
    axes[0].text(x, v + 40, f"{v:,}", ha="center")
axes[0].set_xlabel("每位肺炎病人的框數")
axes[0].set_ylabel("病人數")
axes[0].set_title("每張片子有幾個框")
axes[1].hist(boxes.width, bins=40, alpha=0.7, label="寬", color="#3B6EA8")
axes[1].hist(boxes.height, bins=40, alpha=0.7, label="高", color="#E0A030")
axes[1].set_xlabel("像素（原始 1024 尺度）")
axes[1].set_ylabel("框數")
axes[1].set_title(f"框的寬高分佈（共 {len(boxes):,} 個框）")
axes[1].legend()
plt.tight_layout()
plt.savefig(f"{OUT}/02_box_statistics.png", dpi=DPI)
plt.close()

# 3. 範例影像：三種類別各挑幾張，肺炎的畫上框
per_patient_cls = classes.set_index("patientId")["class"]
show = (list(per_patient_cls[per_patient_cls == "Normal"].index[:1])
        + list(per_patient_cls[per_patient_cls == "No Lung Opacity / Not Normal"].index[:1])
        + list(n_boxes[n_boxes == 1].index[:1]) + list(n_boxes[n_boxes == 2].index[:2])
        + list(n_boxes[n_boxes >= 3].index[:1]))
fig, axes = plt.subplots(2, 3, figsize=(13, 9))
for ax, pid in zip(axes.flat, show):
    img = pydicom.dcmread(f"{DATA}/stage_2_train_images/{pid}.dcm").pixel_array
    ax.imshow(img, cmap="gray")
    pb = boxes[boxes.patientId == pid]
    for r in pb.itertuples():
        ax.add_patch(patches.Rectangle((r.x, r.y), r.width, r.height, fill=False, edgecolor="red", linewidth=2))
    ax.set_title(f"{CLASS_ZH[per_patient_cls[pid]]}，{len(pb)} 個框", fontsize=12)
    ax.axis("off")
plt.suptitle("訓練資料範例（紅框為醫師標註的肺炎區域）", fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUT}/03_samples.png", dpi=DPI)
plt.close()

# 4. 前處理前後對照
pid = n_boxes[n_boxes == 2].index[0]
ds = pydicom.dcmread(f"{DATA}/stage_2_train_images/{pid}.dcm")
small = Image.open(f"{DATA}/train_512/{pid}.png")
fig, axes = plt.subplots(1, 2, figsize=(11, 5.8))
for ax, img, scale, title in [
    (axes[0], ds.pixel_array, 1.0, f"原始 DICOM：{ds.pixel_array.shape[1]}x{ds.pixel_array.shape[0]}"),
    (axes[1], small, 0.5, f"前處理後 PNG：{small.width}x{small.height}（框座標 x0.5）"),
]:
    ax.imshow(img, cmap="gray")
    for r in boxes[boxes.patientId == pid].itertuples():
        ax.add_patch(patches.Rectangle((r.x * scale, r.y * scale), r.width * scale, r.height * scale,
                                       fill=False, edgecolor="red", linewidth=2))
    ax.set_title(title, fontsize=12)
    ax.axis("on")
plt.suptitle("前處理：DICOM 縮成 512x512 PNG，標註框同比例縮放", fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUT}/04_preprocess.png", dpi=DPI)
plt.close()

print("saved:", sorted(os.listdir(OUT)))
