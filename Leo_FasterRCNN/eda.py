# [AI 協助] 本檔由 Claude（Anthropic，透過 Claude Code）協助撰寫，Leo 負責執行與驗證
# RSNA Pneumonia 資料初步觀察：類別比例、每張幾個框、框的大小，並畫幾張有框的 X 光片
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
import matplotlib.patches as patches

DATA = "data"
labels = pd.read_csv(f"{DATA}/stage_2_train_labels.csv")
classes = pd.read_csv(f"{DATA}/stage_2_detailed_class_info.csv")

print("labels 列數:", len(labels), "/ 病人數:", labels.patientId.nunique())
per_patient = labels.groupby("patientId").Target.max()
print("有肺炎的病人:", int(per_patient.sum()), f"({per_patient.mean():.1%})")

boxes = labels[labels.Target == 1]
n_boxes = boxes.groupby("patientId").size()
print("有肺炎者每人框數分佈:\n", n_boxes.value_counts().sort_index().to_string())
print("框寬 / 高 (像素):\n", boxes[["width", "height"]].describe().round(1).to_string())

print("詳細類別 (每位病人一筆):\n",
      classes.drop_duplicates("patientId")["class"].value_counts().to_string())

ds = pydicom.dcmread(f"{DATA}/stage_2_train_images/{labels.patientId[0]}.dcm")
print("影像大小:", ds.pixel_array.shape, ds.pixel_array.dtype,
      "/ 拍攝方向:", ds.get("ViewPosition"), "/ 性別:", ds.get("PatientSex"), "/ 年齡:", ds.get("PatientAge"))

# 畫 2 張沒肺炎 + 4 張有肺炎（框數 1、2、2、3+）
show = list(per_patient[per_patient == 0].index[:2])
for k in [1, 2, 2, 3]:
    cand = [p for p in n_boxes[n_boxes >= k].index if p not in show]
    show.append(cand[0])

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, pid in zip(axes.flat, show):
    img = pydicom.dcmread(f"{DATA}/stage_2_train_images/{pid}.dcm").pixel_array
    ax.imshow(img, cmap="gray")
    pb = boxes[boxes.patientId == pid]
    for _, r in pb.iterrows():
        ax.add_patch(patches.Rectangle((r.x, r.y), r.width, r.height, fill=False, edgecolor="red", linewidth=2))
    cls = classes[classes.patientId == pid]["class"].iloc[0]
    ax.set_title(f"{cls}\nboxes: {len(pb)}", fontsize=11)
    ax.axis("off")
plt.tight_layout()
plt.savefig("eda_samples.png", dpi=80, bbox_inches="tight")
print("saved eda_samples.png")
