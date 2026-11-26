# [AI 協助] 本檔由 Claude（Anthropic，透過 Claude Code）協助撰寫，Leo 負責執行與驗證
# 進度報告用的統計：DICOM 標頭特徵 vs 肺炎、相關係數、驗證集錯誤分析
import sys, pickle, pandas as pd, pydicom

DATA = "data"
lab = pd.read_csv(f"{DATA}/stage_2_train_labels.csv")
cls = pd.read_csv(f"{DATA}/stage_2_detailed_class_info.csv").drop_duplicates("patientId").set_index("patientId")["class"]
g = lab.groupby("patientId")
target, nbox = g.Target.max(), g.Target.sum()

# 1. 讀每位病人的 DICOM 標頭（不讀像素，約 2 分鐘）
rows = []
for pid in target.index:
    d = pydicom.dcmread(f"{DATA}/stage_2_train_images/{pid}.dcm", stop_before_pixels=True)
    rows.append((pid, d.ViewPosition, d.PatientSex, int(d.PatientAge)))
m = pd.DataFrame(rows, columns=["patientId", "view", "sex", "age"]).set_index("patientId")
m["target"], m["nbox"] = target, nbox

print("拍攝方向的肺炎比例:\n", m.groupby("view").target.agg(["size", "mean"]).round(4))
print("性別的肺炎比例:\n", m.groupby("sex").target.agg(["size", "mean"]).round(4))
m["age_bin"] = pd.cut(m.age, [0, 20, 30, 40, 50, 60, 70, 200], right=False)
print("年齡的肺炎比例:\n", m.groupby("age_bin", observed=False).target.agg(["size", "mean"]).round(4))
print("年齡 > 100 的錯誤值:", int((m.age > 100).sum()))

# 2. 相關係數（AP、男性、是否肺炎編成 0/1，排除年齡 > 100）
c = m[m.age <= 100].assign(AP=lambda x: (x.view == "AP").astype(float), male=lambda x: (x.sex == "M").astype(float))
print("相關係數 n =", len(c), "\n", c[["AP", "age", "male", "nbox", "target"]].astype(float).corr().round(3))

# 3. 驗證集錯誤分析：v1aug2 單模型、門檻 0.65 下，各類別有多少片子被畫了框
V = pickle.load(open("runs/v1aug2/val_preds.pkl", "rb"))
r = pd.DataFrame([(pid, cls[pid], int((s >= 0.65).sum()) > 0) for pid, (b, s) in V.items()], columns=["pid", "cls", "drew"])
print("各類別被畫框的比例:\n", r.groupby("cls").drew.agg(["size", "mean"]).round(4))
fp = r[(r.cls != "Lung Opacity") & r.drew]
print("誤畫框的片子類別組成:\n", fp.cls.value_counts(normalize=True).round(4))
