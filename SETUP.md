# RSNA Pneumonia 期末專案：環境操作手冊

比賽：<https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge>
專案說明：`training_tips_1021.pdf` 第 37–67 頁

## 1. 取得專案

```bash
git clone https://github.com/Wei-ChengTW/2026DL.git
cd 2026DL
git checkout William
```

資料集（`data/`）不在 git 裡，每台電腦都要重新下載，見第 3 步。

## 2. 安裝 Kaggle CLI

先安裝 uv：<https://docs.astral.sh/uv/getting-started/installation/>

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

uv tool install kaggle
kaggle --version
```

登入 Kaggle（會開瀏覽器，授權後就完成）：

```bash
kaggle auth login
```

不想用瀏覽器登入的話：到 Kaggle → Settings → API → Create New Token，把下載的 `kaggle.json` 放到
`~/.kaggle/kaggle.json`（Windows 是 `C:\Users\<你>\.kaggle\kaggle.json`）。

確認可以存取比賽資料（要先在比賽頁面接受規則；帳號已經接受過的話，換電腦不用重按）：

```bash
kaggle competitions files rsna-pneumonia-detection-challenge | head
```

## 3. 下載資料（約 3.9 GB 下載，解壓後 3.7 GB）

```bash
mkdir -p data/rsna
kaggle competitions download -c rsna-pneumonia-detection-challenge -p data/rsna
cd data/rsna && unzip -q rsna-pneumonia-detection-challenge.zip && rm rsna-pneumonia-detection-challenge.zip && cd ../..
```

檢查：

| 內容 | 應有數量 |
|---|---|
| `data/rsna/stage_2_train_images/` | 26,684 個 `.dcm` |
| `data/rsna/stage_2_test_images/` | 3,000 個 `.dcm` |
| `stage_2_train_labels.csv` | 30,227 列，欄位 `patientId,x,y,width,height,Target` |
| `stage_2_detailed_class_info.csv` | 30,227 列 |
| `stage_2_sample_submission.csv` | 3,000 列 |

```bash
ls data/rsna/stage_2_train_images | wc -l   # 26684
ls data/rsna/stage_2_test_images | wc -l    # 3000
```

## 4. Kaggle Notebook GPU

1. **驗證手機（只要做一次）**：頭像 → Settings → Phone Verification。沒驗證就不能用 GPU，也不能連網路。
2. **開 Notebook**：比賽頁面 → Code → New Notebook，資料會自動掛在
   `/kaggle/input/rsna-pneumonia-detection-challenge/`。
3. **開 GPU**：右側 Session options → Accelerator → **GPU T4 x2** → Turn on。
   要 `pip install` 或下載權重的話，Internet 也要打開。
4. **確認 GPU**：
   ```python
   import torch
   print(torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
   # 2 ['Tesla T4', 'Tesla T4']
   ```

注意事項：
- 每週 30 小時 GPU。session 開著就會扣時數，不用的時候按右上角 **Stop Session**。
- 開 GPU 後 CPU 會變少，`DataLoader(num_workers=2~4)` 就好。
- 單次 session 最多約 12 小時。`/kaggle/working/` 的檔案在 session 結束後就會消失。
- 長時間訓練用 **Save Version → Save & Run All (Commit)**：在背景跑完，輸出會保存下來。
- 寫程式和 EDA 先用 CPU session 做，要訓練時再開 GPU。
- 讀 DICOM 很慢：先轉成 256/512 的 png 存成 output，之後的實驗都讀 png。

## 5. 用 CLI 操作 Kaggle（本機寫 notebook，丟到 Kaggle 跑）

| 做什麼 | 指令 |
|---|---|
| 上傳 notebook 並執行 | `kaggle kernels push -p <資料夾>`（資料夾內要有 `kernel-metadata.json`） |
| 看執行狀態 | `kaggle kernels status <user>/<slug>` |
| 下載輸出 | `kaggle kernels output <user>/<slug> -p out/` |
| 提交 | `kaggle competitions submit -c rsna-pneumonia-detection-challenge -f submission.csv -m "說明"` |
| 看提交紀錄 | `kaggle competitions submissions -c rsna-pneumonia-detection-challenge` |

`kernel-metadata.json` 範例：

```json
{
  "id": "<kaggle帳號>/rsna-train",
  "title": "rsna-train",
  "code_file": "train.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "competition_sources": ["rsna-pneumonia-detection-challenge"]
}
```

## 6. 比賽重點

- 任務是 **object detection**：要輸出肺炎的 bounding box。
- 提交格式：`patientId,PredictionString`，PredictionString 是 `conf x y w h conf x y w h ...`。沒有框就留空。
- 評分：IoU 門檻 0.40–0.75（間隔 0.05）的平均 precision `TP/(TP+FP+FN)`，每張影像各算一次再平均。
- ⚠️ **沒有肺炎的影像只要輸出任何框，那張就是 0 分**。所以要先分類（有沒有肺炎），再偵測（畫框）。

## 7. 進度與待辦

- [x] 安裝 Kaggle CLI、登入、接受比賽規則
- [x] 下載並解壓資料到 `data/rsna/`
- [x] 確認 Kaggle GPU T4 x2 可以用
- [ ] 問助教（鄭茹云 m1461010@cgu.edu.tw）：可不可以用 pretrained model？分組、繳交內容、截止日期、評分方式？
- [ ] EDA：Target 比例、每人框數、疊框範例、框大小與位置分布
- [ ] 依 patientId 分層切 train/val
- [ ] 階段 A 分類器 + 階段 B 偵測器
- [ ] 在 val 上實作 mAP，調門檻
- [ ] 產生 submission.csv 並提交
