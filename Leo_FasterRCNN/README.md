# Leo 版本：Faster R-CNN baseline

RSNA Pneumonia Detection：在胸部 X 光上框出肺炎區域。

## 成績

| 版本 | 自切驗證集 | Kaggle Private | Kaggle Public |
|---|---|---|---|
| baseline（3 epoch） | 0.2285 | 0.13139 | 0.06349 |
| baseline + 框縮成 87.5% | 0.2027 | **0.15278** | 0.07142 |

![Kaggle 分數](figures/07_kaggle_score.png)（截圖是 baseline 那次）

### 縮框實驗（`shrink_boxes.py`、`shrink_results.csv`）

參考第 1 名（[i-pan/kaggle-rsna18](https://github.com/i-pan/kaggle-rsna18)，寬高縮成 87.5%）和第 3 名（[pmcheng/rsna-pneumonia](https://github.com/pmcheng/rsna-pneumonia)，每邊縮 17%）的做法，以中心點為準把預測框縮小。

| 縮放比例 | 0.800 | 0.850 | 0.875 | 0.900 | 0.950 | 1.000 |
|---|---|---|---|---|---|---|
| 驗證分數 | 0.1675 | 0.1935 | 0.2027 | 0.2120 | 0.2238 | 0.2285 |

在自己的驗證集上越縮越差，但用第 1 名的 87.5% 上傳後 Kaggle Private 從 0.131 升到 0.153。兩邊結論相反，推測測試集的框標得比訓練資料緊，這也可能是驗證分數和 Kaggle 分數落差大的原因之一。87.5% 是沿用第 1 名的數字，沒有拿 Kaggle 分數去挑比例。

## 做法

| 步驟 | 內容 | 檔案 |
|---|---|---|
| 前處理 | DICOM（1024x1024）縮成 512x512 PNG；依三種類別（正常 / 不正常但非肺炎 / 肺炎）分層切 80/20 | `preprocess.py` |
| 模型 | torchvision `fasterrcnn_resnet50_fpn`，COCO 預訓練，最後分類層換成 2 類（背景、肺炎） | `train_baseline.py` |
| 資料擴增 | 50% 機率左右翻轉（框座標跟著翻） | `train_baseline.py` |
| 訓練 | SGD momentum 0.9、lr 0.01、weight decay 1e-4、前 500 步 warmup + cosine、batch 8、混合精度 | `train_baseline.py` |
| 評分 | 照比賽規則：IoU 0.40–0.75 共 8 個門檻的平均 TP/(TP+FP+FN)；每個 epoch 在驗證集上挑最佳信心門檻 | `train_baseline.py` |
| 預測 / 出圖 | 用最佳模型預測 3,000 張測試圖，座標換回 1024 尺度，產生 submission.csv 與報告圖 | `finish.py`、`make_process_figures.py` |

原本排 8 個 epoch，因時間只跑 3 個（每個約 25 分鐘），驗證分數仍在上升，見 `train_log.txt`：

| epoch | 訓練 loss | 驗證分數 | 最佳信心門檻 |
|---|---|---|---|
| 1 | 0.0849 | 0.2146 | 0.60 |
| 2 | 0.0780 | 0.2235 | 0.70 |
| 3 | 0.0751 | 0.2285 | 0.75 |

## 怎麼跑

環境：Windows 11、RTX 4060 Laptop（8GB）、Python 3.12，套件見 `requirements.txt`。

```
kaggle competitions download -c rsna-pneumonia-detection-challenge -p data
tar -xf data/rsna-pneumonia-detection-challenge.zip -C data
python preprocess.py                      # 約 2 分鐘
python train_baseline.py --epochs 8       # 每個 epoch 約 25 分鐘
python finish.py                          # 產生 runs/baseline/submission.csv 與 figures/05、06
python shrink_boxes.py                    # 縮框實驗：驗證集上試 80%–100%
```

比賽資料依 Kaggle 規則不能再散布，模型權重（158MB）超過 GitHub 單檔上限，所以這兩樣都沒有放上來，照上面的步驟可以重新產生。

## 報告用圖

| 圖 | 用途 |
|---|---|
| `figures/01_class_distribution.png` | 資料介紹：77.5% 的片子沒有肺炎 |
| `figures/02_box_statistics.png` | 資料介紹：每人框數、框的寬高 |
| `figures/03_samples.png` | 資料介紹：三種類別的 X 光範例 |
| `figures/04_preprocess.png` | 前處理：1024 → 512，框跟著縮 |
| `figures/05_training_curve.png` | 訓練過程 |
| `figures/06_predictions.png` | 錯誤分析：畫準的、漏掉的、健康卻畫框的 |
| `figures/07_kaggle_score.png` | 實驗結果 |

## AI 使用聲明

程式碼與本 README 由 Claude（Anthropic，透過 Claude Code）協助撰寫，每個 `.py` 檔第一行有標註。
