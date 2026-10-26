# Leo 版本：Faster R-CNN baseline

RSNA Pneumonia Detection：在胸部 X 光上框出肺炎區域。

## 成績

| 版本 | 自切驗證集 | Kaggle Private | Kaggle Public |
|---|---|---|---|
| baseline（3 epoch） | 0.2285 | 0.13139 | 0.06349 |
| baseline + 框縮成 87.5% | 0.2027 | 0.15278 | 0.07142 |
| **full8 第 3 輪 + 框縮成 87.5%（目前最好）** | 0.2325（不縮框） | **0.16831** | 0.07936 |

![Kaggle 分數](figures/07_kaggle_score.png)（截圖是 baseline 那次）

正式排名看 Kaggle Private。驗證分數 > Private > Public 的順序是合理的：驗證集和訓練資料是同一種標法，信心門檻也是在驗證集上挑的；Public 只用了測試集的一小部分。

### 縮框實驗（`shrink_boxes.py`、`shrink_results.csv`）

參考第 1 名（[i-pan/kaggle-rsna18](https://github.com/i-pan/kaggle-rsna18)，寬高縮成 87.5%）和第 3 名（[pmcheng/rsna-pneumonia](https://github.com/pmcheng/rsna-pneumonia)，每邊縮 17%）的做法，以中心點為準把預測框縮小。

| 縮放比例 | 0.800 | 0.850 | 0.875 | 0.900 | 0.950 | 1.000 |
|---|---|---|---|---|---|---|
| 驗證分數 | 0.1675 | 0.1935 | 0.2027 | 0.2120 | 0.2238 | 0.2285 |

在自己的驗證集上越縮越差，但用第 1 名的 87.5% 上傳後 Kaggle Private 從 0.131 升到 0.153。兩邊結論相反，推測測試集的框標得比訓練資料緊，這也可能是驗證分數和 Kaggle 分數落差大的原因之一。87.5% 是沿用第 1 名的數字，沒有拿 Kaggle 分數去挑比例。

### 訓練輪數實驗（`full8_log.txt`、`full8b_log.txt`）

同一支 `train_baseline.py` 排 8 輪跑了兩次（full8 跑到第 4 輪時電腦記憶體不足被中斷，所以又重跑一次完整 8 輪的 full8b）：

| 輪 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| full8 驗證分數 | 0.2175 | 0.2179 | **0.2325** | 0.2244 | – | – | – | – |
| full8b 驗證分數 | 0.2194 | 0.2186 | **0.2280** | 0.2279 | 0.2201 | 0.2272 | 0.2232 | 0.2210 |
| full8b 訓練 loss | 0.0844 | 0.0778 | 0.0750 | 0.0717 | 0.0675 | 0.0640 | 0.0603 | 0.0573 |

兩次都在第 3 輪最好。之後訓練 loss 一路下降、驗證分數卻不再上升，是過擬合：模型越來越會做訓練資料，但沒有變得更會看新的片子，所以多訓練幾輪沒有幫助。兩次第 3 輪差 0.0045，來自隨機性（資料順序、翻轉）。最後採用 full8 第 3 輪的模型。

### 有無肺炎分類器（`classify.py`、`combine.py`、`cls_log.txt`、`combine_results.csv`）

參考第 1 名的做法：另外訓練一個只判斷「整張片子有沒有肺炎」的分類器（ImageNet 預訓練 ResNet50、5 輪、AdamW lr 1e-4），用來過濾健康片子上的誤判框（健康片子畫任何框，那張就是 0 分）。

| 輪 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| AUC | 0.8717 | 0.8749 | **0.8831** | 0.8710 | 0.8696 |
| 猜對率 | 83.7% | 84.8% | **84.8%** | 83.8% | 83.3% |

和 full8 的框合併，三種方式在驗證集上比較：

| 方式 | 說明 | 驗證分數 |
|---|---|---|
| none | 不用分類器 | **0.2325** |
| gate g=0.1 | 分類器機率 < 0.1 的片子整張不畫框 | 0.2323 |
| gate g=0.5 | 同上，門檻 0.5 | 0.2015 |
| gate g=0.9 | 同上，門檻 0.9 | 0.0770 |
| geo | 框分數改成 sqrt(框分數 x 分類器機率) | 0.1891 |

分類器沒有幫助，所以沒有上傳。原因是偵測模型本身已經在判斷有沒有肺炎：用它每張圖最高的框分數當「有肺炎的分數」，AUC 是 0.8931、猜對率 84.6%（全猜沒肺炎是 77.5%），比分類器的 0.8831 還高。用比較弱的判斷去否決比較強的判斷，只會把畫對的框也刪掉。第 1 名有效是因為他們的分類器用了多個大模型和更高解析度，明顯強過偵測器；這在 8GB 的筆電顯示卡上做不到。

## 做法

| 步驟 | 內容 | 檔案 |
|---|---|---|
| 前處理 | DICOM（1024x1024）縮成 512x512 PNG；依三種類別（正常 / 不正常但非肺炎 / 肺炎）分層切 80/20 | `preprocess.py` |
| 模型 | torchvision `fasterrcnn_resnet50_fpn`，COCO 預訓練，最後分類層換成 2 類（背景、肺炎） | `train_baseline.py` |
| 資料擴增 | 50% 機率左右翻轉（框座標跟著翻） | `train_baseline.py` |
| 訓練 | SGD momentum 0.9、lr 0.01、weight decay 1e-4、前 500 步 warmup + cosine、batch 8、混合精度 | `train_baseline.py` |
| 評分 | 照比賽規則：IoU 0.40–0.75 共 8 個門檻的平均 TP/(TP+FP+FN)；每個 epoch 在驗證集上挑最佳信心門檻 | `train_baseline.py` |
| 預測 / 出圖 | 用最佳模型預測 3,000 張測試圖，座標換回 1024 尺度，產生 submission.csv 與報告圖 | `finish.py`、`make_process_figures.py` |
| 提交檔（縮框） | 指定訓練結果和縮框比例產生提交檔 | `make_submission.py` |

第一版 baseline 原本排 8 個 epoch，因時間只跑 3 個（每個約 25 分鐘），見 `train_log.txt`（之後的 8 輪實驗見上面「訓練輪數實驗」）：

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
python train_baseline.py --epochs 8 --out runs/full8
python make_submission.py runs/full8 0.875 # 目前最好的提交檔（Private 0.16831）
python classify.py --epochs 5             # 有無肺炎分類器，每輪約 10 分鐘
python combine.py runs/full8 runs/cls 0.875  # 分類器 + 框的三種合併方式比較
```

長時間訓練建議在自己開的終端機跑，並且接上電源（筆電用電池時顯示卡會降速，分類器第 2 輪因此從 10 分鐘拉長到 42 分鐘）。

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
