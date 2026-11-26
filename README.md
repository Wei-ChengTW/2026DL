# RSNA Pneumonia Detection with Faster R-CNN

以 PyTorch Faster R-CNN（ResNet50-FPN）偵測 RSNA 胸部 X 光中的肺部混濁區域，並用本機網頁查看留出驗證集預測、醫師框標註與訓練紀錄。

這是影像研究與課程展示，不是臨床診斷工具。

## 環境

專案使用現有的 Conda 環境 `py3_10`。目前環境已有 PyTorch 2.8、torchvision 0.23、CUDA、Flask；讀取 DICOM 需要 `pydicom`。

```bash
conda run -n py3_10 python -m pip install -r requirements.txt
```

資料目錄預設為專案根目錄下的 `rsna-pneumonia-detection-challenge/`，內容需包括：

```text
stage_2_train_images/
stage_2_train_labels.csv
stage_2_test_images/
stage_2_sample_submission.csv
```

## 訓練

完整訓練預設使用 80/20 病人層級切分、8 個 epoch、batch size 2，並載入 torchvision 的 COCO 預訓練權重做遷移學習：

```bash
conda run -n py3_10 python train.py
```

若先想快速確認流程，可限制影像數與 epoch。這種快速執行的分數只代表該次子集訓練和子集驗證結果：

```bash
conda run -n py3_10 python train.py --limit-train 1000 --limit-val 300 --epochs 2
```

訓練產物會寫入 `outputs/`：

- `best_model.pth`：驗證 mAP 最佳的 Faster R-CNN 權重。
- `last_model.pth`：最後一個 epoch 的權重。
- `metrics.json`：切分大小、訓練紀錄、mAP@IoU 0.4 與網頁樣本清單。

若要調整訓練，常用參數包括 `--epochs`、`--batch-size`、`--workers`、`--learning-rate`、`--limit-train` 和 `--limit-val`。影像在模型輸入前縮放至短邊 512、長邊最多 768；標註座標保留原始像素尺度。

## 網頁

訓練完成後啟動本機網站：

```bash
conda run -n py3_10 python main.py
```

開啟 [http://127.0.0.1:7860](http://127.0.0.1:7860)。網頁可切換驗證影像、比較模型框與醫師標註、調整信心門檻、查看訓練紀錄，也可上傳 DICOM 做單張推論。上傳檔案只在本機記憶體中處理。

## Kaggle submission

建立完整 stage 2 測試集預測檔：

```bash
conda run -n py3_10 python submission.py
```

輸出 `outputs/submission.csv`，欄位為 `patientId,PredictionString`。`upload_submission.py` 會讀取 `~/.kaggle/access_token` 或 `KAGGLE_API_TOKEN` 環境變數，並等待官方評分；token 不會寫入專案或輸出：

```bash
conda run -n py3_10 python upload_submission.py \
  --file outputs/submission.csv \
  --message "Faster R-CNN ResNet50-FPN"
```

Kaggle 測試標籤未公開，因此 `train.py` 的 mAP 是從訓練資料保留出的驗證影像計算；只有 Kaggle 接受提交並完成評分時，才會有官方 leaderboard score。
