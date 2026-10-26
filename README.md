# RSNA Pneumonia Detection with a PyTorch Ensemble

以 PyTorch Faster R-CNN、RetinaNet 與 EfficientNet-B0 組成 RSNA 胸部 X 光肺部混濁偵測集成，並用本機網頁查看留出驗證集預測、醫師框標註與訓練紀錄。

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

單一 detector 訓練預設使用 80/20 病人層級切分、8 個 epoch、batch size 2，並載入 torchvision 的預訓練權重做遷移學習。可用 `--architecture retinanet_resnet50_fpn` 切換 RetinaNet：

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
- `metrics.json`：切分大小、訓練紀錄、RSNA reference-style mAP_IoU 與網頁樣本清單。

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

Kaggle 測試標籤未公開，因此 `train.py` 的分數是從訓練資料保留出的驗證影像計算。現行本地 scorer 採用公開 RSNA 參考實作：每張影像計算 `TP/(TP+FN+FP)`，對 IoU 0.40 到 0.75 的八個門檻取平均；沒有真值也沒有預測的影像不納入平均。只有 Kaggle 接受提交並完成評分時，才會有官方 leaderboard score。

## 一鍵集成訓練與上傳

`run_ensemble_kaggle.py` 會依序訓練 Faster R-CNN 和 RetinaNet，接著訓練 EfficientNet-B0 分類器；各模型預測會以 WBF 融合，最後自動建立 submission 並呼叫 Kaggle API。模型不會同時常駐 GPU，適合 16 GB RTX A4000：

```bash
conda run -n py3_10 python run_ensemble_kaggle.py \
  --data-dir /data/2026DL/rsna-pneumonia-detection-challenge
```

若要在背景執行並保留 log：

```bash
nohup conda run -n py3_10 python -u run_ensemble_kaggle.py \
  --data-dir /data/2026DL/rsna-pneumonia-detection-challenge \
  > outputs/ensemble_run.log 2>&1 < /dev/null &
```

訓練完成後會在 `outputs/ensemble/` 保存 checkpoints、預測快取、`submission.csv` 和 `manifest.json`，並使用 `~/.kaggle/access_token` 或 `KAGGLE_API_TOKEN` 上傳。若只想產生檔案而不上傳，加入 `--skip-upload`。

WBF 會以模型權重平均融合框的位置與信心；預設信心門檻為 `0.1`，與兩個 detector 在留出驗證集選出的門檻一致。若只修改融合參數，可重跑同一指令，既有 checkpoint 與 prediction cache 會被重用，不會重新訓練。
