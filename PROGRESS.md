# RSNA 肺炎偵測：進度總結（2026-10-27）

## 目前最佳
- **Kaggle Private 0.19070**（Public 0.08134），檔案 `submission_ens_c0.03_t0.25.csv`
- 朋友的 Faster R-CNN baseline：Private 0.13139 → 高 0.059（約 +45%）
- 重現指令：`python ensemble.py --conf 0.03 --t-cls 0.25 --shrink 0.8`

## 流程
1. `prep.py`：DICOM → CLAHE → 512px PNG、YOLO labels、病人層級 StratifiedKFold(5)（`SIZE=768` 可產生其他解析度）
2. `train_det.py <fold>`：YOLO11m 偵測器，20 epochs，關掉 mosaic、旋轉、色彩擾動
3. `train_cls.py <fold>`：ConvNeXt-Tiny 三分類（Lung Opacity / Not Normal / Normal），把關哪些圖要輸出框
4. `ensemble.py`：5 個分類器機率取平均，5 個偵測器用 WBF（IoU 0.4）合併框
   - 分類機率 < t_cls → 不輸出框
   - 保留信心值 ≥ conf 的框；如果分類器判有病但沒有框過門檻，保留信心最高的那個
   - 框往中心縮到 shrink 倍
5. `predict.py`：單一 fold 的驗證和預測；`metric.py`：比賽指標（IoU 0.4–0.75 的平均）

## 分數演進（Private）
| 步驟 | Private |
|---|---|
| YOLO11m + 分類器（fold 0） | 0.12748 |
| 框縮小到 0.8 倍 | 0.14290 |
| 分類器門檻 t 0.5 → 0.15 | 0.17669 |
| 5-fold ensemble + 重新調參（t 0.25、conf 0.03） | **0.19070** |

主要貢獻：降低分類器門檻（+0.034）、縮框（+0.015）、5-fold ensemble（+0.014）。

## 關鍵觀察（報告可用）
- **測試集標註是 3 位放射科醫師的交集** → 框比較小、陽性比例比訓練集高。所以驗證集上最佳的參數（shrink 0.95、t 0.4）放到測試集都不對。
- shrink 和 t_cls 是看 Kaggle Private 分數調的 → 分數偏樂觀，報告要註明。
- NMS IoU 0.2 比 ultralytics 預設的 0.7 好很多（驗證集上）。
- 瓶頸在框的定位：分類完全正確（oracle）配上 YOLO 的框，也只有 0.369。

## 試過但沒用的（fold 0 驗證集，比賽指標）
| 偵測器 | val |
|---|---|
| **YOLO11m @512（採用）** | **0.2342** |
| YOLO11l @512 | 0.2214 |
| YOLO11m @768（batch 8） | 0.2205 |
| WBF 11m + 11l | 0.2380 |
| WBF 512 + 768 | 0.2372 |

大模型、高解析度在 20 個 epoch 內都比較差，合併也只多 0.003～0.004，訓練時間卻多 2～3 倍，所以沒有繼續。
其他：WBF IoU 0.5、shrink 0.75 或 0.85、conf 0.015 或 0.05 都沒有提升。

## 接下來可以做
- [ ] 寫報告（方法、分數演進、上面的觀察和失敗實驗）
- [ ] （可選）768 或 11l 多跑幾個 epoch（30+），看能不能追上 512
- [ ] （可選）偵測器 TTA（水平翻轉）
- [ ] 整理 repo：只留最佳的 submission、把 logs 移到 `runs/`

## 環境注意
- Windows：訓練程式要有 `if __name__ == "__main__"`；ultralytics 要傳絕對路徑的 `project=`
- 8GB VRAM：ConvNeXt batch 12、YOLO11m @768 batch 8
- 訓練時 GPU 會到 85～89°C 並降頻，筆電要墊高散熱
- 宿舍網路每天上限 50 GB
