"""RSNA Pneumonia Detection 共用工具。

放在 .py 檔而不是 notebook 裡，是因為 Windows 上 DataLoader(num_workers>0)
與 multiprocessing 需要能被 import 的函式 / 類別。
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "rsna"
PNG_DIR = ROOT / "data" / "png512"
ORIG_SIZE = 1024
IOU_THRESHOLDS = np.arange(0.4, 0.76, 0.05)  # 0.40, 0.45, ..., 0.75


# ---------------------------------------------------------------------------
# 標籤
# ---------------------------------------------------------------------------
def load_labels(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """回傳每列一個框的 DataFrame，並附上細分類別 (class)。

    Target=0 的病人只有一列，x/y/width/height 為 NaN。
    """
    labels = pd.read_csv(raw_dir / "stage_2_train_labels.csv")
    classes = pd.read_csv(raw_dir / "stage_2_detailed_class_info.csv").drop_duplicates("patientId")
    return labels.merge(classes, on="patientId", how="left")


def boxes_by_patient(labels: pd.DataFrame) -> dict[str, np.ndarray]:
    """patientId -> (N, 4) 陣列，格式為 [x, y, w, h]，座標為原圖 1024 尺度。"""
    out = {}
    for pid, g in labels.groupby("patientId"):
        g = g.dropna(subset=["x"])
        out[pid] = g[["x", "y", "width", "height"]].to_numpy(dtype=np.float32)
    return out


def make_split(labels: pd.DataFrame, val_ratio: float = 0.15, seed: int = 42) -> pd.DataFrame:
    """以病人為單位切 train / val，依細分類別分層抽樣，確保兩邊比例一致。"""
    patients = labels.drop_duplicates("patientId")[["patientId", "Target", "class"]]
    train_ids, _ = train_test_split(
        patients.patientId, test_size=val_ratio, random_state=seed, stratify=patients["class"]
    )
    patients = patients.copy()
    patients["split"] = np.where(patients.patientId.isin(set(train_ids)), "train", "val")
    return patients.reset_index(drop=True)


# ---------------------------------------------------------------------------
# DICOM -> PNG
# ---------------------------------------------------------------------------
def read_dicom(path: Path) -> tuple[np.ndarray, dict]:
    ds = pydicom.dcmread(path)
    meta = {
        "patientId": path.stem,
        "age": int(ds.PatientAge) if str(ds.PatientAge).isdigit() else np.nan,
        "sex": ds.PatientSex,
        "view": ds.ViewPosition,
    }
    return ds.pixel_array, meta


def _convert_one(args: tuple[Path, Path, int]) -> dict:
    src, dst_dir, size = args
    img, meta = read_dicom(src)
    dst = dst_dir / f"{src.stem}.png"
    if not dst.exists():
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst), img)
    return meta


def convert_split(split: str, size: int = 512, workers: int = 8,
                  raw_dir: Path = RAW_DIR, out_dir: Path = PNG_DIR) -> pd.DataFrame:
    """把 stage_2_{split}_images 的 DICOM 縮放成 size x size 的 PNG。

    已存在的 PNG 會跳過，可以中斷後重跑。回傳並另存 DICOM 檔頭裡的 metadata。
    """
    src_dir = raw_dir / f"stage_2_{split}_images"
    dst_dir = out_dir / split
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(src_dir.glob("*.dcm"))
    jobs = [(f, dst_dir, size) for f in files]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        metas = list(tqdm(ex.map(_convert_one, jobs, chunksize=64), total=len(jobs), desc=split))
    meta = pd.DataFrame(metas)
    meta.to_csv(out_dir / f"{split}_meta.csv", index=False)
    return meta


# ---------------------------------------------------------------------------
# 評分（與 Kaggle 官方定義一致）
# ---------------------------------------------------------------------------
def iou_xywh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N,4), b: (M,4)，格式 [x,y,w,h]。回傳 (N,M) IoU 矩陣。"""
    ax1, ay1 = a[:, 0:1], a[:, 1:2]
    ax2, ay2 = ax1 + a[:, 2:3], ay1 + a[:, 3:4]
    bx1, by1 = b[:, 0], b[:, 1]
    bx2, by2 = bx1 + b[:, 2], by1 + b[:, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    union = a[:, 2:3] * a[:, 3:4] + b[:, 2] * b[:, 3] - inter
    return inter / np.maximum(union, 1e-9)


def score_image(gt: np.ndarray, pred: np.ndarray, scores: np.ndarray | None = None) -> float | None:
    """單張圖的分數。

    gt / pred: (N,4) [x,y,w,h]；scores: 預測框的信心值，用來決定配對順序。
    沒有真實框且沒有預測框時回傳 None（官方不把這種圖算進平均）。
    沒有真實框但有任何預測框時得 0 分。
    """
    gt = np.asarray(gt, dtype=np.float32).reshape(-1, 4)
    pred = np.asarray(pred, dtype=np.float32).reshape(-1, 4)
    if len(gt) == 0 and len(pred) == 0:
        return None
    if len(gt) == 0 or len(pred) == 0:
        return 0.0

    if scores is not None:
        pred = pred[np.argsort(-np.asarray(scores))]
    ious = iou_xywh(pred, gt)

    precisions = []
    for t in IOU_THRESHOLDS:
        matched = np.zeros(len(gt), dtype=bool)
        tp = 0
        for i in range(len(pred)):
            cand = np.where(~matched, ious[i], -1.0)
            j = int(cand.argmax())
            if cand[j] > t:
                matched[j] = True
                tp += 1
        fp = len(pred) - tp
        fn = len(gt) - tp
        precisions.append(tp / (tp + fp + fn))
    return float(np.mean(precisions))


def mean_score(gt_boxes: dict[str, np.ndarray], preds: dict[str, tuple[np.ndarray, np.ndarray]]) -> float:
    """整個資料集的分數。

    gt_boxes: patientId -> (N,4)
    preds: patientId -> (boxes (M,4), scores (M,))，沒預測的病人可省略。
    """
    vals = []
    empty = (np.zeros((0, 4), np.float32), np.zeros(0, np.float32))
    for pid, gt in gt_boxes.items():
        boxes, scores = preds.get(pid, empty)
        s = score_image(gt, boxes, scores)
        if s is not None:
            vals.append(s)
    return float(np.mean(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# 上傳檔
# ---------------------------------------------------------------------------
def to_prediction_string(boxes: np.ndarray, scores: np.ndarray) -> str:
    """boxes 為原圖 1024 尺度的 [x,y,w,h]。"""
    parts = []
    for (x, y, w, h), s in zip(boxes, scores):
        parts.append(f"{s:.4f} {x:.0f} {y:.0f} {w:.0f} {h:.0f}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# PyTorch Dataset（torchvision 偵測模型格式）
# ---------------------------------------------------------------------------
class RSNADetectionDataset(torch.utils.data.Dataset):
    """讀取 convert_split 產生的 PNG。

    回傳 (image, target)：
      image: float32 (3, H, W)，數值 0~1（灰階複製成 3 通道，方便接 torchvision 模型）
      target: {"boxes": (N,4) xyxy（已縮放到 PNG 尺度）, "labels": (N,) 全為 1, "patientId": str}
    Target=0 的圖 boxes 為空，torchvision 的 Faster R-CNN 可以直接吃。
    """

    def __init__(self, patient_ids, gt_boxes: dict[str, np.ndarray] | None,
                 img_dir: Path, size: int = 512, hflip_prob: float = 0.0):
        self.ids = list(patient_ids)
        self.gt = gt_boxes
        self.img_dir = Path(img_dir)
        self.scale = size / ORIG_SIZE
        self.size = size
        self.hflip_prob = hflip_prob

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        img = cv2.imread(str(self.img_dir / f"{pid}.png"), cv2.IMREAD_GRAYSCALE)
        img = torch.from_numpy(img).float().div(255).unsqueeze(0).repeat(3, 1, 1)

        xywh = self.gt.get(pid, np.zeros((0, 4), np.float32)) if self.gt is not None else np.zeros((0, 4), np.float32)
        boxes = torch.as_tensor(xywh, dtype=torch.float32).reshape(-1, 4) * self.scale
        boxes[:, 2:] += boxes[:, :2]  # xywh -> xyxy

        if self.hflip_prob > 0 and torch.rand(1).item() < self.hflip_prob:
            img = img.flip(-1)
            boxes[:, [0, 2]] = self.size - boxes[:, [2, 0]]

        target = {
            "boxes": boxes,
            "labels": torch.ones(len(boxes), dtype=torch.int64),
            "patientId": pid,
        }
        return img, target


def detection_collate(batch):
    return tuple(zip(*batch))
