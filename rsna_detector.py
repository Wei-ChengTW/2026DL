"""Data and model helpers for the RSNA pneumonia detection project."""

from __future__ import annotations

import csv
import io
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pydicom
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    RetinaNet_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
    retinanet_resnet50_fpn,
)
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "rsna-pneumonia-detection-challenge"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs"
NUM_CLASSES = 2  # background and lung opacity
ARCHITECTURE_RESNET50_FPN = "fasterrcnn_resnet50_fpn"
ARCHITECTURE_RESNET50_FPN_V2 = "fasterrcnn_resnet50_fpn_v2"
ARCHITECTURE_RETINANET_RESNET50_FPN = "retinanet_resnet50_fpn"


def normalize_architecture(architecture: str) -> str:
    aliases = {
        "Faster R-CNN ResNet50-FPN": ARCHITECTURE_RESNET50_FPN,
        "Faster R-CNN ResNet50-FPN v2": ARCHITECTURE_RESNET50_FPN_V2,
        "RetinaNet ResNet50-FPN": ARCHITECTURE_RETINANET_RESNET50_FPN,
    }
    architecture = aliases.get(architecture, architecture)
    if architecture not in (
        ARCHITECTURE_RESNET50_FPN,
        ARCHITECTURE_RESNET50_FPN_V2,
        ARCHITECTURE_RETINANET_RESNET50_FPN,
    ):
        raise ValueError(f"Unsupported detector architecture: {architecture}")
    return architecture


def read_annotations(csv_path: str | Path) -> tuple[dict[str, list[list[float]]], list[str]]:
    """Return positive boxes and all patient IDs from an RSNA labels CSV."""
    boxes_by_patient: dict[str, list[list[float]]] = {}
    patient_ids: list[str] = []
    seen: set[str] = set()
    with Path(csv_path).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            patient_id = row["patientId"]
            if patient_id not in seen:
                patient_ids.append(patient_id)
                seen.add(patient_id)
            if row.get("Target") != "1":
                continue
            try:
                x = float(row["x"])
                y = float(row["y"])
                width = float(row["width"])
                height = float(row["height"])
            except (TypeError, ValueError, KeyError):
                continue
            if width > 0 and height > 0:
                boxes_by_patient.setdefault(patient_id, []).append(
                    [x, y, x + width, y + height]
                )
    return boxes_by_patient, patient_ids


def dicom_to_grayscale(source: str | Path | bytes | bytearray) -> np.ndarray:
    """Decode a DICOM image and apply a stable per-image 8-bit display window."""
    if isinstance(source, (bytes, bytearray)):
        dataset = pydicom.dcmread(io.BytesIO(source))
    else:
        dataset = pydicom.dcmread(str(source))

    pixels = np.asarray(dataset.pixel_array)
    if pixels.ndim == 3 and pixels.shape[0] == 1:
        pixels = pixels[0]
    if pixels.ndim != 2:
        raise ValueError(f"Expected a single 2D DICOM image, received {pixels.shape}.")

    pixels = pixels.astype(np.float32, copy=False)
    pixels = pixels * float(getattr(dataset, "RescaleSlope", 1.0))
    pixels = pixels + float(getattr(dataset, "RescaleIntercept", 0.0))
    pixels = np.nan_to_num(pixels, copy=False)
    if str(getattr(dataset, "PhotometricInterpretation", "MONOCHROME2")).upper() == "MONOCHROME1":
        pixels = np.nanmax(pixels) - pixels

    low, high = np.percentile(pixels, (0.5, 99.5))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(pixels))
        high = float(np.max(pixels))
    if high <= low:
        return np.zeros(pixels.shape, dtype=np.uint8)
    scaled = (np.clip(pixels, low, high) - low) * (255.0 / (high - low))
    return np.asarray(scaled, dtype=np.uint8)


def grayscale_to_tensor(grayscale: np.ndarray) -> torch.Tensor:
    """Convert an 8-bit grayscale image to the three-channel input expected by torchvision."""
    rgb = np.repeat(grayscale[:, :, None], 3, axis=2)
    rgb = np.ascontiguousarray(rgb)
    # Use the buffer protocol instead of torch.from_numpy: the current Conda
    # environment has a NumPy/PyTorch bridge mismatch despite compatible
    # versions, while ndarray's contiguous byte buffer remains interoperable.
    tensor = torch.frombuffer(memoryview(rgb), dtype=torch.uint8).view(*rgb.shape)
    return tensor.permute(2, 0, 1).float().div_(255.0)


class RSNADetectionDataset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        patient_ids: Iterable[str],
        boxes_by_patient: dict[str, list[list[float]]],
        augment: bool = False,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.patient_ids = list(patient_ids)
        self.boxes_by_patient = boxes_by_patient
        self.augment = augment

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, index: int):
        patient_id = self.patient_ids[index]
        path = self.image_dir / f"{patient_id}.dcm"
        grayscale = dicom_to_grayscale(path)
        image_height, image_width = grayscale.shape

        boxes = np.asarray(self.boxes_by_patient.get(patient_id, []), dtype=np.float32)
        if boxes.size == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
        else:
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, image_width)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, image_height)
            valid = (boxes[:, 2] - boxes[:, 0] > 1) & (boxes[:, 3] - boxes[:, 1] > 1)
            boxes = boxes[valid]

        if self.augment and random.random() < 0.5:
            grayscale = np.ascontiguousarray(grayscale[:, ::-1])
            if len(boxes):
                old_x1 = boxes[:, 0].copy()
                boxes[:, 0] = image_width - boxes[:, 2]
                boxes[:, 2] = image_width - old_x1

        tensor = grayscale_to_tensor(grayscale)
        box_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        target = {
            "boxes": box_tensor,
            "labels": torch.ones((len(box_tensor),), dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": (box_tensor[:, 2] - box_tensor[:, 0])
            * (box_tensor[:, 3] - box_tensor[:, 1]),
            "iscrowd": torch.zeros((len(box_tensor),), dtype=torch.int64),
        }
        return tensor, target, patient_id


def detection_collate(batch):
    images, targets, patient_ids = zip(*batch)
    return list(images), list(targets), list(patient_ids)


def build_model(
    pretrained: bool = False,
    min_size: int = 512,
    max_size: int = 768,
    detections_per_image: int = 100,
    box_score_thresh: float = 0.0,
    architecture: str = ARCHITECTURE_RESNET50_FPN,
) -> torch.nn.Module:
    architecture = normalize_architecture(architecture)
    if architecture == ARCHITECTURE_RETINANET_RESNET50_FPN:
        # RetinaNet's COCO head has 91 outputs and cannot be loaded directly
        # after changing the number of classes.  Use the ImageNet backbone
        # weights and create the two-output (background + opacity) head here.
        model = retinanet_resnet50_fpn(
            weights=None,
            weights_backbone=ResNet50_Weights.DEFAULT if pretrained else None,
            num_classes=NUM_CLASSES,
            min_size=min_size,
            max_size=max_size,
            detections_per_img=detections_per_image,
            score_thresh=box_score_thresh,
        )
        return model
    if architecture == ARCHITECTURE_RESNET50_FPN_V2:
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
        builder = fasterrcnn_resnet50_fpn_v2
    else:
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        builder = fasterrcnn_resnet50_fpn
    model = builder(
        weights=weights,
        min_size=min_size,
        max_size=max_size,
        box_detections_per_img=detections_per_image,
        box_score_thresh=box_score_thresh,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    return model


def load_trained_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint.get("model_config", {})
    model = build_model(
        pretrained=False,
        min_size=int(config.get("min_size", 512)),
        max_size=int(config.get("max_size", 768)),
        detections_per_image=int(config.get("detections_per_image", 20)),
        box_score_thresh=float(config.get("box_score_thresh", 0.05)),
        architecture=config.get("architecture", ARCHITECTURE_RESNET50_FPN),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def render_grayscale_png(grayscale: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(grayscale, mode="L").save(output, format="PNG", optimize=True)
    return output.getvalue()
