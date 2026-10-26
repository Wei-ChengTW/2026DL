"""Train a sequential detector ensemble, fuse predictions, and submit to Kaggle.

The default run is designed for a 16 GB RTX A4000:

* Faster R-CNN ResNet50-FPN and RetinaNet ResNet50-FPN are trained one at a
  time, so their weights never occupy the GPU simultaneously.
* An EfficientNet-B0 image classifier is trained after the detectors and is
  used only to calibrate the fused detection confidence.
* Test predictions are combined with weighted boxes fusion (WBF), written in
  RSNA submission format, and uploaded with the existing Kaggle API helper.

The process is restartable. Existing best checkpoints and prediction caches
are reused unless ``--force-retrain`` is supplied.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from rsna_detector import (
    ARCHITECTURE_RESNET50_FPN,
    ARCHITECTURE_RETINANET_RESNET50_FPN,
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    dicom_to_grayscale,
    grayscale_to_tensor,
    load_trained_model,
    normalize_architecture,
    read_annotations,
)
from train import split_patients
from rsna_metrics import METRIC_NAME, RSNA_IOU_THRESHOLDS


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_COMPETITION = "rsna-pneumonia-detection-challenge"
DEFAULT_DETECTORS = (
    ARCHITECTURE_RESNET50_FPN,
    ARCHITECTURE_RETINANET_RESNET50_FPN,
)


def default_data_dir() -> Path:
    configured = os.environ.get("RSNA_DATA_DIR", "").strip()
    candidates = [Path(configured)] if configured else []
    candidates.extend(
        [
            Path("/data/2026DL/rsna-pneumonia-detection-challenge"),
            DEFAULT_DATA_DIR,
        ]
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "ensemble")
    parser.add_argument(
        "--detectors",
        default=",".join(DEFAULT_DETECTORS),
        help="Comma-separated detector architecture names.",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--classifier-epochs", type=int, default=5)
    parser.add_argument("--detector-batch-size", type=int, default=1)
    parser.add_argument("--classifier-batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--classifier-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0025)
    parser.add_argument("--classifier-learning-rate", type=float, default=0.0003)
    parser.add_argument("--min-size", type=int, default=512)
    parser.add_argument("--max-size", type=int, default=768)
    parser.add_argument("--classifier-image-size", type=int, default=512)
    parser.add_argument("--detections-per-image", type=int, default=100)
    parser.add_argument("--box-score-threshold", type=float, default=0.0)
    # Both detector validation sweeps selected 0.1 as the best operating point.
    # Keep the ensemble default aligned with that local threshold so a future
    # cache-only rerun does not reintroduce low-confidence false positives.
    parser.add_argument("--score-threshold", type=float, default=0.1)
    parser.add_argument("--fusion-iou", type=float, default=0.55)
    parser.add_argument(
        "--detector-weights",
        default=None,
        help="Optional comma-separated weights matching --detectors; defaults to equal weights.",
    )
    parser.add_argument(
        "--classifier-alpha",
        type=float,
        default=0.5,
        help="Confidence calibration strength: final score *= (1-alpha)+alpha*P(pneumonia).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--competition", default=DEFAULT_COMPETITION)
    parser.add_argument("--submission-message", default="Faster R-CNN + RetinaNet + EfficientNet-B0 WBF")
    parser.add_argument("--poll-seconds", type=int, default=900)
    parser.add_argument("--poll-interval", type=int, default=20)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--skip-classifier", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not download/use ImageNet or COCO initialization weights.",
    )
    return parser.parse_args()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_data_dir(data_dir: Path) -> None:
    required = (
        data_dir / "stage_2_train_images",
        data_dir / "stage_2_train_labels.csv",
        data_dir / "stage_2_test_images",
        data_dir / "stage_2_sample_submission.csv",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing RSNA data files:\n" + "\n".join(missing))


def run_subprocess(command: list[str], label: str) -> None:
    print(f"[{label}] {' '.join(str(part) for part in command)}", flush=True)
    environment = os.environ.copy()
    environment.setdefault("PYTHONUNBUFFERED", "1")
    subprocess.run(command, cwd=PROJECT_DIR, env=environment, check=True)


class RSNAClassificationDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        patient_ids: Iterable[str],
        positive_ids: set[str],
        image_size: int,
        augment: bool = False,
    ) -> None:
        self.image_dir = image_dir
        self.patient_ids = list(patient_ids)
        self.positive_ids = positive_ids
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        patient_id = self.patient_ids[index]
        image = grayscale_to_tensor(dicom_to_grayscale(self.image_dir / f"{patient_id}.dcm"))
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        if self.augment and random.random() < 0.5:
            image = image.flip(-1)
        image = (image - image.new_tensor((0.485, 0.456, 0.406))[:, None, None]) / image.new_tensor(
            (0.229, 0.224, 0.225)
        )[:, None, None]
        label = torch.tensor(float(patient_id in self.positive_ids), dtype=torch.float32)
        return image, label


class RSNATestDataset(Dataset):
    def __init__(self, image_dir: Path, patient_ids: list[str]) -> None:
        self.image_dir = image_dir
        self.patient_ids = patient_ids

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        patient_id = self.patient_ids[index]
        image = grayscale_to_tensor(dicom_to_grayscale(self.image_dir / f"{patient_id}.dcm"))
        return image, patient_id


def classification_collate(batch):
    images, labels = zip(*batch)
    return torch.stack(images), torch.stack(labels)


def detector_collate(batch):
    images, patient_ids = zip(*batch)
    return list(images), list(patient_ids)


def build_classifier(pretrained: bool = True) -> torch.nn.Module:
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)
    return model


def train_classifier(args: argparse.Namespace, patient_ids: list[str], boxes_by_patient: dict[str, list]) -> Path | None:
    if args.skip_classifier:
        print("[classifier] skipped by --skip-classifier", flush=True)
        return None
    checkpoint_path = args.output_dir / "efficientnet_b0_classifier.pth"
    if checkpoint_path.is_file() and not args.force_retrain:
        print(f"[classifier] reusing {checkpoint_path}", flush=True)
        return checkpoint_path

    train_ids, val_ids = split_patients(patient_ids, boxes_by_patient, args.val_fraction, args.seed)
    if args.limit_train:
        train_ids = train_ids[: args.limit_train]
    if args.limit_val:
        val_ids = val_ids[: args.limit_val]
    positive_ids = {patient_id for patient_id, boxes in boxes_by_patient.items() if boxes}
    train_dataset = RSNAClassificationDataset(
        args.data_dir / "stage_2_train_images",
        train_ids,
        positive_ids,
        args.classifier_image_size,
        augment=True,
    )
    val_dataset = RSNAClassificationDataset(
        args.data_dir / "stage_2_train_images",
        val_ids,
        positive_ids,
        args.classifier_image_size,
        augment=False,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loader_kwargs = {
        "num_workers": args.classifier_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.classifier_workers > 0,
        "collate_fn": classification_collate,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.classifier_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_kwargs,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.classifier_batch_size, shuffle=False, **loader_kwargs)
    model = build_classifier(pretrained=not args.no_pretrained).to(device)
    positive_count = sum(patient_id in positive_ids for patient_id in train_ids)
    negative_count = max(1, len(train_ids) - positive_count)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negative_count / max(1, positive_count)], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.classifier_learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.classifier_epochs))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_val_loss = float("inf")
    history = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[classifier] train={len(train_ids):,}, validation={len(val_ids):,}, "
        f"positive_train={positive_count:,}, device={device}",
        flush=True,
    )
    for epoch in range(1, args.classifier_epochs + 1):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = loss_fn(model(images).squeeze(1), labels)
            else:
                loss = loss_fn(model(images).squeeze(1), labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite classifier loss: {loss.item()}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.detach().cpu())

        model.eval()
        val_loss = 0.0
        correct = 0
        count = 0
        with torch.inference_mode():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model(images).squeeze(1)
                        loss = loss_fn(logits, labels)
                else:
                    logits = model(images).squeeze(1)
                    loss = loss_fn(logits, labels)
                val_loss += float(loss.detach().cpu())
                correct += int(((torch.sigmoid(logits) >= 0.5) == (labels >= 0.5)).sum().cpu())
                count += labels.numel()
        train_loss /= max(1, len(train_loader))
        val_loss /= max(1, len(val_loader))
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "validation_accuracy": correct / max(1, count),
        }
        history.append(record)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": {"architecture": "efficientnet_b0", "image_size": args.classifier_image_size},
            "epoch": epoch,
            "validation_loss": val_loss,
            "history": history,
        }
        torch.save(checkpoint, args.output_dir / "efficientnet_b0_classifier.last.pth")
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, checkpoint_path)
        print(
            f"[classifier] epoch {epoch}/{args.classifier_epochs} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={correct / max(1, count):.4f}",
            flush=True,
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return checkpoint_path


def train_detectors(args: argparse.Namespace, architectures: list[str]) -> dict[str, Path]:
    checkpoints: dict[str, Path] = {}
    for architecture in architectures:
        detector_dir = args.output_dir / architecture
        checkpoint = detector_dir / "best_model.pth"
        if checkpoint.is_file() and not args.force_retrain:
            print(f"[{architecture}] reusing {checkpoint}", flush=True)
            checkpoints[architecture] = checkpoint
            continue
        command = [
            sys.executable,
            str(PROJECT_DIR / "train.py"),
            "--data-dir",
            str(args.data_dir),
            "--output-dir",
            str(detector_dir),
            "--architecture",
            architecture,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.detector_batch_size),
            "--workers",
            str(args.workers),
            "--learning-rate",
            str(args.learning_rate),
            "--min-size",
            str(args.min_size),
            "--max-size",
            str(args.max_size),
            "--detections-per-image",
            str(args.detections_per_image),
            "--box-score-threshold",
            str(args.box_score_threshold),
            "--seed",
            str(args.seed),
            "--val-fraction",
            str(args.val_fraction),
        ]
        if args.limit_train:
            command.extend(["--limit-train", str(args.limit_train)])
        if args.limit_val:
            command.extend(["--limit-val", str(args.limit_val)])
        if args.no_pretrained:
            command.append("--scratch")
        run_subprocess(command, architecture)
        if not checkpoint.is_file():
            raise RuntimeError(f"Detector finished without creating {checkpoint}")
        checkpoints[architecture] = checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return checkpoints


def load_test_ids(data_dir: Path, limit: int | None) -> list[str]:
    with (data_dir / "stage_2_sample_submission.csv").open("r", newline="", encoding="utf-8") as handle:
        patient_ids = [row["patientId"] for row in csv.DictReader(handle)]
    missing = [
        patient_id
        for patient_id in patient_ids
        if not (data_dir / "stage_2_test_images" / f"{patient_id}.dcm").is_file()
    ]
    if missing:
        raise SystemExit(f"{len(missing)} test DICOM files are missing; first missing ID: {missing[0]}")
    return patient_ids[:limit] if limit else patient_ids


def predict_detector(
    checkpoint_path: Path,
    data_dir: Path,
    patient_ids: list[str],
    batch_size: int,
    workers: int,
    cache_path: Path,
    force: bool,
) -> dict[str, list[dict]]:
    if cache_path.is_file() and not force:
        print(f"[inference] reusing {cache_path}", flush=True)
        return load_json(cache_path)  # type: ignore[return-value]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_trained_model(checkpoint_path, device)
    architecture = normalize_architecture(checkpoint.get("model_config", {}).get("architecture", ""))
    dataset = RSNATestDataset(data_dir / "stage_2_test_images", patient_ids)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        collate_fn=detector_collate,
    )
    predictions: dict[str, list[dict]] = {}
    started = time.time()
    model.eval()
    with torch.inference_mode():
        for completed, (images, ids) in enumerate(loader, start=1):
            images = [image.to(device, non_blocking=True) for image in images]
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(images)
            else:
                outputs = model(images)
            for patient_id, output in zip(ids, outputs):
                boxes = output["boxes"].detach().cpu().tolist()
                scores = output["scores"].detach().cpu().tolist()
                labels = output.get("labels")
                if architecture == ARCHITECTURE_RETINANET_RESNET50_FPN and labels is not None:
                    keep = [int(label) == 1 for label in labels.detach().cpu().tolist()]
                    boxes = [box for box, include in zip(boxes, keep) if include]
                    scores = [score for score, include in zip(scores, keep) if include]
                predictions[patient_id] = [
                    {"box": [float(value) for value in box], "score": float(score)}
                    for box, score in zip(boxes, scores)
                    if float(score) >= 0.001
                ]
            if completed % 100 == 0 or completed == len(loader):
                print(
                    f"[inference] {checkpoint_path.parent.name} {len(predictions):,}/{len(dataset):,} "
                    f"elapsed={time.time() - started:.0f}s",
                    flush=True,
                )
    save_json(cache_path, predictions)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions


def predict_classifier(
    checkpoint_path: Path | None,
    args: argparse.Namespace,
    patient_ids: list[str],
    cache_path: Path,
) -> dict[str, float]:
    if checkpoint_path is None:
        return {patient_id: 1.0 for patient_id in patient_ids}
    if cache_path.is_file() and not args.force_retrain:
        print(f"[classifier inference] reusing {cache_path}", flush=True)
        return load_json(cache_path)  # type: ignore[return-value]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    image_size = int(checkpoint.get("model_config", {}).get("image_size", args.classifier_image_size))
    model = build_classifier(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    dataset = RSNATestDataset(args.data_dir / "stage_2_test_images", patient_ids)

    class ResizeNormalize(Dataset):
        def __init__(self, source: Dataset, size: int) -> None:
            self.source = source
            self.size = size

        def __len__(self):
            return len(self.source)

        def __getitem__(self, index):
            image, patient_id = self.source[index]
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0), size=(self.size, self.size), mode="bilinear", align_corners=False
            ).squeeze(0)
            image = (image - image.new_tensor((0.485, 0.456, 0.406))[:, None, None]) / image.new_tensor(
                (0.229, 0.224, 0.225)
            )[:, None, None]
            return image, patient_id

    def collate(batch):
        images, ids = zip(*batch)
        return torch.stack(images), list(ids)

    loader = DataLoader(
        ResizeNormalize(dataset, image_size),
        batch_size=args.classifier_batch_size,
        shuffle=False,
        num_workers=args.classifier_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.classifier_workers > 0,
        collate_fn=collate,
    )
    scores: dict[str, float] = {}
    with torch.inference_mode():
        for images, ids in loader:
            images = images.to(device, non_blocking=True)
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    probabilities = torch.sigmoid(model(images).squeeze(1))
            else:
                probabilities = torch.sigmoid(model(images).squeeze(1))
            scores.update({patient_id: float(probability) for patient_id, probability in zip(ids, probabilities.cpu())})
    save_json(cache_path, scores)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scores


def box_iou(left: list[float], right: list[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = area_left + area_right - intersection
    return intersection / union if union > 0 else 0.0


def weighted_boxes_fusion(
    model_predictions: list[list[dict]],
    model_weights: list[float],
    iou_threshold: float,
    score_threshold: float,
    classifier_probability: float,
    classifier_alpha: float,
) -> list[dict]:
    candidates = []
    for model_index, predictions in enumerate(model_predictions):
        weight = float(model_weights[model_index])
        for prediction in predictions:
            score = float(prediction["score"])
            if score >= 0.001:
                candidates.append((score, weight, prediction["box"]))
    candidates.sort(key=lambda row: row[0], reverse=True)
    clusters: list[dict] = []
    for score, weight, box in candidates:
        best_cluster = None
        best_overlap = iou_threshold
        for cluster in clusters:
            overlap = box_iou(box, cluster["box"])
            if overlap >= best_overlap:
                best_overlap = overlap
                best_cluster = cluster
        contribution = max(1e-6, score * weight)
        if best_cluster is None:
            best_cluster = {
                "box": list(box),
                "weighted_box": [value * contribution for value in box],
                # ``weight`` is the score-weighted contribution used for box
                # coordinates.  Keep the model weight separately for score
                # averaging; dividing ``score_sum`` by ``weight`` would make
                # every fused confidence exactly 1.0.
                "weight": contribution,
                "model_weight_sum": weight,
                "score_sum": score * weight,
            }
            clusters.append(best_cluster)
        else:
            best_cluster["weighted_box"] = [
                old + value * contribution
                for old, value in zip(best_cluster["weighted_box"], box)
            ]
            best_cluster["weight"] += contribution
            best_cluster["model_weight_sum"] += weight
            best_cluster["score_sum"] += score * weight
            best_cluster["box"] = [value / best_cluster["weight"] for value in best_cluster["weighted_box"]]

    output = []
    calibration = (1.0 - classifier_alpha) + classifier_alpha * classifier_probability
    for cluster in clusters:
        score = (cluster["score_sum"] / max(1e-6, cluster["model_weight_sum"])) * calibration
        if score >= score_threshold:
            output.append({"box": cluster["box"], "score": score})
    output.sort(key=lambda prediction: prediction["score"], reverse=True)
    return output


def create_submission(
    output_path: Path,
    data_dir: Path,
    patient_ids: list[str],
    predictions_by_model: list[dict[str, list[dict]]],
    model_weights: list[float],
    classifier_scores: dict[str, float],
    fusion_iou: float,
    score_threshold: float,
    classifier_alpha: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["patientId", "PredictionString"])
        writer.writeheader()
        for patient_id in patient_ids:
            fused = weighted_boxes_fusion(
                [predictions[patient_id] for predictions in predictions_by_model],
                model_weights,
                iou_threshold=fusion_iou,
                score_threshold=score_threshold,
                classifier_probability=float(classifier_scores.get(patient_id, 1.0)),
                classifier_alpha=classifier_alpha,
            )
            fields = []
            for prediction in fused:
                x1, y1, x2, y2 = prediction["box"]
                fields.extend(
                    [
                        f"{prediction['score']:.6f}",
                        f"{x1:.1f}",
                        f"{y1:.1f}",
                        f"{max(0.0, x2 - x1):.1f}",
                        f"{max(0.0, y2 - y1):.1f}",
                    ]
                )
            writer.writerow({"patientId": patient_id, "PredictionString": " ".join(fields)})
    print(f"[submission] wrote {output_path} ({len(patient_ids):,} rows)", flush=True)


def upload(args: argparse.Namespace, submission_path: Path) -> None:
    command = [
        sys.executable,
        str(PROJECT_DIR / "upload_submission.py"),
        "--file",
        str(submission_path),
        "--competition",
        args.competition,
        "--message",
        args.submission_message,
        "--poll-seconds",
        str(args.poll_seconds),
        "--poll-interval",
        str(args.poll_interval),
    ]
    run_subprocess(command, "kaggle upload")


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.classifier_epochs < 1:
        raise SystemExit("--epochs and --classifier-epochs must be at least 1")
    if args.detector_batch_size < 1 or args.classifier_batch_size < 1:
        raise SystemExit("batch sizes must be at least 1")
    if not 0 < args.val_fraction < 1:
        raise SystemExit("--val-fraction must be between 0 and 1")
    if not 0 < args.fusion_iou < 1:
        raise SystemExit("--fusion-iou must be between 0 and 1")
    if not 0 <= args.classifier_alpha <= 1:
        raise SystemExit("--classifier-alpha must be between 0 and 1")
    if not 0 <= args.score_threshold <= 1:
        raise SystemExit("--score-threshold must be between 0 and 1")
    if args.limit_test and not args.skip_upload:
        raise SystemExit("A partial test set cannot be uploaded. Use --skip-upload or remove --limit-test.")

    validate_data_dir(args.data_dir)
    architectures = [normalize_architecture(value.strip()) for value in args.detectors.split(",") if value.strip()]
    if not architectures:
        raise SystemExit("At least one detector is required")
    if args.detector_weights:
        model_weights = [float(value) for value in args.detector_weights.split(",")]
        if len(model_weights) != len(architectures) or any(value <= 0 for value in model_weights):
            raise SystemExit("--detector-weights must contain one positive number per detector")
    else:
        model_weights = [1.0] * len(architectures)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        args.output_dir / "run_config.json",
        {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "data_dir": str(args.data_dir),
            "architectures": architectures,
            "args": {key: str(value) for key, value in vars(args).items()},
        },
    )
    boxes_by_patient, patient_ids = read_annotations(args.data_dir / "stage_2_train_labels.csv")
    patient_ids = [
        patient_id
        for patient_id in patient_ids
        if (args.data_dir / "stage_2_train_images" / f"{patient_id}.dcm").is_file()
    ]
    print(
        f"[run] data={args.data_dir}; train images={len(patient_ids):,}; "
        f"positive={sum(bool(boxes_by_patient.get(pid)) for pid in patient_ids):,}; "
        f"device={'cuda:0' if torch.cuda.is_available() else 'cpu'}",
        flush=True,
    )

    checkpoints = train_detectors(args, architectures)
    classifier_checkpoint = train_classifier(args, patient_ids, boxes_by_patient)
    test_ids = load_test_ids(args.data_dir, args.limit_test)
    detector_predictions = []
    for architecture in architectures:
        detector_predictions.append(
            predict_detector(
                checkpoints[architecture],
                args.data_dir,
                test_ids,
                args.detector_batch_size,
                args.workers,
                args.output_dir / f"{architecture}_test_predictions.json",
                args.force_retrain,
            )
        )
    classifier_scores = predict_classifier(
        classifier_checkpoint,
        args,
        test_ids,
        args.output_dir / "efficientnet_b0_test_scores.json",
    )
    submission_path = args.output_dir / "submission.csv"
    create_submission(
        submission_path,
        args.data_dir,
        test_ids,
        detector_predictions,
        model_weights,
        classifier_scores,
        args.fusion_iou,
        args.score_threshold,
        args.classifier_alpha if classifier_checkpoint is not None else 0.0,
    )
    save_json(
        args.output_dir / "manifest.json",
        {
            "architectures": architectures,
            "checkpoints": {key: str(value) for key, value in checkpoints.items()},
            "classifier_checkpoint": str(classifier_checkpoint) if classifier_checkpoint else None,
            "submission": str(submission_path),
            "test_images": len(test_ids),
            "iou_thresholds": list(RSNA_IOU_THRESHOLDS),
            "metric": METRIC_NAME,
            "fusion_iou": args.fusion_iou,
            "score_threshold": args.score_threshold,
            "classifier_alpha": args.classifier_alpha if classifier_checkpoint is not None else 0.0,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    if args.skip_upload:
        print("[kaggle] skipped by --skip-upload", flush=True)
    else:
        upload(args, submission_path)
    print("[run] completed", flush=True)


if __name__ == "__main__":
    main()
