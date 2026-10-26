"""Create a Kaggle-format test submission from the trained detector."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from rsna_detector import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    dicom_to_grayscale,
    grayscale_to_tensor,
    load_trained_model,
)


class RSNATestDataset(Dataset):
    def __init__(self, image_dir: Path, patient_ids: list[str]):
        self.image_dir = image_dir
        self.patient_ids = patient_ids

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, index):
        patient_id = self.patient_ids[index]
        image = dicom_to_grayscale(self.image_dir / f"{patient_id}.dcm")
        return grayscale_to_tensor(image), patient_id


def collate(batch):
    images, patient_ids = zip(*batch)
    return list(images), list(patient_ids)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_OUTPUT_DIR / "best_model.pth")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "submission.csv")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Optional override; defaults to the threshold selected on validation.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional partial output; do not submit it to Kaggle.")
    return parser.parse_args()


def main():
    args = parse_args()
    image_dir = args.data_dir / "stage_2_test_images"
    sample_path = args.data_dir / "stage_2_sample_submission.csv"
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}. Train the model first.")
    if not image_dir.is_dir() or not sample_path.is_file():
        raise SystemExit(f"Could not find stage_2_test_images and stage_2_sample_submission.csv in {args.data_dir}.")

    with sample_path.open("r", newline="", encoding="utf-8") as handle:
        sample_rows = list(csv.DictReader(handle))
    patient_ids = [row["patientId"] for row in sample_rows]
    if args.limit:
        patient_ids = patient_ids[: args.limit]
    missing = [pid for pid in patient_ids if not (image_dir / f"{pid}.dcm").is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} test DICOM files are missing; first missing ID: {missing[0]}.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_trained_model(args.checkpoint, device)
    score_threshold = (
        args.score_threshold
        if args.score_threshold is not None
        else float(checkpoint.get("score_threshold", 0.05))
    )
    dataset = RSNATestDataset(image_dir, patient_ids)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        collate_fn=collate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    completed = 0
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["patientId", "PredictionString"])
        writer.writeheader()
        model.eval()
        with torch.inference_mode():
            for images, batch_ids in loader:
                images = [image.to(device, non_blocking=True) for image in images]
                if device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = model(images)
                else:
                    outputs = model(images)
                for patient_id, output in zip(batch_ids, outputs):
                    boxes = output["boxes"].detach().cpu().tolist()
                    scores = output["scores"].detach().cpu().tolist()
                    fields = []
                    for box, score in zip(boxes, scores):
                        if score < score_threshold:
                            continue
                        x1, y1, x2, y2 = box
                        fields.extend(
                            [
                                f"{score:.6f}",
                                f"{x1:.1f}",
                                f"{y1:.1f}",
                                f"{max(0.0, x2 - x1):.1f}",
                                f"{max(0.0, y2 - y1):.1f}",
                            ]
                        )
                    writer.writerow(
                        {"patientId": patient_id, "PredictionString": " ".join(fields)}
                    )
                    completed += 1
                if completed % 200 == 0 or completed == len(dataset):
                    print(
                        f"predicted {completed}/{len(dataset)} images "
                        f"in {time.time() - started:.0f}s",
                        flush=True,
                    )

    print(
        f"Created {args.output} with {completed} rows; checkpoint epoch="
        f"{checkpoint.get('epoch', 'unknown')}, local validation RSNA mAP_IoU="
        f"{float(checkpoint.get('map_iou_0_4', 0.0)):.4f}, "
        f"score threshold={score_threshold:g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
