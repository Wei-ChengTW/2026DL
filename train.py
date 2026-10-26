"""Train Faster R-CNN on the local RSNA Pneumonia Detection Challenge data."""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rsna_detector import (
    ARCHITECTURE_RESNET50_FPN,
    ARCHITECTURE_RESNET50_FPN_V2,
    ARCHITECTURE_RETINANET_RESNET50_FPN,
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    RSNADetectionDataset,
    build_model,
    detection_collate,
    normalize_architecture,
    read_annotations,
)
from rsna_metrics import METRIC_NAME, RSNA_IOU_THRESHOLDS, mean_average_precision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0025)
    parser.add_argument("--lr-step-size", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int, default=None, help="Optional training subset for a quick run.")
    parser.add_argument("--limit-val", type=int, default=None, help="Optional validation subset for a quick run.")
    parser.add_argument("--min-size", type=int, default=512)
    parser.add_argument("--max-size", type=int, default=768)
    parser.add_argument(
        "--architecture",
        choices=(
            ARCHITECTURE_RESNET50_FPN,
            ARCHITECTURE_RESNET50_FPN_V2,
            ARCHITECTURE_RETINANET_RESNET50_FPN,
        ),
        default=ARCHITECTURE_RESNET50_FPN,
    )
    parser.add_argument("--detections-per-image", type=int, default=100)
    parser.add_argument("--box-score-threshold", type=float, default=0.0)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--scratch", action="store_true", help="Initialize Faster R-CNN without COCO weights.")
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Initialize from a local checkpoint while using the current training settings.",
    )
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def split_patients(patient_ids, boxes_by_patient, val_fraction, seed):
    positive_ids = [pid for pid in patient_ids if boxes_by_patient.get(pid)]
    negative_ids = [pid for pid in patient_ids if not boxes_by_patient.get(pid)]
    rng = random.Random(seed)
    rng.shuffle(positive_ids)
    rng.shuffle(negative_ids)

    def split_group(ids):
        val_count = max(1, round(len(ids) * val_fraction)) if len(ids) > 1 else 0
        return ids[val_count:], ids[:val_count]

    train_positive, val_positive = split_group(positive_ids)
    train_negative, val_negative = split_group(negative_ids)
    train_ids = train_positive + train_negative
    val_ids = val_positive + val_negative
    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    return train_ids, val_ids


def move_targets(targets, device):
    return [{key: value.to(device) for key, value in target.items()} for target in targets]


@torch.inference_mode()
def evaluate(model, loader, device, score_threshold, architecture):
    model.eval()
    architecture = normalize_architecture(architecture)
    prediction_map: dict[str, list[dict]] = {}
    truth_map: dict[str, list[list[float]]] = {}
    use_amp = device.type == "cuda"
    for images, targets, patient_ids in loader:
        images = [image.to(device, non_blocking=True) for image in images]
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
        else:
            outputs = model(images)
        for patient_id, target, output in zip(patient_ids, targets, outputs):
            truth_map[patient_id] = target["boxes"].tolist()
            boxes = output["boxes"].detach().cpu().tolist()
            scores = output["scores"].detach().cpu().tolist()
            if architecture == ARCHITECTURE_RETINANET_RESNET50_FPN:
                labels = output.get("labels")
                if labels is not None:
                    filtered = [
                        (box, score)
                        for box, score, label in zip(
                            boxes,
                            scores,
                            labels.detach().cpu().tolist(),
                        )
                        if int(label) == 1
                    ]
                    boxes = [box for box, _ in filtered]
                    scores = [score for _, score in filtered]
            prediction_map[patient_id] = [
                {"box": box, "score": score}
                for box, score in zip(boxes, scores)
            ]
    score = mean_average_precision(
        prediction_map,
        truth_map,
        iou_threshold=0.4,
        score_threshold=score_threshold,
    )
    return score, prediction_map, truth_map


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("--epochs and --batch-size must be at least 1.")
    if args.lr_step_size is not None and args.lr_step_size < 1:
        raise SystemExit("--lr-step-size must be at least 1.")
    if not 0 < args.val_fraction < 1:
        raise SystemExit("--val-fraction must be between 0 and 1.")
    if args.resume_checkpoint and not args.resume_checkpoint.is_file():
        raise SystemExit(f"Resume checkpoint not found: {args.resume_checkpoint}")
    if args.resume_checkpoint and args.scratch:
        raise SystemExit("--resume-checkpoint and --scratch cannot be used together.")

    train_image_dir = args.data_dir / "stage_2_train_images"
    labels_path = args.data_dir / "stage_2_train_labels.csv"
    if not train_image_dir.is_dir() or not labels_path.is_file():
        raise SystemExit(
            f"Could not find {train_image_dir} and {labels_path}. "
            "Set --data-dir to the extracted competition data directory."
        )

    boxes_by_patient, patient_ids = read_annotations(labels_path)
    patient_ids = [pid for pid in patient_ids if (train_image_dir / f"{pid}.dcm").is_file()]
    train_ids, val_ids = split_patients(patient_ids, boxes_by_patient, args.val_fraction, args.seed)
    if args.limit_train:
        train_ids = train_ids[: args.limit_train]
    if args.limit_val:
        val_ids = val_ids[: args.limit_val]
    if not train_ids or not val_ids:
        raise SystemExit("The selected train or validation split is empty.")

    print(
        f"Data: {len(patient_ids):,} images; train={len(train_ids):,}, "
        f"validation={len(val_ids):,}; train positives="
        f"{sum(bool(boxes_by_patient.get(pid)) for pid in train_ids):,}, "
        f"validation positives={sum(bool(boxes_by_patient.get(pid)) for pid in val_ids):,}",
        flush=True,
    )
    print(f"Device: {'cuda:0' if torch.cuda.is_available() else 'cpu'}", flush=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = RSNADetectionDataset(
        train_image_dir, train_ids, boxes_by_patient, augment=True
    )
    val_dataset = RSNADetectionDataset(
        train_image_dir, val_ids, boxes_by_patient, augment=False
    )
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "collate_fn": detection_collate,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_args,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)

    resume = None
    if args.resume_checkpoint:
        resume = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=True)
        saved_config = resume.get("model_config", {})
        saved_architecture = normalize_architecture(
            saved_config.get("architecture", ARCHITECTURE_RESNET50_FPN)
        )
        if saved_architecture != normalize_architecture(args.architecture):
            raise SystemExit(
                f"Checkpoint architecture {saved_architecture} does not match "
                f"requested architecture {normalize_architecture(args.architecture)}."
            )
    model = build_model(
        pretrained=not args.scratch and resume is None,
        min_size=args.min_size,
        max_size=args.max_size,
        detections_per_image=args.detections_per_image,
        box_score_thresh=args.box_score_threshold,
        architecture=args.architecture,
    ).to(device)
    if resume is not None:
        model.load_state_dict(resume["model_state_dict"])
        print(
            f"Initialized from checkpoint epoch={resume.get('epoch', 'unknown')}, "
            f"validation {METRIC_NAME}={float(resume.get('map_iou_0_4', 0.0)):.4f}",
            flush=True,
        )
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=0.0005,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step_size or max(1, (args.epochs + 2) // 3),
        gamma=0.1,
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_config = {
        "min_size": args.min_size,
        "max_size": args.max_size,
        "detections_per_image": args.detections_per_image,
        "box_score_thresh": args.box_score_threshold,
        "architecture": normalize_architecture(args.architecture),
        "num_classes": 2,
    }
    best_map = -1.0
    best_score_threshold = args.score_threshold
    best_threshold_scores: dict[str, float] = {}
    history = []
    started_at = datetime.now(timezone.utc).isoformat()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        epoch_started = time.time()
        for step, (images, targets, _) in enumerate(train_loader, start=1):
            images = [image.to(device, non_blocking=True) for image in images]
            targets = move_targets(targets, device)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss_dict = model(images, targets)
                    loss = sum(loss_dict.values())
            else:
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, step {step}: {loss.item()}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().cpu())
            if step % args.log_every == 0 or step == len(train_loader):
                mean_loss = running_loss / step
                print(
                    f"epoch {epoch}/{args.epochs} step {step}/{len(train_loader)} "
                    f"loss={mean_loss:.4f} elapsed={time.time() - epoch_started:.0f}s",
                    flush=True,
                )

        train_loss = running_loss / max(1, len(train_loader))
        _, prediction_map, truth_map = evaluate(
            model,
            val_loader,
            device,
            score_threshold=0.0,
            architecture=args.architecture,
        )
        thresholds = sorted(
            set((0.001, 0.005, 0.01, 0.02, 0.05, 0.1, args.score_threshold))
        )
        threshold_scores = {
            f"{threshold:g}": mean_average_precision(
                prediction_map,
                truth_map,
                iou_threshold=0.4,
                score_threshold=threshold,
            )
            for threshold in thresholds
        }
        requested_threshold_key = f"{args.score_threshold:g}"
        val_map = threshold_scores[requested_threshold_key]
        selected_threshold_key = max(threshold_scores, key=threshold_scores.get)
        selected_threshold = float(selected_threshold_key)
        selected_map = threshold_scores[selected_threshold_key]
        scheduler.step()
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "map_iou_0_4": val_map,
            "best_threshold_map_iou_0_4": selected_map,
            "best_score_threshold": selected_threshold,
            "score_threshold_sweep": threshold_scores,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - epoch_started,
        }
        history.append(epoch_record)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "epoch": epoch,
            "map_iou_0_4": selected_map,
            "score_threshold": selected_threshold,
        }
        torch.save(checkpoint, args.output_dir / "last_model.pth")
        if selected_map >= best_map:
            best_map = selected_map
            best_score_threshold = selected_threshold
            best_threshold_scores = threshold_scores
            torch.save(checkpoint, args.output_dir / "best_model.pth")
        elapsed = time.time() - epoch_started
        print(
            f"epoch {epoch} complete: train_loss={train_loss:.4f}, "
            f"validation {METRIC_NAME}={val_map:.4f} at score threshold {args.score_threshold:g}; "
            f"best={selected_map:.4f} at threshold {selected_threshold:g}, "
            f"time={elapsed:.0f}s",
            flush=True,
        )

        sample_ids = []
        positive_val = [pid for pid in val_ids if boxes_by_patient.get(pid)]
        negative_val = [pid for pid in val_ids if not boxes_by_patient.get(pid)]
        sample_ids.extend(positive_val[:60])
        sample_ids.extend(negative_val[:60])
        metrics = {
            "model": model_config["architecture"],
            "metric": METRIC_NAME,
            "best_map_iou_0_4": best_map,
            "best_map_percent": best_map * 100.0,
            "epoch": epoch,
            "epochs_requested": args.epochs,
            "train_images": len(train_ids),
            "validation_images": len(val_ids),
            "train_positive_images": sum(bool(boxes_by_patient.get(pid)) for pid in train_ids),
            "validation_positive_images": len(positive_val),
            "score_threshold": best_score_threshold,
            "requested_score_threshold": args.score_threshold,
            "score_threshold_sweep": best_threshold_scores,
            "iou_thresholds": list(RSNA_IOU_THRESHOLDS),
            "device": str(device),
            "started_at_utc": started_at,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "history": history,
            "sample_ids": sample_ids,
            "split_seed": args.seed,
            "validation_fraction": args.val_fraction,
            "model_config": model_config,
        }
        save_json(args.output_dir / "metrics.json", metrics)

    print(
        f"FINAL validation score: {METRIC_NAME} = {best_map:.4f} "
        f"({best_map * 100:.2f}%) at score threshold {best_score_threshold:g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
