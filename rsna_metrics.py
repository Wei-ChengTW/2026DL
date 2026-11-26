"""Local mAP@IoU evaluation helpers for one foreground class."""

from __future__ import annotations


def box_iou(left: list[float], right: list[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def mean_average_precision(
    predictions: dict[str, list[dict]],
    ground_truths: dict[str, list[list[float]]],
    iou_threshold: float = 0.4,
    score_threshold: float = 0.05,
) -> float:
    """Compute single-class AP with a 101-point interpolated precision-recall curve."""
    ground_truth_count = sum(len(boxes) for boxes in ground_truths.values())
    if ground_truth_count == 0:
        return 0.0

    ranked_predictions = []
    for patient_id, boxes in predictions.items():
        for prediction in boxes:
            score = float(prediction["score"])
            if score >= score_threshold:
                ranked_predictions.append((score, patient_id, prediction["box"]))
    ranked_predictions.sort(key=lambda row: row[0], reverse=True)

    matched: dict[str, set[int]] = {patient_id: set() for patient_id in ground_truths}
    true_positives: list[float] = []
    false_positives: list[float] = []
    for _, patient_id, predicted_box in ranked_predictions:
        truths = ground_truths.get(patient_id, [])
        best_index = -1
        best_iou = 0.0
        for index, truth_box in enumerate(truths):
            if index in matched.setdefault(patient_id, set()):
                continue
            overlap = box_iou(predicted_box, truth_box)
            if overlap > best_iou:
                best_index = index
                best_iou = overlap
        if best_index >= 0 and best_iou >= iou_threshold:
            matched[patient_id].add(best_index)
            true_positives.append(1.0)
            false_positives.append(0.0)
        else:
            true_positives.append(0.0)
            false_positives.append(1.0)

    if not true_positives:
        return 0.0

    cumulative_tp = 0.0
    cumulative_fp = 0.0
    recalls: list[float] = []
    precisions: list[float] = []
    for tp, fp in zip(true_positives, false_positives):
        cumulative_tp += tp
        cumulative_fp += fp
        recalls.append(cumulative_tp / ground_truth_count)
        precisions.append(cumulative_tp / (cumulative_tp + cumulative_fp))

    ap = 0.0
    for point in range(101):
        recall_level = point / 100.0
        precision_at_level = max(
            (precision for recall, precision in zip(recalls, precisions) if recall >= recall_level),
            default=0.0,
        )
        ap += precision_at_level / 101.0
    return ap
