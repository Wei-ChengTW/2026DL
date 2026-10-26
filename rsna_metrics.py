"""RSNA Pneumonia Detection Challenge local metric.

This follows the public RSNA reference/champion implementation: each image
gets TP/(TP+FN+FP), averaged over IoU thresholds 0.40 through 0.75. Images
with neither ground truth nor predictions are omitted; a negative image with
a prediction contributes zero.
"""

from __future__ import annotations


METRIC_NAME = "RSNA reference mAP_IoU (0.40:0.05:0.75)"
RSNA_IOU_THRESHOLDS = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75)


def box_iou(left: list[float], right: list[float]) -> float:
    """IoU for boxes represented as [x1, y1, x2, y2]."""
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(
        0.0, float(left[3]) - float(left[1])
    )
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(
        0.0, float(right[3]) - float(right[1])
    )
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def image_rsna_map_iou(
    predictions: list[dict],
    ground_truths: list[list[float]],
    iou_thresholds: tuple[float, ...] = RSNA_IOU_THRESHOLDS,
    score_threshold: float = 0.0,
) -> float | None:
    """Return the reference mAP_IoU value for one patient image."""
    ranked = [
        (float(prediction["score"]), prediction["box"])
        for prediction in predictions
        if float(prediction.get("score", 0.0)) >= score_threshold
    ]
    ranked.sort(key=lambda row: row[0], reverse=True)
    if not ground_truths and not ranked:
        return None

    scores = []
    for threshold in iou_thresholds:
        matched_predictions: set[int] = set()
        true_positive = 0
        false_negative = 0
        # This is the matching order used by the public reference code:
        # iterate ground-truth boxes and consume the highest-scoring eligible
        # prediction at most once.
        for truth_box in ground_truths:
            matched = False
            for index, (_, predicted_box) in enumerate(ranked):
                if index in matched_predictions:
                    continue
                if box_iou(truth_box, predicted_box) >= threshold:
                    matched = True
                    true_positive += 1
                    matched_predictions.add(index)
                    break
            if not matched:
                false_negative += 1
        false_positive = len(ranked) - len(matched_predictions)
        denominator = true_positive + false_negative + false_positive
        scores.append(true_positive / denominator if denominator else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def mean_average_precision(
    predictions: dict[str, list[dict]],
    ground_truths: dict[str, list[list[float]]],
    iou_threshold: float = 0.4,
    score_threshold: float = 0.0,
) -> float:
    """Return the local RSNA reference-style image mean.

    ``iou_threshold`` is retained for compatibility with the existing
    training API. The public reference averages the eight thresholds in
    :data:`RSNA_IOU_THRESHOLDS`, so the argument is intentionally ignored.
    """
    del iou_threshold
    patient_ids = set(ground_truths) | set(predictions)
    image_scores = []
    for patient_id in patient_ids:
        score = image_rsna_map_iou(
            predictions.get(patient_id, []),
            ground_truths.get(patient_id, []),
            score_threshold=score_threshold,
        )
        if score is not None:
            image_scores.append(score)
    return sum(image_scores) / len(image_scores) if image_scores else 0.0
