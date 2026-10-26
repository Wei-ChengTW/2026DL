"""Flask dashboard for Faster R-CNN RSNA predictions and validation examples."""

from __future__ import annotations

import base64
import json
import re
import threading
from functools import lru_cache
from pathlib import Path

import torch
from flask import Flask, jsonify, render_template, request

from rsna_detector import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    dicom_to_grayscale,
    grayscale_to_tensor,
    load_trained_model,
    read_annotations,
    render_grayscale_png,
)


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = DEFAULT_DATA_DIR
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
CHECKPOINT_PATH = OUTPUT_DIR / "best_model.pth"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

_load_lock = threading.Lock()
_inference_lock = threading.Lock()
_runtime = None
_runtime_mtime = None


@lru_cache(maxsize=1)
def dataset_annotations():
    labels = DATA_DIR / "stage_2_train_labels.csv"
    if not labels.is_file():
        return {}, []
    return read_annotations(labels)


def metrics_data() -> dict:
    if not METRICS_PATH.is_file():
        return {}
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def runtime_model():
    global _runtime, _runtime_mtime
    if not CHECKPOINT_PATH.is_file():
        return None
    checkpoint_mtime = CHECKPOINT_PATH.stat().st_mtime_ns
    if _runtime is not None and _runtime_mtime == checkpoint_mtime:
        return _runtime
    with _load_lock:
        checkpoint_mtime = CHECKPOINT_PATH.stat().st_mtime_ns
        if _runtime is None or _runtime_mtime != checkpoint_mtime:
            model, checkpoint = load_trained_model(CHECKPOINT_PATH, DEVICE)
            _runtime = (model, checkpoint)
            _runtime_mtime = checkpoint_mtime
    return _runtime


def infer_grayscale(grayscale):
    loaded = runtime_model()
    if loaded is None:
        return []
    model, _ = loaded
    tensor = grayscale_to_tensor(grayscale).to(DEVICE)
    with _inference_lock, torch.inference_mode():
        if DEVICE.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model([tensor])[0]
        else:
            output = model([tensor])[0]
    boxes = output["boxes"].detach().cpu().tolist()
    scores = output["scores"].detach().cpu().tolist()
    return [{"box": box, "score": float(score)} for box, score in zip(boxes, scores)]


def image_response(grayscale, patient_id=None, ground_truth=None, source="validation"):
    predictions = infer_grayscale(grayscale)
    return {
        "patient_id": patient_id,
        "source": source,
        "width": int(grayscale.shape[1]),
        "height": int(grayscale.shape[0]),
        "image": "data:image/png;base64," + base64.b64encode(render_grayscale_png(grayscale)).decode("ascii"),
        "predictions": predictions,
        "ground_truth": ground_truth or [],
        "is_positive": bool(ground_truth),
        "model_ready": CHECKPOINT_PATH.is_file(),
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def api_status():
    metrics = metrics_data()
    loaded = runtime_model() if CHECKPOINT_PATH.is_file() else None
    return jsonify(
        {
            "model_ready": loaded is not None,
            "model_name": metrics.get("model", "Faster R-CNN ResNet50-FPN"),
            "metric_name": metrics.get("metric", "RSNA reference mAP_IoU (0.40:0.05:0.75)"),
            "best_map": metrics.get("best_map_iou_0_4"),
            "best_map_percent": metrics.get("best_map_percent"),
            "epoch": metrics.get("epoch", 0),
            "epochs_requested": metrics.get("epochs_requested", 0),
            "train_images": metrics.get("train_images", 0),
            "validation_images": metrics.get("validation_images", 0),
            "validation_positive_images": metrics.get("validation_positive_images", 0),
            "device": str(DEVICE),
            "history": metrics.get("history", []),
            "data_available": (DATA_DIR / "stage_2_train_images").is_dir(),
        }
    )


@app.get("/api/samples")
def api_samples():
    metrics = metrics_data()
    boxes_by_patient, patient_ids = dataset_annotations()
    requested = metrics.get("sample_ids") or patient_ids[:100]
    samples = [
        {"patient_id": patient_id, "positive": bool(boxes_by_patient.get(patient_id))}
        for patient_id in requested
        if (DATA_DIR / "stage_2_train_images" / f"{patient_id}.dcm").is_file()
    ]
    return jsonify(samples)


@app.get("/api/sample/<patient_id>")
def api_sample(patient_id: str):
    if not re.fullmatch(r"[A-Za-z0-9-]{1,80}", patient_id):
        return jsonify({"error": "Invalid patient ID."}), 400
    image_path = DATA_DIR / "stage_2_train_images" / f"{patient_id}.dcm"
    if not image_path.is_file():
        return jsonify({"error": "This validation image is not available."}), 404
    boxes_by_patient, patient_ids = dataset_annotations()
    if patient_id not in set(patient_ids):
        return jsonify({"error": "This image does not have training annotations."}), 404
    try:
        grayscale = dicom_to_grayscale(image_path)
        result = image_response(
            grayscale,
            patient_id=patient_id,
            ground_truth=boxes_by_patient.get(patient_id, []),
        )
        return jsonify(result)
    except Exception as error:  # noqa: BLE001 - return a useful decoding message to the UI.
        return jsonify({"error": f"Could not decode DICOM: {error}"}), 422


@app.post("/api/predict")
def api_predict():
    if not CHECKPOINT_PATH.is_file():
        return jsonify({"error": "Train a model before uploading an image."}), 503
    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "Choose a DICOM file first."}), 400
    if not uploaded.filename.lower().endswith((".dcm", ".dicom")):
        return jsonify({"error": "Please upload a .dcm or .dicom file."}), 400
    try:
        grayscale = dicom_to_grayscale(uploaded.read())
        return jsonify(image_response(grayscale, source="uploaded DICOM"))
    except Exception as error:  # noqa: BLE001 - malformed uploads should be reported in the UI.
        return jsonify({"error": f"Could not read this DICOM file: {error}"}), 422


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"error": "The DICOM file is larger than the 32 MB upload limit."}), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=False, threaded=True)
