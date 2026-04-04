#!/usr/bin/env python3

from io import BytesIO
from pathlib import Path
import json
import os
import time

import h5py
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)


class CompatibleDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
	@classmethod
	def from_config(cls, config):
		config.pop("groups", None)
		return super().from_config(config)


WORKSPACE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.getenv("MODEL_CONFIG_PATH", WORKSPACE_DIR / "model_runtime_config.json"))


def discover_models() -> list[Path]:
	models = []
	for pattern in ("**/*.h5", "**/*.keras"):
		for path in WORKSPACE_DIR.glob(pattern):
			rel = path.relative_to(WORKSPACE_DIR)
			if str(rel).startswith(".venv"):
				continue
			models.append(path)
	models.sort(key=lambda p: p.stat().st_mtime, reverse=True)
	return models


def load_runtime_config() -> dict:
	cfg = {
		"selected_model": None,
		"preprocess_mode": "auto",
		"threshold": 0.5,
		"positive_label": "REAL",
		"negative_label": "FAKE",
		"input_size": [224, 224],
	}
	if CONFIG_PATH.exists():
		user_cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
		cfg.update(user_cfg)
	return cfg


def choose_model_path(cfg: dict) -> Path:
	selected = cfg.get("selected_model")
	if selected:
		model_path = (WORKSPACE_DIR / selected).resolve()
		if model_path.exists() and model_path.is_file():
			return model_path
		raise FileNotFoundError(f"Configured selected_model not found: {selected}")

	discovered = discover_models()
	if discovered:
		return discovered[0]
	raise FileNotFoundError("No model files (.h5/.keras) found in workspace")


def detect_preprocess_mode(loaded_model: tf.keras.Model) -> str:
	layer_names = " ".join(layer.name.lower() for layer in loaded_model.layers)
	if "mobilenet" in layer_names:
		return "mobilenet_v2"
	if "resnet" in layer_names:
		return "resnet50"
	if "efficientnet" in layer_names:
		return "efficientnet"
	return "rescale_255"


def load_model_robust(model_path: Path) -> tf.keras.Model:
	try:
		return tf.keras.models.load_model(
			str(model_path),
			compile=False,
			custom_objects={"DepthwiseConv2D": CompatibleDepthwiseConv2D},
		)
	except Exception:
		pass

	with h5py.File(str(model_path), "r") as h5_file:
		cfg_raw = h5_file.attrs.get("model_config")
		if cfg_raw is None:
			raise ValueError("model_config not found in H5 file")
		if isinstance(cfg_raw, bytes):
			cfg_raw = cfg_raw.decode("utf-8")
		cfg = json.loads(cfg_raw)

	def _strip_groups(obj):
		if isinstance(obj, dict):
			if obj.get("class_name") == "DepthwiseConv2D" and isinstance(obj.get("config"), dict):
				obj["config"].pop("groups", None)
			for key in list(obj.keys()):
				_strip_groups(obj[key])
		elif isinstance(obj, list):
			for item in obj:
				_strip_groups(item)

	_strip_groups(cfg)
	model_from_cfg = tf.keras.models.model_from_json(json.dumps(cfg))
	model_from_cfg.load_weights(str(model_path))
	return model_from_cfg


RUNTIME_CONFIG = load_runtime_config()
MODEL_PATH = choose_model_path(RUNTIME_CONFIG)
THRESHOLD = float(RUNTIME_CONFIG.get("threshold", 0.5))
POSITIVE_LABEL = str(RUNTIME_CONFIG.get("positive_label", "REAL"))
NEGATIVE_LABEL = str(RUNTIME_CONFIG.get("negative_label", "FAKE"))
INPUT_SIZE = tuple(RUNTIME_CONFIG.get("input_size", [224, 224]))

model = None
MODEL_LOAD_ERROR = None
PREPROCESS_MODE = str(RUNTIME_CONFIG.get("preprocess_mode", "auto"))


def ensure_model_loaded() -> None:
	global model, MODEL_LOAD_ERROR, PREPROCESS_MODE
	if model is not None or MODEL_LOAD_ERROR is not None:
		return
	try:
		model = load_model_robust(MODEL_PATH)
		if PREPROCESS_MODE == "auto":
			PREPROCESS_MODE = detect_preprocess_mode(model)
	except Exception as exc:  # noqa: BLE001
		MODEL_LOAD_ERROR = str(exc)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
	image = Image.open(BytesIO(image_bytes)).convert("RGB")
	image = image.resize(INPUT_SIZE)
	img_array = np.array(image, dtype=np.float32)
	img_array = np.expand_dims(img_array, axis=0)

	if PREPROCESS_MODE == "mobilenet_v2":
		return tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
	if PREPROCESS_MODE == "resnet50":
		return tf.keras.applications.resnet50.preprocess_input(img_array)
	if PREPROCESS_MODE == "efficientnet":
		return tf.keras.applications.efficientnet.preprocess_input(img_array)
	return img_array / 255.0


@app.route("/model-status", methods=["GET"])
def model_status():
	ensure_model_loaded()
	return jsonify(
		{
			"config_path": str(CONFIG_PATH),
			"model_path": str(MODEL_PATH),
			"input_shape": None if model is None else model.input_shape,
			"output_shape": None if model is None else model.output_shape,
			"preprocess_mode": PREPROCESS_MODE,
			"threshold": THRESHOLD,
			"positive_label": POSITIVE_LABEL,
			"negative_label": NEGATIVE_LABEL,
			"input_size": list(INPUT_SIZE),
			"model_loaded": model is not None,
			"model_load_error": MODEL_LOAD_ERROR,
		}
	)


@app.route("/predict", methods=["GET", "POST"])
def predict():
	if request.method == "GET":
		return jsonify(
			{
				"message": "Use POST /predict with form-data key 'image' to run inference.",
				"example": "curl -X POST http://127.0.0.1:8080/predict -F \"image=@/path/to/image.jpg\"",
				"model_status": "http://127.0.0.1:8080/model-status",
			}
		)

	ensure_model_loaded()
	if model is None:
		return jsonify({"error": f"Model failed to load: {MODEL_LOAD_ERROR}"}), 500

	if "image" not in request.files:
		return jsonify({"error": "No image file found in form-data with key 'image'."}), 400

	image_file = request.files["image"]
	if image_file.filename == "":
		return jsonify({"error": "Empty file name."}), 400

	try:
		start_time = time.perf_counter()
		image_bytes = image_file.read()
		arr = preprocess_image(image_bytes)

		pred = model.predict(arr, verbose=0)
		raw_score = float(pred[0][0])

		if raw_score > THRESHOLD:
			label = POSITIVE_LABEL
			confidence = raw_score * 100.0
		else:
			label = NEGATIVE_LABEL
			confidence = (1.0 - raw_score) * 100.0

		elapsed_seconds = time.perf_counter() - start_time

		return jsonify(
			{
				"label": label,
				"confidence": round(confidence, 2),
				"raw_score": round(raw_score, 4),
				"processing_time_sec": round(elapsed_seconds, 3),
				"message": f"This image is {label} with {round(confidence, 2)}% confidence",
				"model_path": str(MODEL_PATH),
				"preprocess_mode": PREPROCESS_MODE,
				"threshold": THRESHOLD,
				"positive_label": POSITIVE_LABEL,
				"negative_label": NEGATIVE_LABEL,
			}
		)
	except Exception as exc:  # noqa: BLE001
		return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
	print(f"Config path: {CONFIG_PATH}")
	print(f"Selected model: {MODEL_PATH}")
	print(f"Preprocess mode: {PREPROCESS_MODE}")
	print(f"Threshold: {THRESHOLD}")
	print(f"Positive label: {POSITIVE_LABEL}")
	print(f"Negative label: {NEGATIVE_LABEL}")
	app.run(debug=False, use_reloader=False, host="0.0.0.0", port=8080)
