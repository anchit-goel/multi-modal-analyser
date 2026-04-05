#!/usr/bin/env python3

from io import BytesIO
from pathlib import Path
import json
import os
import sys
import tempfile
import threading
import time
import traceback

import h5py
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request, send_from_directory
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
	non_demo = [p for p in models if p.name != "_demo_classifier.keras"]
	if non_demo:
		models = non_demo
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
		"allow_demo_model": True,
	}
	if CONFIG_PATH.exists():
		user_cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
		cfg.update(user_cfg)
	return cfg


def resolve_model_path(cfg: dict) -> Path | None:
	"""Return path to a weights file, or None if none are available yet."""
	selected = cfg.get("selected_model")
	if selected:
		model_path = (WORKSPACE_DIR / selected).resolve()
		if model_path.exists() and model_path.is_file():
			return model_path

	discovered = discover_models()
	if discovered:
		return discovered[0]
	return None


def write_demo_classifier(path: Path, input_size: tuple[int, ...]) -> None:
	"""Tiny random-init classifier so the API runs when real weights are not in the repo."""
	if path.exists() and path.stat().st_size > 0:
		return
	path.parent.mkdir(parents=True, exist_ok=True)
	h, w = int(input_size[0]), int(input_size[1])
	demo = tf.keras.Sequential(
		[
			tf.keras.layers.Input(shape=(h, w, 3)),
			tf.keras.layers.Conv2D(16, 5, activation="relu", padding="same"),
			tf.keras.layers.MaxPooling2D(2),
			tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
			tf.keras.layers.GlobalAveragePooling2D(),
			tf.keras.layers.Dense(1, activation="sigmoid"),
		]
	)
	demo.save(str(path))


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
THRESHOLD = float(RUNTIME_CONFIG.get("threshold", 0.5))
POSITIVE_LABEL = str(RUNTIME_CONFIG.get("positive_label", "REAL"))
NEGATIVE_LABEL = str(RUNTIME_CONFIG.get("negative_label", "FAKE"))
INPUT_SIZE = tuple(RUNTIME_CONFIG.get("input_size", [224, 224]))

MODEL_PATH = resolve_model_path(RUNTIME_CONFIG)
MODEL_LOAD_ERROR: str | None = None
DEMO_MODEL_PATH = WORKSPACE_DIR / "results" / "_demo_classifier.keras"
IS_DEMO_MODEL = False

if MODEL_PATH is None:
	if RUNTIME_CONFIG.get("allow_demo_model", True):
		try:
			write_demo_classifier(DEMO_MODEL_PATH, INPUT_SIZE)
			MODEL_PATH = DEMO_MODEL_PATH
			IS_DEMO_MODEL = True
		except Exception as exc:  # noqa: BLE001
			MODEL_LOAD_ERROR = f"Could not create demo model: {exc}"
	else:
		MODEL_LOAD_ERROR = (
			"No model weights found. Add a .h5/.keras file under the project, "
			"set selected_model in model_runtime_config.json, or set allow_demo_model to true."
		)
elif MODEL_PATH.resolve() == DEMO_MODEL_PATH.resolve():
	IS_DEMO_MODEL = True

model = None
PREPROCESS_MODE = str(RUNTIME_CONFIG.get("preprocess_mode", "auto"))


def ensure_model_loaded() -> None:
	global model, MODEL_LOAD_ERROR, PREPROCESS_MODE
	if model is not None or MODEL_LOAD_ERROR is not None:
		return
	if MODEL_PATH is None:
		MODEL_LOAD_ERROR = MODEL_LOAD_ERROR or "No model file configured or found."
		return
	try:
		model = load_model_robust(MODEL_PATH)
		if PREPROCESS_MODE == "auto":
			PREPROCESS_MODE = detect_preprocess_mode(model)
	except Exception as exc:  # noqa: BLE001
		MODEL_LOAD_ERROR = str(exc)

AUDIO_ROOT = WORKSPACE_DIR.parent / "audio_spoof_system"
AUDIO_LOAD_ERROR: str | None = None
audio_predict = None

MULTIMODAL_ROOT = WORKSPACE_DIR.parent / "multimodal_injection"
MULTIMODAL_LOAD_ERROR: str | None = None
multimodal_analyze = None
MULTIMODAL_LOADING = False
MULTIMODAL_LOADING_LOCK = threading.Lock()


def ensure_audio_loaded() -> None:
    global audio_predict, AUDIO_LOAD_ERROR
    if audio_predict is not None or AUDIO_LOAD_ERROR is not None:
        return
    if not AUDIO_ROOT.exists():
        AUDIO_LOAD_ERROR = f"Audio spoof system folder not found: {AUDIO_ROOT}"
        return

    audio_root_str = str(AUDIO_ROOT.resolve())
    original_cwd = Path.cwd()
    if audio_root_str not in sys.path:
        sys.path.insert(0, audio_root_str)
    try:
        os.chdir(audio_root_str)
        from inference import predict as _predict
        audio_predict = _predict
    except Exception as exc:  # noqa: BLE001
        AUDIO_LOAD_ERROR = str(exc)
    finally:
        if audio_root_str in sys.path:
            sys.path.remove(audio_root_str)
        try:
            os.chdir(original_cwd)
        except OSError:
            pass


def ensure_multimodal_loaded() -> None:
    global multimodal_analyze, MULTIMODAL_LOAD_ERROR, MULTIMODAL_LOADING
    with MULTIMODAL_LOADING_LOCK:
        if multimodal_analyze is not None or MULTIMODAL_LOAD_ERROR is not None:
            return
        if MULTIMODAL_LOADING:
            return
        MULTIMODAL_LOADING = True

    if not MULTIMODAL_ROOT.exists():
        MULTIMODAL_LOAD_ERROR = f"Multimodal injection folder not found: {MULTIMODAL_ROOT}"
        MULTIMODAL_LOADING = False
        return

    multimodal_root_str = str(MULTIMODAL_ROOT.resolve())
    original_cwd = Path.cwd()
    if multimodal_root_str not in sys.path:
        sys.path.insert(0, multimodal_root_str)
    try:
        os.chdir(multimodal_root_str)
        from multimodal_detector import analyze_multimodal as _analyze
        multimodal_analyze = _analyze
    except Exception:
        MULTIMODAL_LOAD_ERROR = traceback.format_exc()
    finally:
        if multimodal_root_str in sys.path:
            sys.path.remove(multimodal_root_str)
        try:
            os.chdir(original_cwd)
        except OSError:
            pass
        MULTIMODAL_LOADING = False


def start_multimodal_background_load() -> None:
    if multimodal_analyze is None and MULTIMODAL_LOAD_ERROR is None:
        threading.Thread(target=ensure_multimodal_loaded, daemon=True).start()


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
	ensure_audio_loaded()
	start_multimodal_background_load()
	return jsonify(
		{
			"config_path": str(CONFIG_PATH),
			"model_path": None if MODEL_PATH is None else str(MODEL_PATH),
			"input_shape": None if model is None else model.input_shape,
			"output_shape": None if model is None else model.output_shape,
			"preprocess_mode": PREPROCESS_MODE,
			"threshold": THRESHOLD,
			"positive_label": POSITIVE_LABEL,
			"negative_label": NEGATIVE_LABEL,
			"input_size": list(INPUT_SIZE),
			"model_loaded": model is not None,
			"model_load_error": MODEL_LOAD_ERROR,
			"is_demo_model": IS_DEMO_MODEL,
			"audio_root": str(AUDIO_ROOT),
			"audio_model_loaded": audio_predict is not None,
			"audio_model_error": AUDIO_LOAD_ERROR,
			"multimodal_root": str(MULTIMODAL_ROOT),
			"multimodal_model_loaded": multimodal_analyze is not None,
			"multimodal_model_loading": MULTIMODAL_LOADING,
			"multimodal_model_error": MULTIMODAL_LOAD_ERROR,
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
				"model_path": None if MODEL_PATH is None else str(MODEL_PATH),
				"preprocess_mode": PREPROCESS_MODE,
				"threshold": THRESHOLD,
				"positive_label": POSITIVE_LABEL,
				"negative_label": NEGATIVE_LABEL,
				"is_demo_model": IS_DEMO_MODEL,
			}
		)
	except Exception as exc:  # noqa: BLE001
		return jsonify({"error": str(exc)}), 500


@app.route("/predict-audio", methods=["GET", "POST"])
def predict_audio():
	if request.method == "GET":
		return jsonify(
			{
				"message": "Use POST /predict-audio with form-data key 'audio' to run audio inference.",
				"example": "curl -X POST http://127.0.0.1:8080/predict-audio -F \"audio=@/path/to/audio.wav\"",
				"model_status": "http://127.0.0.1:8080/model-status",
			}
		)

	ensure_audio_loaded()
	if audio_predict is None:
		return jsonify({"error": f"Audio model failed to load: {AUDIO_LOAD_ERROR}"}), 500

	if "audio" not in request.files:
		return jsonify({"error": "No audio file found in form-data with key 'audio'."}), 400

	audio_file = request.files["audio"]
	if audio_file.filename == "":
		return jsonify({"error": "Empty file name."}), 400

	tmp_file = tempfile.NamedTemporaryFile(suffix=Path(audio_file.filename).suffix or ".wav", delete=False)
	tmp_file.write(audio_file.read())
	tmp_file.close()
	try:
		start_time = time.perf_counter()
		result = audio_predict(tmp_file.name)
		elapsed_seconds = time.perf_counter() - start_time
		result["processing_time_sec"] = round(elapsed_seconds, 3)
		result["audio_file"] = audio_file.filename
		return jsonify(result)
	finally:
		try:
			os.remove(tmp_file.name)
		except OSError:
			pass


@app.route("/predict-multimodal", methods=["GET", "POST"])
def predict_multimodal():
	if request.method == "GET":
		return jsonify(
			{
				"message": "Use POST /predict-multimodal with form-data keys 'image' and/or 'audio'.",
				"example": "curl -X POST http://127.0.0.1:8080/predict-multimodal -F \"image=@/path/to/image.jpg\" -F \"audio=@/path/to/audio.wav\"",
				"model_status": "http://127.0.0.1:8080/model-status",
			}
		)

	if multimodal_analyze is None and MULTIMODAL_LOAD_ERROR is None:
		return jsonify({"error": "Multimodal model is still loading. Please try again in a minute."}), 503
	if multimodal_analyze is None:
		return jsonify({"error": f"Multimodal model failed to load: {MULTIMODAL_LOAD_ERROR}"}), 500

	image_path = None
	audio_path = None

	if "image" in request.files:
		image_file = request.files["image"]
		if image_file.filename != "":
			tmp_image = tempfile.NamedTemporaryFile(suffix=Path(image_file.filename).suffix or ".png", delete=False)
			tmp_image.write(image_file.read())
			tmp_image.close()
			image_path = tmp_image.name
	else:
		tmp_image = None

	if "audio" in request.files:
		audio_file = request.files["audio"]
		if audio_file.filename != "":
			tmp_audio = tempfile.NamedTemporaryFile(suffix=Path(audio_file.filename).suffix or ".wav", delete=False)
			tmp_audio.write(audio_file.read())
			tmp_audio.close()
			audio_path = tmp_audio.name
	else:
		tmp_audio = None

	if image_path is None and audio_path is None:
		return jsonify({"error": "Upload at least one file using the 'image' or 'audio' form-data keys."}), 400

	try:
		start_time = time.perf_counter()
		result = multimodal_analyze(audio_path=audio_path, image_path=image_path)
		elapsed_seconds = time.perf_counter() - start_time
		result["processing_time_sec"] = round(elapsed_seconds, 3)
		return jsonify(result)
	finally:
		for path in (image_path, audio_path):
			if path:
				try:
					os.remove(path)
				except OSError:
					pass


@app.route("/")
def serve_index():
	return send_from_directory(WORKSPACE_DIR, "index.html")


@app.route("/style.css")
def serve_style():
	return send_from_directory(WORKSPACE_DIR, "style.css")


if __name__ == "__main__":
	print(f"Config path: {CONFIG_PATH}")
	print(f"Selected model: {MODEL_PATH}")
	print(f"Demo model (untrained placeholder): {IS_DEMO_MODEL}")
	print(f"Preprocess mode: {PREPROCESS_MODE}")
	print(f"Threshold: {THRESHOLD}")
	print(f"Positive label: {POSITIVE_LABEL}")
	print(f"Negative label: {NEGATIVE_LABEL}")
	start_multimodal_background_load()
	app.run(debug=False, use_reloader=False, host="127.0.0.1", port=8080)
