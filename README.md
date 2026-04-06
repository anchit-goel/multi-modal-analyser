# GuardianAI - Multimodal Deepfake Detection

GuardianAI is a Flask-based API and web UI for deepfake and manipulation detection across three pipelines:

- Visual detection (image classifier)
- Audio spoof detection
- Multimodal injection and consistency detection (image + audio)

It exposes simple HTTP endpoints for inference and a lightweight frontend.

## Features

- Image deepfake scoring with configurable threshold
- Audio spoof analysis endpoint
- Multimodal risk scoring with signal breakdown
- Unified model health/status endpoint
- Render deployment support (`render.yaml`)

## Project Structure

- `app.py` - Main Flask server and API routes
- `model_runtime_config.json` - Runtime config (selected visual model, threshold, labels)
- `models/` - Visual model files (`.h5` / `.keras`)
- `modules/audio_spoof_system/` - Audio spoof module
- `modules/multimodal_injection/` - Multimodal analysis module
- `index.html`, `style.css` - Frontend UI
- `render.yaml` - Render Blueprint deployment config
- `.python-version` - Python version pin for Render

## Requirements

- Python 3.10.x (recommended: 3.10.13)
- pip

This repo is configured for Python 3.10 in cloud deploys.

## Local Setup

1. Create and activate a virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. Start the app:

```bash
python app.py
```

Server defaults:

- Host: `0.0.0.0`
- Port: `8080` (or `$PORT` if set)

Open:

- `http://127.0.0.1:8080/`

## Configuration

Edit `model_runtime_config.json`:

- `selected_model`: path to visual model (currently `models/visual_model.h5`)
- `threshold`: decision threshold for visual classifier
- `positive_label` / `negative_label`
- `input_size`

Example:

```json
{
  "selected_model": "models/visual_model.h5",
  "allow_demo_model": true,
  "preprocess_mode": "rescale_255",
  "threshold": 0.01,
  "positive_label": "REAL",
  "negative_label": "FAKE",
  "input_size": [224, 224]
}
```

## API Endpoints

### 1) Health and model status

- `GET /model-status`

Returns load status for visual, audio, and multimodal models.

```bash
curl -s http://127.0.0.1:8080/model-status
```

### 2) Visual prediction

- `POST /predict`
- Form-data key: `image`

```bash
curl -X POST http://127.0.0.1:8080/predict \
  -F "image=@/path/to/image.jpg"
```

### 3) Audio prediction

- `POST /predict-audio`
- Form-data key: `audio`

```bash
curl -X POST http://127.0.0.1:8080/predict-audio \
  -F "audio=@/path/to/audio.wav"
```

### 4) Multimodal prediction

- `POST /predict-multimodal`
- Form-data keys: `image` and/or `audio`

```bash
curl -X POST http://127.0.0.1:8080/predict-multimodal \
  -F "image=@/path/to/image.jpg" \
  -F "audio=@/path/to/audio.wav"
```

## Render Deployment

This project includes Render Blueprint config in `render.yaml`.

### Deploy steps

1. Push code to GitHub.
2. In Render dashboard: **New + -> Blueprint**.
3. Select this repository and branch.
4. Deploy.

### Important notes

- Python version is pinned via `.python-version` (`3.10.13`).
- Build can take time because ML dependencies are heavy.
- First request is usually slower due to model loading/cache warmup.

## Troubleshooting

### Port already in use

If you see "Address already in use" for port 8080, stop the existing process first.

### Old model path still showing

If `/model-status` shows an old model path after config updates, restart the Flask process so it reloads `model_runtime_config.json`.

### Render build fails for TensorFlow

Ensure Render is using Python 3.10.x (check deploy logs). If needed, set `PYTHON_VERSION=3.10.13` in Render environment variables and redeploy.

## License

Add your preferred license here.
