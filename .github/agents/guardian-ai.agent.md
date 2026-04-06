---
name: guardian-ai-runner
description: "Use when running, starting, or verifying the Guardian AI project on Windows, macOS, or Linux."
---

# Guardian AI Run Guide

Use this agent guide when you need to start or verify the Guardian AI app from a fresh checkout.

## Project Layout

Run the project from the `guardian-ai` folder that contains:
- `app.py`
- `modules/`
- `models/`
- `docs/`
- `scripts/`
- `requirements.txt`

## Runtime Rules

- Always run commands from the `guardian-ai` project root.
- Keep the existing `.venv` in the parent workspace only as a local development convenience.
- Do not move the model files out of `models/`.
- Do not move `audio_spoof_system` or `multimodal_injection` out of `modules/`.
- The Flask app serves on `http://127.0.0.1:8080`.

## Windows Setup and Run

If the environment already has a virtual environment:

```powershell
cd guardian-ai
.\.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

If a virtual environment does not exist yet:

```powershell
cd guardian-ai
py -3 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

If PowerShell blocks script execution, use the Python executable directly:

```powershell
cd guardian-ai
& .\.venv\Scripts\python.exe app.py
```

## macOS Setup and Run

If you are on macOS, use Python 3 and a POSIX shell:

```bash
cd guardian-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

If you already have a virtual environment:

```bash
cd guardian-ai
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Verification Checklist

After startup, confirm:
- `http://127.0.0.1:8080/model-status` responds successfully.
- `model_path` points to a file in `models/`.
- `audio_root` points to `modules/audio_spoof_system`.
- `multimodal_root` points to `modules/multimodal_injection`.
- The audio, image, and multimodal endpoints all load without errors.

## Notes

- The app uses relative paths, so it must be started from the project root.
- Keep `requirements.txt` in sync with any dependency changes.
- On macOS, you may need `brew install python` or a Python 3 build that includes `venv` support.
