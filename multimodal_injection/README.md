# Multimodal Prompt Injection Detection

Detects hidden/adversarial instructions in images, audio, and cross-modal mismatches.

## Setup
pip install -r requirements.txt

## Run API
uvicorn main:app --reload --host 0.0.0.0 --port 8001

## Endpoints
- POST /detect/image     — image injection detection
- POST /detect/audio     — audio injection detection  
- POST /detect/multimodal — full cross-modal analysis
- GET  /health           — health check

## From-scratch components
- Hidden text detector (pixel-level contrast analysis)
- Prompt injection keyword scanner
- TF-IDF + Jaccard semantic similarity
- Mismatch risk scoring and ensemble fusion

## Pretrained models (used as sensors only)
- openai/whisper-base — speech to text
- Salesforce/blip-image-captioning-base — image to text
- openai/clip-vit-base-patch32 — image-text alignment