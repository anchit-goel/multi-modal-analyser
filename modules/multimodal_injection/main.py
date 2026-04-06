from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from multimodal_detector import analyze_multimodal
import shutil, os, uuid

app = FastAPI(title="Multimodal Prompt Injection Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain.com"],  # ← restrict in prod
    max_age=3600,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/detect/image")
async def detect_image(image: UploadFile = File(...)):
    image_path = f"temp_{uuid.uuid4().hex}_{image.filename}"
    with open(image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)
    try:
        result = analyze_multimodal(image_path=image_path)
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)
    return result

@app.post("/detect/audio")
async def detect_audio(audio: UploadFile = File(...)):
    audio_path = f"temp_{uuid.uuid4().hex}_{audio.filename}"
    with open(audio_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)
    try:
        result = analyze_multimodal(audio_path=audio_path)
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
    return result

@app.post("/detect/multimodal")
async def detect_multimodal(
    audio: UploadFile = File(None),
    image: UploadFile = File(None)
):
    audio_path = None
    image_path = None

    if audio:
        audio_path = f"temp_{uuid.uuid4().hex}_{audio.filename}"
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)

    if image:
        image_path = f"temp_{uuid.uuid4().hex}_{image.filename}"
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

    try:
        result = analyze_multimodal(audio_path=audio_path, image_path=image_path)
    finally:
        for path in [audio_path, image_path]:
            if path and os.path.exists(path):
                os.remove(path)

    return result

@app.get("/health")
def health():
    return {"status": "running", "system": "multimodal injection detector"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)