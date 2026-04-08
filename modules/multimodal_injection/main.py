from contextlib import asynccontextmanager
from pathlib import Path
import logging
import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from multimodal_detector import (
    DETECTOR_VERSION,
    analyze_multimodal,
    load_multimodal_models,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_multimodal_models()
        logger.info("Multimodal models loaded successfully. detector_version=%s", DETECTOR_VERSION)
    except Exception as exc:
        logger.exception("Model startup failed. detector_version=%s error=%s", DETECTOR_VERSION, exc)
    yield


app = FastAPI(
    title="Multimodal Prompt Injection Detection API",
    version=DETECTOR_VERSION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://your-frontend-domain.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,
)


def save_temp_upload(upload: UploadFile, suffix: str = "") -> str:
    ext = os.path.splitext(upload.filename or "")[1]
    effective_suffix = ext or suffix
    fd, tmp_path = tempfile.mkstemp(prefix="mm_", suffix=effective_suffix)

    try:
        with os.fdopen(fd, "wb") as tmp_file:
            while True:
                chunk = upload.file.read(1024 * 1024)
                if not chunk:
                    break
                tmp_file.write(chunk)
        return tmp_path
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        safe_remove(tmp_path)
        raise
    finally:
        upload.file.close()


def safe_remove(path: str | None):
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            logger.warning("Could not delete temp file: %s", path)


def with_detector_version(payload: dict) -> dict:
    payload["detector_version"] = DETECTOR_VERSION
    return payload


def validate_upload(upload: UploadFile | None, kind: str) -> None:
    if upload is None:
        raise HTTPException(status_code=400, detail=f"{kind} file is required.")
    if not (upload.filename or "").strip():
        raise HTTPException(status_code=400, detail=f"{kind} filename is missing.")


@app.get("/health")
def health():
    return with_detector_version({
        "status": "running",
        "system": "multimodal injection detector"
    })


@app.post("/detect/image")
async def detect_image(image: UploadFile = File(...)):
    validate_upload(image, "Image")

    image_path = None
    try:
        image_path = save_temp_upload(image)
        logger.info("/detect/image request accepted. file=%s", Path(image_path).name)
        result = analyze_multimodal(image_path=image_path)
        return with_detector_version(result)
    except HTTPException:
        raise
    except ValueError as exc:
        logger.exception("Image detection validation failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Image detection failed: %s", exc)
        raise HTTPException(status_code=500, detail="Image detection failed")
    finally:
        safe_remove(image_path)


@app.post("/detect/audio")
async def detect_audio(audio: UploadFile = File(...)):
    validate_upload(audio, "Audio")

    audio_path = None
    try:
        audio_path = save_temp_upload(audio)
        logger.info("/detect/audio request accepted. file=%s", Path(audio_path).name)
        result = analyze_multimodal(audio_path=audio_path)
        return with_detector_version(result)
    except HTTPException:
        raise
    except ValueError as exc:
        logger.exception("Audio detection validation failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Audio detection failed: %s", exc)
        raise HTTPException(status_code=500, detail="Audio detection failed")
    finally:
        safe_remove(audio_path)


@app.post("/detect/multimodal")
async def detect_multimodal(
    audio: UploadFile | None = File(None),
    image: UploadFile | None = File(None)
):
    if audio is None and image is None:
        raise HTTPException(status_code=400, detail="Upload at least one file: audio or image.")

    audio_path = None
    image_path = None

    try:
        if audio is not None:
            validate_upload(audio, "Audio")
            audio_path = save_temp_upload(audio)

        if image is not None:
            validate_upload(image, "Image")
            image_path = save_temp_upload(image)

        logger.info(
            "/detect/multimodal request accepted. audio=%s image=%s",
            Path(audio_path).name if audio_path else None,
            Path(image_path).name if image_path else None,
        )
        result = analyze_multimodal(audio_path=audio_path, image_path=image_path)
        return with_detector_version(result)

    except HTTPException:
        raise
    except ValueError as exc:
        logger.exception("Multimodal detection validation failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Multimodal detection failed: %s", exc)
        raise HTTPException(status_code=500, detail="Multimodal detection failed")

    finally:
        safe_remove(audio_path)
        safe_remove(image_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)