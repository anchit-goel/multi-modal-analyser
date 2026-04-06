import torch
import numpy as np
import librosa

DEVICE = -1

print("Loading calibrator model...")

detector = None
MODEL_LOADED = False

try:
    from transformers import pipeline
    detector = pipeline(
        "audio-classification",
        model="Gustking/wav2vec2-large-xlsr-deepfake-audio-classification",
        device=DEVICE
    )
    MODEL_LOADED = True
    print("Calibrator loaded successfully!")
except Exception as e:
    print(f"Calibrator failed to load: {e}")
    MODEL_LOADED = False


def predict_pretrained(audio_path):
    if not MODEL_LOADED or detector is None:
        return None

    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    audio = audio.astype(np.float32)

    result = detector(audio, sampling_rate=16000)

    spoof_score = 0.5
    for item in result:
        if item['label'].lower() in ['fake', 'spoof', 'synthetic']:
            spoof_score = item['score']
            break
        elif item['label'].lower() in ['real', 'genuine', 'bonafide']:
            spoof_score = 1 - item['score']
            break

    return float(spoof_score)