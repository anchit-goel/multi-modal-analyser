from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
import librosa
import numpy as np

from models_architecture import CNNModel, LCNNModel

DEVICE = torch.device("cpu")
BASE_DIR = Path(__file__).resolve().parent

# Load your models (built from scratch)
cnn_model = CNNModel().to(DEVICE)
cnn_model.load_state_dict(torch.load(BASE_DIR / "models" / "cnn_best_clean.pth", map_location=DEVICE))
cnn_model.eval()

lcnn_model = LCNNModel().to(DEVICE)
lcnn_model.load_state_dict(torch.load(BASE_DIR / "models" / "lcnn_best_clean.pth", map_location=DEVICE))
lcnn_model.eval()

# Load pretrained as calibrator (secondary, not primary)
try:
    from pretrained_detector import predict_pretrained
    CALIBRATOR_AVAILABLE = True
    print("All models loaded — CNN + LCNN (primary) + Calibrator (secondary)")
except Exception as e:
    CALIBRATOR_AVAILABLE = False
    print(f"Running with CNN + LCNN only: {e}")


def extract_logmel(audio, sr=16000):
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=80,
        n_fft=400, hop_length=160
    )
    logmel = librosa.power_to_db(mel)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-8)
    return logmel


def extract_lfcc(audio, sr=16000):
    waveform = torch.FloatTensor(audio).unsqueeze(0)
    lfcc_transform = torchaudio.transforms.LFCC(
        sample_rate=sr,
        n_lfcc=60,
        speckwargs={"n_fft": 512, "hop_length": 160}
    )
    lfcc = lfcc_transform(waveform).squeeze(0).numpy()
    lfcc = (lfcc - lfcc.mean()) / (lfcc.std() + 1e-8)
    return lfcc


def get_risk_tier(score):
    if score < 0.3:
        return "LOW"
    elif score < 0.6:
        return "MEDIUM"
    elif score < 0.85:
        return "HIGH"
    else:
        return "CRITICAL"


def calibrate_score(cnn_prob, lcnn_prob, calibrator_prob):
    """
    Your CNN+LCNN are primary.
    Calibrator only corrects when your models are overconfident
    and calibrator strongly disagrees.
    """
    # Base score from your models (primary — built from scratch)
    base_score = (cnn_prob * 0.55) + (lcnn_prob * 0.45)

    # Calibration logic
    both_overconfident = cnn_prob > 0.85 and lcnn_prob > 0.85
    calibrator_says_real = calibrator_prob < 0.25

    if both_overconfident and calibrator_says_real:
        # Your models are overconfident, calibrator disagrees strongly
        # Apply correction but your models still lead
        corrected = (base_score * 0.55) + (calibrator_prob * 0.45)
        return corrected, "calibrated"
    else:
        # Normal case — your models decide, calibrator has small influence
        final = (base_score * 0.75) + (calibrator_prob * 0.25)
        return final, "standard"


def predict(audio_path):
    # Load and preprocess
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    audio = audio.astype(np.float32)

    # Extract features
    logmel = extract_logmel(audio)
    lfcc   = extract_lfcc(audio)

    logmel_tensor = torch.tensor(logmel).unsqueeze(0).unsqueeze(0).float()
    lfcc_tensor   = torch.tensor(lfcc).unsqueeze(0).unsqueeze(0).float()

    with torch.no_grad():
        cnn_logits  = cnn_model(logmel_tensor)
        lcnn_logits = lcnn_model(lfcc_tensor)

        cnn_prob  = F.softmax(cnn_logits,  dim=1)[0, 1].item()
        lcnn_prob = F.softmax(lcnn_logits, dim=1)[0, 1].item()

    # Base score from your scratch-built models
    base_score = (cnn_prob * 0.55) + (lcnn_prob * 0.45)

    if CALIBRATOR_AVAILABLE:
        try:
            calibrator_prob = predict_pretrained(audio_path)
            if calibrator_prob is None:
                raise ValueError("Calibrator unavailable")
            final_score, mode = calibrate_score(cnn_prob, lcnn_prob, calibrator_prob)
            models_used = f"CNN + LCNN (primary) | Calibrator ({mode})"
        except Exception as e:
            print(f"Calibrator failed: {e}")
            final_score = base_score
            calibrator_prob = None
            models_used = "CNN + LCNN"
    else:
        final_score = base_score
        calibrator_prob = None
        models_used = "CNN + LCNN"

    risk_tier = get_risk_tier(final_score)
    verdict   = "SPOOF" if final_score >= 0.5 else "GENUINE"

    # Build explanation
    if calibrator_prob is not None:
        explanation = (
            f"Primary models — CNN: {cnn_prob:.1%}, LCNN: {lcnn_prob:.1%}. "
            f"Calibrator: {calibrator_prob:.1%}. "
            f"Final risk: {final_score:.1%} ({risk_tier})"
        )
    else:
        explanation = (
            f"CNN: {cnn_prob:.1%} | LCNN: {lcnn_prob:.1%} | "
            f"Final risk: {final_score:.1%} ({risk_tier})"
        )

    return {
        "cnn_score":        round(cnn_prob,         4),
        "lcnn_score":       round(lcnn_prob,         4),
        "calibrator_score": round(calibrator_prob,   4) if calibrator_prob is not None else None,
        "risk_score":       round(final_score,       4),
        "verdict":          verdict,
        "risk_tier":        risk_tier,
        "models_used":      models_used,
        "explanation":      explanation
    }


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "sample.wav"
    result = predict(path)
    for k, v in result.items():
        print(f"{k}: {v}")