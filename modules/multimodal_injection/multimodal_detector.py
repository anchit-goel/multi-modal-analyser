import logging
import os
import re
import traceback
import importlib.util
import sys
from collections import Counter

import easyocr
import librosa
import numpy as np
import torch
from PIL import Image
from rapidfuzz import fuzz
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    CLIPModel,
    CLIPProcessor,
    pipeline,
)

MODELS_LOADED = False
MODEL_LOAD_ERROR = None

DETECTOR_VERSION = "v3-audio-balanced-2026-04-09"

logger = logging.getLogger(__name__)

ocr_reader = None
whisper = None
blip_processor = None
blip_model = None
clip_processor = None
clip_model = None
audio_spoof_predict_fn = None
audio_spoof_predict_load_attempted = False
audio_spoof_predict_load_error = None


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(max(low, min(high, value)))


def normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return clamp((value - low) / (high - low), 0.0, 1.0)


def load_multimodal_models():
    global MODELS_LOADED, MODEL_LOAD_ERROR
    global ocr_reader, whisper, blip_processor, blip_model, clip_processor, clip_model

    if MODELS_LOADED:
        return
    if MODEL_LOAD_ERROR is not None:
        raise RuntimeError(MODEL_LOAD_ERROR)

    try:
        ocr_reader = easyocr.Reader(["en"], gpu=False)
        whisper = pipeline("automatic-speech-recognition", model="openai/whisper-base")

        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model.eval()

        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_model.eval()

        MODELS_LOADED = True
        logger.info("Multimodal detector models ready. detector_version=%s", DETECTOR_VERSION)
    except Exception:
        MODEL_LOAD_ERROR = traceback.format_exc()
        logger.exception("Failed to load multimodal detector models. detector_version=%s", DETECTOR_VERSION)
        raise


INJECTION_KEYWORDS = [
    "new instruction",
    "system prompt",
    "you are now",
    "act as",
    "from now on",
    "ignore previous instructions",
    "ignore above instructions",
    "do not follow previous instructions",
    "print your instructions",
    "what are your instructions",
    "reveal your instructions",
]


def tokenize(text: str):
    return re.findall(r"\b\w+\b", text.lower())


def compute_tfidf_vector(text, vocab):
    tokens = tokenize(text)
    tf = Counter(tokens)
    vector = np.array([tf.get(word, 0) for word in vocab], dtype=float)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector


def from_scratch_similarity(text1, text2):
    tokens1 = set(tokenize(text1))
    tokens2 = set(tokenize(text2))

    if not tokens1 or not tokens2:
        return 0.0

    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    jaccard = len(intersection) / (len(union) + 1e-8)

    vocab = list(union)
    vec1 = compute_tfidf_vector(text1, vocab)
    vec2 = compute_tfidf_vector(text2, vocab)
    cosine = float(np.dot(vec1, vec2))

    similarity = (0.4 * jaccard) + (0.6 * cosine)
    return round(float(similarity), 4)


def detect_low_contrast_text(image_path, block_size=32):
    img = np.array(Image.open(image_path).convert("L"))
    h, w = img.shape

    suspicious_blocks = 0
    total_blocks = 0

    for i in range(0, max(h - block_size + 1, 1), block_size):
        for j in range(0, max(w - block_size + 1, 1), block_size):
            block = img[i:i + block_size, j:j + block_size]
            if block.size == 0:
                continue

            std = float(np.std(block))
            mean = float(np.mean(block))
            total_blocks += 1

            if 4 <= std <= 18 and 25 <= mean <= 230:
                suspicious_blocks += 1

    suspicion_ratio = suspicious_blocks / (total_blocks + 1e-8)
    hidden_risk = min(suspicion_ratio * 0.25, 0.12)

    return {
        "suspicious_block_ratio": round(suspicion_ratio, 4),
        "hidden_content_risk": round(hidden_risk, 4),
        "verdict": "SUSPICIOUS" if hidden_risk > 0.18 else "CLEAN",
    }


def extract_text_ocr(image_path, min_confidence=0.70):
    results = ocr_reader.readtext(image_path)

    high_conf = []
    low_conf = []

    for _, text, conf in results:
        cleaned = text.strip()
        if not cleaned:
            continue

        item = {
            "text": cleaned,
            "confidence": float(conf),
            "length": len(cleaned),
        }

        if conf >= min_confidence:
            high_conf.append(item)
        else:
            low_conf.append(item)

    extracted = " ".join(x["text"] for x in high_conf)
    avg_conf = sum(x["confidence"] for x in high_conf) / len(high_conf) if high_conf else 0.0

    return {
        "text": extracted.strip(),
        "high_confidence_tokens": high_conf,
        "low_confidence_tokens": low_conf,
        "avg_confidence": round(avg_conf, 3),
        "token_count": len(high_conf),
    }


def is_reliable_ocr(ocr_result: dict) -> bool:
    token_count = int(ocr_result.get("token_count", 0))
    avg_confidence = float(ocr_result.get("avg_confidence", 0.0))
    text = (ocr_result.get("text") or "").strip()
    compact_len = len(re.sub(r"\s+", "", text))
    return token_count >= 4 and avg_confidence >= 0.80 and compact_len >= 12


def scan_for_injection(text, fuzzy_threshold=95):
    text_lower = text.lower().strip()
    if not text_lower:
        return {
            "exact_matches": [],
            "fuzzy_matches": [],
            "injection_keywords_found": [],
            "injection_risk": 0.0,
            "verdict": "CLEAN",
        }

    words = text_lower.split()
    found_exact = []
    found_fuzzy = []

    for keyword in INJECTION_KEYWORDS:
        exact_pattern = r"\\b" + re.escape(keyword) + r"\\b"
        if re.search(exact_pattern, text_lower):
            found_exact.append(keyword)
            continue

        kw_words = keyword.split()
        window_size = len(kw_words)

        for i in range(len(words) - window_size + 1):
            window = " ".join(words[i:i + window_size])
            score = fuzz.partial_ratio(keyword, window)
            if score >= fuzzy_threshold:
                found_fuzzy.append({
                    "keyword": keyword,
                    "matched": window,
                    "confidence": score,
                })
                break

    all_found = found_exact + [f["keyword"] for f in found_fuzzy]
    unique_found = sorted(set(all_found))

    if len(unique_found) >= 5:
        risk = 0.90
    elif len(unique_found) == 4:
        risk = 0.72
    elif len(unique_found) == 3:
        risk = 0.45
    elif len(unique_found) == 2:
        risk = 0.22
    elif len(unique_found) == 1:
        risk = 0.08
    else:
        risk = 0.0

    return {
        "exact_matches": found_exact,
        "fuzzy_matches": found_fuzzy,
        "injection_keywords_found": unique_found,
        "injection_risk": round(risk, 4),
        "verdict": "INJECTION DETECTED" if risk >= 0.6 else "CLEAN",
    }


def transcribe_audio(audio_path):
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    result = whisper(audio.astype(np.float32))
    return result["text"].strip()


def load_optional_audio_spoof_predictor():
    global audio_spoof_predict_fn, audio_spoof_predict_load_attempted, audio_spoof_predict_load_error

    if audio_spoof_predict_load_attempted:
        return audio_spoof_predict_fn

    audio_spoof_predict_load_attempted = True

    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        inference_path = os.path.join(base_dir, "..", "audio_spoof_system", "inference.py")
        inference_path = os.path.abspath(inference_path)
        audio_spoof_dir = os.path.dirname(inference_path)

        if not os.path.exists(inference_path):
            audio_spoof_predict_load_error = f"inference.py not found: {inference_path}"
            logger.info("Optional audio spoof model unavailable: %s", audio_spoof_predict_load_error)
            return None

        spec = importlib.util.spec_from_file_location("guardian_audio_spoof_inference", inference_path)
        if spec is None or spec.loader is None:
            audio_spoof_predict_load_error = "Could not create import spec for audio spoof inference"
            logger.warning("Optional audio spoof model unavailable: %s", audio_spoof_predict_load_error)
            return None

        if audio_spoof_dir not in sys.path:
            sys.path.insert(0, audio_spoof_dir)

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        candidate = getattr(module, "predict", None)

        if callable(candidate):
            audio_spoof_predict_fn = candidate
            logger.info("Optional audio spoof predictor loaded for multimodal fusion")
            return audio_spoof_predict_fn

        audio_spoof_predict_load_error = "predict() not found in audio spoof inference module"
        logger.warning("Optional audio spoof model unavailable: %s", audio_spoof_predict_load_error)
        return None
    except Exception:
        audio_spoof_predict_load_error = traceback.format_exc()
        logger.warning("Optional audio spoof model failed to load: %s", audio_spoof_predict_load_error)
        return None


def _safe_stat(arr, stat="std"):
    if arr is None or len(arr) == 0:
        return 0.0
    if stat == "mean":
        return float(np.mean(arr))
    if stat == "median":
        return float(np.median(arr))
    return float(np.std(arr))


def analyze_audio_authenticity(audio_path: str):
    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
    except Exception as exc:
        return {
            "audio_spoof_risk": 0.5,
            "verdict": "UNCERTAIN",
            "feature_summary": {"error": str(exc)},
            "analysis_confidence": 0.2,
            "note": "Audio load failed; using neutral fallback risk",
        }

    duration_sec = len(y) / float(sr + 1e-8)
    if duration_sec < 0.6:
        return {
            "audio_spoof_risk": 0.55,
            "verdict": "UNCERTAIN",
            "feature_summary": {
                "duration_sec": round(duration_sec, 3),
                "reason": "audio too short for stable authenticity analysis",
            },
            "analysis_confidence": 0.25,
            "note": "Short audio can be unreliable for authenticity profiling",
        }

    y = librosa.util.normalize(y)
    frame_length = 2048
    hop = 512

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop).flatten()
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop).flatten()
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop).flatten()

    rms_median = _safe_stat(rms, "median")
    voiced_mask = rms > max(0.008, rms_median * 0.65)

    f0 = librosa.yin(y, fmin=60, fmax=350, sr=sr, frame_length=frame_length, hop_length=hop)
    finite_f0 = np.isfinite(f0)
    voiced_f0 = f0[finite_f0]
    if len(voiced_mask) == len(f0):
        voiced_f0 = f0[np.logical_and(finite_f0, voiced_mask)]

    pitch_mean = _safe_stat(voiced_f0, "mean")
    pitch_std = _safe_stat(voiced_f0, "std")
    pitch_cv = pitch_std / (pitch_mean + 1e-8) if pitch_mean > 0 else 0.0

    energy_cv = _safe_stat(rms, "std") / (_safe_stat(rms, "mean") + 1e-8)
    zcr_mean = _safe_stat(zcr, "mean")
    zcr_std = _safe_stat(zcr, "std")
    spectral_cv = _safe_stat(centroid, "std") / (_safe_stat(centroid, "mean") + 1e-8)

    rms_p10 = float(np.percentile(rms, 10)) if len(rms) else 0.0
    rms_p90 = float(np.percentile(rms, 90)) if len(rms) else 0.0
    rms_dynamic_range = max(0.0, rms_p90 - rms_p10)
    pause_ratio = float(np.mean(rms < max(0.0045, rms_median * 0.45))) if len(rms) else 0.0

    pitch_flat_risk = 1.0 - normalize(pitch_cv, 0.08, 0.30)
    energy_flat_risk = 1.0 - normalize(energy_cv, 0.22, 0.70)
    zcr_flat_risk = 1.0 - normalize(zcr_std, 0.015, 0.065)
    spectral_flat_risk = 1.0 - normalize(spectral_cv, 0.10, 0.36)
    rms_dynamic_risk = 1.0 - normalize(rms_dynamic_range, 0.03, 0.20)

    zcr_profile_risk = 0.0
    if zcr_mean < 0.018 or zcr_mean > 0.19:
        zcr_profile_risk = 0.20

    pause_profile_risk = 0.0
    if pause_ratio < 0.03:
        pause_profile_risk = 0.18
    elif pause_ratio > 0.52:
        pause_profile_risk = 0.10

    voiced_frames = int(np.sum(voiced_mask))
    total_frames = int(len(voiced_mask))
    voiced_ratio = voiced_frames / float(total_frames + 1e-8)
    confidence = clamp(0.35 + normalize(duration_sec, 1.0, 8.0) * 0.35 + normalize(voiced_ratio, 0.12, 0.55) * 0.30)

    weighted_risk = (
        0.28 * clamp(pitch_flat_risk)
        + 0.22 * clamp(energy_flat_risk)
        + 0.16 * clamp(zcr_flat_risk)
        + 0.18 * clamp(spectral_flat_risk)
        + 0.16 * clamp(rms_dynamic_risk)
        + zcr_profile_risk
        + pause_profile_risk
    )

    # Confidence-aware smoothing:
    # low-confidence clips stay closer to neutral, high-confidence clips preserve stronger risk.
    audio_spoof_risk = clamp(0.5 + (weighted_risk - 0.5) * (0.55 + 0.45 * confidence))

    if len(voiced_f0) < 12:
        audio_spoof_risk = max(audio_spoof_risk, 0.58)

    model_assisted = {
        "available": False,
        "risk_score": None,
        "verdict": None,
        "models_used": None,
    }

    model_predictor = load_optional_audio_spoof_predictor()
    if model_predictor is not None:
        try:
            model_result = model_predictor(audio_path)
            model_risk = clamp(float(model_result.get("risk_score", 0.5)))

            model_assisted = {
                "available": True,
                "risk_score": round(model_risk, 4),
                "verdict": model_result.get("verdict"),
                "models_used": model_result.get("models_used"),
                "calibration_mode": model_result.get("calibration_mode"),
            }

            combined = clamp((0.35 * audio_spoof_risk) + (0.65 * model_risk))

            if model_risk >= 0.80:
                combined = max(combined, 0.78)
            if model_risk >= 0.72 and audio_spoof_risk >= 0.45:
                combined = max(combined, 0.75)
            if model_risk <= 0.25 and audio_spoof_risk <= 0.35:
                combined = min(combined, 0.35)

            audio_spoof_risk = combined
        except Exception as exc:
            model_assisted = {
                "available": False,
                "risk_score": None,
                "verdict": None,
                "models_used": None,
                "error": str(exc),
            }

    if audio_spoof_risk >= 0.75:
        verdict = "LIKELY_SPOOF"
    elif audio_spoof_risk >= 0.60:
        verdict = "SUSPICIOUS"
    else:
        verdict = "LIKELY_GENUINE"

    feature_summary = {
        "duration_sec": round(duration_sec, 3),
        "pitch_variation_cv": round(pitch_cv, 4),
        "energy_variation_cv": round(energy_cv, 4),
        "zcr_mean": round(zcr_mean, 4),
        "zcr_std": round(zcr_std, 4),
        "spectral_centroid_variation_cv": round(spectral_cv, 4),
        "rms_dynamic_range": round(rms_dynamic_range, 4),
        "pause_ratio": round(pause_ratio, 4),
        "voiced_ratio": round(voiced_ratio, 4),
        "component_risks": {
            "pitch_flat_risk": round(clamp(pitch_flat_risk), 4),
            "energy_flat_risk": round(clamp(energy_flat_risk), 4),
            "zcr_flat_risk": round(clamp(zcr_flat_risk), 4),
            "spectral_flat_risk": round(clamp(spectral_flat_risk), 4),
            "rms_dynamic_risk": round(clamp(rms_dynamic_risk), 4),
            "zcr_profile_risk": round(clamp(zcr_profile_risk), 4),
            "pause_profile_risk": round(clamp(pause_profile_risk), 4),
        },
    }

    analysis_note = "Heuristic authenticity estimator"
    if model_assisted.get("available"):
        analysis_note = "Heuristic + dedicated audio spoof model fusion"
    elif audio_spoof_predict_load_error:
        analysis_note = "Heuristic authenticity estimator (dedicated anti-spoof model unavailable)"

    return {
        "audio_spoof_risk": round(audio_spoof_risk, 4),
        "verdict": verdict,
        "feature_summary": feature_summary,
        "model_assisted_analysis": model_assisted,
        "analysis_confidence": round(confidence, 4),
        "note": analysis_note,
    }


def caption_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(image, return_tensors="pt")
    with torch.no_grad():
        output = blip_model.generate(**inputs, max_length=40)
    return blip_processor.decode(output[0], skip_special_tokens=True).strip()


def get_clip_alignment(image_path, text):
    clean_text = re.sub(r"[\W_]+", "", text)
    if len(clean_text) < 12:
        return None, "SKIPPED — transcript too short/noisy"

    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(text=[text], images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        score = torch.sigmoid(outputs.logits_per_image).item()

    return round(float(score), 4), "OK"


def score_mismatch_risk(similarity, image_caption="", transcript="", audio_spoof_risk=0.0):
    caption_lower = image_caption.lower()
    transcript_len = len(tokenize(transcript))

    generic_speaking_scene = any(
        phrase in caption_lower
        for phrase in [
            "a person speaking",
            "a woman speaking",
            "a man speaking",
            "speaking at a podium",
            "standing at a podium",
            "at a microphone",
            "giving a speech",
            "person giving a speech",
            "person speaking to a crowd",
            "woman giving a speech",
            "man giving a speech",
        ]
    )

    if generic_speaking_scene and transcript_len >= 8:
        if audio_spoof_risk >= 0.65 and similarity < 0.12:
            return 0.24, "MEDIUM", "Generic speaking scene; elevated due to high audio spoof + very low similarity"
        return 0.08, "LOW", "Generic speaking scene — transcript/caption mismatch unreliable"

    if similarity > 0.50:
        base_risk = 0.05
        level = "LOW"
        note = "Modalities consistent"
    elif similarity > 0.30:
        base_risk = 0.16
        level = "LOW"
        note = "Weak mismatch"
    elif similarity > 0.15:
        base_risk = 0.28
        level = "MEDIUM"
        note = "Moderate mismatch"
    else:
        base_risk = 0.42
        level = "MEDIUM"
        note = "Strong mismatch"

    if audio_spoof_risk >= 0.65 and similarity < 0.18:
        base_risk = min(0.58, base_risk + 0.10)
        note = f"{note}; boosted by high audio spoof risk"
        if base_risk >= 0.34:
            level = "MEDIUM"

    return round(base_risk, 4), level, note


def compute_overall_risk(signal_scores):
    weights = {
        "audio_spoof": 0.36,
        "ocr_injection": 0.29,
        "mismatch": 0.18,
        "hidden_text": 0.07,
        "clip": 0.07,
        "audio_injection": 0.10,
    }

    weighted_sum = 0.0
    total_weight = 0.0
    breakdown = {}

    for signal, score in signal_scores.items():
        w = weights.get(signal, 0.08)
        contribution = score * w
        breakdown[signal] = {
            "raw_score": round(score, 4),
            "weight": w,
            "contribution": round(contribution, 4),
        }
        weighted_sum += contribution
        total_weight += w

    overall = weighted_sum / total_weight if total_weight > 0 else 0.0

    return {
        "raw_overall_risk_score": round(overall, 4),
        "overall_risk_score": round(overall, 4),
        "signal_breakdown": breakdown,
    }


def _tier_from_score(score: float) -> str:
    if score >= 0.80:
        return "CRITICAL"
    if score >= 0.60:
        return "HIGH"
    if score >= 0.35:
        return "MEDIUM"
    return "LOW"


def _verdict_from_tier(tier: str) -> str:
    if tier == "CRITICAL":
        return "CRITICAL — injection attack detected"
    if tier == "HIGH":
        return "HIGH RISK — likely manipulation"
    if tier == "MEDIUM":
        return "SUSPICIOUS — review required"
    return "SAFE"


def _apply_rule_escalations(raw_score: float, signal_scores: dict):
    final_score = clamp(raw_score)
    decision_notes = []
    escalation_applied = False

    ocr_signal = float(signal_scores.get("ocr_injection", 0.0))
    hidden_signal = float(signal_scores.get("hidden_text", 0.0))
    mismatch_signal = float(signal_scores.get("mismatch", 0.0))
    clip_signal = float(signal_scores.get("clip", 0.0))
    audio_spoof_signal = float(signal_scores.get("audio_spoof", 0.0))

    # Rule 1: spoof audio should not stay LOW.
    if audio_spoof_signal >= 0.65 and final_score < 0.35:
        final_score = 0.35
        escalation_applied = True
        decision_notes.append("Escalation rule: audio_spoof_risk >= 0.65 forces at least MEDIUM tier")

    # Rule 2: strong spoof audio should not stay below HIGH.
    if audio_spoof_signal >= 0.75 and final_score < 0.60:
        final_score = 0.60
        escalation_applied = True
        decision_notes.append("Escalation rule: audio_spoof_risk >= 0.75 forces at least HIGH tier")

    # Rule 3: strong spoof + corroborating multimodal evidence can push higher in HIGH band.
    corroborating_signal = mismatch_signal >= 0.20 or ocr_signal >= 0.22
    if audio_spoof_signal >= 0.75 and corroborating_signal:
        boost = 0.08
        if mismatch_signal >= 0.35 or ocr_signal >= 0.45:
            boost = 0.12
        target_score = min(0.79, max(final_score, 0.60) + boost)
        if target_score > final_score:
            final_score = target_score
            escalation_applied = True
            decision_notes.append("Escalation rule: high audio spoof risk plus mismatch/OCR corroboration increased final score")

    # Rule 4: no strong spoof and weak signals -> keep conservative low score.
    weak_signals = (
        audio_spoof_signal < 0.45
        and ocr_signal < 0.18
        and mismatch_signal < 0.20
        and hidden_signal < 0.08
        and clip_signal < 0.18
    )
    if weak_signals and raw_score < 0.35 and not escalation_applied:
        decision_notes.append("No escalation: weak multimodal signals with no strong spoof evidence")

    final_score = clamp(final_score)
    final_tier = _tier_from_score(final_score)
    final_verdict = _verdict_from_tier(final_tier)

    return {
        "raw_overall_risk_score": round(clamp(raw_score), 4),
        "final_overall_risk_score": round(final_score, 4),
        "risk_tier": final_tier,
        "overall_verdict": final_verdict,
        "decision_notes": decision_notes,
        "escalation_applied": escalation_applied,
    }


def analyze_multimodal(audio_path=None, image_path=None):
    load_multimodal_models()

    results = {
        "detector_version": DETECTOR_VERSION,
        "hidden_text_analysis": None,
        "ocr_analysis": None,
        "injection_scan": None,
        "audio_visual_mismatch": None,
        "clip_alignment": None,
        "image_caption": None,
        "audio_transcript": None,
        "semantic_similarity": None,
        "audio_authenticity_analysis": None,
        "audio_spoof_risk": 0.0,
        "raw_overall_risk_score": 0.0,
        "final_overall_risk_score": 0.0,
        "overall_risk_score": 0.0,
        "overall_verdict": "SAFE",
        "risk_tier": "LOW",
        "flags": [],
        "decision_notes": [],
        "explanation": "",
        "signal_breakdown": {},
    }

    signal_scores = {}
    image_caption = None
    transcript = None

    if image_path:
        hidden = detect_low_contrast_text(image_path)
        results["hidden_text_analysis"] = hidden

        ocr_result = extract_text_ocr(image_path, min_confidence=0.75)
        results["ocr_analysis"] = ocr_result

        image_caption = caption_image(image_path)
        results["image_caption"] = image_caption

        if is_reliable_ocr(ocr_result):
            hidden_risk = min(hidden["hidden_content_risk"] + 0.10, 0.28)
        else:
            hidden_risk = min(hidden["hidden_content_risk"], 0.06)

        signal_scores["hidden_text"] = hidden_risk

        if hidden_risk >= 0.18:
            results["flags"].append(
                f"Possible hidden/embedded text in image (weak signal, ratio: {hidden['suspicious_block_ratio']:.1%})"
            )

        # Keep protection: OCR-driven injection is trusted only when OCR is reliable.
        ocr_is_reliable = is_reliable_ocr(ocr_result)
        if ocr_is_reliable:
            ocr_injection = scan_for_injection(ocr_result["text"])
        else:
            ocr_injection = {
                "exact_matches": [],
                "fuzzy_matches": [],
                "injection_keywords_found": [],
                "injection_risk": 0.0,
                "verdict": "CLEAN",
                "note": "OCR confidence too low to trust for injection detection",
            }

        results["injection_scan"] = ocr_injection
        signal_scores["ocr_injection"] = ocr_injection["injection_risk"]

        if ocr_injection["injection_keywords_found"]:
            results["flags"].append(
                f"Prompt-injection keywords in OCR text: {ocr_injection['injection_keywords_found']}"
            )

    if audio_path:
        transcript = transcribe_audio(audio_path)
        results["audio_transcript"] = transcript

        audio_auth = analyze_audio_authenticity(audio_path)
        results["audio_authenticity_analysis"] = audio_auth
        results["audio_spoof_risk"] = float(audio_auth.get("audio_spoof_risk", 0.0))
        signal_scores["audio_spoof"] = results["audio_spoof_risk"]

        if results["audio_spoof_risk"] >= 0.65:
            results["flags"].append(
                f"Audio appears synthetic/spoofed (risk {results['audio_spoof_risk']:.1%})"
            )

        audio_injection = scan_for_injection(transcript)
        if not results.get("injection_scan"):
            results["injection_scan"] = audio_injection
        signal_scores["audio_injection"] = audio_injection["injection_risk"]

        if audio_injection["injection_keywords_found"]:
            results["flags"].append(
                f"Prompt-injection keywords in audio transcript: {audio_injection['injection_keywords_found']}"
            )

    if audio_path and image_path:
        similarity = from_scratch_similarity(transcript or "", image_caption or "")
        results["semantic_similarity"] = similarity

        mismatch_risk, risk_level, mismatch_note = score_mismatch_risk(
            similarity,
            image_caption=image_caption or "",
            transcript=transcript or "",
            audio_spoof_risk=results["audio_spoof_risk"],
        )

        results["audio_visual_mismatch"] = {
            "audio_transcript": transcript,
            "image_caption": image_caption,
            "similarity_score": similarity,
            "mismatch_risk": round(mismatch_risk, 4),
            "risk_level": risk_level,
            "note": mismatch_note,
        }
        signal_scores["mismatch"] = mismatch_risk

        if mismatch_risk >= 0.35:
            results["flags"].append(
                f"Audio-visual mismatch: similarity {similarity:.1%} — {mismatch_note}"
            )

        clip_score, clip_status = get_clip_alignment(image_path, transcript or "")
        if clip_score is not None:
            clip_risk = max(0.0, min(1.0 - clip_score, 1.0)) * 0.35
            results["clip_alignment"] = {
                "score": clip_score,
                "status": clip_status,
                "note": "Support signal only",
            }
            signal_scores["clip"] = clip_risk
        else:
            results["clip_alignment"] = {
                "score": None,
                "status": clip_status,
                "note": "Skipped",
            }

    if signal_scores:
        overall = compute_overall_risk(signal_scores)
        results["raw_overall_risk_score"] = overall["raw_overall_risk_score"]
        results["signal_breakdown"] = overall["signal_breakdown"]

    escalation = _apply_rule_escalations(results["raw_overall_risk_score"], signal_scores)
    results["raw_overall_risk_score"] = escalation["raw_overall_risk_score"]
    results["final_overall_risk_score"] = escalation["final_overall_risk_score"]
    results["overall_risk_score"] = escalation["final_overall_risk_score"]
    results["overall_verdict"] = escalation["overall_verdict"]
    results["risk_tier"] = escalation["risk_tier"]
    results["decision_notes"] = escalation["decision_notes"]

    modality = "audio + image" if audio_path and image_path else "audio" if audio_path else "image"
    raw_pct = results["raw_overall_risk_score"] * 100.0
    final_pct = results["final_overall_risk_score"] * 100.0

    if escalation["escalation_applied"] and abs(results["final_overall_risk_score"] - results["raw_overall_risk_score"]) > 1e-9:
        decision_context = (
            f"Raw fusion risk: {raw_pct:.1f}%, escalated to {final_pct:.1f}% due to rule-based evidence. "
            f"{'; '.join(results['decision_notes']) if results['decision_notes'] else ''}"
        ).strip()
    else:
        decision_context = (
            f"Raw fusion risk: {raw_pct:.1f}%. Final risk: {final_pct:.1f}% (no escalation)."
        )

    results["explanation"] = (
        f"Analyzed {modality}. "
        f"{decision_context} "
        f"Risk tier: {results['risk_tier']}. Flags: {len(results['flags'])}. "
        f"{'; '.join(results['flags']) if results['flags'] else 'No suspicious activity detected.'}"
    )

    mismatch_score_log = (results.get("audio_visual_mismatch") or {}).get("mismatch_risk")

    logger.info("detector_version=%s audio_spoof_risk=%s", DETECTOR_VERSION, results.get("audio_spoof_risk"))
    logger.info("detector_version=%s ocr_avg_confidence=%s", DETECTOR_VERSION, (results.get("ocr_analysis") or {}).get("avg_confidence"))
    logger.info("detector_version=%s image_caption=%s", DETECTOR_VERSION, results.get("image_caption"))
    logger.info("detector_version=%s audio_transcript=%s", DETECTOR_VERSION, results.get("audio_transcript"))
    logger.info("detector_version=%s mismatch_score=%s", DETECTOR_VERSION, mismatch_score_log)
    logger.info("detector_version=%s raw_overall_risk_score=%s", DETECTOR_VERSION, results.get("raw_overall_risk_score"))
    logger.info("detector_version=%s final_overall_risk_score=%s", DETECTOR_VERSION, results.get("final_overall_risk_score"))
    logger.info(
        "detector_version=%s final_fusion_result={'raw_overall_risk_score': %s, 'final_overall_risk_score': %s, 'risk_tier': %s, 'overall_verdict': %s}",
        DETECTOR_VERSION,
        results.get("raw_overall_risk_score"),
        results.get("final_overall_risk_score"),
        results.get("risk_tier"),
        results.get("overall_verdict"),
    )

    return results


if __name__ == "__main__":
    import json
    import sys

    audio = sys.argv[1] if len(sys.argv) > 1 else None
    image = sys.argv[2] if len(sys.argv) > 2 else None

    result = analyze_multimodal(audio_path=audio, image_path=image)
    print(json.dumps(result, indent=2))