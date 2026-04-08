import os
import re
import traceback
from collections import Counter
import logging

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

DETECTOR_VERSION = "v2-fixed-2026-04-09"

logger = logging.getLogger(__name__)

ocr_reader = None
whisper = None
blip_processor = None
blip_model = None
clip_processor = None
clip_model = None


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
    """
    Conservative heuristic:
    low local variance alone is NOT hidden text.
    We keep this as a weak signal only.
    """
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

            # Very conservative band for faint overlays.
            # Avoid flagging ordinary flat backgrounds or portraits.
            if 4 <= std <= 18 and 25 <= mean <= 230:
                suspicious_blocks += 1

    suspicion_ratio = suspicious_blocks / (total_blocks + 1e-8)

    # Weak signal only: low local contrast alone should never dominate.
    hidden_risk = min(suspicion_ratio * 0.25, 0.12)

    return {
        "suspicious_block_ratio": round(suspicion_ratio, 4),
        "hidden_content_risk": round(hidden_risk, 4),
        "verdict": "SUSPICIOUS" if hidden_risk > 0.18 else "CLEAN"
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
            "length": len(cleaned)
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
        "token_count": len(high_conf)
    }


def is_reliable_ocr(ocr_result: dict) -> bool:
    token_count = int(ocr_result.get("token_count", 0))
    avg_confidence = float(ocr_result.get("avg_confidence", 0.0))
    text = (ocr_result.get("text") or "").strip()
    compact_len = len(re.sub(r"\s+", "", text))

    # Conservative gate: require enough readable content before trusting OCR-driven injection detection.
    return token_count >= 4 and avg_confidence >= 0.80 and compact_len >= 12


def scan_for_injection(text, fuzzy_threshold=95):
    text_lower = text.lower().strip()
    if not text_lower:
        return {
            "exact_matches": [],
            "fuzzy_matches": [],
            "injection_keywords_found": [],
            "injection_risk": 0.0,
            "verdict": "CLEAN"
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
                    "confidence": score
                })
                break

    all_found = found_exact + [f["keyword"] for f in found_fuzzy]

    unique_found = sorted(set(all_found))

    # More conservative scoring: one weak keyword should not trigger high risk.
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
        "verdict": "INJECTION DETECTED" if risk >= 0.6 else "CLEAN"
    }


def transcribe_audio(audio_path):
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    result = whisper(audio.astype(np.float32))
    return result["text"].strip()


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


def score_mismatch_risk(similarity, image_caption="", transcript=""):
    """
    This is now conservative, because transcript-vs-caption mismatch
    is a weak signal for normal speech photos.
    """
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

    # If the image is just a generic speaking scene and transcript is long,
    # lexical overlap is expected to be weak. Treat as uncertain, not risky.
    if generic_speaking_scene and transcript_len >= 8:
        return 0.08, "LOW", "Generic speaking scene — transcript/caption mismatch unreliable"

    if similarity > 0.50:
        return 0.05, "LOW", "Modalities consistent"
    if similarity > 0.30:
        return 0.15, "LOW", "Weak mismatch"
    if similarity > 0.15:
        return 0.25, "MEDIUM", "Moderate mismatch"
    return 0.38, "MEDIUM", "Possible mismatch, low confidence"


def compute_overall_risk(signal_scores):
    weights = {
        "hidden_text": 0.07,
        "ocr_injection": 0.50,
        "mismatch": 0.17,
        "clip": 0.06,
        "audio_injection": 0.32,
    }

    weighted_sum = 0.0
    total_weight = 0.0
    breakdown = {}

    for signal, score in signal_scores.items():
        w = weights.get(signal, 0.10)
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
        "overall_risk_score": round(overall, 4),
        "signal_breakdown": breakdown
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
        "overall_risk_score": 0.0,
        "overall_verdict": "SAFE",
        "risk_tier": "LOW",
        "flags": [],
        "explanation": "",
        "signal_breakdown": {}
    }

    signal_scores = {}
    image_caption = None

    if image_path:
        hidden = detect_low_contrast_text(image_path)
        results["hidden_text_analysis"] = hidden

        ocr_result = extract_text_ocr(image_path, min_confidence=0.75)
        results["ocr_analysis"] = ocr_result

        image_caption = caption_image(image_path)
        results["image_caption"] = image_caption

        # Hidden text only matters meaningfully if OCR also found readable text
        if is_reliable_ocr(ocr_result):
            hidden_risk = min(hidden["hidden_content_risk"] + 0.10, 0.28)
        else:
            hidden_risk = min(hidden["hidden_content_risk"], 0.06)

        signal_scores["hidden_text"] = hidden_risk

        if hidden_risk >= 0.18:
            results["flags"].append(
                f"Possible hidden/embedded text in image (weak signal, ratio: {hidden['suspicious_block_ratio']:.1%})"
            )

        # Important fix:
        # Only OCR text is scanned for injection. Caption is not.
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
                "note": "OCR confidence too low to trust for injection detection"
            }

        results["injection_scan"] = ocr_injection
        signal_scores["ocr_injection"] = ocr_injection["injection_risk"]

        if ocr_injection["injection_keywords_found"]:
            results["flags"].append(
                f"Prompt-injection keywords in OCR text: {ocr_injection['injection_keywords_found']}"
            )

    if audio_path and not image_path:
        transcript = transcribe_audio(audio_path)
        results["audio_transcript"] = transcript

        audio_injection = scan_for_injection(transcript)
        results["injection_scan"] = audio_injection
        signal_scores["audio_injection"] = audio_injection["injection_risk"]

        if audio_injection["injection_keywords_found"]:
            results["flags"].append(
                f"Prompt-injection keywords in audio transcript: {audio_injection['injection_keywords_found']}"
            )

    if audio_path and image_path:
        transcript = transcribe_audio(audio_path)
        results["audio_transcript"] = transcript

        similarity = from_scratch_similarity(transcript, image_caption or "")
        results["semantic_similarity"] = similarity

        mismatch_risk, risk_level, mismatch_note = score_mismatch_risk(
            similarity,
            image_caption=image_caption or "",
            transcript=transcript,
        )

        results["audio_visual_mismatch"] = {
            "audio_transcript": transcript,
            "image_caption": image_caption,
            "similarity_score": similarity,
            "mismatch_risk": round(mismatch_risk, 4),
            "risk_level": risk_level,
            "note": mismatch_note
        }
        signal_scores["mismatch"] = mismatch_risk

        if mismatch_risk >= 0.35:
            results["flags"].append(
                f"Audio-visual mismatch: similarity {similarity:.1%} — {mismatch_note}"
            )

        clip_score, clip_status = get_clip_alignment(image_path, transcript)
        if clip_score is not None:
            clip_risk = max(0.0, min(1.0 - clip_score, 1.0))
            clip_risk *= 0.35
            results["clip_alignment"] = {
                "score": clip_score,
                "status": clip_status,
                "note": "Support signal only"
            }
            signal_scores["clip"] = clip_risk
        else:
            results["clip_alignment"] = {
                "score": None,
                "status": clip_status,
                "note": "Skipped"
            }

    if signal_scores:
        overall = compute_overall_risk(signal_scores)
        results["overall_risk_score"] = overall["overall_risk_score"]
        results["signal_breakdown"] = overall["signal_breakdown"]

    score = results["overall_risk_score"]

    ocr_signal = float(signal_scores.get("ocr_injection", 0.0))
    hidden_signal = float(signal_scores.get("hidden_text", 0.0))
    strong_evidence = (
        ocr_signal >= 0.72
        or (ocr_signal >= 0.45 and hidden_signal >= 0.18)
    )

    if score >= 0.80:
        if strong_evidence:
            results["overall_verdict"] = "CRITICAL — injection attack detected"
            results["risk_tier"] = "CRITICAL"
        else:
            results["overall_verdict"] = "SUSPICIOUS — review required"
            results["risk_tier"] = "MEDIUM"
            results["flags"].append("Risk gated: no strong reliable evidence for CRITICAL verdict")
    elif score >= 0.68:
        if strong_evidence:
            results["overall_verdict"] = "HIGH RISK — likely manipulation"
            results["risk_tier"] = "HIGH"
        else:
            results["overall_verdict"] = "SUSPICIOUS — review required"
            results["risk_tier"] = "MEDIUM"
            results["flags"].append("Risk gated: no strong reliable evidence for HIGH verdict")
    elif score >= 0.40:
        results["overall_verdict"] = "SUSPICIOUS — review required"
        results["risk_tier"] = "MEDIUM"
    else:
        results["overall_verdict"] = "SAFE"
        results["risk_tier"] = "LOW"

    modality = "audio + image" if audio_path and image_path else "audio" if audio_path else "image"

    results["explanation"] = (
        f"Analyzed {modality}. "
        f"Overall risk: {results['overall_risk_score']:.1%} ({results['risk_tier']}). "
        f"Flags: {len(results['flags'])}. "
        f"{'; '.join(results['flags']) if results['flags'] else 'No suspicious activity detected.'}"
    )

    logger.info("detector_version=%s ocr_text=%s", DETECTOR_VERSION, (results.get("ocr_analysis") or {}).get("text"))
    logger.info("detector_version=%s ocr_avg_confidence=%s", DETECTOR_VERSION, (results.get("ocr_analysis") or {}).get("avg_confidence"))
    logger.info("detector_version=%s image_caption=%s", DETECTOR_VERSION, results.get("image_caption"))
    logger.info("detector_version=%s audio_transcript=%s", DETECTOR_VERSION, results.get("audio_transcript"))
    logger.info("detector_version=%s signal_breakdown=%s", DETECTOR_VERSION, results.get("signal_breakdown"))

    return results


if __name__ == "__main__":
    import json
    import sys

    audio = sys.argv[1] if len(sys.argv) > 1 else None
    image = sys.argv[2] if len(sys.argv) > 2 else None

    result = analyze_multimodal(audio_path=audio, image_path=image)
    print(json.dumps(result, indent=2))