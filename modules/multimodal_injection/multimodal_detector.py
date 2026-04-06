import torch
import numpy as np
import librosa
from PIL import Image
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    CLIPProcessor, CLIPModel,
    pipeline
)
from collections import Counter
import math
import os
import traceback
import easyocr
from rapidfuzz import fuzz

MODELS_LOADED = False
MODEL_LOAD_ERROR = None

ocr_reader = None
whisper = None
blip_processor = None
blip_model = None
clip_processor = None
clip_model = None


def load_multimodal_models():
    global MODELS_LOADED, MODEL_LOAD_ERROR
    global ocr_reader, whisper, blip_processor, blip_model, clip_processor, clip_model

    if MODELS_LOADED or MODEL_LOAD_ERROR is not None:
        return

    try:
        ocr_reader = easyocr.Reader(['en'], gpu=False)
        whisper = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base"
        )
        blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        blip_model.eval()
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_model.eval()
        MODELS_LOADED = True
    except Exception as exc:
        MODEL_LOAD_ERROR = traceback.format_exc()
        raise


# ── FROM SCRATCH: Hidden text detector ───────────────────────

def detect_low_contrast_text(image_path, threshold=30):
    img = np.array(Image.open(image_path).convert('L'))
    h, w = img.shape
    block_size = 32

    suspicious_blocks = 0      # ghost text (low contrast)
    high_contrast_blocks = 0   # embedded text (high contrast)
    total_blocks = 0

    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = img[i:i+block_size, j:j+block_size]
            std = np.std(block)
            total_blocks += 1

            if 2 < std < threshold:
                suspicious_blocks += 1          # hidden/ghost text
            elif std >= threshold and std < 80:
                high_contrast_blocks += 1       # likely embedded readable text

    suspicion_ratio = suspicious_blocks / (total_blocks + 1e-8)
    text_presence_ratio = high_contrast_blocks / (total_blocks + 1e-8)
    hidden_risk = round(min(suspicion_ratio * 3, 1.0), 4)

    return {
        "suspicious_block_ratio": round(suspicion_ratio, 4),
        "text_presence_ratio": round(text_presence_ratio, 4),
        "hidden_content_risk": hidden_risk,
        "has_embedded_text": text_presence_ratio > 0.05,
        "verdict": "SUSPICIOUS" if hidden_risk > 0.4 else (
            "TEXT PRESENT" if text_presence_ratio > 0.05 else "CLEAN"
        )
    }


def extract_text_ocr(image_path, min_confidence=0.4):
    """
    EasyOCR with confidence filtering.
    Returns structured results including high and low confidence tokens.
    """
    results = ocr_reader.readtext(image_path)
    
    high_conf = [(text, conf) for (_, text, conf) in results if conf >= min_confidence]
    low_conf  = [(text, conf) for (_, text, conf) in results if conf < min_confidence]

    extracted = " ".join([text for text, _ in high_conf])

    return {
        "text": extracted.strip(),
        "high_confidence_tokens": high_conf,
        "low_confidence_tokens": low_conf,
        "avg_confidence": round(
            sum(c for _, c in high_conf) / (len(high_conf) + 1e-8), 3
        ),
        "token_count": len(high_conf)
    }


# ── FROM SCRATCH: Prompt injection keyword scanner ───────────

INJECTION_KEYWORDS = [
    "ignore", "forget", "new instruction", "system prompt",
    "you are now", "disregard", "override", "jailbreak",
    "act as", "pretend", "from now on", "ignore previous",
    "ignore above", "do not follow", "bypass", "reveal",
    "print your instructions", "what are your instructions"
]

def scan_for_injection(text, fuzzy_threshold=80):
    """
    Enhanced scanner with fuzzy matching to catch OCR errors
    and deliberate leetspeak/typo evasion.
    """
    text_lower = text.lower()
    words = text_lower.split()
    found_exact = []
    found_fuzzy = []

    for keyword in INJECTION_KEYWORDS:
        # Exact match
        if keyword in text_lower:
            found_exact.append(keyword)
            continue

        # Fuzzy match against sliding word windows
        kw_words = keyword.split()
        window_size = len(kw_words)
        for i in range(len(words) - window_size + 1):
            window = " ".join(words[i:i + window_size])
            score = fuzz.ratio(keyword, window)
            if score >= fuzzy_threshold:
                found_fuzzy.append({
                    "keyword": keyword,
                    "matched": window,
                    "confidence": score
                })
                break

    all_found = found_exact + [f["keyword"] for f in found_fuzzy]
    risk = 0.0
    if len(all_found) >= 3:
        risk = 1.0
    elif len(all_found) == 2:
        risk = 0.85
    elif len(all_found) == 1:
        risk = 0.65

    return {
        "exact_matches": found_exact,
        "fuzzy_matches": found_fuzzy,
        "injection_keywords_found": all_found,
        "injection_risk": round(risk, 4),
        "verdict": "INJECTION DETECTED" if risk > 0.5 else "CLEAN"
    }


def compute_overall_risk(risk_scores_dict):
    """
    Computes overall risk with weighted signal breakdown for auditability.
    """
    weights = {
        "hidden_text":    0.15,
        "ocr_injection":  0.35,   # highest weight — most reliable signal
        "mismatch":       0.30,
        "clip":           0.20
    }

    weighted_sum = 0.0
    total_weight = 0.0
    breakdown = {}

    for signal, score in risk_scores_dict.items():
        w = weights.get(signal, 0.1)
        contribution = round(score * w, 4)
        breakdown[signal] = {
            "raw_score": score,
            "weight": w,
            "contribution": contribution
        }
        weighted_sum += contribution
        total_weight += w

    overall = round(weighted_sum / total_weight, 4)

    return {
        "overall_risk_score": overall,
        "signal_breakdown": breakdown
    }


# ── FROM SCRATCH: Semantic similarity (TF-IDF + Jaccard) ─────

def tokenize(text):
    """Simple tokenizer"""
    import re
    return re.findall(r'\b\w+\b', text.lower())

def compute_tfidf_vector(text, vocab):
    """Compute TF vector for a text given a vocabulary"""
    tokens = tokenize(text)
    tf = Counter(tokens)
    vector = np.array([tf.get(word, 0) for word in vocab], dtype=float)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector

def from_scratch_similarity(text1, text2):
    """
    From-scratch text similarity using TF cosine similarity + Jaccard.
    Returns similarity score between 0 and 1.
    """
    tokens1 = set(tokenize(text1))
    tokens2 = set(tokenize(text2))

    # Jaccard similarity
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    jaccard = len(intersection) / (len(union) + 1e-8)

    # TF cosine similarity
    vocab = list(union)
    if vocab:
        vec1 = compute_tfidf_vector(text1, vocab)
        vec2 = compute_tfidf_vector(text2, vocab)
        cosine = float(np.dot(vec1, vec2))
    else:
        cosine = 0.0

    # Combined similarity
    similarity = (jaccard * 0.4) + (cosine * 0.6)
    return round(similarity, 4)


# ── FROM SCRATCH: Mismatch risk scorer ───────────────────────

def score_mismatch_risk(similarity):
    """
    From-scratch mismatch risk scoring based on similarity.
    Low similarity between modalities = high injection risk.
    """
    if similarity > 0.5:
        return 0.05, "LOW", "Modalities consistent"
    elif similarity > 0.3:
        return 0.45, "MEDIUM", "Moderate mismatch — review recommended"
    elif similarity > 0.15:
        return 0.75, "HIGH", "Significant mismatch — likely manipulation"
    else:
        return 0.95, "CRITICAL", "Severe mismatch — injection attack likely"


# ── SENSORS: Pretrained models used only for extraction ──────

def transcribe_audio(audio_path):
    """Whisper used as sensor — extracts text from audio"""
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    result = whisper(audio.astype(np.float32))
    return result["text"].strip()


def caption_image(image_path):
    """BLIP used as sensor — extracts text description from image"""
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(image, return_tensors="pt")
    with torch.no_grad():
        output = blip_model.generate(**inputs, max_length=60)
    return blip_processor.decode(output[0], skip_special_tokens=True)


def get_clip_alignment(image_path, text):
    """CLIP alignment with degenerate input guard"""
    
    # Guard: if transcript is effectively empty/noise, skip CLIP
    clean_text = text.replace(".", "").replace(" ", "").strip()
    if len(clean_text) < 5:
        return None, "SKIPPED — transcript too short/noisy for reliable CLIP scoring"

    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(
        text=[text],
        images=image,
        return_tensors="pt",
        padding=True
    )
    with torch.no_grad():
        outputs = clip_model(**inputs)
        score = torch.sigmoid(outputs.logits_per_image).item()
    return round(score, 4), "OK"


# ── MAIN: Full multimodal analysis ───────────────────────────

def analyze_multimodal(audio_path=None, image_path=None):
    """
    Full multimodal prompt injection detection.
    Core detection logic is from scratch.
    Pretrained models used only as feature/text extractors.
    
    Args:
        audio_path: path to audio file (optional)
        image_path: path to image file (optional)
    
    Returns:
        dict with full analysis results
    """
    load_multimodal_models()
    if MODEL_LOAD_ERROR is not None:
        raise RuntimeError(
            "Multimodal model initialization failed. Check model cache, disk space, and network access."
        )

    results = {
        "hidden_text_analysis": None,
        "injection_scan": None,
        "audio_visual_mismatch": None,
        "clip_alignment": None,
        "overall_risk_score": 0.0,
        "overall_verdict": "SAFE",
        "risk_tier": "LOW",
        "flags": [],
        "explanation": "",
        "signal_breakdown": {}
    }

    signal_scores = {}

    # ── Image analysis ────────────────────────────────────────
    if image_path:
        # 1. From-scratch hidden text detection
        hidden = detect_low_contrast_text(image_path)
        results["hidden_text_analysis"] = hidden
        signal_scores["hidden_text"] = hidden["hidden_content_risk"]

        if hidden["hidden_content_risk"] > 0.4:
            results["flags"].append(
                f"Hidden content detected in image "
                f"(suspicion ratio: {hidden['suspicious_block_ratio']:.1%})"
            )

        # 2. BLIP for scene understanding (caption)
        image_caption = caption_image(image_path)
        results["image_caption"] = image_caption

        # 3. OCR for actual text extraction
        ocr_result = extract_text_ocr(image_path)
        ocr_text = ocr_result["text"]
        results["ocr_analysis"] = ocr_result

        # 4. Scan BOTH caption and OCR text for injection keywords
        injection_caption = scan_for_injection(image_caption)
        injection_ocr = scan_for_injection(ocr_text)

        # Merge findings — take highest risk
        combined_keywords = list(set(
            injection_caption["injection_keywords_found"] +
            injection_ocr["injection_keywords_found"]
        ))
        combined_risk = max(
            injection_caption["injection_risk"],
            injection_ocr["injection_risk"]
        )

        results["injection_scan"] = {
            "caption_scan": injection_caption,
            "ocr_scan": injection_ocr,
            "combined_keywords": combined_keywords,
            "injection_risk": round(combined_risk, 4),
            "verdict": "INJECTION DETECTED" if combined_risk > 0.5 else "CLEAN",
            "source": "ocr" if injection_ocr["injection_risk"] > injection_caption["injection_risk"] else "caption"
        }
        signal_scores["ocr_injection"] = combined_risk

        if combined_keywords:
            results["flags"].append(
                f"Prompt injection keywords found: {combined_keywords}"
            )

    # ── Audio + Image cross-modal analysis ───────────────────
    if audio_path and image_path:
        # 4. Transcribe audio (Whisper as sensor)
        audio_transcript = transcribe_audio(audio_path)
        results["audio_transcript"] = audio_transcript

        # 5. From-scratch semantic similarity
        similarity = from_scratch_similarity(audio_transcript, image_caption)
        results["semantic_similarity"] = similarity

        # 6. From-scratch mismatch risk scoring
        mismatch_risk, risk_level, mismatch_note = score_mismatch_risk(similarity)
        signal_scores["mismatch"] = mismatch_risk

        results["audio_visual_mismatch"] = {
            "audio_transcript": audio_transcript,
            "image_caption": image_caption,
            "similarity_score": similarity,
            "mismatch_risk": round(mismatch_risk, 4),
            "risk_level": risk_level,
            "note": mismatch_note
        }

        if mismatch_risk > 0.7:
            results["flags"].append(
                f"Audio-visual MISMATCH: similarity only {similarity:.1%} — "
                f"{mismatch_note}"
            )

        # 7. CLIP alignment check (CLIP as sensor)
        clip_score, clip_status = get_clip_alignment(image_path, audio_transcript)

        if clip_score is not None:
            results["clip_alignment"] = {
                "score": clip_score,
                "status": clip_status,
                "note": "High score = audio and image are consistent"
            }
            # Low CLIP alignment also contributes to risk
            clip_risk = round(1.0 - clip_score, 4)
            signal_scores["clip"] = clip_risk
        else:
            results["clip_alignment"] = {
                "score": None,
                "status": clip_status,
                "note": "CLIP skipped — unreliable for this input"
            }
            # Still penalize: degenerate transcript = likely noise injection
            signal_scores["clip"] = 0.5
            results["flags"].append("Audio transcript was empty/noise — CLIP skipped, risk penalized")

    # ── Audio only analysis ───────────────────────────────────
    if audio_path and not image_path:
        audio_transcript = transcribe_audio(audio_path)
        results["audio_transcript"] = audio_transcript

        # Scan transcript for injection keywords
        injection = scan_for_injection(audio_transcript)
        results["injection_scan"] = injection
        signal_scores["ocr_injection"] = injection["injection_risk"] # Reuse key for audio transcript scan

        if injection["injection_keywords_found"]:
            results["flags"].append(
                f"Prompt injection keywords found in audio: "
                f"{injection['injection_keywords_found']}"
            )

    # ── Overall risk computation (Breakdown) ──────────────────
    if signal_scores:
        explanation = compute_overall_risk(signal_scores)
        results["overall_risk_score"] = explanation["overall_risk_score"]
        results["signal_breakdown"] = explanation["signal_breakdown"]

    score = results["overall_risk_score"]
    if score > 0.85:
        results["overall_verdict"] = "CRITICAL — injection attack detected"
        results["risk_tier"] = "CRITICAL"
    elif score > 0.6:
        results["overall_verdict"] = "HIGH RISK — likely manipulation"
        results["risk_tier"] = "HIGH"
    elif score > 0.35:
        results["overall_verdict"] = "SUSPICIOUS — review required"
        results["risk_tier"] = "MEDIUM"
    else:
        results["overall_verdict"] = "SAFE"
        results["risk_tier"] = "LOW"

    results["explanation"] = (
        f"Analyzed {('audio + image' if audio_path and image_path else 'audio' if audio_path else 'image')}. "
        f"Overall risk: {score:.1%} ({results['risk_tier']}). "
        f"Flags: {len(results['flags'])}. "
        f"{'; '.join(results['flags']) if results['flags'] else 'No suspicious activity detected.'}"
    )

    return results


if __name__ == "__main__":
    import json
    import sys

    audio = sys.argv[1] if len(sys.argv) > 1 else None
    image = sys.argv[2] if len(sys.argv) > 2 else None

    result = analyze_multimodal(audio_path=audio, image_path=image)
    print(json.dumps(result, indent=2))