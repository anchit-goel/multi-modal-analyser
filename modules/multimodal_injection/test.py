from multimodal_detector import analyze_multimodal
import json

# Test 1: Image only
print("=== Test 1: Image only ===")
result = analyze_multimodal(image_path="test_image.jpg")
print(json.dumps(result, indent=2))

# Test 2: Audio only
print("\n=== Test 2: Audio only ===")
result = analyze_multimodal(audio_path="test_audio.wav")
print(json.dumps(result, indent=2))

# Test 3: Both
print("\n=== Test 3: Full multimodal ===")
result = analyze_multimodal(
    audio_path="test_audio.wav",
    image_path="test_image.jpg"
)
print(json.dumps(result, indent=2))