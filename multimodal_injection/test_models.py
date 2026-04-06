from multimodal_detector import analyze_multimodal
import json
import os

# Function to print test name clearly
def print_test_header(title):
    print(f"\n{'='*50}\n{title}\n{'='*50}")

# Test 1: Image only - Clean
if os.path.exists("test_image.jpg"):
    print_test_header("Test 1: Clean Image (Hello World)")
    result1 = analyze_multimodal(image_path="test_image.jpg")
    print(json.dumps(result1, indent=2))
else:
    print("test_image.jpg NOT FOUND")

# Test 2: Image only - Injection
if os.path.exists("test_injection_image.jpg"):
    print_test_header("Test 2: Injection Image (Ignore instructions...)")
    result2 = analyze_multimodal(image_path="test_injection_image.jpg")
    print(json.dumps(result2, indent=2))
else:
    print("test_injection_image.jpg NOT FOUND")

# Test 3: Audio only
if os.path.exists("test_audio.wav"):
    print_test_header("Test 3: Audio (Sine Wave)")
    result3 = analyze_multimodal(audio_path="test_audio.wav")
    print(json.dumps(result3, indent=2))
else:
    print("test_audio.wav NOT FOUND")

# Test 4: Multimodal (Audio + Injection Image)
if os.path.exists("test_audio.wav") and os.path.exists("test_injection_image.jpg"):
    print_test_header("Test 4: Multimodal - Audio & Injection Image")
    result4 = analyze_multimodal(
        audio_path="test_audio.wav",
        image_path="test_injection_image.jpg"
    )
    print(json.dumps(result4, indent=2))
else:
    print("Test 4 skipped - files missing")
