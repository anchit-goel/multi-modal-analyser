# Deepfake Pipeline Plan (Using best_pretrained_model2.h5)

## Current Serving Pipeline

1. Input image upload in frontend (JPG/PNG/WebP/BMP).
2. Backend endpoint `/predict` receives form-data key `image`.
3. Preprocess:
- Resize to 224x224
- Convert RGB
- Convert to float32
- Normalize with `arr / 255.0`
4. Inference using `results/best_pretrained_model2.h5` only.
5. Sigmoid score -> label using threshold 0.5.
6. Return JSON with `label`, `confidence`, `raw_score`, and `processing_time_sec`.

## Why This Matches Your Project Files

The training scripts in your Deep-Fake-Detection folder repeatedly use `ImageDataGenerator(rescale=1./255.)` and binary classification:
- Deep-Fake-Detection-using-advanced-CNN-architectures-and-its-comparison-main/Proposed_Custom_Hybrid-CNN(Optmized).py
- Deep-Fake-Detection-using-advanced-CNN-architectures-and-its-comparison-main/Deepfake_detection_hybrid_model_structure.ipynb
- Deep-Fake-Detection-using-advanced-CNN-architectures-and-its-comparison-main/custom_cnn (2).ipynb

This is why serving currently uses `/255.0` normalization and thresholded sigmoid output.

## Fast Improvement Pipeline

1. Validate class mapping once from data loader
- During training, print `train_flow.class_indices` and save it.
- Store mapping in JSON (example: `{"fake": 0, "real": 1}`).
- Use this exact mapping in backend label assignment.

2. Tune threshold on validation set
- Do not lock to 0.5 blindly.
- Compute ROC/PR threshold from validation predictions.
- Save best threshold (for F1 or balanced error) and load in backend.

3. Add calibration checks
- Track score distributions for real/fake separately.
- If scores collapse near 0 or 1 for all samples, flag model drift.

4. Improve data robustness
- Keep augmentation from your scripts (rotation, shear, zoom, flip).
- Add compression/noise/blur augmentations to simulate social-media artifacts.

5. Add face-centric preprocessing
- Detect/crop face before classification.
- Run model on aligned face crop, not full frame.

6. Add test-time augmentation (TTA)
- Predict on original + horizontal flip.
- Average scores for a more stable output.

7. Add explainability gate
- For low-confidence outputs (for example, 45-55%), generate saliency/Grad-CAM.
- Surface these as "review needed" instead of confident hard labels.

8. Version control model artifacts
- Keep model file + metadata together:
  - model path
  - preprocessing mode
  - threshold
  - class mapping
  - training dataset version

## Recommended Next Iteration

1. Run a 100-image labeled benchmark (50 real, 50 fake).
2. Export score CSV (`image, true_label, score, pred_label`).
3. Fit optimal threshold from validation ROC/PR.
4. Update backend threshold and class mapping JSON.
5. Re-test and freeze version as `best_pretrained_model2_calibrated`.
