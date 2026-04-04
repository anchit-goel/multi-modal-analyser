#!/usr/bin/env python3

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf


def list_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts]


def preprocess_batch(arr, mode):
    if mode == "mobilenet_v2":
        return tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    if mode == "resnet50":
        return tf.keras.applications.resnet50.preprocess_input(arr)
    if mode == "efficientnet":
        return tf.keras.applications.efficientnet.preprocess_input(arr)
    return arr / 255.0


def score_paths(model, paths, true_label, size, mode, writer, batch_size):
    for i in range(0, len(paths), batch_size):
        chunk = paths[i : i + batch_size]
        imgs = []
        keep = []
        for p in chunk:
            try:
                arr = np.array(Image.open(p).convert("RGB").resize(size), dtype=np.float32)
                imgs.append(arr)
                keep.append(p)
            except Exception:
                continue
        if not imgs:
            continue
        x = np.stack(imgs, axis=0)
        x = preprocess_batch(x, mode)
        scores = model.predict(x, verbose=0).reshape(-1)
        for p, s in zip(keep, scores):
            writer.writerow({"image": str(p), "score": float(s), "true_label": true_label})


def main():
    parser = argparse.ArgumentParser(description="Generate calibration CSV with local model inference")
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--fake-dir", required=True)
    parser.add_argument("--out", default="calibration_scores.csv")
    parser.add_argument("--config", default="model_runtime_config.json")
    parser.add_argument("--max-per-class", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    model_path = Path(cfg["selected_model"])
    mode = cfg.get("preprocess_mode", "rescale_255")
    size = tuple(cfg.get("input_size", [224, 224]))

    real_images = list_images(Path(args.real_dir))
    fake_images = list_images(Path(args.fake_dir))

    random.seed(42)
    if args.max_per_class > 0:
        if len(real_images) > args.max_per_class:
            real_images = random.sample(real_images, args.max_per_class)
        if len(fake_images) > args.max_per_class:
            fake_images = random.sample(fake_images, args.max_per_class)

    model = tf.keras.models.load_model(str(model_path), compile=False)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "score", "true_label"])
        writer.writeheader()
        score_paths(model, real_images, 1, size, mode, writer, args.batch_size)
        score_paths(model, fake_images, 0, size, mode, writer, args.batch_size)

    print(f"Wrote {len(real_images) + len(fake_images)} samples to {args.out}")


if __name__ == "__main__":
    main()
