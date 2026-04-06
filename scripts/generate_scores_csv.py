#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path
from urllib import request
from urllib.error import HTTPError, URLError
import uuid


def list_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts]


def post_image(api_url: str, image_path: Path):
    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
    image_bytes = image_path.read_bytes()

    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        f'Content-Disposition: form-data; name="image"; filename="{image_path.name}"\r\n'.encode()
    )
    body.extend(b"Content-Type: application/octet-stream\r\n\r\n")
    body.extend(image_bytes)
    body.extend(f"\r\n--{boundary}--\r\n".encode())

    req = request.Request(api_url, method="POST", data=bytes(body))
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")

    with request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Generate score CSV for threshold calibration")
    parser.add_argument("--real-dir", required=True, help="Folder containing REAL images")
    parser.add_argument("--fake-dir", required=True, help="Folder containing FAKE images")
    parser.add_argument("--out", default="calibration_scores.csv", help="Output CSV path")
    parser.add_argument("--api", default="http://127.0.0.1:8080/predict", help="Predict API URL")
    parser.add_argument("--max-per-class", type=int, default=0, help="Limit per class (0 = all)")
    args = parser.parse_args()

    real_dir = Path(args.real_dir)
    fake_dir = Path(args.fake_dir)

    if not real_dir.exists() or not fake_dir.exists():
        raise SystemExit("Both --real-dir and --fake-dir must exist.")

    real_images = list_images(real_dir)
    fake_images = list_images(fake_dir)

    if args.max_per_class > 0:
        real_images = real_images[: args.max_per_class]
        fake_images = fake_images[: args.max_per_class]

    rows = []

    for path in real_images:
        try:
            payload = post_image(args.api, path)
            rows.append(
                {
                    "image": str(path),
                    "score": float(payload.get("raw_score", 0.0)),
                    "true_label": 1,
                    "pred_label": payload.get("label", ""),
                }
            )
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            print(f"REAL failed: {path} -> {exc}")

    for path in fake_images:
        try:
            payload = post_image(args.api, path)
            rows.append(
                {
                    "image": str(path),
                    "score": float(payload.get("raw_score", 0.0)),
                    "true_label": 0,
                    "pred_label": payload.get("label", ""),
                }
            )
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            print(f"FAKE failed: {path} -> {exc}")

    out_path = Path(args.out)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "score", "true_label", "pred_label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
