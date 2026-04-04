#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path


def f1(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else (2 * tp) / denom


def evaluate(rows, threshold):
    tp = fp = tn = fn = 0
    for score, y_true in rows:
        y_pred = 1 if score > threshold else 0
        if y_true == 1 and y_pred == 1:
            tp += 1
        elif y_true == 0 and y_pred == 1:
            fp += 1
        elif y_true == 0 and y_pred == 0:
            tn += 1
        else:
            fn += 1
    f1_score = f1(tp, fp, fn)
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    return f1_score, acc, {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def main():
    parser = argparse.ArgumentParser(description="Calibrate binary threshold from score CSV")
    parser.add_argument("--csv", required=True, help="CSV with columns: score,true_label")
    parser.add_argument("--out", default="model_runtime_config.json", help="Config JSON to update")
    args = parser.parse_args()

    rows = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((float(r["score"]), int(r["true_label"])))

    best = (0.5, -1.0, -1.0, None)
    for i in range(1, 100):
        thr = i / 100.0
        f1_score, acc, cm = evaluate(rows, thr)
        if f1_score > best[1]:
            best = (thr, f1_score, acc, cm)

    out_path = Path(args.out)
    cfg = {}
    if out_path.exists():
        cfg = json.loads(out_path.read_text(encoding="utf-8"))

    cfg["threshold"] = round(best[0], 2)
    out_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    print("best_threshold", round(best[0], 2))
    print("best_f1", round(best[1], 4))
    print("best_acc", round(best[2], 4))
    print("confusion", best[3])


if __name__ == "__main__":
    main()
