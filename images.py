"""Research-only image baseline on two separate public NLM cohorts."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split


def load_images(folder, prefix):
    images, labels, names = [], [], []
    for path in sorted(folder.rglob("*.png")):
        label = path.stem.rsplit("_", 1)[-1]
        if not path.name.startswith(prefix) or label not in {"0", "1"}:
            continue
        with Image.open(path) as image:
            thumbnail = image.convert("L").resize((48, 48), Image.Resampling.LANCZOS)
            images.append(np.asarray(thumbnail, dtype=np.float32).ravel() / 255)
        labels.append(int(label))
        names.append(path.name)
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate filenames in {folder}")
    if set(labels) != {0, 1}:
        raise ValueError(f"Missing PNG images from either class in {folder}")
    return np.array(images), np.array(labels), names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shenzhen", type=Path)
    parser.add_argument("montgomery", type=Path)
    parser.add_argument("--out", type=Path, default=Path("image_results"))
    args = parser.parse_args()
    source, labels, _ = load_images(args.shenzhen, "CHNCXR_")
    test, test_labels, names = load_images(args.montgomery, "MCUCXR_")
    train, val, train_labels, val_labels = train_test_split(
        source, labels, test_size=.25, stratify=labels, random_state=42
    )
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(train, train_labels)
    val_scores = model.predict_proba(val)[:, 1]
    # Pick the highest validation threshold that flags at least 90% of positives.
    positive_scores = np.sort(val_scores[val_labels == 1])[::-1]
    threshold = float(positive_scores[int(np.ceil(.90 * len(positive_scores))) - 1])
    scores = model.predict_proba(test)[:, 1]
    predicted = scores >= threshold
    tn, fp, fn, tp = map(int, confusion_matrix(test_labels, predicted, labels=[0, 1]).ravel())
    metrics = {
        "train_images": len(train), "validation_images": len(val), "test_images": len(test),
        "threshold_from_shenzhen_validation": threshold, "test_roc_auc": roc_auc_score(test_labels, scores),
        "test_average_precision": average_precision_score(test_labels, scores),
        "test_sensitivity": tp / (tp + fn), "test_specificity": tn / (tn + fp),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with (args.out / "predictions.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["file", "label", "score", "flag"])
        writer.writerows(zip(names, test_labels, scores, predicted.astype(int)))


if __name__ == "__main__":
    main()
