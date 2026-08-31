"""Evaluates every available expression model individually, and every
combination of them as a prediction-averaging ensemble, with and without
test-time augmentation (TTA, averaging the prediction for a face and its
horizontal mirror). Prints a clear comparison table so you can see exactly
which combination performs best before deciding what to deploy.

Looks for whichever of these exist and skips any that don't:
  saved_models/expression_model.keras        (MobileNetV2 alpha=0.35)
  saved_models/expression_model_wide.keras   (MobileNetV2 alpha=1.0)
  saved_models/expression_scratch_model.keras (from-scratch CNN, 48x48 grayscale)

Usage:
  python evaluate_expression_ensemble.py
"""
import itertools
import json
import pathlib

import numpy as np
import tensorflow as tf

from prepare_fer2013 import DATA_DIR, IMG_SIZE, KEPT_LABELS, NATIVE_IMG_SIZE

ROOT = pathlib.Path(__file__).resolve().parent
MODELS_DIR = ROOT / "saved_models"
RESULTS_DIR = ROOT / "results"

CANDIDATES = {
    "mobilenet_035": {"path": MODELS_DIR / "expression_model.keras", "kind": "mobilenet"},
    "mobilenet_1.0": {"path": MODELS_DIR / "expression_model_wide.keras", "kind": "mobilenet"},
    "scratch_cnn": {"path": MODELS_DIR / "expression_scratch_model.keras", "kind": "scratch"},
}


def prep_mobilenet(images: np.ndarray) -> np.ndarray:
    x = tf.expand_dims(images, axis=-1)
    x = tf.image.grayscale_to_rgb(x)
    x = tf.image.resize(x, [IMG_SIZE, IMG_SIZE])
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    return x.numpy()


def prep_scratch(images: np.ndarray) -> np.ndarray:
    x = np.expand_dims(images, axis=-1).astype("float32") / 255.0
    return x


def get_probs(model, kind: str, images: np.ndarray, flipped_images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prep = prep_mobilenet if kind == "mobilenet" else prep_scratch
    probs = model.predict(prep(images), batch_size=64, verbose=0)
    probs_flipped = model.predict(prep(flipped_images), batch_size=64, verbose=0)
    return probs, probs_flipped


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.argmax(probs, axis=1) == labels))


def confusion_matrix(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    preds = np.argmax(probs, axis=1)
    n = len(KEPT_LABELS)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    return cm


def print_confusion_matrix(cm: np.ndarray) -> None:
    """Per-class recall plus the confusion matrix, so class-level weaknesses (e.g. Sad
    being confused for Neutral) are visible, not just one overall accuracy number --
    added after a real-world misread (Surprise predicted as Happy) turned out, once
    actually measured, not to be the ensemble's real weak point (Fear and Sad are)."""
    n = len(KEPT_LABELS)
    header = "true\\pred".ljust(10) + "".join(l[:6].rjust(8) for l in KEPT_LABELS)
    print(header)
    for i, lbl in enumerate(KEPT_LABELS):
        row = lbl.ljust(10) + "".join(str(cm[i, j]).rjust(8) for j in range(n))
        print(row)
    print("\nPer-class recall and top confusion:")
    for i, lbl in enumerate(KEPT_LABELS):
        total = cm[i].sum()
        correct = cm[i, i]
        recall = correct / total if total else 0.0
        row_wo_correct = cm[i].copy()
        row_wo_correct[i] = 0
        top_idx = int(np.argmax(row_wo_correct))
        top_n = int(row_wo_correct[top_idx])
        print(f"  {lbl:10s} recall={recall*100:5.1f}%  n={total:4d}  "
              f"top confused as: {KEPT_LABELS[top_idx]} ({top_n} times, {top_n/total*100:.1f}%)")


def main() -> None:
    data = np.load(DATA_DIR / "test.npz")
    images, labels = data["images"], data["labels"]
    flipped = images[:, :, ::-1]

    loaded = {}
    for name, spec in CANDIDATES.items():
        if spec["path"].exists():
            loaded[name] = {"model": tf.keras.models.load_model(spec["path"]), "kind": spec["kind"]}
            print(f"Loaded {name} from {spec['path'].name}")
        else:
            print(f"Skipping {name}: {spec['path'].name} not found")

    if not loaded:
        raise SystemExit("No expression models found in saved_models/ -- train at least one first.")

    probs_plain = {}
    probs_flip = {}
    for name, info in loaded.items():
        p, pf = get_probs(info["model"], info["kind"], images, flipped)
        probs_plain[name] = p
        probs_flip[name] = pf
        print(f"{name}: plain accuracy = {accuracy(p, labels)*100:.2f}%, "
              f"TTA accuracy = {accuracy((p + pf) / 2, labels)*100:.2f}%")

    print("\n--- Ensemble combinations (averaging softmax across models) ---")
    results = {}
    names = list(loaded.keys())
    best = (None, 0.0)
    for r in range(1, len(names) + 1):
        for combo in itertools.combinations(names, r):
            plain_avg = np.mean([probs_plain[n] for n in combo], axis=0)
            tta_avg = np.mean([(probs_plain[n] + probs_flip[n]) / 2 for n in combo], axis=0)
            plain_acc = accuracy(plain_avg, labels)
            tta_acc = accuracy(tta_avg, labels)
            label = "+".join(combo)
            results[label] = {"plain_accuracy": plain_acc, "tta_accuracy": tta_acc}
            print(f"{label}: plain = {plain_acc*100:.2f}%, with TTA = {tta_acc*100:.2f}%")
            if tta_acc > best[1]:
                best = (label + " (with TTA)", tta_acc)
                best_combo = combo

    print(f"\nBest combination: {best[0]} -> {best[1]*100:.2f}%")

    print("\n--- Confusion matrix for the best combination (rows=true, cols=predicted) ---")
    best_tta_avg = np.mean([(probs_plain[n] + probs_flip[n]) / 2 for n in best_combo], axis=0)
    cm = confusion_matrix(best_tta_avg, labels)
    print_confusion_matrix(cm)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "expression_ensemble_comparison.json"
    with open(out_path, "w") as f:
        json.dump({
            "combinations": results,
            "best": {"label": best[0], "accuracy": best[1]},
            "best_confusion_matrix": cm.tolist(),
            "labels": KEPT_LABELS,
        }, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
