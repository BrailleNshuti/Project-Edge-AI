"""Evaluates the expression model on the test set with test-time augmentation
(average softmax over the original image and its horizontal flip), matching
exactly what web_app/app.js does at inference. Reported here so the number in
the report reflects the model's actual deployed behavior, not just the
straight (non-TTA) test accuracy from training's model.evaluate() call.

Usage:
  python evaluate_expression_tta.py
"""
import json
import pathlib

import numpy as np
import tensorflow as tf

from prepare_fer2013 import DATA_DIR, IMG_SIZE

ROOT = pathlib.Path(__file__).resolve().parent
MODELS_DIR = ROOT / "saved_models"
RESULTS_DIR = ROOT / "results"


def main() -> None:
    model = tf.keras.models.load_model(MODELS_DIR / "expression_model.keras")
    data = np.load(DATA_DIR / "test.npz")
    images, labels = data["images"], data["labels"]

    def to_input(imgs):
        x = tf.expand_dims(imgs, axis=-1)
        x = tf.image.grayscale_to_rgb(x)
        x = tf.image.resize(x, [IMG_SIZE, IMG_SIZE])
        return tf.keras.applications.mobilenet_v2.preprocess_input(x)

    batch = to_input(images)
    flipped = tf.image.flip_left_right(batch)

    probs_a = model.predict(batch, batch_size=64, verbose=0)
    probs_b = model.predict(flipped, batch_size=64, verbose=0)
    tta_probs = (probs_a + probs_b) / 2

    plain_preds = np.argmax(probs_a, axis=1)
    tta_preds = np.argmax(tta_probs, axis=1)

    plain_acc = float(np.mean(plain_preds == labels))
    tta_acc = float(np.mean(tta_preds == labels))

    print(f"Plain (no TTA) test accuracy: {plain_acc * 100:.2f}%")
    print(f"With TTA (flip-averaged) test accuracy: {tta_acc * 100:.2f}%")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "expression_tta_cpu.json"
    with open(out_path, "w") as f:
        json.dump({"plain_test_accuracy": plain_acc, "tta_test_accuracy": tta_acc}, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
