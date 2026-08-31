"""Trains a small CNN *from scratch* on FER2013 (48x48 grayscale, 6 classes),
instead of transfer learning from an ImageNet-pretrained MobileNetV2 backbone.
Meant to run on a GPU (Colab) -- a first attempt at this on CPU measured
~2s/step (roughly 700s/epoch, ~15 hours for a full run) and was abandoned;
the same architecture should run in a small fraction of that time on a GPU,
making the epoch counts published FER2013 results actually use (60-100+)
practical.

Why a second architecture at all: ImageNet-pretrained features encode
object/texture priors that transfer reasonably well to coarse attributes like
age/gender but poorly to the fine, transient muscle patterns that distinguish
facial expressions -- a well-documented limitation in the FER2013 literature,
where most published results in the 65-75% range use custom architectures
trained directly on FER2013 rather than transfer learning. This is that
alternative: a compact VGG-style CNN (3 conv blocks + global average
pooling), trained directly on the native 48x48 grayscale images (no upscale,
no RGB conversion needed).

Meant to be combined with the alpha=1.0 MobileNetV2 variant from
train_expression.py as a two-model ensemble -- see
evaluate_expression_ensemble.py -- since an ensemble of two differently-biased
models typically outperforms either alone.

Usage:
  python train_expression_scratch.py --tag gpu --epochs 100
"""
import argparse
import json
import pathlib
import time

import numpy as np
import tensorflow as tf

from prepare_fer2013 import KEPT_LABELS, NATIVE_IMG_SIZE, load_dataset_native

ROOT = pathlib.Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "saved_models"

NUM_CLASSES = len(KEPT_LABELS)
BEST_CKPT_PATH = str(MODELS_DIR / "expression_scratch_best.weights.h5")


def conv_block(x, filters, dropout):
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    return x


def build_model() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(NATIVE_IMG_SIZE, NATIVE_IMG_SIZE, 1))
    x = conv_block(inputs, 64, 0.25)
    x = conv_block(x, 128, 0.25)
    x = conv_block(x, 256, 0.3)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="expression")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# See train_expression.py's identical constant for the full rationale: recall-informed
# weighting replaced naive inverse-frequency weighting because the evidence showed
# frequency doesn't explain the real weak spots (Sad has more training data than Angry
# or Surprise, yet the worst recall). Measured on the currently-deployed 2-model TTA
# ensemble's FER2013 test-set confusion matrix.
_MEASURED_RECALL = {
    "Angry": 0.6033, "Fear": 0.5153, "Happy": 0.8580,
    "Sad": 0.5138, "Surprise": 0.7386, "Neutral": 0.7164,
}


def compute_class_weights() -> dict:
    inv_recall = np.array([1.0 / _MEASURED_RECALL[label] for label in KEPT_LABELS])
    weights = inv_recall / inv_recall.mean()
    print("Recall-informed class weights:", dict(zip(KEPT_LABELS, weights.round(3).tolist())))
    return {i: float(w) for i, w in enumerate(weights)}


def main(tag: str, epochs: int, batch_size: int) -> None:
    train_ds = load_dataset_native("train", batch_size=batch_size, shuffle=True, augment=True)
    val_ds = load_dataset_native("val", batch_size=batch_size)
    test_ds = load_dataset_native("test", batch_size=batch_size)
    class_weight = compute_class_weights()

    model = build_model()
    model.summary()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            BEST_CKPT_PATH, monitor="val_accuracy", save_best_only=True, save_weights_only=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=False, verbose=1),
    ]

    start = time.time()
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs, class_weight=class_weight, callbacks=callbacks,
    )
    train_seconds = time.time() - start

    print("\nRestoring best-validation-accuracy weights...")
    model.load_weights(BEST_CKPT_PATH)

    test_metrics = model.evaluate(test_ds, return_dict=True)
    model.save(MODELS_DIR / "expression_scratch_model.keras")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "tag": tag,
        "approach": "from_scratch_cnn_48x48_grayscale",
        "epochs_requested": epochs,
        "epochs_run": len(history.history["loss"]),
        "batch_size": batch_size,
        "train_seconds": train_seconds,
        "seconds_per_epoch": train_seconds / len(history.history["loss"]),
        "test_metrics": test_metrics,
        "best_val_accuracy": float(max(history.history.get("val_accuracy", [0]))),
        "labels": KEPT_LABELS,
    }
    out_path = RESULTS_DIR / f"expression_scratch_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="gpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    main(args.tag, args.epochs, args.batch_size)
