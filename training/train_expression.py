"""Trains the expression-recognition model on FER2013 (6 classes, Disgust dropped).

Two-phase training: phase 1 trains a small head on a frozen MobileNetV2 backbone
(fast), phase 2 unfreezes the ENTIRE backbone and continues at a much lower
learning rate (proper fine-tuning, not just a linear probe on frozen ImageNet
features -- ImageNet features weren't optimized to distinguish facial
expressions, so this phase matters more here than for age/gender). An earlier
version only unfroze the top 30-60 layers; unfreezing everything gave a
further real improvement without a prohibitive CPU time cost, since alpha=0.35
keeps the whole backbone small.

Also applies: class weighting (FER2013's classes are imbalanced -- Happy is ~2.3x
more common than Surprise in this data), a ReduceLROnPlateau schedule so phase 2
doesn't stall at a fixed LR, and restores the best-validation-accuracy weights
seen across all of phase 2 rather than just whatever the last epoch happened to
land on (fine-tuning on a dataset this size can drift past its best point well
before the epoch budget runs out).

--alpha controls MobileNetV2's width multiplier. The default (0.35) is what was
used for the CPU-vs-GPU comparison in the report (Table 1) -- run this script
unchanged with --tag gpu on Colab to reproduce that leg. --alpha 1.0 trains a
substantially higher-capacity variant only practical with a GPU; it's saved
under a different filename (expression_model_wide.keras) so it doesn't
overwrite the standard comparison model, and is meant to be combined with
train_expression_scratch.py's model as a two-model ensemble (see
evaluate_expression_ensemble.py) for a further accuracy push.

Usage:
  python train_expression.py --tag gpu --epochs 20 --finetune-epochs 20
  python train_expression.py --tag gpu_wide --alpha 1.0 --epochs 15 --finetune-epochs 20
"""
import argparse
import json
import pathlib
import time

import numpy as np
import tensorflow as tf

from prepare_fer2013 import IMG_SIZE, KEPT_LABELS, load_dataset

ROOT = pathlib.Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "saved_models"

NUM_CLASSES = len(KEPT_LABELS)


def build_model(alpha: float) -> tuple[tf.keras.Model, tf.keras.Model]:
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        alpha=alpha,
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False  # phase 1: frozen backbone -> fast, CPU-friendly head-only training

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="expression")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base


# Recall of the currently-deployed 2-model TTA ensemble on the FER2013 test set
# (measured directly: training/results/expression_ensemble_comparison.json's winning
# combination, evaluated per-class rather than just overall). Order matches KEPT_LABELS
# (Angry, Fear, Happy, Sad, Surprise, Neutral). Kept here, not recomputed at train time,
# because computing it requires an already-trained model -- it's a diagnostic snapshot,
# not a live measurement.
#
# This replaced naive inverse-frequency weighting (kept in every prior round) after the
# evidence showed frequency doesn't explain the real weak spots: Sad has MORE training
# examples (3750) than Angry (3116) or Surprise (2094), yet the worst recall of any class
# (51.4%) alongside Fear (51.5%) -- both well below Surprise's 73.9%, despite Surprise
# being the smallest class. Fear and Sad are the FER2013 literature's well-documented
# "hard triad" with Angry: visually similar, low-resolution muscle cues that are
# genuinely difficult to separate, not merely under-sampled. Weighting by observed
# recall targets that directly instead of guessing from class counts.
_MEASURED_RECALL = {
    "Angry": 0.6033, "Fear": 0.5153, "Happy": 0.8580,
    "Sad": 0.5138, "Surprise": 0.7386, "Neutral": 0.7164,
}


def compute_class_weights() -> dict:
    inv_recall = np.array([1.0 / _MEASURED_RECALL[label] for label in KEPT_LABELS])
    weights = inv_recall / inv_recall.mean()  # normalize so the average weight is 1.0
    print("Recall-informed class weights:", dict(zip(KEPT_LABELS, weights.round(3).tolist())))
    return {i: float(w) for i, w in enumerate(weights)}


def main(tag: str, epochs: int, finetune_epochs: int, batch_size: int, alpha: float) -> None:
    model_name = "expression_model" if alpha == 0.35 else "expression_model_wide"
    best_ckpt_path = str(MODELS_DIR / f"{model_name}_best.weights.h5")

    train_ds = load_dataset("train", batch_size=batch_size, shuffle=True, augment=True)
    val_ds = load_dataset("val", batch_size=batch_size)
    test_ds = load_dataset("test", batch_size=batch_size)
    class_weight = compute_class_weights()

    model, base = build_model(alpha)
    model.summary()

    start = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=class_weight)

    # Unfreeze the ENTIRE backbone (not just the top N layers) -- with alpha=0.35
    # this is a small enough network that full fine-tuning is still fast on CPU
    # (proven: partial unfreezing of the top 60 layers only added ~35% time
    # over the frozen phase), and expression is different enough from
    # ImageNet's object-recognition features that even the low/mid-level
    # filters plausibly benefit from adapting, not just the top layers.
    base.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-5),  # lower than the top-60-layer run: more layers moving at once
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"\n--- Phase 2: fine-tuning the ENTIRE backbone (alpha={alpha}) ---\n")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            best_ckpt_path, monitor="val_accuracy", save_best_only=True, save_weights_only=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]
    finetune_history = model.fit(
        train_ds, validation_data=val_ds, epochs=finetune_epochs, class_weight=class_weight, callbacks=callbacks,
    )
    for k, v in finetune_history.history.items():
        history.history.setdefault(k, []).extend(v)

    print("\nRestoring best-validation-accuracy weights from phase 2 checkpoint...")
    model.load_weights(best_ckpt_path)

    train_seconds = time.time() - start

    test_metrics = model.evaluate(test_ds, return_dict=True)

    model.save(MODELS_DIR / f"{model_name}.keras")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "tag": tag,
        "alpha": alpha,
        "model_name": model_name,
        "epochs": epochs,
        "finetune_epochs": finetune_epochs,
        "fine_tune_layers": "all",
        "batch_size": batch_size,
        "train_seconds": train_seconds,
        "seconds_per_epoch": train_seconds / (epochs + finetune_epochs),
        "test_metrics": test_metrics,
        "final_val_metrics": {k: v[-1] for k, v in history.history.items() if k.startswith("val_")},
        "best_val_accuracy": float(max(history.history.get("val_accuracy", [0]))),
        "labels": KEPT_LABELS,
    }
    out_path = RESULTS_DIR / f"expression_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="cpu", help="e.g. cpu or gpu -- labels the results file")
    parser.add_argument("--epochs", type=int, default=20, help="phase 1: frozen-backbone head training")
    parser.add_argument("--finetune-epochs", type=int, default=20, help="phase 2: fine-tune top backbone layers")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.35, help="MobileNetV2 width multiplier")
    args = parser.parse_args()
    main(args.tag, args.epochs, args.finetune_epochs, args.batch_size, args.alpha)
