"""Trains the age (regression) + gender (binary) model on UTKFace.

Run the exact same script locally (CPU) and on Google Colab (GPU) with a
different --tag so both legs of the CPU-vs-GPU report table come from
identical code/hyperparameters -- only the hardware differs.

Usage:
  python train_age_gender.py --tag cpu --epochs 15
"""
import argparse
import json
import pathlib
import time

import numpy as np
import tensorflow as tf

from prepare_utkface import IMG_SIZE, load_dataset

ROOT = pathlib.Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "saved_models"

# Must match ELDERLY_AGE_THRESHOLD in android_app/.../EvaluateActivity.kt -- the
# 20-image evaluation only has adult/elderly ground truth (per the brief), not
# exact ages, so bucket accuracy at this threshold is the only age metric that
# can be reported consistently across the CPU/GPU test set *and* the edge device.
ELDERLY_AGE_THRESHOLD = 60.0


def _age_bucket_accuracy(model: tf.keras.Model, test_ds: tf.data.Dataset) -> float:
    age_preds, _ = model.predict(test_ds, verbose=0)
    true_ages = np.concatenate([labels["age"].numpy() for _, labels in test_ds.unbatch().batch(1024)])
    pred_bucket = age_preds.flatten() >= ELDERLY_AGE_THRESHOLD
    true_bucket = true_ages >= ELDERLY_AGE_THRESHOLD
    return float(np.mean(pred_bucket == true_bucket))


FINE_TUNE_LAYERS = 30  # how many of MobileNetV2's top layers to unfreeze in phase 2


def build_model(alpha: float = 0.35) -> tuple[tf.keras.Model, tf.keras.Model]:
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

    age_output = tf.keras.layers.Dense(1, name="age")(x)
    gender_output = tf.keras.layers.Dense(1, activation="sigmoid", name="gender")(x)

    model = tf.keras.Model(inputs, [age_output, gender_output])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={"age": "mse", "gender": "binary_crossentropy"},
        loss_weights={"age": 1.0 / 400.0, "gender": 1.0},  # roughly balance loss scales
        metrics={"age": ["mae"], "gender": ["accuracy"]},
    )
    return model, base


def main(tag: str, epochs: int, finetune_epochs: int, batch_size: int, alpha: float,
         beard_augment: bool = True) -> None:
    model_name = "age_gender_model" if alpha == 0.35 else "age_gender_model_wide"

    train_ds = load_dataset("train", batch_size=batch_size, shuffle=True, augment=True,
                             beard_augment=beard_augment)
    val_ds = load_dataset("val", batch_size=batch_size)
    test_ds = load_dataset("test", batch_size=batch_size)

    model, base = build_model(alpha=alpha)
    model.summary()

    start = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    # Phase 2: unfreeze the top FINE_TUNE_LAYERS of the backbone and continue
    # training at a much lower learning rate, so the pretrained ImageNet
    # features get adapted to faces/expressions instead of being destroyed by
    # large early-training gradients.
    base.trainable = True
    for layer in base.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss={"age": "mse", "gender": "binary_crossentropy"},
        loss_weights={"age": 1.0 / 400.0, "gender": 1.0},
        metrics={"age": ["mae"], "gender": ["accuracy"]},
    )
    print(f"\n--- Phase 2: fine-tuning top {FINE_TUNE_LAYERS} backbone layers ---\n")
    finetune_history = model.fit(train_ds, validation_data=val_ds, epochs=finetune_epochs)
    for k, v in finetune_history.history.items():
        history.history.setdefault(k, []).extend(v)

    train_seconds = time.time() - start

    test_metrics = model.evaluate(test_ds, return_dict=True)
    test_metrics["age_bucket_accuracy"] = _age_bucket_accuracy(model, test_ds)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / f"{model_name}.keras")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "tag": tag,
        "alpha": alpha,
        "model_name": model_name,
        "beard_augment": beard_augment,
        "epochs": epochs,
        "finetune_epochs": finetune_epochs,
        "batch_size": batch_size,
        "train_seconds": train_seconds,
        "seconds_per_epoch": train_seconds / (epochs + finetune_epochs),
        "test_metrics": test_metrics,
        "final_val_metrics": {k: v[-1] for k, v in history.history.items() if k.startswith("val_")},
    }
    out_path = RESULTS_DIR / f"age_gender_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="cpu", help="e.g. cpu or gpu -- labels the results file")
    parser.add_argument("--epochs", type=int, default=15, help="phase 1: frozen-backbone head training")
    parser.add_argument("--finetune-epochs", type=int, default=10, help="phase 2: fine-tune top backbone layers")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.35, help="MobileNetV2 width multiplier")
    parser.add_argument("--no-beard-aug", action="store_true",
                         help="disable the counterfactual beard augmentation, to isolate its effect from the crop-consistency fix")
    args = parser.parse_args()
    main(args.tag, args.epochs, args.finetune_epochs, args.batch_size, args.alpha,
         beard_augment=not args.no_beard_aug)
