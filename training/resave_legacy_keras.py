"""Rebuilds both trained models under TF's legacy Keras-2 compatibility shim
(tf_keras) and re-saves them as .h5, so tensorflowjs_converter -- which only
understands Keras 2's older config/graph serialization format -- can read
them. Keras 3's native format (inbound_nodes graph structure, DTypePolicy
dtype wrapper, batch_shape key) is structurally different, not just a
renamed field, so hand-patching model.json isn't viable; rebuilding under
the real legacy implementation and loading the already-trained weights
(portable, version-agnostic numpy arrays) is the correct fix.

Must be run with TF_USE_LEGACY_KERAS=1 set before Python starts (setting it
mid-process after tensorflow is already imported elsewhere has no effect).

Usage:
  TF_USE_LEGACY_KERAS=1 python resave_legacy_keras.py
"""
import os

if os.environ.get("TF_USE_LEGACY_KERAS") != "1":
    raise SystemExit("Run this with TF_USE_LEGACY_KERAS=1 set in the environment before Python starts.")

import numpy as np
import tensorflow as tf

print("tf.keras implementation:", tf.keras.__name__)

IMG_SIZE = 96


def build_age_gender_model(alpha: float = 0.35) -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), alpha=alpha, include_top=False, weights=None, pooling="avg",
    )
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    age_output = tf.keras.layers.Dense(1, name="age")(x)
    gender_output = tf.keras.layers.Dense(1, activation="sigmoid", name="gender")(x)
    return tf.keras.Model(inputs, [age_output, gender_output])


def build_expression_model(num_classes: int, alpha: float = 0.35) -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), alpha=alpha, include_top=False, weights=None, pooling="avg",
    )
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="expression")(x)
    return tf.keras.Model(inputs, outputs)


NATIVE_IMG_SIZE = 48


def _scratch_conv_block(x, filters, dropout):
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    return x


def build_expression_scratch_model(num_classes: int) -> tf.keras.Model:
    """Mirrors train_expression_scratch.py's build_model() exactly -- same
    layer order/params are required for the flat weights list to line up."""
    inputs = tf.keras.Input(shape=(NATIVE_IMG_SIZE, NATIVE_IMG_SIZE, 1))
    x = _scratch_conv_block(inputs, 64, 0.25)
    x = _scratch_conv_block(x, 128, 0.25)
    x = _scratch_conv_block(x, 256, 0.3)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="expression")(x)
    return tf.keras.Model(inputs, outputs)


def load_weights_npz(path: str) -> list:
    data = np.load(path)
    return [data[f"arr_{i}"] for i in range(len(data.files))]


def resave(name: str, model: tf.keras.Model) -> None:
    weights = load_weights_npz(f"saved_models/{name}_weights.npz")
    current = model.get_weights()
    if len(current) != len(weights):
        raise SystemExit(f"{name}: weight count mismatch -- legacy model has {len(current)}, saved file has {len(weights)}")
    for i, (c, w) in enumerate(zip(current, weights)):
        if c.shape != w.shape:
            raise SystemExit(f"{name}: shape mismatch at weight {i} -- legacy {c.shape} vs saved {w.shape}")
    model.set_weights(weights)
    out_path = f"saved_models/{name}_legacy.h5"
    model.save(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["age_gender", "expression", "expression_wide", "expression_scratch", "both"],
        default="both",
    )
    args = parser.parse_args()

    if args.model in ("age_gender", "both"):
        resave("age_gender_model", build_age_gender_model())
    if args.model in ("expression", "both"):
        with open("../data/fer2013/emotion_labels.json") as f:
            labels = json.load(f)
        resave("expression_model", build_expression_model(num_classes=len(labels)))
    if args.model == "expression_wide":
        with open("../data/fer2013/emotion_labels.json") as f:
            labels = json.load(f)
        resave("expression_model_wide", build_expression_model(num_classes=len(labels), alpha=1.0))
    if args.model == "expression_scratch":
        with open("../data/fer2013/emotion_labels.json") as f:
            labels = json.load(f)
        resave("expression_scratch_model", build_expression_scratch_model(num_classes=len(labels)))
