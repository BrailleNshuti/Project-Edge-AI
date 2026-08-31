"""Builds train/val/test arrays from FER2013.

Source: https://huggingface.co/datasets/JimmyUnleashed/FER-2013 (public mirror,
no login required). Columns: emotion (int, standard FER2013 challenge ids
0=Angry 1=Disgust 2=Fear 3=Happy 4=Sad 5=Surprise 6=Neutral), pixels (str, 2304
space-separated grayscale values for a 48x48 image).

We drop "Disgust" (too rare/noisy, ~1.5% of the dataset) and remap the rest to
0..5. The resulting label order is written to fer2013/emotion_labels.json and
MUST be reused by the Android app and evaluation scripts so indices line up.

Note on this mirror's quirks (verified by inspection): its "test" split has
emotion=None for every row (unusable), and its "train" split contains each
labeled example twice. So we only use "train", de-duplicate by the raw pixel
string, and then make our own 80/10/10 train/val/test split from the result.

Usage:
  python prepare_fer2013.py --build-arrays
"""
import argparse
import json
import pathlib

import numpy as np
from datasets import load_dataset as hf_load_dataset

HF_DATASET_NAME = "JimmyUnleashed/FER-2013"
DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "fer2013"

ORIGINAL_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
DROP_LABEL_ID = ORIGINAL_LABELS.index("Disgust")
KEPT_LABELS = [l for l in ORIGINAL_LABELS if l != "Disgust"]
HAPPY_INDEX = KEPT_LABELS.index("Happy")
SAD_INDEX = KEPT_LABELS.index("Sad")

IMG_SIZE = 96


def _remap_and_filter(emotions: np.ndarray, pixels_list) -> tuple[np.ndarray, np.ndarray]:
    remap = {}
    new_id = 0
    for old_id in range(len(ORIGINAL_LABELS)):
        if old_id == DROP_LABEL_ID:
            continue
        remap[old_id] = new_id
        new_id += 1

    keep_mask = emotions != DROP_LABEL_ID
    kept_emotions = emotions[keep_mask]
    kept_pixels = [p for p, keep in zip(pixels_list, keep_mask) if keep]

    labels = np.array([remap[e] for e in kept_emotions], dtype=np.int64)
    images = np.stack(
        [np.fromstring(p, sep=" ", dtype=np.uint8).reshape(48, 48) for p in kept_pixels]
    )
    return images, labels


def build_arrays(seed: int = 42) -> None:
    raw_train = hf_load_dataset(HF_DATASET_NAME, split="train")

    emotions = np.array(raw_train["emotion"])
    pixels = raw_train["pixels"]

    # de-duplicate: this mirror repeats every labeled row twice
    seen = set()
    keep = np.zeros(len(pixels), dtype=bool)
    for i, p in enumerate(pixels):
        if p not in seen:
            seen.add(p)
            keep[i] = True
    emotions, pixels = emotions[keep], [p for p, k in zip(pixels, keep) if k]
    print(f"After de-duplication: {len(pixels)} unique labeled images (from {len(keep)} raw rows).")

    images, labels = _remap_and_filter(emotions, pixels)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(labels))
    n = len(labels)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA_DIR / "train.npz", images=images[train_idx], labels=labels[train_idx])
    np.savez_compressed(DATA_DIR / "val.npz", images=images[val_idx], labels=labels[val_idx])
    np.savez_compressed(DATA_DIR / "test.npz", images=images[test_idx], labels=labels[test_idx])

    print(f"train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")

    with open(DATA_DIR / "emotion_labels.json", "w") as f:
        json.dump(KEPT_LABELS, f, indent=2)
    print(f"Label order ({len(KEPT_LABELS)} classes): {KEPT_LABELS}")


def _preprocess(image, label, augment):
    import tensorflow as tf

    image = tf.expand_dims(image, axis=-1)  # 48x48 -> 48x48x1
    image = tf.image.grayscale_to_rgb(image)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    if augment:
        # Random flip/brightness/contrast on the 0-255-range image, before
        # preprocess_input scales it to [-1, 1] -- doing this after that
        # scaling would feed out-of-expected-range values into these ops.
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.15 * 255)
        image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
        image = tf.clip_by_value(image, 0.0, 255.0)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label


def load_dataset(split: str, batch_size: int = 32, shuffle: bool = False, augment: bool = False):
    import tensorflow as tf

    npz_path = DATA_DIR / f"{split}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} not found. Run `python prepare_fer2013.py --build-arrays` first.")
    data = np.load(npz_path)
    images, labels = data["images"], data["labels"]
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(labels), seed=42)
    ds = ds.map(lambda img, lbl: _preprocess(img, lbl, augment), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


NATIVE_IMG_SIZE = 48


def _preprocess_native(image, label, augment):
    """Keeps images at native 48x48 grayscale (no upscale, no RGB conversion)
    -- for a small CNN trained from scratch on FER2013 rather than transfer
    learning from an ImageNet-pretrained RGB backbone. Simple [0,1] scaling
    (not MobileNetV2's preprocess_input) since this isn't feeding a
    MobileNetV2 backbone."""
    import tensorflow as tf

    image = tf.expand_dims(image, axis=-1)  # 48x48 -> 48x48x1
    image = tf.cast(image, tf.float32)
    if augment:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.15 * 255)
        image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
        image = tf.clip_by_value(image, 0.0, 255.0)
        # small zoom via a FIXED-size crop (pad up first, then crop back down
        # to a constant shape). A dynamic crop size here (e.g. from
        # tf.random.uniform) forces TF to retrace its graph per-example
        # instead of running a vectorized batch op -- measured ~10x slower
        # than the model's actual compute cost when this was tried.
        pad = 4
        image = tf.image.resize_with_crop_or_pad(image, NATIVE_IMG_SIZE + pad, NATIVE_IMG_SIZE + pad)
        image = tf.image.random_crop(image, [NATIVE_IMG_SIZE, NATIVE_IMG_SIZE, 1])
    image = image / 255.0
    return image, label


def load_dataset_native(split: str, batch_size: int = 64, shuffle: bool = False, augment: bool = False):
    import tensorflow as tf

    npz_path = DATA_DIR / f"{split}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} not found. Run `python prepare_fer2013.py --build-arrays` first.")
    data = np.load(npz_path)
    images, labels = data["images"], data["labels"]
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(labels), seed=42)
    ds = ds.map(lambda img, lbl: _preprocess_native(img, lbl, augment), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-arrays", action="store_true")
    args = parser.parse_args()
    if args.build_arrays:
        build_arrays()
    else:
        parser.print_help()
