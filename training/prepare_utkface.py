"""Builds train/val/test arrays from UTKFace, face-detected and cropped to match the
deployed app's preprocessing exactly.

Source: https://huggingface.co/datasets/nu-delta/utkface (public mirror of UTKFace, no
login required). Columns used: image (PIL, 200x200 RGB), age (int, 1-116), gender (str,
"Male"/"Female" -> mapped to 0/1 respectively, matching the original UTKFace convention).

Why face-detect-and-crop at all, when the original 200x200 images are already
close-cropped: the deployed web app always runs a face detector (BlazeFace) and crops to
its box (expanded by a margin) before resizing to 96x96 -- see web_app/app.js's
cropFaceToCanvas/FACE_CROP_MARGIN. The first version of this script instead resized the
*full* 200x200 image directly, with no detection or crop step at all. That mismatch
between training (zero crop) and deployment (always cropped) was root-caused after a
real user-reported misprediction (a close-up selfie scored 82.4 years for someone in
their 20s) -- see training/analyze_age_bias.py and the report's Limitations section for
the full investigation. This script fixes it by running face detection during data prep
too, with the same style of margin-expanded crop, so training and deployment see the same
kind of input. (Exact pixel-for-pixel matching to BlazeFace isn't the point and isn't
attempted -- OpenCV's bundled Haar cascade detector is used here since it needs no extra
model download and runs fast in a plain training loop; what matters is that both sides of
the pipeline crop-to-face-plus-margin instead of one side cropping and the other not.)

Usage:
  python prepare_utkface.py --build-manifest   # downloads (first run) + writes index manifests
  python prepare_utkface.py --build-crops      # face-detects + crops + writes train/val/test.npz
"""
import argparse
import pathlib

import cv2
import numpy as np
from datasets import load_dataset as hf_load_dataset

HF_DATASET_NAME = "nu-delta/utkface"
DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "utkface"
MANIFEST_DIR = DATA_DIR

IMG_SIZE = 96
# Matches web_app/app.js's FACE_CROP_MARGIN exactly -- both sides of the pipeline should
# expand the detected box by the same fraction before cropping.
FACE_CROP_MARGIN = 0.3

_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def _gender_to_int(gender_str: str) -> int:
    return 0 if gender_str == "Male" else 1


def _detect_and_crop(pil_image) -> np.ndarray:
    """Detects the largest face, crops to its box (expanded by FACE_CROP_MARGIN,
    clamped to image bounds), and resizes to IMG_SIZE x IMG_SIZE. Falls back to
    resizing the full image if no face is detected, matching the pre-fix behavior for
    exactly the examples a detector can't handle (rather than dropping them).

    Histogram equalization + loosened detector parameters were tuned against a random
    150-image sample of UTKFace (which spans very different lighting/exposure, being
    sourced from in-the-wild photos): default Haar cascade parameters on plain grayscale
    only detected 53% of faces; equalizeHist plus these parameters raised that to 86% on
    that sample, without a manual check turning up obviously spurious (wrong-region)
    detections. On the full dataset the rate settled at 81.4% (consistent across the
    train/val/test splits: 81.5%/81.3%/81.3%) -- the remaining ~19% fall back to a full
    resize, same as the original pre-fix behavior, rather than being dropped."""
    arr = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))

    if len(faces) == 0:
        resized = cv2.resize(arr, (IMG_SIZE, IMG_SIZE))
        return resized, False

    # largest by area
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    ih, iw = arr.shape[:2]
    marginX, marginY = w * FACE_CROP_MARGIN, h * FACE_CROP_MARGIN
    left = max(0, int(x - marginX))
    top = max(0, int(y - marginY))
    right = min(iw, int(x + w + marginX))
    bottom = min(ih, int(y + h + marginY))

    cropped = arr[top:bottom, left:right]
    resized = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))
    return resized, True


def build_manifest(seed: int = 42) -> None:
    ds = hf_load_dataset(HF_DATASET_NAME, split="train")
    n = len(ds)
    print(f"Loaded {n} UTKFace examples from {HF_DATASET_NAME}.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    splits = {
        "train": indices[:n_train],
        "val": indices[n_train : n_train + n_val],
        "test": indices[n_train + n_val :],
    }

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    for name, idx in splits.items():
        np.save(MANIFEST_DIR / f"{name}_indices.npy", idx)
        print(f"{name}: {len(idx)} examples -> {name}_indices.npy")


def build_crops() -> None:
    """Reads the index manifests (run --build-manifest first) and writes
    train/val/test.npz with pre-cropped 96x96 images, so training doesn't re-run face
    detection every epoch."""
    ds = hf_load_dataset(HF_DATASET_NAME, split="train")

    for split in ("train", "val", "test"):
        indices_path = MANIFEST_DIR / f"{split}_indices.npy"
        if not indices_path.exists():
            raise FileNotFoundError(f"{indices_path} not found. Run --build-manifest first.")
        indices = np.load(indices_path)

        images = np.zeros((len(indices), IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        ages = np.zeros(len(indices), dtype=np.int32)
        genders = np.zeros(len(indices), dtype=np.int32)
        n_detected = 0
        for out_i, idx in enumerate(indices):
            ex = ds[int(idx)]
            cropped, detected = _detect_and_crop(ex["image"])
            images[out_i] = cropped
            ages[out_i] = ex["age"]
            genders[out_i] = _gender_to_int(ex["gender"])
            n_detected += detected
            if (out_i + 1) % 2000 == 0:
                print(f"  {split}: {out_i + 1}/{len(indices)} ({n_detected} faces detected so far)")

        np.savez_compressed(DATA_DIR / f"{split}.npz", images=images, ages=ages, genders=genders)
        print(f"{split}: wrote {len(indices)} examples, face detected in "
              f"{n_detected}/{len(indices)} ({100 * n_detected / len(indices):.1f}%) -> {split}.npz")


# ---------- counterfactual beard augmentation (round 3: reduced strength) ----------
# A controlled test (training/analyze_age_bias.py) found the age model treats jaw-region
# darkness as a spurious "older" cue, likely because full facial hair is
# disproportionately associated with older subjects in UTKFace's own age distribution.
# During training, randomly darkening the jaw of some YOUNG faces gives the model direct
# counterexamples -- "this jaw-darkness pattern, on a young face, is still young" -- which
# a purely accuracy-driven objective has no other way to learn if the correlation in the
# real data is one-directional. This doesn't add any new images; it modifies existing
# young-face examples in place, only during training (never for val/test/inference).
#
# Round 1 used PROBABILITY=0.25 and strength in [0.6, 0.95] (near-opaque). It genuinely
# fixed the targeted age effect, but also corrupted the shared MobileNetV2 backbone badly
# enough to cause a real regression: bearded photos misclassified as Female 23% of the
# time (vs. 7% baseline), confirmed via a controlled --no-beard-aug ablation. Round 2
# deployed with the augmentation disabled entirely rather than ship that regression.
# Round 3 (below) is a deliberately weaker signal -- lower probability, and a much more
# translucent patch -- on the theory that a lighter counterfactual nudge can still teach
# "jaw-darkness isn't reliably 'older'" without as strongly corrupting the gender-relevant
# features the same backbone depends on. Must be re-validated against BOTH age AND gender
# (via analyze_age_bias.py) before deploying -- checking only age is exactly what let the
# Round 1 regression ship unnoticed the first time.
BEARD_AUG_MAX_AGE = 35.0
BEARD_AUG_PROBABILITY = 0.12


def _make_jaw_mask(size: int):
    import tensorflow as tf

    yy, xx = tf.meshgrid(tf.range(size, dtype=tf.float32), tf.range(size, dtype=tf.float32), indexing="ij")
    cx, cy = size * 0.5, size * 0.75
    rx, ry = size * 0.28, size * 0.20
    mask = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0
    return tf.cast(mask, tf.float32)[..., tf.newaxis]  # (size, size, 1)


def _apply_beard_augmentation(image, age):
    """image is float32 in [0, 255], RGB, shape (IMG_SIZE, IMG_SIZE, 3). Darkens the jaw
    region with random probability, only for young faces, only during training."""
    import tensorflow as tf

    should_apply = tf.logical_and(
        age < BEARD_AUG_MAX_AGE,
        tf.random.uniform([]) < BEARD_AUG_PROBABILITY,
    )

    def darken():
        mask = _make_jaw_mask(IMG_SIZE)
        # dark, slightly textured tone rather than flat black, closer to real facial hair
        noise = tf.random.uniform([IMG_SIZE, IMG_SIZE, 1], 15.0, 45.0)
        dark_color = tf.concat([noise, noise * 0.9, noise * 0.85], axis=-1)
        # Round 3: [0.25, 0.5], a translucent shadow rather than Round 1's near-opaque
        # [0.6, 0.95] patch -- a deliberately weaker counterfactual signal, see the
        # module comment above for why.
        strength = tf.random.uniform([], 0.25, 0.5)
        return image * (1 - mask * strength) + dark_color * (mask * strength)

    return tf.cond(should_apply, darken, lambda: image)


def load_dataset(split: str, batch_size: int = 32, shuffle: bool = False, augment: bool = False,
                  beard_augment: bool = True):
    """split is one of: train, val, test. Requires --build-crops to have been run.

    beard_augment exists to isolate the beard augmentation's effect from the
    crop-consistency fix's effect (they were evaluated together the first time, which
    made it impossible to tell which one caused a real gender-accuracy regression found
    afterward -- see training/analyze_age_bias.py and the report's Limitations section).
    Set to False to get crop-consistency without the augmentation."""
    import tensorflow as tf

    npz_path = DATA_DIR / f"{split}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} not found. Run `python prepare_utkface.py --build-crops` first.")
    data = np.load(npz_path)
    images, ages, genders = data["images"], data["ages"], data["genders"]

    ds = tf.data.Dataset.from_tensor_slices((images, ages, genders))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(ages), seed=42)

    def _prep(image, age, gender):
        image = tf.cast(image, tf.float32)
        if augment:
            # Random flip/brightness/contrast before preprocess_input's [-1,1] scaling,
            # same reasoning as before: these ops expect the 0-255 range.
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.15 * 255)
            image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
            image = tf.clip_by_value(image, 0.0, 255.0)
            if beard_augment:
                image = _apply_beard_augmentation(image, tf.cast(age, tf.float32))
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image, {"age": tf.cast(age, tf.float32), "gender": tf.cast(gender, tf.float32)}

    ds = ds.map(_prep, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-manifest", action="store_true")
    parser.add_argument("--build-crops", action="store_true")
    args = parser.parse_args()
    if args.build_manifest:
        build_manifest()
    elif args.build_crops:
        build_crops()
    else:
        parser.print_help()
