"""Diagnoses the deployed age/gender model for gender-accuracy bias, specifically an
ethnicity interaction that has never been checked before.

Written after real-world testing (both the formal 20-photo Evaluate-mode set and casual
Analyze-tab testing) repeatedly found the model calling real women "Male" far more often
than it should, despite 86.8-87.3% overall gender accuracy on UTKFace's own held-out test
set. The age-bias investigation (analyze_age_bias.py) already broke UTKFace down by its
`ethnicity` label for the AGE question and found no demographic effect for males -- but
that check never looked at GENDER accuracy, and never looked at women at all. This script
closes that gap: does UTKFace's own held-out test set show a gender-accuracy gap for any
specific ethnicity group, particularly the one most likely to match this project's real
evaluation photos (Black subjects)?

Usage:
  python analyze_gender_bias.py
"""
import pathlib

import numpy as np
import tensorflow as tf
from datasets import load_dataset as hf_load_dataset

from prepare_utkface import _detect_and_crop

HF_DATASET_NAME = "nu-delta/utkface"
IMG_SIZE = 96
ROOT = pathlib.Path(__file__).resolve().parent
MODELS_DIR = ROOT / "saved_models"
DATA_DIR = ROOT.parent / "data" / "utkface"


def _prep(pil_img) -> np.ndarray:
    cropped, _ = _detect_and_crop(pil_img)
    arr = cropped.astype(np.float32)
    return tf.keras.applications.mobilenet_v2.preprocess_input(arr)


def main() -> None:
    model = tf.keras.models.load_model(MODELS_DIR / "age_gender_model.keras")
    test_idx = np.load(DATA_DIR / "test_indices.npy")
    ds = hf_load_dataset(HF_DATASET_NAME, split="train")

    images, true_genders, true_ethnicities, true_ages = [], [], [], []
    for i in test_idx:
        ex = ds[int(i)]
        images.append(_prep(ex["image"]))
        true_genders.append(ex["gender"])
        true_ethnicities.append(ex["ethnicity"])
        true_ages.append(ex["age"])
    images = np.stack(images)
    true_genders = np.array(true_genders)
    true_ethnicities = np.array(true_ethnicities)
    true_ages = np.array(true_ages)

    preds = model.predict(images, batch_size=64, verbose=0)
    gender_raw = preds[1].flatten()  # <0.5 = male
    pred_genders = np.where(gender_raw < 0.5, "Male", "Female")
    correct = pred_genders == true_genders

    print("=== 1. Overall gender accuracy (sanity check against training-set baseline) ===")
    print(f"  n={len(correct)}  accuracy={100*correct.mean():.2f}%")
    for g in ["Male", "Female"]:
        mask = true_genders == g
        print(f"  {g:8s} n={mask.sum():4d}  accuracy={100*correct[mask].mean():.2f}%")

    print("\n=== 2. Gender accuracy x ethnicity (the never-before-checked interaction) ===")
    print(f"{'ethnicity':10s} {'gender':8s} {'n':>5s} {'accuracy':>9s}")
    ethnicities_present = sorted(set(true_ethnicities))
    for eth in ethnicities_present:
        for g in ["Male", "Female"]:
            mask = (true_ethnicities == eth) & (true_genders == g)
            n = mask.sum()
            if n < 5:
                print(f"{eth:10s} {g:8s} {n:5d}  (n<5, skipped)")
                continue
            acc = 100 * correct[mask].mean()
            print(f"{eth:10s} {g:8s} {n:5d} {acc:8.2f}%")

    print("\n=== 3. Gender accuracy x ethnicity x age range, Female only (finer view) ===")
    buckets = [(0, 29), (30, 49), (50, 116)]
    print(f"{'ethnicity':10s} {'age range':10s} {'n':>4s} {'accuracy':>9s}")
    for eth in ethnicities_present:
        for lo, hi in buckets:
            mask = (true_ethnicities == eth) & (true_genders == "Female") & (true_ages >= lo) & (true_ages <= hi)
            n = mask.sum()
            if n < 5:
                continue
            acc = 100 * correct[mask].mean()
            print(f"{eth:10s} {lo}-{hi:<7d} {n:4d} {acc:8.2f}%")

    print("\n=== 4. Mean raw gender score by ethnicity x gender (calibration check) ===")
    print("(raw score: 0=confident Male, 1=confident Female; a group whose Female mean")
    print(" sits far below 0.5 is one the model is systematically miscalibrated on)")
    print(f"{'ethnicity':10s} {'gender':8s} {'n':>5s} {'mean_raw':>9s}")
    for eth in ethnicities_present:
        for g in ["Male", "Female"]:
            mask = (true_ethnicities == eth) & (true_genders == g)
            n = mask.sum()
            if n < 5:
                continue
            print(f"{eth:10s} {g:8s} {n:5d} {gender_raw[mask].mean():9.3f}")


if __name__ == "__main__":
    main()
