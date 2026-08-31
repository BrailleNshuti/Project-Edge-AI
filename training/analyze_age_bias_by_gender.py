"""Extends analyze_age_bias.py with checks it never ran: the original demographic-bias
check only ever looked at males (since the facial-hair investigation it was built for is
inherently male-coded). This script asks the questions that were left open:

1. Is age accuracy (bias and MAE) different for Female subjects than Male subjects,
   overall and per ethnicity? The original script's `check_demographic_bias` hard-filters
   to `genders == "Male"` and never reports the Female side at all.
2. Does a subject's gender-prediction error correlate with the size of their age error?
   Motivated by a real Analyze-tab case (a young adult woman scored 2.3 years, far beyond
   anything the facial-hair effect alone explains, on a photo the model also got gender
   wrong on) -- if age and gender errors are correlated, that points to a shared
   representation failure on certain inputs rather than two independent, separately
   explained biases.

Usage:
  python analyze_age_bias_by_gender.py
"""
import pathlib

import numpy as np
import tensorflow as tf
from datasets import load_dataset as hf_load_dataset

from prepare_utkface import _detect_and_crop

HF_DATASET_NAME = "nu-delta/utkface"
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

    images, true_ages, true_genders, true_ethnicities = [], [], [], []
    for i in test_idx:
        ex = ds[int(i)]
        images.append(_prep(ex["image"]))
        true_ages.append(ex["age"])
        true_genders.append(ex["gender"])
        true_ethnicities.append(ex["ethnicity"])
    images = np.stack(images)
    true_ages = np.array(true_ages)
    true_genders = np.array(true_genders)
    true_ethnicities = np.array(true_ethnicities)

    preds = model.predict(images, batch_size=64, verbose=0)
    pred_ages = preds[0].flatten()
    gender_raw = preds[1].flatten()
    pred_genders = np.where(gender_raw < 0.5, "Male", "Female")
    gender_correct = pred_genders == true_genders
    age_err = pred_ages - true_ages
    age_abs_err = np.abs(age_err)

    print("=== 1. Age bias/MAE by gender, all ethnicities pooled (never checked before) ===")
    for g in ["Male", "Female"]:
        mask = true_genders == g
        print(f"  {g:8s} n={mask.sum():4d}  bias={age_err[mask].mean():+6.2f}  MAE={age_abs_err[mask].mean():6.2f}")

    print("\n=== 2. Age bias/MAE by ethnicity x gender, adults only (age>=18; never checked for Female) ===")
    print(f"{'ethnicity':10s} {'gender':8s} {'n':>4s} {'bias':>7s} {'MAE':>6s}")
    for eth in sorted(set(true_ethnicities)):
        for g in ["Male", "Female"]:
            mask = (true_ethnicities == eth) & (true_genders == g) & (true_ages >= 18)
            n = mask.sum()
            if n < 5:
                continue
            print(f"{eth:10s} {g:8s} {n:4d} {age_err[mask].mean():+7.2f} {age_abs_err[mask].mean():6.2f}")

    print("\n=== 3. Does gender-prediction error correlate with age-error size? ===")
    print("(tests whether wrong-gender cases are ALSO the worst age misses -- a shared")
    print(" representation failure -- rather than two independent, unrelated biases)")
    wrong_gender_mae = age_abs_err[~gender_correct].mean() if (~gender_correct).sum() > 0 else float("nan")
    right_gender_mae = age_abs_err[gender_correct].mean()
    print(f"  MAE when gender WRONG   (n={(~gender_correct).sum():4d}): {wrong_gender_mae:.2f} years")
    print(f"  MAE when gender CORRECT (n={gender_correct.sum():4d}): {right_gender_mae:.2f} years")
    try:
        from scipy import stats
        # point-biserial-style: correlate |age error| with a 0/1 wrong-gender indicator
        r, p = stats.pointbiserialr((~gender_correct).astype(int), age_abs_err)
        print(f"  point-biserial correlation (wrong-gender vs |age error|): r={r:.3f}  p={p:.4f}")
    except ImportError:
        print("  (install scipy for the correlation test: pip install scipy)")

    # Also report the single worst age misses and whether gender was also wrong on them --
    # a direct, concrete look at whether the "2.3 years" style outlier has company.
    print("\n=== 4. The 15 worst age misses in the test set -- was gender also wrong? ===")
    worst = np.argsort(-age_abs_err)[:15]
    print(f"{'true_age':>8s} {'pred_age':>8s} {'abs_err':>7s} {'gender':>16s}")
    for i in worst:
        g_str = "CORRECT" if gender_correct[i] else "WRONG"
        print(f"{true_ages[i]:8.0f} {pred_ages[i]:8.1f} {age_abs_err[i]:7.1f} {true_genders[i]:>8s}->{g_str}")


if __name__ == "__main__":
    main()
