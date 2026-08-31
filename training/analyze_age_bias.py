"""Diagnoses the age/gender model for demographic bias and spurious visual cues.

Written after a real user-reported misprediction (a close-up selfie of a bearded man in
his 20s scored 82.4 years). Runs checks in the order the investigation actually
proceeded, including one added after a second real incident:

1. General age-range bias: mean signed error (bias) and MAE per age bucket, across all
   ethnicities pooled. Checks whether the model systematically over-predicts age for any
   broad age range.
2. Demographic bias: the same breakdown, but crossed with UTKFace's `ethnicity` label
   *and* a young-adult age range, since a demographic group's aggregate accuracy can look
   fine while masking worse accuracy in an underrepresented sub-range.
3. Causal facial-hair test, on BOTH age and gender: paints a crude, synthetic dark region
   over the jaw/chin of real young-adult test photos (simulating a beard, changing
   nothing else) and compares the resulting age shift AND gender-misclassification rate
   against the same manipulation applied to the forehead and a cheek as controls. The
   gender half of this check was added only after a real regression: a fix aimed
   entirely at the age output (evaluated only for its effect on age) was deployed, and a
   user then reported a bearded selfie misclassified as female -- the fix had degraded
   backbone features shared with the gender head, which nothing had checked for before
   shipping. Anyone tuning the beard augmentation further should treat a jaw-specific
   effect on EITHER output as a problem, not just age.

Usage:
  python analyze_age_bias.py
"""
import pathlib

import numpy as np
import tensorflow as tf
from datasets import load_dataset as hf_load_dataset
from PIL import Image, ImageDraw

from prepare_utkface import _detect_and_crop

HF_DATASET_NAME = "nu-delta/utkface"
IMG_SIZE = 96
ROOT = pathlib.Path(__file__).resolve().parent
MODELS_DIR = ROOT / "saved_models"
DATA_DIR = ROOT.parent / "data" / "utkface"


def _predict_batch(model, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    preds = model.predict(images, batch_size=64, verbose=0)
    return preds[0].flatten(), preds[1].flatten()


def _cropped_array(pil_img) -> np.ndarray:
    """Runs the exact same face-detect-and-crop step training now uses (see
    prepare_utkface.py), so these diagnostics test the model against inputs shaped like
    what it was actually trained and deployed on -- not a plain resize, which stopped
    matching training once the crop-consistency fix landed."""
    cropped, _ = _detect_and_crop(pil_img)
    return cropped  # uint8, IMG_SIZE x IMG_SIZE x 3


def _prep(pil_img) -> np.ndarray:
    arr = _cropped_array(pil_img).astype(np.float32)
    return tf.keras.applications.mobilenet_v2.preprocess_input(arr)


def check_age_range_bias(model, ds, test_idx) -> None:
    print("\n=== 1. General age-range bias (all ethnicities pooled) ===")
    images, true_ages = [], []
    for i in test_idx:
        ex = ds[int(i)]
        images.append(_prep(ex["image"]))
        true_ages.append(ex["age"])
    images = np.stack(images)
    true_ages = np.array(true_ages)
    pred_ages, _ = _predict_batch(model, images)

    buckets = [(0, 12), (13, 19), (20, 29), (30, 39), (40, 49), (50, 59), (60, 116)]
    print(f"{'age range':12s} {'n':>4s} {'bias(pred-true)':>16s} {'MAE':>6s}")
    for lo, hi in buckets:
        mask = (true_ages >= lo) & (true_ages <= hi)
        if mask.sum() == 0:
            continue
        bias = (pred_ages[mask] - true_ages[mask]).mean()
        mae = np.abs(pred_ages[mask] - true_ages[mask]).mean()
        print(f"{lo:3d}-{hi:<8d} {mask.sum():4d} {bias:16.1f} {mae:6.1f}")

    return true_ages, pred_ages


def check_demographic_bias(model, ds, test_idx) -> None:
    print("\n=== 2. Demographic bias: ethnicity x age range, males only ===")
    images, true_ages, ethnicities, genders = [], [], [], []
    for i in test_idx:
        ex = ds[int(i)]
        images.append(_prep(ex["image"]))
        true_ages.append(ex["age"])
        ethnicities.append(ex["ethnicity"])
        genders.append(ex["gender"])
    images = np.stack(images)
    true_ages = np.array(true_ages)
    ethnicities = np.array(ethnicities)
    genders = np.array(genders)
    pred_ages, _ = _predict_batch(model, images)

    buckets = [(18, 29), (30, 39), (40, 59), (60, 116)]
    print(f"{'ethnicity':8s} {'age range':10s} {'n':>4s} {'bias':>7s} {'MAE':>6s}")
    for eth in ["White", "Black", "Indian", "Asian"]:
        for lo, hi in buckets:
            mask = (ethnicities == eth) & (true_ages >= lo) & (true_ages <= hi) & (genders == "Male")
            n = mask.sum()
            if n < 5:
                continue
            bias = (pred_ages[mask] - true_ages[mask]).mean()
            mae = np.abs(pred_ages[mask] - true_ages[mask]).mean()
            print(f"{eth:8s} {lo}-{hi:<6d} {n:4d} {bias:7.1f} {mae:6.1f}")


def check_facial_hair_cue(model, ds, test_idx, n_examples: int = 30) -> None:
    """Checks BOTH age and gender shifts from the same synthetic patches -- added after
    a real regression: a first fix (crop-consistency + a counterfactual beard
    augmentation) was evaluated for its effect on age only, deployed, and a user then
    reported a bearded selfie misclassified as female. A follow-up test found the same
    jaw patch that used to inflate age now flipped gender to female in 23% of cases (vs.
    7% normally) -- age and gender share the same backbone, so a fix aimed at one output
    can silently break the other if it's only evaluated against the one it targeted. This
    function now checks both every time, specifically to catch that class of mistake
    before deployment rather than after a user hits it.

    Separately: n=6 in early exploration gave a dramatic-looking +12.1 year mean jaw
    shift, 100% positive -- re-running at n=30 pulled that back to a more modest, honest
    +5.5 years, 77% positive. Small samples on a noisy per-image effect are easy to
    overstate; this defaults to 30 so the printed numbers are the trustworthy ones."""
    print(f"\n=== 3. Causal test: dark jaw patch (simulated beard) vs. forehead/cheek controls (n={n_examples}) ===")
    candidates = []
    for i in test_idx:
        ex = ds[int(i)]
        if ex["gender"] == "Male" and 20 <= ex["age"] <= 29:
            candidates.append(int(i))
        if len(candidates) >= n_examples:
            break

    def predict_arr(arr_uint8: np.ndarray) -> tuple[float, float]:
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr_uint8.astype(np.float32))
        p = model.predict(arr[None, ...], verbose=0)
        return float(p[0][0][0]), float(p[1][0][0])  # age, gender_raw (<0.5 = male)

    regions = {"jaw": [], "forehead": [], "cheek": []}
    age_shifts = {k: [] for k in regions}
    gender_vals = {"orig": [], **{k: [] for k in regions}}

    for idx in candidates:
        ex = ds[idx]
        # Crop first, THEN paint the synthetic patch -- the patch needs to land on the
        # jaw/forehead/cheek of the actual model input (the face-cropped 96x96 image),
        # not on the raw uncropped photo where those relative coordinates would be wrong.
        cropped = _cropped_array(ex["image"].convert("RGB"))
        img = Image.fromarray(cropped)
        w, h = img.size
        age_orig, gender_orig = predict_arr(np.array(img))
        gender_vals["orig"].append(gender_orig)

        boxes = {
            "jaw": [w * 0.22, h * 0.55, w * 0.78, h * 0.95],
            "forehead": [w * 0.22, h * 0.05, w * 0.78, h * 0.35],
            "cheek": [w * 0.05, h * 0.35, w * 0.40, h * 0.65],
        }
        for name, box in boxes.items():
            patched = img.copy()
            ImageDraw.Draw(patched, "RGBA").ellipse(box, fill=(25, 20, 18, 235))
            age_p, gender_p = predict_arr(np.array(patched))
            age_shifts[name].append(age_p - age_orig)
            gender_vals[name].append(gender_p)

    for k in age_shifts:
        age_shifts[k] = np.array(age_shifts[k])
    for k in gender_vals:
        gender_vals[k] = np.array(gender_vals[k])

    def summarize_age(name: str, shifts: np.ndarray) -> None:
        print(f"  age  {name:10s} mean={shifts.mean():+.2f}  median={np.median(shifts):+.2f}  "
              f"std={shifts.std():.2f}  pct_positive={100 * (shifts > 0).mean():.0f}%")

    def summarize_gender(name: str, vals: np.ndarray) -> None:
        print(f"  gender {name:10s} mean_raw={vals.mean():.3f}  pct_misclassified_female={100 * (vals >= 0.5).mean():.0f}%")

    print("Age shift (years, from the unpatched prediction):")
    for name in ("jaw", "forehead", "cheek"):
        summarize_age(name, age_shifts[name])

    try:
        from scipy import stats

        t_jf, p_jf = stats.ttest_rel(age_shifts["jaw"], age_shifts["forehead"])
        t_jc, p_jc = stats.ttest_rel(age_shifts["jaw"], age_shifts["cheek"])
        print(f"  paired t-test jaw vs forehead: t={t_jf:.2f} p={p_jf:.4f}")
        print(f"  paired t-test jaw vs cheek:    t={t_jc:.2f} p={p_jc:.4f}")
    except ImportError:
        print("  (install scipy for significance tests: pip install scipy)")

    print("\nGender (raw output, <0.5 = male; this is the check that would have caught")
    print("the round-1 regression before deployment):")
    summarize_gender("orig", gender_vals["orig"])
    for name in ("jaw", "forehead", "cheek"):
        summarize_gender(name, gender_vals[name])


def main() -> None:
    model = tf.keras.models.load_model(MODELS_DIR / "age_gender_model.keras")
    test_idx = np.load(DATA_DIR / "test_indices.npy")
    ds = hf_load_dataset(HF_DATASET_NAME, split="train")

    check_age_range_bias(model, ds, test_idx)
    check_demographic_bias(model, ds, test_idx)
    check_facial_hair_cue(model, ds, test_idx)


if __name__ == "__main__":
    main()
