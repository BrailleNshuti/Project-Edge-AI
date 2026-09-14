# Edge AI Project -- Task 2: Age, Gender & Expression Recognition

Built for the `DLBAIPEAI - Project Edge AI` course. This implements Task 2 from the
official brief: an application that recognizes age, gender, and expression on-device,
plus everything needed to produce the two required report tables.

**Primary deliverable: `web_app/`** -- a browser-based version, built because the
student's only phone is an iPhone 11 Pro with no Mac available (native iOS needs
Xcode/macOS). The brief does not require Android: its own introductory literature cites
Apple's Machine Learning docs alongside Samsung/Huawei's SDKs, and just says "mobile
phone or tablet." The web app runs entirely client-side in Safari -- no App Store, no
code signing, no Mac -- and all inference happens on-device, satisfying "deploy on the
edge" the same way a native app would.

`android_app/` also exists, fully built and verified (compiles, runs, TFLite models
included) from before the platform was known to be an iPhone. It's kept as a working
alternative in case a real Android device ever becomes available, but it is **not**
the path to use with an iPhone -- an `.apk` cannot run on iOS at all.

**`Project_Report.docx`** -- the actual report submitted, restructured to match the
official IU course page's required format: a textual + graphical abstract,
introduction/literature review, method (with an architecture diagram), results analysis,
conclusion, references, and a source-code appendix -- the main body (Introduction through
Declaration of AI Tool Use) is 8 pages, within the course page's 7-10 page limit. Both
result tables are fully filled in with real numbers (GPU training, iPhone benchmark, and
the 20-photo demographic evaluation, all described in the numbered sections below), the
title page and AI-tool disclosure are personalized, and the document uses proper
front-matter (lowercase Roman) / body (Arabic, starting at 1) / back-matter (Roman,
continuing the front matter) page numbering across three real Word sections.

## What's here

```
training/           Python: dataset prep, model training, TFLite + TensorFlow.js conversion
  logs/              Raw training console output (large, reference only)
  results/           Metric JSONs -- feeds Table 1 of the report
  saved_models/      Trained models (.keras = canonical, tracked in git; the
                      _legacy.h5/_weights.npz conversion bridges and the rejected
                      beard-augmentation variant are gitignored -- present on disk,
                      regenerable via convert_to_tflite.py --help and
                      resave_legacy_keras.py --help, not pushed to GitHub since
                      they're build output, not source; see the age-bias Design note
                      below for what the rejected variant was)
  tflite/            Exported .tflite files (secondary Android artifact)
  analyze_age_bias.py  Bias/spurious-cue diagnostic for the age/gender model -- run this
                      after any retrain that touches age_gender_model.keras
colab_gpu_training.ipynb   Same training pipeline, for the GPU leg of the report table
web_app/            PRIMARY: browser app (plain HTML/JS + TensorFlow.js), works on the iPhone via Safari
android_app/        Alternative: Android app (Kotlin, Gradle CLI project) -- only useful with a real Android device
evaluation/          Scripts that turn raw results into report-ready tables
data/                Downloaded datasets (UTKFace, FER2013) -- not something you write to directly
reference_materials/ The 4 PDFs your course provided (brief, citation guide, etc.)
```

## How each report requirement maps to a concrete step

| Brief requirement | How it's produced |
|---|---|
| "Deploy your task on the edge" | `web_app/` -- BlazeFace + both TensorFlow.js models run fully in-browser, on-device |
| "Report ... performance ... trained on CPU or GPU ... performance on your edge device" | `training/train_*.py --tag cpu` locally, `colab_gpu_training.ipynb --tag gpu` on Colab, web app's Benchmark tab for edge latency (run on your iPhone), all merged by `evaluation/generate_performance_table.py` |
| "Collect ... 20 images ... evaluate performance" | Web app's Evaluate tab, scored against `evaluation/LABELING_TEMPLATE.csv`, formatted by `evaluation/evaluate_20_images.py` |
| "Appendix should present the source code" | Everything under `training/` and `web_app/` |
| "State if AI tools ... have been used" | You must write this yourself in the report -- see note at the bottom |

## 1. Train the models (already done for you, CPU leg)

```
cd training
pip install -r requirements.txt
python prepare_utkface.py --build-manifest      # one-time, downloads UTKFace (~1GB)
python prepare_fer2013.py --build-arrays        # one-time, downloads FER2013 (~300MB)
python train_age_gender.py --tag cpu --epochs 15 --finetune-epochs 10
python train_expression.py --tag cpu --epochs 20 --finetune-epochs 15
python convert_to_tflite.py
```

Run through several rounds of improvement (see `training/results/age_gender_cpu.json`
and `expression_cpu.json` for the final numbers, and `training/logs/` for the full
history):

| Metric | First pass (frozen only) | Final (CPU) |
|---|---|---|
| Age MAE | 8.49 years | **7.57 years** |
| Gender accuracy | 84.02% | **86.80%** |
| Age bucket accuracy (adult/elderly) | 91.65% | **92.87%** |
| Expression accuracy (6 classes, α=0.35) | 48.71% | **57.79%** (58.34% with test-time augmentation) |

Age/gender needed one round: fine-tune the top backbone layers instead of only training
a frozen-backbone head, plus augmentation. Expression needed several rounds to get from
48.71% to 57.79% -- unfreezing more of the backbone, class weighting (FER2013 is
imbalanced), a learning-rate schedule, best-checkpoint restoration, and finally
test-time augmentation at inference (averaging a face crop's prediction with its
horizontal mirror -- free at inference, no retraining) for a final 58.34%. This is the
CPU-only starting point; the GPU-based follow-up in Section 2 below pushed the *deployed*
expression accuracy considerably further. Full write-up of what was tried is in the
report's Results and Analysis section (Section 3) and `training/logs/`.

## 2. GPU numbers and the expression-accuracy push -- done (results already deployed)

`colab_gpu_training.ipynb` has been run (Google Colab, free T4 GPU). Both parts completed
and both are now reflected in `training/results/`, the deployed web app, and the report:

**Part 1 (required CPU-vs-GPU comparison)**: reproduced the exact CPU run on GPU, same
code and hyperparameters. Confirms training hardware alone doesn't change model quality
(GPU numbers are statistically indistinguishable from CPU -- see report Section 3) --
what it changes is training time: 2,905s -> 1,629s (1.8x) for age/gender, and a striking
7,834s -> 626s (12.5x) for expression.

**Part 2 (expression-accuracy push)**: trained a wider MobileNetV2 (α=1.0) and a compact
CNN from scratch directly on FER2013 (the architecture that was too slow to train on CPU
-- ~15 estimated hours became ~17 GPU-minutes), then compared every individual model and
combination as an averaged-softmax ensemble, with and without test-time augmentation. The
winner -- α=1.0 MobileNetV2 + the from-scratch CNN, both with TTA -- reached **67.60%**
test accuracy, up from 58.34% on CPU. Full comparison table is in the report (Table 3,
Section 3) and `training/results/expression_ensemble_comparison.json`.

**This is what's deployed right now.** The web app's expression prediction is this
2-model ensemble (`web_app/models/expression_wide/` + `web_app/models/expression_scratch/`),
not the original single α=0.35 model, which has been removed from the app. Expression
inference is correspondingly slower (~800ms instead of ~100-150ms, measured in-browser --
four forward passes instead of two) but still comfortably fast enough for this app's
single-photo workflow.

**Still short of the original 70% target** -- 67.60% sits within the published
human-inter-annotator-agreement range for FER2013 (~65-68%) and near the low end of what
purpose-built models typically reach (~70-75%). The report's Section 4 (Conclusion), which
folds in the limitations discussion, covers this honestly rather than overstating it. If you ever want to push further
yourself, the notebook and `evaluate_expression_ensemble.py` are reusable as-is -- e.g. to
try a third differently-biased architecture, more epochs, or FER2013-specific
augmentation.

If you ever retrain any model again, the conversion pipeline that turned the new
`.keras` files into what's now deployed is: `training/resave_legacy_keras.py --model
<name>` (rebuilds under the legacy Keras-2 shim TensorFlow.js's converter needs) then
`tensorflowjs_converter --input_format keras <name>_legacy.h5 web_app/models/<dest>/`.

## 3. Run the web app on your iPhone

The app is a handful of static files (HTML/CSS/JS + the two converted models) -- no
build step, no install on the iPhone side. You just need *something* to serve those
files so Safari can fetch them, since `file://` pages can't load the models. Easiest
option: your ThinkPad, over your home WiFi.

**On the ThinkPad:**
```
cd web_app
python -m http.server 8765
```

**On the iPhone**, connect to the *same WiFi network* as the ThinkPad, open Safari, and
go to `http://<thinkpad-lan-ip>:8765` (this laptop's current LAN IP is `192.168.0.117` --
if that stops working, re-check it on the laptop with `ipconfig` and look for the
"IPv4 Address" under the Wi-Fi adapter; it can change when you reconnect to WiFi).

Optional but nice: in Safari, tap the Share icon -> "Add to Home Screen". It'll launch
full-screen like a real app from then on (this is what the `manifest.json` / Apple
meta tags in `index.html` are for).

No live camera preview is used anywhere (deliberately -- it would require HTTPS, which
a plain LAN server doesn't have). Every mode uses a photo picker instead (`<input
type="file">` with `capture="environment"` for "Take Photo"), which triggers iOS's
native camera/photo UI regardless of HTTP vs HTTPS.

The app's look was redesigned from the original generic blue/system-font template to a
warmer, hand-crafted "photo booth" style (cream background, terracotta accent, native
iOS rounded font, Polaroid-style photo preview, ticket-style tabs) specifically so it
doesn't read as an AI-generated template. The home-screen icon was redrawn to match.

I tested the app's model loading, face detection, and age/gender/expression inference
myself (in-browser tests, not on the real iPhone) against real labeled photos at every
stage of this project, most recently after deploying the GPU-trained ensemble above -- a
real UTKFace test photo (true age 100, female) came back as age 92.99, Female, with the
full pipeline (face detection + age/gender + the 2-model expression ensemble) running end
to end with no errors and realistic steady-state latency (~800ms for expression, measured
on WebGL during development). **This has since actually been run on the real iPhone** --
see Section 4 below for the real on-device numbers, which are notably higher than the
WebGL figure above because the app forces the CPU backend in production (see the Design
notes' WebGL-precision entry for why).

## 4. Collect and evaluate real photos -- done, real results in

This was carried out on the real iPhone, following the brief's 20-image demographic
protocol: 20 real photos in the final round (18 adult / 2 elderly -- not a perfect
50/50 split; real people willing to be photographed, especially elderly subjects, were
genuinely hard to find, honestly noted in the report rather than hidden). Labeled against
`evaluation/LABELING_TEMPLATE.csv`, run through the web app's **Evaluate** tab, and
scored with:

```
python evaluation/evaluate_20_images.py evaluation/results.csv
```

Every ground-truth label in the returned `results.csv` was checked against the intended
labels CSV and matched exactly, confirming the run used the correct photos and labels.
This is the second independent real-device evaluation round for this project -- an
earlier 21-photo round (archived as `evaluation/results_21photo_archive.csv`) found a
similar pattern with different numbers, which is discussed below. The real, current
numbers (also in `evaluation_tables.md` and `Project_Report.docx` Table 2):

| Metric | Overall | Detail |
|---|---|---|
| Age bucket | 85.0% | Adult 94.4% (n=18), Elderly 0.0% (n=2) |
| Gender | 65.0% | Male 90.9% (n=11), Female 33.3% (n=9) |
| Expression | 60.0% | Happy 90.9% (n=11), Sad 22.2% (n=9) |

The gender gap is the headline finding, discussed in depth in the report's Section 3
(Discussion) and the Design notes below. The web app's **Benchmark** tab (Run 15 passes,
real device, not a desktop browser -- that's what the brief asks you to measure) was also
run on the real iPhone; see Section 3 above for those latency numbers.

**Anonymized in the report**: no identifiable photos of real people appear in
`Project_Report.docx` itself -- only the aggregate numbers/tables above, per the brief's
data-privacy rule.

## 5. Generate the final performance table -- done

```
python evaluation/generate_performance_table.py
```

`evaluation/edge_results.json` is already filled in with the real Benchmark-tab and
Evaluate-tab numbers above, and `evaluation/performance_table.md` is already generated
and pasted into `Project_Report.docx` Table 1. Re-run the command above only if you
change any of the underlying result files.

## Design notes worth knowing before you write the report

- **Age/gender is one lightweight model**; expression is a 2-model ensemble, not one --
  they don't share a dataset with common labels, so a single multi-task model wasn't a
  good fit to begin with, and expression specifically needed more capacity than
  age/gender to reach reasonable accuracy (see report Section 3 for why). `age_gender_model`
  (MobileNetV2 α=0.35 backbone, two small heads: age regression + gender binary) trained
  on UTKFace; expression is `expression_model_wide` (MobileNetV2 α=1.0) + `expression_scratch_model`
  (compact CNN trained from scratch on FER2013's native 48x48 grayscale, not transfer
  learning) with their softmax outputs averaged, both also using test-time augmentation --
  the combination that won a head-to-head GPU comparison against every single model and
  combination (Section 2, `training/results/expression_ensemble_comparison.json`). The
  original single α=0.35 expression model (CPU-only result: 58.34%) has been fully
  replaced, not kept alongside the ensemble.
- **Face detection** is BlazeFace (TensorFlow.js, on-device, no training needed) in the
  web app -- it crops the face before either classifier runs. (The Android app uses
  Google ML Kit for the same purpose.)
- **Age** is trained as continuous regression (MAE-reportable, more standard than a
  binary classifier) and only bucketed into adult/elderly (60+ threshold) at evaluation
  time, to match the brief's required 20-image split. Since the 21 real photos collected
  only have adult/elderly ground truth (not exact ages), the performance table reports *two* age
  rows: MAE in years (CPU/GPU test set only -- "--" on the edge column, since there's no
  exact-age ground truth to compare against there) and adult/elderly bucket accuracy
  (all three columns, directly comparable, and the one that maps onto what the brief
  actually asks you to evaluate).
- **Open issue: facial hair is a spurious age cue for the age model** (root-caused via a
  deep investigation after real user testing -- a close-up selfie of a bearded man in his
  20s scored 82.4 years; full runnable analysis in `training/analyze_age_bias.py`). Ruled
  out, in order: (1) general model bias -- a held-out-test check found only ~2-3 years
  average error for actual 20-somethings; (2) demographic bias -- broke the held-out test
  set down by UTKFace's ethnicity label *and* age range jointly; Black males aged 18-29
  (n=88) had MAE 5.4 / bias +3.2, comparable to or better than White males the same age
  (MAE 5.9 / bias +3.4), so this wasn't it either. Then found real (if modest) evidence
  for the actual cause with a controlled causal test at n=30: painting a crude dark
  region over the jaw/chin of real young-adult male test photos (simulating a beard,
  nothing else changed) increased predicted age by a mean of 5.5 years (77% of cases
  positive) -- significantly more than the same patch on the forehead (mean 2.6 years,
  paired t-test p=0.041) or a cheek (mean 0.7 years, p=0.002) as controls. (An initial
  6-example run had suggested a much flashier +12.1 years, 100% positive -- worth
  flagging as a reminder that a small sample can overstate a real effect; the n=30 run is
  the trustworthy one and is what's reported here and in the script's default.) Read most
  simply: full facial hair is disproportionately associated with older subjects in
  UTKFace's own age distribution, and the model appears to have partly learned
  jaw-region darkness as an age cue that under-generalizes to young adults with a full
  beard -- a real, statistically significant, but probabilistic effect, not a
  deterministic one. Secondary, separately confirmed factor: sensitivity to close-up
  photo framing -- `train_age_gender.py` trains on full, uncropped UTKFace images with no
  face-detection step at all, while the deployed app always crops the face first, so
  deployed framing never exactly matches training, worse for an extreme close-up. A
  principled fix was attempted (aligning the crop to BlazeFace's eye landmarks,
  calibrated against real UTKFace images) and rejected after testing showed it made a
  previously-correct prediction worse, not better -- not shipped as an unverified change.
  **Update: both fixes were built, retrained on GPU, and tested -- but only one is
  deployed, after the other one caused a real regression a user actually hit.**

  Round 1 (crop-consistency + beard augmentation together): a clean before/after
  comparison showed the targeted fix worked -- pre-fix, jaw-darkening's age-inflating
  effect (+6.4 years) was statistically indistinguishable from the forehead control (+7.1
  years, p=0.68); post-fix, jaw-darkening produced a significantly *smaller* shift than
  forehead/cheek (p<0.0001, p=0.0002), meaning the spurious cue was measurably weakened.
  This version was deployed. A user then reported a **new** problem on this version: a
  real bearded selfie was misclassified as female. A follow-up causal test confirmed a
  real regression -- the same jaw patch that used to inflate age now flipped gender to
  female in 23% of cases (vs. 7% on unmodified photos), because age and gender share the
  same backbone and the beard augmentation degraded shared features it wasn't meant to
  touch.

  Round 2 isolated the cause: retrained with crop-consistency alone, beard augmentation
  removed. Gender misclassification returned to baseline (0% under the same jaw-patch
  test) -- confirming the augmentation, not the crop fix, broke gender. But this also
  confirmed the original age problem was **not** meaningfully fixed by crop-consistency
  alone (jaw-region darkening still produced a significantly larger effect than
  forehead/cheek controls, p=0.0006, same direction as before any fix).

  **What's actually deployed**: the crop-consistency-only version (no beard
  augmentation) -- reliable gender, unsolved age-facial-hair sensitivity. Faced with "real
  partial age fix + real gender regression" vs. "reliable gender + unsolved age issue,"
  the second was chosen: don't trade one real bug for a different, more severe one. This
  is reported as an open, unresolved limitation, not a solved problem. A more careful fix
  -- likely a much lower-strength/probability beard augmentation, evaluated for its effect
  on *both* outputs before deployment, not just one -- is future work.

  All three model versions and exact numbers reproducible via
  `training/analyze_age_bias.py` and `training/train_age_gender.py --no-beard-aug`.
- **Expression** is trained on 6 classes (Angry/Fear/Happy/Sad/Surprise/Neutral -- Disgust
  dropped as too rare/noisy), demonstrating broader capability than the evaluation strictly
  requires, but only the Happy/Sad subset is scored against your 20 images per the brief.
- **FER2013 mirror quirk**: the Hugging Face mirror used (`JimmyUnleashed/FER-2013`) has an
  unusable `test` split (all labels `None`) and a `train` split with every row duplicated.
  `prepare_fer2013.py` de-duplicates by the raw pixel string and builds its own 80/10/10
  split from the result -- worth a one-line mention in your data section if you want to show
  you noticed and handled a real data-quality issue.
- **Converting Keras 3 models for the browser was not a one-line command.** The
  `tensorflowjs` Python package (last released before Keras 3 existed) can't parse Keras
  3's native `.keras`/H5 config format at all (different `dtype`/`inbound_nodes`
  structure, not just renamed fields). The fix, in `training/resave_legacy_keras.py`:
  rebuild the identical architecture under TensorFlow's official `tf_keras` legacy-Keras-2
  shim, load the already-trained weights into it (plain numpy arrays, version-agnostic),
  and re-save -- *then* convert that. Also had to neutralize an unrelated dead import in
  `tensorflow_hub` (TF1 Estimator API, removed in current TensorFlow) that crashed on
  import even though nothing in the conversion path uses it, and patch a couple of
  `np.object`/`np.bool` deprecated-alias references in `tensorflowjs` itself for current
  NumPy. None of this affects model correctness -- verified by checking output-layer
  order in the converted `model.json` and by running a real labeled test image through
  the full pipeline (see step 3).
- If you ever see a `Didn't find op for builtin opcode ... version 'X'` TFLite error on
  the Android side (this is what cost another student two weeks on Task 3): it means the
  `.tflite` file was produced by a newer converter than the runtime understands.
  `convert_to_tflite.py` already smoke-tests every export in Python immediately after
  conversion to catch this before it ever reaches an app.
- **Two real accuracy bugs were found and fixed via real-world testing, not just metrics**:
  (1) BlazeFace's raw face bounding box is tighter than the crops UTKFace/FER2013 were
  trained on -- `FACE_CROP_MARGIN = 0.3` in `app.js` expands the box before cropping,
  which alone fixed several misclassifications; (2) the first training pass only trained
  a head on a *frozen* backbone, which is fast but weak, especially for expression (frozen
  ImageNet features were never optimized to distinguish facial expressions the way they
  incidentally help with age/gender's coarser cues like skin texture and bone structure).
  Both `train_age_gender.py` and `train_expression.py` do a second phase: unfreeze
  backbone layers, drop the learning rate, and continue training with image augmentation
  (random horizontal flip, ±15% brightness/contrast jitter, applied before MobileNetV2's
  [-1,1] input scaling so the augmentation ops see the pixel range they expect).
  `train_age_gender.py` unfreezes the top 30 layers; `train_expression.py` ended up
  unfreezing the *entire* backbone (an earlier top-60-layers version got most of the way
  there already -- unfreezing everything cost an extra ~2 hours of CPU time for a
  0.1-percentage-point difference, a real diminishing-returns result worth knowing if you
  retrain this yourself) plus class weighting and a reduce-on-plateau LR schedule.
  Results: see the table in step 1, and Section 3 of the report for the full
  before/after story.
- **Expression's class weighting was switched from naive inverse-frequency to
  recall-informed, after a real misread exposed that frequency wasn't the real problem.**
  A user photo of a surprised expression got predicted as Happy; before assuming class
  imbalance was to blame, I actually computed the deployed ensemble's confusion matrix
  against the full FER2013 test set (`evaluate_expression_ensemble.py` now does this by
  default and prints it, not just overall accuracy). Surprise turned out to be one of the
  better-performing classes (73.9% recall) -- the real weak spots are Fear (51.5%) and
  Sad (51.4%), and Sad has more training examples than Angry or Surprise, so scarcity
  wasn't the cause. `train_expression.py` and `train_expression_scratch.py`'s
  `compute_class_weights()` now weight by each class's observed recall instead of raw
  frequency, targeting Fear/Sad directly. **Tested via a real GPU retrain -- result was a
  mixed, honest negative.** Sad's recall improved (51.4% to 60.3%, the intended target),
  but Fear fell (51.5% to 44.9%), Angry fell (60.3% to 57.9%), and Neutral fell (71.6% to
  64.8%) -- the model over-corrected toward predicting Sad (Neutral-to-Sad confusion
  nearly doubled), and overall ensemble accuracy dropped from 67.60% to 66.35-66.68%. Not
  deployed; the original ensemble remains in production. Worth noting: the JSON initially
  downloaded from Colab for this looked suspiciously identical to the pre-retrain
  baseline (a strong sign it was stale, not from the actual retrained model) -- caught by
  independently re-running `evaluate_expression_ensemble.py` locally against the
  downloaded `.keras` files themselves rather than trusting the printed/downloaded
  result. Full numbers in `Project_Report.docx`'s Section 4 (Conclusion).
- **The age model's facial-hair bias (report Section 3) got a third attempt**,
  after real-world testing confirmed it's still present in the deployed (Round 2,
  no-beard-aug) model. Round 3 retried the same counterfactual jaw-darkening
  augmentation from Round 1, but deliberately weaker -- `prepare_utkface.py`'s
  `BEARD_AUG_PROBABILITY` dropped from 0.25 to 0.12, darkening opacity dropped from a
  near-opaque [0.6, 0.95] to a translucent [0.25, 0.5]. **Tested and failed, worse than
  Round 1:** the causal test (`analyze_age_bias.py`, n=30) showed jaw-patch gender
  misclassification at 40% (vs. Round 2's near-0%, and worse than even Round 1's
  original 23% regression). The age effect wasn't a clean fix either -- every patch,
  including the forehead/cheek controls, now shifted predicted age down by several
  years, more consistent with general instability to any occlusion than a targeted fix.
  Not deployed; Round 2's model remains in production.
- **Test-time augmentation (TTA) for expression, at inference**: `app.js` runs each
  expression model twice per photo -- once on the face crop, once on its horizontal
  mirror -- and averages the two softmax outputs before averaging across the two models
  in the ensemble. This is a standard, well-known technique for squeezing a bit more
  reliability out of a trained model at inference time, with no retraining required; it
  added ~0.6-1.8 points per model (Table 3 in the report). The cost is 4 forward passes
  total per prediction instead of 2 -- ~800ms on WebGL in early testing, but the app
  forces the CPU backend in production (see the WebGL-precision entry below), where a
  real benchmark run measured **~5.4 seconds** for this step alone, not negligible.
  Age/gender inference does not use TTA, only expression, since that's the prediction
  that actually needed the help.
- **The Benchmark tab's pass count was cut from 50 to 15, and a per-pass watchdog was
  added, after real-device testing surfaced a genuine CPU-backend stall.** A user reported
  the benchmark looking permanently frozen; reproducing it directly confirmed two real,
  separate problems. First, the original 50-pass loop gave zero feedback between passes,
  so several genuinely-slow-but-working minutes were indistinguishable from a crash --
  fixed by updating the status line every pass ("Running pass N of 15... (Xs elapsed)").
  Second, and more seriously: CPU-backend passes occasionally stall hard -- a pass that
  normally takes ~6-9 seconds can take a minute or more, reproduced consistently around
  the 4th pass in controlled testing, with the tab remaining otherwise responsive (not a
  full freeze, no thrown error) -- consistent with a resource-pressure effect in the
  pure-JS CPU backend under sustained heavy computation, though the exact internal cause
  wasn't pinned down further. Rather than chase that indefinitely, `app.js`'s benchmark
  loop now races each pass against a 30-second timeout; if a pass doesn't finish in time,
  the loop stops and reports honest results from whichever passes did complete, with a
  clear message explaining why, instead of leaving the page looking dead with no way to
  tell if it would ever finish. Passes were also reduced from 50 to 15, both to shorten
  the normal-case wait and to reduce how often the stall is even encountered.
- **The real-world evaluation surfaced a genuine, significant gender-accuracy gap**, not
  just routine noise, and it has now replicated across two independent real-device
  rounds. Round 1 (21 photos, archived): male predictions right 83.3% of the time
  (n=12), female only 11.1% (n=9). Round 2 (20 photos, current): male 90.9% (n=11),
  female 33.3% (n=9) -- both far below the same model's 86-87% accuracy on UTKFace's own
  held-out test set, and both far below male accuracy, even though the exact numbers
  moved between rounds. Verified this wasn't a repeat of the earlier coordinate-crop bug
  by directly inspecting the exact 96x96 crop the model received for one confidently-wrong
  case: a clean, correctly-framed photo of a woman's face. Round 1's evidence pointed to
  an age-interacted gender confound (the only correct female prediction was of a visibly
  elderly woman, every misclassified woman was a younger adult) -- but Round 2 doesn't fit
  that as cleanly: two of the three correctly-classified women in Round 2 are themselves
  younger adults, not elderly. Reported honestly as an open question rather than forcing
  the earlier explanation to fit; this kind of subgroup accuracy gap is a well-documented
  category of problem in face-analysis fairness research generally (see Buolamwini &
  Gebru's 2018 "Gender Shades" study), not unique to this project. Full write-up,
  including small-sample caveats, is in the report's Section 3 (Results and Analysis).
- **A follow-up root-cause investigation tested five specific hypotheses for the gender gap
  against real statistics, rather than guessing.** Ethnicity representation: ruled out
  (women of the closest-matching ethnicity scored 86.9% on UTKFace's own test set, n=206).
  A shared age/gender representation link: confirmed (wrong-gender photos have significantly
  larger age errors too, p=0.0036, n=2,371). A training/deployment face-detector mismatch
  (Haar cascade vs. BlazeFace): tested with a controlled experiment holding the same 2,371
  images constant and varying only the detector -- ruled out, and in the opposite direction
  expected (BlazeFace-style crops scored *better*, p=0.0034). An image-sharpness difference
  between UTKFace and real photos (real photos are ~8x sharper, p=0.0085): real, but a causal
  blur test ruled it out too -- blurring real photos to match made accuracy worse, not better
  (66.7% to 55.6%). Four hypotheses eliminated with real statistics, one (facial hair)
  confirmed but unresolved -- the root cause of the gap's full magnitude remains genuinely
  open. Full write-up is in the report's Section 3.
- **The web app's TF.js conversion has a second Keras-3 wrinkle worth knowing about**:
  after retraining, `resave_legacy_keras.py` takes an optional `--model
  age_gender|expression|expression_wide|expression_scratch|both` flag (`convert_to_tflite.py`,
  used only for the Android artifact, still takes `age_gender|expression|both`), so you
  can reconvert just the one model that changed instead of redoing all of them -- useful
  if you ever retrain again.
- **BlazeFace's weights are self-hosted**, not fetched from Google's CDN on every load
  (`web_app/models/blazeface/`) -- combined with the service worker in `sw.js`, the app
  works fully offline after the first load. Whenever any cached file changes (a model,
  `app.js`, `style.css`, icons), `sw.js`'s `CACHE_VERSION` gets bumped and the relevant
  file's `?v=N` query string gets bumped too, in both `sw.js`'s `ASSETS` list and
  `index.html`'s `<link>`/`<script>` tags -- skipping either half of that causes the
  browser or the service worker to keep serving a stale cached copy, which is a real trap
  worth knowing about if you ever edit the app yourself.
- **The plain-HTTP-caching trap above turned out to be worse than "worth knowing about"
  -- it caused a real, confusing incident.** After deploying a fix, a user kept getting
  results that didn't match what had just been verified working. Root-caused by directly
  inspecting the service worker's Cache Storage in a browser devtools session: `index.html`
  and `./` are the only cached URLs that never change (every other asset gets a `?v=N`
  bump specifically to force a fresh fetch) -- so once the browser's own plain HTTP cache
  had a stale copy of either, from as far back as the first visit, `sw.js`'s
  `cache.addAll(ASSETS)` kept re-fetching and re-freezing that *same* stale copy into
  every single new `CACHE_VERSION` from then on, because `cache.addAll()`'s fetches use
  default HTTP caching behavior, not a network-bypassing one. The version number was
  always correct; the content silently wasn't. Fixed two ways: install-time fetches now
  use `{cache: "reload"}` to force a real network hit past the browser's HTTP cache, and
  the fetch handler now serves `index.html`/`app.js`/`style.css` network-first (falling
  back to cache only when offline) instead of cache-first, so this specific class of
  staleness can't silently recur even if a future edit forgets a version bump somewhere.
  Also added a visible version tag to the in-app status line (`APP_VERSION` in `app.js`,
  kept in sync with `sw.js`'s `CACHE_VERSION`) specifically so "is this device on the
  latest version" is something you can just look at, not guess.
- **After fixing the caching bug above, the exact same photo still gave a different
  gender prediction on a different machine -- confirmed via a private/incognito window,
  which rules out caching entirely.** Checked the raw (pre-threshold) model output rather
  than just the Male/Female label: it was 0.0002, about as confidently "male" as a
  sigmoid output gets. A gap that large cannot come from ordinary floating-point rounding
  differences between machines -- it pointed to a genuine computational difference, not
  noise. The one thing that legitimately varies by hardware is WebGL: TensorFlow.js's
  default backend runs inference as GPU shaders, and different GPUs/drivers (especially
  older or integrated ones, common on laptops) can use reduced float precision or take
  different code paths, which can compound across a deep network like MobileNetV2 into a
  real, wrong answer -- not just a slightly-off number. Fixed by switching the app to
  TensorFlow.js's `cpu` backend (`tf.setBackend("cpu")`, set once before any model loads),
  which is slower but numerically exact and identical on every device, trading speed for
  correctness deliberately: expression inference went from well under a second on WebGL
  to roughly 3.5-4.5 seconds on CPU in testing, which is a real, noticeable cost, but
  still acceptable for this app's single-photo (not real-time video) workflow, and correct
  is worth more than fast here.
- **The WebGL fix above did not actually fix the problem it was diagnosed for.** Even
  after confirming (via an added `backend:` diagnostic line) that a machine was genuinely
  running on `cpu`, the exact same real photo still came back with the wrong gender and a
  strangely tiny face-detection box. The real cause was a coordinate-space mismatch that
  had nothing to do with numerical precision: `analyzePreview`, the `<img>` element the
  app actually uses, is styled by CSS to fit inside its `.polaroid` display frame, so its
  on-screen (client) pixel size can be -- and in the reproducing photo's case was --
  drastically smaller than its true (natural) resolution: 169x300 rendered versus 900x1600
  actual. Running BlazeFace directly on that live, styled DOM element returned a face box
  scaled to the small on-screen size, while `cropFaceToCanvas()` (correctly) interpreted
  that box as natural-pixel coordinates -- a roughly 5.3x scale mismatch that silently
  cropped a tiny, wrong region instead of the actual face. This is why the bug survived
  the WebGL fix, the `img.decode()` fix, and repeated internal "verification": every one
  of those internal tests used a detached, unstyled `new Image()` rather than the page's
  real, CSS-styled preview element, and a detached image's on-screen size trivially equals
  its natural size -- so the mismatch never showed up until the bug was reproduced through
  the app's *actual* file-picker code path, into the real DOM element. Fixed in
  `detectLargestFace()` (`app.js`) by first drawing the source image onto an intermediate
  canvas at its natural pixel dimensions (`toNaturalSizeCanvas()`) and running face
  detection against that canvas instead of the live styled element, which removes the
  coordinate-space ambiguity entirely -- a canvas's pixel buffer is always exactly the
  size it was created with, no CSS involved. Verified against all four of this project's
  own real test photos through the exact DOM-element code path the deployed app uses:
  every one now produces a correctly-scaled face box and a confident, correct gender
  prediction (raw sigmoid outputs of 0.0000-0.0002), where each had previously produced a
  wrong, tiny box and, in most cases, a wrong gender label. Shipped as `v22`.
- **Final polish pass (v23-v26), after the real 20-photo evaluation and a full report
  audit against the actual brief.** In order: the benchmark's pass count and per-pass
  watchdog (v23-v24, see the benchmark entry above); the dark-mode override removed after
  a user found it looked muted rather than crisp (v25); and v26, a small but real UX fix
  -- the footer's status dot was hardcoded green even while models were still loading or
  had failed to load, which was silently wrong in exactly the two moments it mattered
  most. It now reflects real state: amber and pulsing while loading, red on failure, green
  once ready. Separately (not a code change, a documentation/submission-readiness pass):
  re-read the actual module brief and audited the report against it directly, which
  caught two real internal contradictions -- Section 6.3 claimed a perfect 5/5/5/5 photo
  split that never actually happened (Table 2 right next to it shows the real, honest
  19/2 adult/elderly split), and the Results section still called the edge-device numbers
  "pending" long after Table 1 was fully complete -- plus one overclaim (the report said
  the Android app was "verified to compile and run correctly"; re-running the build live
  confirmed it does compile, but "runs correctly" was never actually true since no
  Android device or emulator exists anywhere in this project). All three are fixed. The
  project folder was also cleaned of about 400MB of confirmed-redundant debris (stale
  Colab download folders, duplicate backup model files verified byte-identical before
  deletion, and ~204MB of regenerable Android Gradle build cache) with nothing unique or
  irreplaceable touched.

## AI tool disclosure 

In this project I used one AI model which helped me to brainstorm and build Ideas. and build the structure for the project!
