# Next steps: finishing the project

Everything that required your Google account, your phone, and real photographed faces is
now done: the GPU notebook has been run, the app has been benchmarked and evaluated on
your real iPhone, and the results are in the report. Of the four real-world actions that
originally required you personally, three are complete (Steps 1-3 below); the only one
left is Step 4, personalizing the report. This document walks through what was done for
each of the first three, and exactly what's left to do for the fourth.

Steps below are numbered in the order they were done — later steps depended on earlier
ones being done first.

| # | Step | Fills in |
|---|---|---|
| 1 | ~~Run the Colab GPU notebook~~ — **done** | Table 1's "Trained on GPU" column |
| 1b | ~~Fix the age-prediction bug~~ — **done (partial; reliable gender prioritized)** | Gender regression fixed; age/beard sensitivity still open |
| 1c | ~~Fix the real wrong-prediction bug on real photos~~ — **done, verified** | The actual root cause of every "it's still wrong" report — see below |
| 1d | ~~Run Colab notebook Parts 4-5~~ — **done, both tested and not deployed** | Two honest negative results — see below |
| 2 | ~~Benchmark on your iPhone~~ — **done** | Table 1's "On edge device" latency rows |
| 3 | ~~Take + label photos, run Evaluate~~ — **done, real results in** | Table 1's edge accuracy cells, Table 2 |
| 4 | Personalize the report | Title page, AI disclosure, discussion, references |
| — | ~~Full report audit against the real brief + folder cleanup~~ — **done** | Fixed 2 internal contradictions + 1 overclaim; ~400MB debris removed |
| — | ~~Cross-document consistency + citation compliance audit~~ — **done** | Fixed stale README/NEXT_STEPS sections, missing citation page numbers, one reference-formatting bug |
| — | ~~Restructured report to the official 7-10 page format~~ — **done** | Was ~19-20 pages with no abstract/figures; now ~7-9 pages with a graphical abstract and architecture diagram, matching IU's required structure |
| — | ~~Re-collected and re-evaluated the demographic set (Round 2)~~ — **done** | Dropped 2 face-detection failures, added 1 confirmed-real elderly photo, landed on a clean 20-photo set; re-ran on the real iPhone, Table 1/2 and the graphical abstract all updated with the real Round 2 numbers |
| — | ~~Deep root-cause investigation into the gender-accuracy gap~~ — **done** | Tested 5 hypotheses with real statistics; 4 ruled out (ethnicity, general quality, crop-detector, sharpness), 1 confirmed (facial hair, unresolved); written into report Section 3 |

---

## Step 1 — Run the Colab GPU notebook — done ✓

You ran `colab_gpu_training.ipynb` on Colab and brought both the required Part 1 (CPU vs.
GPU comparison) and optional Part 2 (expression-accuracy push) results back. Both are now
processed:

- **Table 1's GPU column is filled in** — GPU-trained accuracy is statistically the same
  as CPU (as expected, since it's identical code/hyperparameters), but training was 1.8x
  and 12.5x faster for the age/gender and expression models respectively.
- **Part 2's ensemble won, and is now deployed**: the app's expression prediction is a
  2-model ensemble (α=1.0 MobileNetV2 + a from-scratch CNN, both with test-time
  augmentation) that reached **67.60%** test accuracy, up from 58.34% on the CPU-only
  single model. I verified it end-to-end in-browser before confirming this — face
  detection, age/gender, and the new expression ensemble all run correctly.
- **The report has been updated** — `Project_Report.docx`'s Table 1, the Discussion
  section (with a new comparison table, Table 3), the Limitations section, and the
  Conclusion all now reflect the real numbers.
- Still short of the original 70% target (67.60% is honestly reported, not rounded up or
  overstated) — see the report's Limitations section for why, and what would plausibly
  close the remaining gap if you wanted to pursue it further.

**One thing to do yourself:** `Project_Report.docx` was open in Word while I updated it.
I was able to write the new version to disk, but Word itself still has the *old* version
in memory. **Close Word without saving**, then reopen `Project_Report.docx`, to see the
updated report. If you save from the still-open Word window first, it will overwrite my
changes with the stale version.

---

## Step 1b — Fixing the real age-prediction bug — done, but only partially (honestly reported) ✓

You reported a close-up selfie scoring 82.4 years. I investigated deeply, built two
fixes, and things did not go cleanly — here's exactly what happened, across two rounds.
Full writeup in the report's Limitations section; reproducible via
`training/analyze_age_bias.py` and `training/train_age_gender.py --no-beard-aug`.

**The two fixes I built:**
1. **Crop-consistency**: the age model used to train on full, uncropped photos, but the
   deployed app always detects and crops the face first — a real mismatch. Training data
   is now face-detected and cropped the same way (81.4% detection rate on the full
   dataset; the rest fall back to a full resize, nothing dropped).
2. **Counterfactual beard augmentation**: training randomly darkens the jaw of young
   faces, directly teaching the model that jaw-darkness doesn't reliably mean "older."

**Round 1 — deployed both fixes together.** A clean before/after test showed the targeted
mechanism genuinely improved: pre-fix, a dark patch on the jaw and the same patch on the
forehead moved predicted age by statistically indistinguishable amounts (p=0.68); post-fix,
jaw-darkening moved age significantly *less* than forehead/cheek darkening (p<0.0001,
p=0.0002). I deployed it. **You then reported a new, real problem**: a bearded selfie
came back as Female. I tested it — confirmed a genuine regression: the same jaw patch
that used to inflate age now flipped gender to Female in 23% of cases (vs. 7% normally).
Age and gender share the same backbone, and the beard augmentation had damaged shared
features it wasn't meant to touch.

**Round 2 — isolated the cause.** Retrained with crop-consistency alone, no beard
augmentation. Gender misclassification dropped back to baseline (0% under the same
jaw-patch test — actually more robust than normal). This confirmed the augmentation, not
the crop fix, caused the gender regression. It also confirmed the original age problem
was **not** meaningfully fixed by crop-consistency alone — jaw-darkening still produces a
significantly larger age effect than the forehead/cheek controls (p=0.0006), same
direction as before any fix.

**What's deployed now**: crop-consistency only, no beard augmentation — reliable gender,
the original age/facial-hair sensitivity still open and unsolved. I chose this over
keeping the age fix, because a new gender-misclassification bug is worse than an
old, already-documented age limitation. Verified end-to-end in-browser after deploying
(age, gender, and expression all work correctly together on real test photos) — this
time including a specific bearded-face gender check, which I should have done the first
round.

**If you want to push further**: the most promising next step is a much
lower-strength/probability beard augmentation, evaluated against *both* age and gender
before deploying — not just age, which is the mistake that caused Round 1's regression.
That's real, well-scoped future work, not a dead end.

---

## Step 1c — Fixing the real wrong-prediction bug on real photos — done, verified ✓

After Round 2 of Step 1b shipped, you kept testing with your own real photos and kept
getting genuinely wrong results — wrong age, wrong gender, sometimes both — even on
photos that should have been easy. Two more fix attempts (switching to the CPU backend
after a suspected GPU-precision issue, then switching to `img.decode()` after a suspected
image-loading timing issue) each seemed reasonable and got deployed, but you kept proving
they didn't actually fix it. That was the right call on your part — both were real,
legitimate engineering improvements, but neither was the actual cause.

**The real cause, finally found by testing through the app's *actual* file-picker code
path instead of a shortcut:** the app's on-screen preview image is styled by CSS to fit
its display frame, so it can render much smaller on screen than its real resolution — in
the reproducing case, 169x300 pixels displayed versus 900x1600 pixels actual. The face
detector was reading coordinates in that small, on-screen size, while the app's own
cropping code was (correctly) treating those same numbers as if they were in the real,
full-resolution size — a roughly 5.3x scale error. The result: instead of cropping out
your actual face, the app was cropping a tiny, essentially random sliver of the photo and
feeding *that* to the age/gender/expression models. No wonder the answers looked
unrelated to the actual photo.

This had been there the entire time, on every version you tested, which is also the most
likely explanation for the very first "82.4 years" report that kicked off this whole
investigation. Every one of my own internal checks kept missing it because I was testing
with a plain, undisplayed image object rather than the app's real, styled on-screen
element — the exact difference that caused the bug to only ever show up on your machine,
never mine, until I deliberately reproduced it the same way you actually use the app.

**Fixed by** making face detection run against a plain, natural-resolution copy of the
photo instead of the on-screen styled version, so there's no longer any ambiguity about
what size the coordinates are in. **Verified** against all four of the project's own real
test photos (`pictures/uno.jpeg` through `four.jpeg`), through the exact same code path
the real app uses (not a shortcut) — every one now gets a correctly-placed face crop and
a confident, correct gender prediction, where every one previously got a wrong, tiny crop.
Shipped as `v22`. Full technical write-up in the report's Limitations section.

**Please re-test on your own phone/laptop once more** (fresh Incognito/private window,
confirm the status line reads `v22` before testing) — this has been rigorously verified
on my end, but you catching a real-world case I haven't thought of is exactly what found
every genuine bug so far in this project, so one more round of your own testing is worth
it before treating this as fully closed.

---

## Step 1d — Round 3 accuracy attempts for age and expression — done, both honest negatives ✓

You ran Parts 4 and 5 on Colab and brought back both results. I didn't just trust the
printed numbers — I independently re-ran both diagnostics locally against the actual
`.keras` files you brought back (good thing too: the expression ensemble JSON you
downloaded turned out to be stale, identical to the pre-retrain baseline — re-running it
myself against the real retrained model files caught that and got the true numbers).

**Age, Round 3 — failed, worse than Round 1.** The same beard augmentation at roughly
half strength was meant to fix the facial-hair bias without repeating Round 1's gender
regression. It did neither cleanly: the causal test's jaw-patch gender misclassification
came back at **40%** (Round 2's baseline is near-0%, and Round 1's original regression was
23% — this is worse than the failure it was trying to avoid). The age effect wasn't a
clean win either: jaw-darkening no longer differed significantly from a forehead control,
but only because *every* patch, including the forehead/cheek controls, now shifted
predicted age down by several years — general instability to any dark patch, not a
targeted fix. **Not deployed.** Round 2's model (crop-consistency only, no beard
augmentation) remains what's actually running in the app.

**Expression, recall-weighted retrain — mixed, not a net win.** Sad's recall did improve,
51.4% to 60.3%, exactly the class this was meant to help. But it came at a real cost: Fear
fell 51.5% to 44.9%, Angry fell 60.3% to 57.9%, Neutral fell 71.6% to 64.8% — the model
over-corrected toward predicting Sad (Neutral-to-Sad confusion nearly doubled), and
overall ensemble accuracy dropped from 67.60% to 66.35-66.68%. **Not deployed.** The
original ensemble remains what's actually running in the app.

**Nothing changed in your app** — both currently-deployed models (age/gender and
expression) are exactly what they were before this round. Both attempts are documented
with their real numbers in `Project_Report.docx`'s Limitations section, README.md, and
here — an honestly-reported "we tried this specific fix and it didn't work, here's why"
is legitimate, valuable content for the report, not something to hide. This closes out
the open accuracy-improvement threads; what's left is Steps 2-4 below, which don't depend
on any of this.

---

## Step 2 — Benchmark on your iPhone (edge-device latency) — done ✓

You ran **Run 15 passes** on your iPhone and all 15 completed cleanly — no timeout, no
stall, so the watchdog fix (see Step 1's app-fix history) wasn't even needed this time.
Real, on-device numbers:

```
Passes: 15 (face found in 15)
Avg face detection:       576.8 ms
Avg age/gender inference: 219.7 ms
Avg expression inference: 6,394.6 ms
Avg total pipeline:       7,191.2 ms
```

These are written into `evaluation/edge_results.json` and already in
`Project_Report.docx`'s Table 1 ("On edge device" column, the three latency rows).
Expression is the dominant cost by far, consistent with what CPU-backend testing showed
earlier -- it runs 2 models with test-time augmentation each (4 forward passes) rather
than 1, which is what pushed its accuracy from 58% to 67.6% (Step 1). Table 1's edge
*accuracy* cells (age bucket, gender, expression) are still `[pending device test]` --
those come from Step 3's 20-photo evaluation, not this benchmark.

---

## Step 3 — The 20-photo demographic evaluation — done, real results in ✓

This went through two real rounds. **Round 1** used 21 real photos (not a perfect 5/5/5/5
split — real people willing to be photographed, especially elderly subjects, were
genuinely hard to find), labeled and run through the Evaluate tab on your iPhone, and
independently reproduced in a browser against the same photos with byte-identical
predictions. That run is archived as `evaluation/results_21photo_archive.csv` and is no
longer the one used in the report.

**Round 2 (current, final)** trimmed to exactly 20 photos: the two photos that failed
face detection outright in Round 1 were dropped, and one new confirmed-real, confirmed-age
elderly photo was added, landing on 18 adult / 2 elderly. This was run for real on your
iPhone, and every ground-truth label in the returned `results.csv` was checked against the
intended labels CSV and matched exactly, confirming the run used the correct photos and
labels.

**The results surfaced a real, significant finding that replicated across both rounds**,
not just a routine accuracy number:

| Metric | Overall | Detail |
|---|---|---|
| Age bucket | 85.0% | Adult 94.4% (n=18), Elderly 0.0% (n=2) |
| Gender | 65.0% | Male 90.9% (n=11), **Female 33.3% (n=9)** |
| Expression | 60.0% | Happy 90.9% (n=11), Sad 22.2% (n=9) |

The gender gap is the headline: the model is right on men roughly 91% of the time but on
women only 33% of the time (Round 1 found an even starker 83% vs. 11% with different
photos) — both far below the same model's 86.8-87.3% accuracy on its own training
benchmark. I checked this wasn't a repeat of the earlier coordinate-crop bug (inspected
the exact 96x96 image the model actually received for one case — the crop was clean and
correct), so this is genuine model behavior, not a pipeline bug. Round 1's evidence
pointed to an age-interacted gender confound (the one correct female prediction was of a
visibly elderly woman, every misclassified woman was a younger adult) — but Round 2
doesn't fit that as cleanly: two of the three correctly-classified women in Round 2 are
themselves younger adults, not elderly. I reported that honestly as an open question
rather than forcing the earlier explanation to fit new data that doesn't support it as
well. This kind of subgroup accuracy gap is a well-documented category of problem in
face-analysis fairness research generally, not something unique to this project.

**This is genuinely good report material, not something to hide** — a real evaluation
surfacing a real, specific finding that holds up under a second independent test is
exactly what this kind of assignment is testing for. I've written the full analysis
directly into the report's Results and Analysis section (Section 3), including the
small-sample caveats (n=9 female, n=2 elderly) and the honest note about the confound
explanation not fully replicating. `Table 1` and `Table 2` are both fully updated with
the real Round 2 numbers.

---

## Step 4 — Personalize the report

The report was restructured since the last version of this document: the official IU
course page (not just the task brief) turned out to require a 7-10 page report with a
specific structure -- textual abstract + graphical abstract, introduction/literature
review, method with a diagram, results analysis, conclusion, references, appendix for
code -- and the report was roughly 19-20 pages against that limit, with no abstract and
no images at all. It's now been rebuilt around that exact structure (Abstract, 1.
Introduction and Related Work, 2. Method, 3. Results and Analysis, 4. Conclusion, 5.
Declaration of AI Tool Use, References, Appendix A, Appendix B), cut to about 3,000 words
of body text (roughly 7-9 pages including the two new figures), with two new diagrams
added (a graphical abstract and a system architecture diagram, both captioned "Fig. N /
Own representation" per the citation guidelines) and a citation for the Goodfellow et al.
(2016) *Deep Learning* textbook added, since IU's own course page recommends it as further
reading and it wasn't cited anywhere before. All the real numbers, the gender-accuracy
finding, the beard-bias investigation, and the honest negative results are still in the
report -- condensed, not removed.

Open `Project_Report.docx` in Word. There are **3 shaded ✎ callout boxes left**, plus the
title page:

1. **Title page** — replace `[Your Full Name]`, `[Your Matriculation Number]`, and
   `[Date]` with your real details.
2. **AI-disclosure callout** (Section 5) — this is the most important one to check
   carefully. It currently states AI assistance was used for scaffolding code, drafting
   scripts, and report structure. **Read it and adjust it to accurately reflect what you
   actually did** — IU's academic integrity policy requires the disclosure to be true, not
   just present.
3. **Reference-list callout** (right after References) — add a citation for the BlazeFace
   paper (Bazarevsky et al., 2019) only if you expand the face-detection discussion beyond
   what's already drafted; otherwise no action needed. (Buolamwini & Gebru's 2018 "Gender
   Shades" study, cited in Section 3's discussion of the gender-accuracy gap, and the
   Goodfellow et al. 2016 textbook, cited in Section 1, have already been added to the
   reference list for you.)
4. **Gantt chart callout** (Appendix B, optional) — swap the relative week numbers for
   real calendar dates if you want a dated timeline instead of "Week 1, Week 2...".

The standalone "Discuss and personalize" callout that used to sit in the old Discussion
section is gone -- the factual comparison it asked for (GPU vs. CPU accuracy, the
67.60% ensemble result, the full gender-accuracy-gap analysis) is now written directly
into Section 3 (Results and Analysis) as part of the condensed report. If you want to add
a sentence of your own personal reflection alongside that analysis, you're welcome to, but
it's no longer a required callout.

Everything else in the report (method, results, limitations folded into the conclusion,
literature) is already complete and doesn't need editing unless you want to adjust the
phrasing to sound more like your own voice.

---

## Once all four are done

Tell me, and I'll do a final pass: re-check the report for the same kind of
copy/formatting bugs I found and fixed before (e.g. the concatenated-percentage cell), and
confirm every ✎ box and `[pending]` placeholder is actually resolved before you submit.
