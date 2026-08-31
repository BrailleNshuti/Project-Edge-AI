"use strict";

// Bump alongside sw.js's CACHE_VERSION. Shown in the model-status line so a stale
// cached copy on any device is immediately, visibly obvious instead of silently giving
// wrong predictions -- a real gap that let a stale-cache-vs-real-bug mixup happen.
const APP_VERSION = "v27";

const FACE_INPUT_SIZE = 96;
const SCRATCH_INPUT_SIZE = 48; // native FER2013 resolution, for the from-scratch expression CNN
const ELDERLY_AGE_THRESHOLD = 60;
// BlazeFace's raw box is tighter than the face crops UTKFace/FER2013 were
// trained on (which include some forehead/chin/ear context) -- expand the
// box by this fraction on each side before feeding it to the classifiers.
const FACE_CROP_MARGIN = 0.3;

const state = {
  blazeface: null,
  ageGenderModel: null,
  // Expression is an ensemble of two differently-biased models -- a wider
  // MobileNetV2 (alpha=1.0) and a small CNN trained from scratch directly on
  // FER2013 -- averaged together with TTA. This combination won a head-to-head
  // comparison against every other single model and combination (measured on
  // Google Colab GPU: 67.60% test accuracy vs 58.34% for the single alpha=0.35
  // model this app shipped with before), so it replaced the single model
  // entirely rather than running alongside it. See
  // training/results/expression_ensemble_comparison.json for the full comparison.
  expressionModelWide: null,
  expressionModelScratch: null,
  emotionLabels: null,
};

// ---------- model loading ----------

async function loadModels() {
  const statusEl = document.getElementById("modelStatus");
  const v = Date.now(); // cache-bust: guarantees fresh model files whenever they're updated

  // WebGL inference is not guaranteed to be numerically consistent across different
  // GPUs/drivers -- found via a real incident: an extremely confident prediction
  // (raw gender output 0.0002, about as unambiguous as a sigmoid gets) on one machine
  // came back as the opposite class on another, for the byte-identical photo and code.
  // A gap that large rules out normal floating-point rounding noise; it points to a
  // GPU-specific precision issue (common on integrated/older GPUs, which WebGL often
  // runs at reduced float precision for). The CPU backend is slower but numerically
  // exact and consistent on every device, which matters more than speed here for a
  // single-photo workflow.
  await tf.setBackend("cpu");
  await tf.ready();

  statusEl.textContent = "Loading face detector...";
  state.blazeface = await blazeface.load({ modelUrl: `models/blazeface/model.json?v=${v}` });

  statusEl.textContent = "Loading age/gender model...";
  state.ageGenderModel = await tf.loadLayersModel(`models/age_gender/model.json?v=${v}`);

  statusEl.textContent = "Loading expression models (1/2)...";
  state.expressionModelWide = await tf.loadLayersModel(`models/expression_wide/model.json?v=${v}`);

  statusEl.textContent = "Loading expression models (2/2)...";
  state.expressionModelScratch = await tf.loadLayersModel(`models/expression_scratch/model.json?v=${v}`);

  const res = await fetch(`models/emotion_labels.json?v=${v}`, { cache: "no-store" });
  state.emotionLabels = await res.json();

  statusEl.textContent = `Models loaded (${APP_VERSION}). Running fully on-device -- nothing is uploaded anywhere.`;
  const dot = document.getElementById("statusDot");
  dot.classList.remove("loading");
  dot.classList.add("ready");
}

// ---------- face detection + preprocessing ----------

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

/**
 * Draws imgElement onto a canvas sized to its NATURAL pixel dimensions.
 *
 * Root cause of the persistent wrong-box/wrong-gender bug: the <img> elements this
 * app actually uses (analyzePreview, benchmarkPreview) are CSS-styled inside a
 * .polaroid frame, so their on-screen/client size is often drastically smaller than
 * their natural resolution (e.g. a 900x1600 photo rendered at 169x300). Passing such
 * a live, styled <img> element straight into BlazeFace's estimateFaces() returned a
 * face box scaled to that small CSS-rendered size, not the natural pixel size --
 * while cropFaceToCanvas() below (correctly) treats box coordinates as natural-pixel.
 * The mismatch (about a 5.3x scale error in the case that exposed it) silently
 * produced a tiny, wrongly-placed crop, which is why age/gender/expression came back
 * wrong on real photos even though every model itself was fine. It went unnoticed
 * through many earlier fix attempts because every test used a detached `new Image()`
 * (never attached to the DOM, so never CSS-styled, so its client size always equals
 * its natural size) -- only testing through the real file-picker path, into the real
 * styled DOM element, reproduced it. A canvas has no such ambiguity: its pixel buffer
 * is exactly the size it's created with, so routing through one here guarantees
 * BlazeFace and cropFaceToCanvas() always agree on the coordinate space.
 */
function toNaturalSizeCanvas(imgElement) {
  const iw = imgElement.naturalWidth || imgElement.width;
  const ih = imgElement.naturalHeight || imgElement.height;
  const canvas = document.createElement("canvas");
  canvas.width = iw;
  canvas.height = ih;
  canvas.getContext("2d").drawImage(imgElement, 0, 0, iw, ih);
  return canvas;
}

/** Runs BlazeFace and returns the largest detected face's bounding box in image pixel space, or null. */
async function detectLargestFace(imgElement) {
  // Re-assert the CPU backend immediately before running BlazeFace -- found necessary
  // after a real incident: setting it once in loadModels() was silently undone by the
  // time inference actually ran (backend reported "webgl" again mid-session). Cheap
  // no-op when nothing has changed it.
  await tf.setBackend("cpu");
  const naturalCanvas = toNaturalSizeCanvas(imgElement);
  const predictions = await state.blazeface.estimateFaces(naturalCanvas, false);
  if (!predictions.length) return null;

  let best = null;
  let bestArea = -1;
  for (const p of predictions) {
    const w = p.bottomRight[0] - p.topLeft[0];
    const h = p.bottomRight[1] - p.topLeft[1];
    const area = w * h;
    if (area > bestArea) {
      bestArea = area;
      best = { left: p.topLeft[0], top: p.topLeft[1], right: p.bottomRight[0], bottom: p.bottomRight[1] };
    }
  }
  return best;
}

/** Crops [box] out of imgElement (expanded by FACE_CROP_MARGIN, clamped to bounds) and scales it onto a size x size canvas. */
function cropFaceToCanvas(imgElement, box, size = FACE_INPUT_SIZE) {
  const iw = imgElement.naturalWidth || imgElement.width;
  const ih = imgElement.naturalHeight || imgElement.height;

  const marginX = (box.right - box.left) * FACE_CROP_MARGIN;
  const marginY = (box.bottom - box.top) * FACE_CROP_MARGIN;

  const left = clamp(box.left - marginX, 0, iw - 1);
  const top = clamp(box.top - marginY, 0, ih - 1);
  const right = clamp(box.right + marginX, left + 1, iw);
  const bottom = clamp(box.bottom + marginY, top + 1, ih);

  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(imgElement, left, top, right - left, bottom - top, 0, 0, size, size);
  return canvas;
}

/** Matches tf.keras.applications.mobilenet_v2.preprocess_input: scales RGB to [-1, 1]. Used for age/gender and the alpha=1.0 expression model, both MobileNetV2-based. */
function canvasToModelInput(canvas) {
  return tf.tidy(() => {
    const img = tf.browser.fromPixels(canvas).toFloat();
    const scaled = img.div(127.5).sub(1);
    return scaled.expandDims(0);
  });
}

/** Matches prepare_fer2013.py's load_dataset_native: grayscale, scaled to [0, 1], no MobileNetV2 preprocessing. Used for the from-scratch expression CNN, which trains directly on native 48x48 grayscale FER2013 images. */
function canvasToScratchInput(canvas) {
  return tf.tidy(() => {
    const img = tf.browser.fromPixels(canvas).toFloat();
    const gray = tf.image.rgbToGrayscale(img);
    const scaled = gray.div(255.0);
    return scaled.expandDims(0);
  });
}

/** Mirrors a canvas horizontally onto a new canvas of the same size. */
function flipCanvasHorizontal(canvas) {
  const flipped = document.createElement("canvas");
  flipped.width = canvas.width;
  flipped.height = canvas.height;
  const ctx = flipped.getContext("2d");
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(canvas, 0, 0);
  return flipped;
}

// ---------- inference ----------

/**
 * Runs the full pipeline on an <img> element.
 * Returns { faceFound, age, genderLabel, expressionLabel, faceDetectMs, ageGenderMs, expressionMs }.
 */
async function analyzeImage(imgElement) {
  const t0 = performance.now();
  const box = await detectLargestFace(imgElement);
  const faceDetectMs = performance.now() - t0;

  if (!box) {
    return { faceFound: false, faceDetectMs };
  }

  const faceCanvas = cropFaceToCanvas(imgElement, box);
  const input = canvasToModelInput(faceCanvas);

  await tf.setBackend("cpu"); // defensive re-assert, see detectLargestFace's comment
  const t1 = performance.now();
  const [ageTensor, genderTensor] = state.ageGenderModel.predict(input);
  const age = (await ageTensor.data())[0];
  const genderRaw = (await genderTensor.data())[0];
  ageTensor.dispose();
  genderTensor.dispose();
  const ageGenderMs = performance.now() - t1;

  // Expression is a 2-model ensemble (alpha=1.0 MobileNetV2 + from-scratch
  // CNN), each also using test-time augmentation (averaging the softmax over
  // the face and its horizontal mirror) -- the combination that won the
  // Colab GPU comparison (see training/results/expression_ensemble_comparison.json).
  // 4 forward passes total (2 models x plain+flipped); still on the order of
  // a few hundred ms even on a phone, negligible for this app's single-photo
  // workflow.
  const t2 = performance.now();
  const scratchCanvas = cropFaceToCanvas(imgElement, box, SCRATCH_INPUT_SIZE);
  const flippedInput = canvasToModelInput(flipCanvasHorizontal(faceCanvas));
  const scratchInput = canvasToScratchInput(scratchCanvas);
  const scratchFlippedInput = canvasToScratchInput(flipCanvasHorizontal(scratchCanvas));

  const wideTensor = state.expressionModelWide.predict(input);
  const wideTensorFlipped = state.expressionModelWide.predict(flippedInput);
  const scratchTensor = state.expressionModelScratch.predict(scratchInput);
  const scratchTensorFlipped = state.expressionModelScratch.predict(scratchFlippedInput);

  const [wideA, wideB, scratchA, scratchB] = await Promise.all([
    wideTensor.data(),
    wideTensorFlipped.data(),
    scratchTensor.data(),
    scratchTensorFlipped.data(),
  ]);
  // Matches evaluate_expression_ensemble.py: average each model's own
  // (plain + flipped) TTA prediction, then average the two models together.
  const exprProbs = wideA.map((_, i) => ((wideA[i] + wideB[i]) / 2 + (scratchA[i] + scratchB[i]) / 2) / 2);

  wideTensor.dispose();
  wideTensorFlipped.dispose();
  scratchTensor.dispose();
  scratchTensorFlipped.dispose();
  flippedInput.dispose();
  scratchInput.dispose();
  scratchFlippedInput.dispose();
  const expressionMs = performance.now() - t2;

  input.dispose();

  let bestIdx = 0;
  for (let i = 1; i < exprProbs.length; i++) {
    if (exprProbs[i] > exprProbs[bestIdx]) bestIdx = i;
  }

  return {
    faceFound: true,
    age,
    genderLabel: genderRaw < 0.5 ? "Male" : "Female",
    genderRaw,
    expressionLabel: state.emotionLabels[bestIdx],
    faceDetectMs,
    ageGenderMs,
    expressionMs,
    // Diagnostic info, always shown in the result text (not hidden in devtools) --
    // added after a real, hard-to-pin-down incident where the backend silently
    // reverted to webgl on one specific machine despite being explicitly set to cpu,
    // producing a wrong face-detection box and a wrong prediction. Baking this into
    // every visible result means that class of problem is diagnosable from a
    // screenshot, not a separate console command.
    backend: tf.getBackend(),
    box,
    imgSize: { w: imgElement.naturalWidth || imgElement.width, h: imgElement.naturalHeight || imgElement.height },
  };
}

function formatResult(r) {
  if (!r.faceFound) return `No face detected. (face detection: ${r.faceDetectMs.toFixed(1)}ms, backend: ${tf.getBackend()})`;
  const b = r.box;
  return (
    `Age: ${Math.round(r.age)}\n` +
    `Gender: ${r.genderLabel}\n` +
    `Expression: ${r.expressionLabel}\n\n` +
    `Latency -- detect: ${r.faceDetectMs.toFixed(1)}ms  ` +
    `age/gender: ${r.ageGenderMs.toFixed(1)}ms  expression: ${r.expressionMs.toFixed(1)}ms\n` +
    `[diagnostic] backend: ${r.backend}  gender_raw: ${r.genderRaw.toFixed(4)}  ` +
    `image: ${r.imgSize.w}x${r.imgSize.h}  ` +
    `face_box: (${b.left.toFixed(0)},${b.top.toFixed(0)})-(${b.right.toFixed(0)},${b.bottom.toFixed(0)})`
  );
}

/** Loads a File (from a file input) into an <img> element and waits for it to decode. */
function loadFileToImage(file, imgElement) {
  // Uses decode() rather than the onload event -- found necessary after a real,
  // hard-to-reproduce incident: onload fires once the image's metadata is ready, which
  // is NOT the same guarantee as "fully decoded and safe to read pixels from" on every
  // platform. On one specific machine this meant face detection ran against a
  // partially-decoded frame, finding a small, wrongly-placed "face" instead of the
  // real one -- while onload-based loading worked fine on every machine this was
  // originally tested on, which is exactly why a timing bug like this is easy to miss.
  // decode() is the standard API that actually guarantees full decode before resolving.
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    imgElement.onerror = reject;
    imgElement.src = url;
    imgElement
      .decode()
      .then(() => {
        imgElement.hidden = false;
        resolve();
      })
      .catch(reject);
  });
}

// ---------- tab switching ----------

function setupTabs() {
  const buttons = document.querySelectorAll(".tab-btn");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      buttons.forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
    });
  });
}

// ---------- analyze tab ----------

function setupAnalyzeTab() {
  const preview = document.getElementById("analyzePreview");
  const result = document.getElementById("analyzeResult");

  async function handleFile(file) {
    if (!file) return;
    await loadFileToImage(file, preview);
    result.textContent = "Analyzing...";
    const r = await analyzeImage(preview);
    result.textContent = formatResult(r);
  }

  document.getElementById("analyzeCameraInput").addEventListener("change", (e) => handleFile(e.target.files[0]));
  document.getElementById("analyzeFileInput").addEventListener("change", (e) => handleFile(e.target.files[0]));
}

// ---------- benchmark tab ----------

function setupBenchmarkTab() {
  const preview = document.getElementById("benchmarkPreview");
  const result = document.getElementById("benchmarkResult");
  const runBtn = document.getElementById("runBenchmarkBtn");

  document.getElementById("benchmarkFileInput").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    await loadFileToImage(file, preview);
    runBtn.disabled = false;
    result.textContent = "Photo loaded. Ready to benchmark.";
  });

  runBtn.addEventListener("click", async () => {
    runBtn.disabled = true;
    const t0 = performance.now();
    // Everything here runs on the CPU backend (see loadModels()'s comment on why),
    // which is numerically correct but slow -- expression alone can take several
    // seconds per pass. Without a per-pass status update, dozens of passes running
    // for minutes with the text frozen on one line is indistinguishable from the
    // page actually being stuck, which is exactly the confusion a real user hit
    // before this was added. Updating the status line every pass makes "slow but
    // working" visibly different from "actually frozen."
    result.textContent = "Warming up...";

    await analyzeImage(preview); // warm-up, excluded from the average

    // Found via real-device testing: repeated CPU-backend passes can occasionally
    // stall hard on one specific pass (seconds-long passes suddenly taking a minute
    // or more, with no error, no crash -- just a very long wait), a resource-pressure
    // effect on the pure-JS CPU backend under sustained heavy load, not a bug that
    // throws. PASS_TIMEOUT_MS bounds the damage: if a single pass runs unusually
    // long, the benchmark gives up on that pass and reports honest partial results
    // from whatever passes DID complete, rather than leaving the page looking dead
    // with no way to tell if it will ever finish. Passes reduced from the original 50
    // to 15 for the same reason -- a shorter run is far less likely to hit this at all
    // and still gives a reasonably stable average.
    const passes = 15;
    const PASS_TIMEOUT_MS = 30000;
    let faceMs = 0, ageGenderMs = 0, expressionMs = 0, found = 0;
    let stalledAt = -1;
    for (let i = 0; i < passes; i++) {
      let r;
      try {
        r = await Promise.race([
          analyzeImage(preview),
          new Promise((_, reject) => setTimeout(() => reject(new Error("pass timed out")), PASS_TIMEOUT_MS)),
        ]);
      } catch (err) {
        stalledAt = i + 1;
        break;
      }
      if (r.faceFound) {
        found++;
        faceMs += r.faceDetectMs;
        ageGenderMs += r.ageGenderMs;
        expressionMs += r.expressionMs;
      }
      const elapsedS = ((performance.now() - t0) / 1000).toFixed(1);
      result.textContent = `Running pass ${i + 1} of ${passes}... (${elapsedS}s elapsed)`;
    }

    if (stalledAt !== -1 && found === 0) {
      result.textContent =
        `Pass ${stalledAt} ran for over ${PASS_TIMEOUT_MS / 1000}s without finishing and was abandoned -- ` +
        `no earlier passes completed either, so there are no results to show. This can happen under ` +
        `sustained load on some devices; closing and reopening the app before trying again usually helps.`;
    } else if (found === 0) {
      result.textContent = "No face detected in the picked photo -- pick a clearer one.";
    } else {
      result.textContent =
        (stalledAt !== -1
          ? `Pass ${stalledAt} ran for over ${PASS_TIMEOUT_MS / 1000}s without finishing and was abandoned -- ` +
            `showing results from the ${found} pass(es) that did complete first.\n\n`
          : "") +
        `Passes: ${stalledAt !== -1 ? stalledAt - 1 : passes} (face found in ${found})\n` +
        `Avg face detection:      ${(faceMs / found).toFixed(1)} ms\n` +
        `Avg age/gender inference: ${(ageGenderMs / found).toFixed(1)} ms\n` +
        `Avg expression inference: ${(expressionMs / found).toFixed(1)} ms\n` +
        `Avg total pipeline:       ${((faceMs + ageGenderMs + expressionMs) / found).toFixed(1)} ms`;
    }
    runBtn.disabled = false;
  });
}

// ---------- evaluate tab ----------

function parseLabelsCsv(text) {
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
  const groundTruth = new Map();
  for (const line of lines.slice(1)) {
    const parts = line.split(",").map((s) => s.trim());
    if (parts.length >= 4) {
      groundTruth.set(parts[0], { ageGroup: parts[1], gender: parts[2], expression: parts[3] });
    }
  }
  return groundTruth;
}

function subgroupAccuracyReport(label, rows, groupOf, matches) {
  const overall = rows.filter(matches).length / rows.length * 100;
  let out = `${label} accuracy (overall: ${overall.toFixed(1)}%)\n`;
  const groups = new Map();
  for (const row of rows) {
    const g = groupOf(row);
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(row);
  }
  for (const g of [...groups.keys()].sort()) {
    const groupRows = groups.get(g);
    const acc = groupRows.filter(matches).length / groupRows.length * 100;
    out += `  ${g}: ${acc.toFixed(1)}% (n=${groupRows.length})\n`;
  }
  return out + "\n";
}

function setupEvaluateTab() {
  const labelsInput = document.getElementById("labelsInput");
  const photosInput = document.getElementById("photosInput");
  const runBtn = document.getElementById("runEvaluateBtn");
  const result = document.getElementById("evaluateResult");
  const downloadLink = document.getElementById("downloadResultsLink");

  let groundTruth = null;
  let photoFiles = null;

  function maybeEnableRun() {
    runBtn.disabled = !(groundTruth && photoFiles && photoFiles.length > 0);
  }

  labelsInput.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    groundTruth = parseLabelsCsv(await file.text());
    result.textContent = `Loaded ${groundTruth.size} labeled rows from ${file.name}.`;
    maybeEnableRun();
  });

  photosInput.addEventListener("change", (e) => {
    photoFiles = e.target.files;
    result.textContent = `${photoFiles.length} photo(s) selected.`;
    maybeEnableRun();
  });

  runBtn.addEventListener("click", async () => {
    runBtn.disabled = true;
    downloadLink.hidden = true;
    result.textContent = "Running evaluation...";

    const tempImg = new Image();
    const rows = [];

    for (const file of photoFiles) {
      const gt = groundTruth.get(file.name);
      if (!gt) continue;

      await loadFileToImage(file, tempImg);
      const r = await analyzeImage(tempImg);

      if (!r.faceFound) {
        rows.push({ filename: file.name, gt, predAge: "no-face", predGender: "no-face", predExpr: "no-face" });
        continue;
      }
      const predAgeGroup = r.age >= ELDERLY_AGE_THRESHOLD ? "elderly" : "adult";
      rows.push({ filename: file.name, gt, predAge: predAgeGroup, predGender: r.genderLabel, predExpr: r.expressionLabel });
    }

    if (rows.length === 0) {
      result.textContent = "No photos matched a row in the labels CSV (check filenames match exactly).";
      runBtn.disabled = false;
      return;
    }

    let summary = `Evaluated ${rows.length} of ${groundTruth.size} labeled images.\n\n`;
    summary += subgroupAccuracyReport("Age group", rows, (r) => r.gt.ageGroup, (r) => r.predAge.toLowerCase() === r.gt.ageGroup.toLowerCase());
    summary += subgroupAccuracyReport("Gender", rows, (r) => r.gt.gender, (r) => r.predGender.toLowerCase() === r.gt.gender.toLowerCase());
    summary += subgroupAccuracyReport("Expression", rows, (r) => r.gt.expression, (r) => r.predExpr.toLowerCase() === r.gt.expression.toLowerCase());
    result.textContent = summary;

    let csv = "filename,gt_age_group,pred_age_group,gt_gender,pred_gender,gt_expression,pred_expression\n";
    for (const r of rows) {
      csv += `${r.filename},${r.gt.ageGroup},${r.predAge},${r.gt.gender},${r.predGender},${r.gt.expression},${r.predExpr}\n`;
    }
    const blob = new Blob([csv], { type: "text/csv" });
    downloadLink.href = URL.createObjectURL(blob);
    downloadLink.hidden = false;

    runBtn.disabled = false;
  });
}

// ---------- boot ----------

function registerServiceWorker() {
  if ("serviceWorker" in navigator) {
    // Caches everything (app + both models + BlazeFace) so the app keeps
    // working offline after the first successful load -- this is what makes
    // "Add to Home Screen" behave like a real installed/downloaded app.
    navigator.serviceWorker.register("sw.js").catch((err) => console.error("Service worker registration failed:", err));
  }
}

(async function main() {
  setupTabs();
  setupAnalyzeTab();
  setupBenchmarkTab();
  setupEvaluateTab();
  registerServiceWorker();
  try {
    await loadModels();
  } catch (err) {
    document.getElementById("modelStatus").textContent = `Failed to load models: ${err.message}`;
    const dot = document.getElementById("statusDot");
    dot.classList.remove("loading");
    dot.classList.add("error");
    console.error(err);
  }
})();
