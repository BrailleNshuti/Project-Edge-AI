// Caches every app asset (including both trained models and BlazeFace) on
// first load, so once you've opened the app once, it keeps working even with
// WiFi off -- this is what makes "Add to Home Screen" behave like a genuinely
// downloaded/installed app rather than just a bookmark.
//
// Bump CACHE_VERSION whenever any cached file changes (new model weights,
// app.js edits, etc.) so old clients pick up the new version instead of
// serving stale cached files forever.
const CACHE_VERSION = "v27";
const CACHE_NAME = `edge-ai-${CACHE_VERSION}`;

const ASSETS = [
  "./",
  "index.html",
  "style.css?v=4",
  "app.js?v=17",
  "manifest.json",
  "icon-180.png",
  "icon-192.png",
  "icon-512.png",
  "lib/tf.min.js",
  "lib/blazeface.min.js",
  "models/blazeface/model.json",
  "models/blazeface/group1-shard1of1.bin",
  "models/age_gender/model.json",
  "models/age_gender/group1-shard1of1.bin",
  "models/emotion_labels.json",
  // Expression ensemble: alpha=1.0 MobileNetV2 (3 shards) + from-scratch CNN
  // (2 shards) -- replaced the single alpha=0.35 model this app shipped with
  // before (58.34% -> 67.60% test accuracy, see training/results/expression_ensemble_comparison.json).
  "models/expression_wide/model.json",
  "models/expression_wide/group1-shard1of3.bin",
  "models/expression_wide/group1-shard2of3.bin",
  "models/expression_wide/group1-shard3of3.bin",
  "models/expression_scratch/model.json",
  "models/expression_scratch/group1-shard1of2.bin",
  "models/expression_scratch/group1-shard2of2.bin",
];

// Root-caused a real, hard-to-spot bug: cache.addAll() fetches with default HTTP
// caching behavior. index.html and "./" never change their URL (unlike every other
// asset here, which get a ?v=N bump on every real change), so once the browser's own
// plain HTTP cache had a stale copy of either -- from as far back as the very first
// visit -- every single future install() kept re-fetching and re-freezing that SAME
// stale copy into each new CACHE_VERSION, even though the version number itself was
// correct. { cache: "reload" } forces each install-time fetch to bypass HTTP cache and
// hit the network for real, which is what should have been happening all along.
async function fetchFresh(url) {
  return fetch(url, { cache: "reload" });
}

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) =>
      Promise.all(ASSETS.map((url) => fetchFresh(url).then((res) => cache.put(url, res))))
    )
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k))))
  );
  self.clients.claim();
});

// App-shell files (navigation, HTML, JS, CSS) use network-first: always try to get
// the latest version when online, since these are exactly the files that were going
// stale; the cached copy is only a fallback for offline use. Everything else (model
// weights, icons, libraries) stays cache-first, since those are large, already
// version-stamped in their URL when they change, and not worth re-downloading on
// every single load.
const NETWORK_FIRST = [".", "index.html", "style.css", "app.js"];

function isNetworkFirst(request) {
  if (request.mode === "navigate") return true;
  const path = new URL(request.url).pathname.replace(/^\//, "") || ".";
  return NETWORK_FIRST.some((f) => path === f || path.endsWith("/" + f));
}

self.addEventListener("fetch", (event) => {
  if (isNetworkFirst(event.request)) {
    event.respondWith(
      fetchFresh(event.request.url)
        .then((res) => {
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, res.clone()));
          return res;
        })
        .catch(() => caches.match(event.request))
    );
    return;
  }
  event.respondWith(
    caches.match(event.request).then((cached) => cached || fetch(event.request))
  );
});
