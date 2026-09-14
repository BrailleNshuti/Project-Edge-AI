# Quickstart: testing the app on my iPhone

## On my laptop

1. Click Start, type `PowerShell`, press Enter.
2. Type and press Enter:
   ```
   cd "C:\Users\Admin\Downloads\edge Ai project\web_app"
   ```
3. Type and press Enter:
   ```
   python -m http.server 8765
   ```
   The window goes quiet and just sits there -- that's correct, it's running. Leave this
   window open the whole time I'm testing.

## On my iPhone

4. Connect to the **same WiFi** as the laptop.
5. Open **Safari**.
6. Type this in the address bar and press Go:
   ```
   192.168.0.117:8765
   ```
   (If this stops working later -- e.g. after reconnecting to WiFi -- go back to the
   laptop, open PowerShell, type `ipconfig`, and look for "IPv4 Address" under the Wi-Fi
   section. Use that number instead.)
7. Wait for "Models loaded..." to appear at the bottom of the page.

The app has a warm, hand-crafted look (cream background, coral accent, Polaroid-style
photo preview) instead of a generic blue template -- that's intentional, not a bug.

## Using the app

- **Analyze** tab: Take Photo or Choose Photo, see age/gender/expression results.
- **Benchmark** tab: pick a photo, tap Run -- measures how fast the app runs on my phone.
  These are the numbers for the report's "edge device" column.
- **Evaluate** tab: once the 20 labeled photos + `labels.csv` are ready (see main
  README's step 4), load them here and tap Run Evaluation.

## "Downloading" / installing the app

There's no separate installer -- this step *is* the download:

1. Open the app in Safari at least once (step 6 above) while connected to the laptop's server.
2. Tap the **Share** icon (square with an arrow, along the bottom of Safari).
3. Scroll down and tap **Add to Home Screen**, then tap **Add**.
4. An "Edge AI" icon now appears on the home screen. Opening it from there launches
   full-screen like a real app -- and it keeps working even with WiFi off, because
   everything (both models, the face detector, the app itself) gets cached on the phone
   on first load.

## When done

Go back to the laptop's PowerShell window and press `Ctrl + C` to stop the server.

## If something goes wrong

- Page won't load on iPhone: confirm both devices show the *same* WiFi network name in
  their WiFi settings.
- PowerShell window closed by accident: just repeat steps 1-3.
- **Getting predictions that seem stuck on an old version of the app** (e.g. results
  that don't match what was just fixed): the iPhone is very likely showing a cached
  copy. This app caches itself for offline use, and a real, now-fixed bug meant that
  cache could get stuck on old content indefinitely, even across "new" versions. One-time
  fix: **Settings → Safari → Advanced → Website Data → find the site's address (e.g.
  `192.168.x.x`) → swipe to delete it** -- then reopen the app fresh (remove and re-add
  the Home Screen icon too, if using one). This clears every layer of caching at once --
  only needed once, since the underlying bug is fixed and future updates should show up
  on a normal reload from here on. The "Models loaded" line at the bottom of the Analyze
  tab shows a version tag (currently "v27") to double-check which version is running,
  without guessing.
- Full technical details and the rest of the project (training, evaluation, report
  requirements): see the main [README.md](README.md).
- The actual report submitted is at `Project_Report.docx` in the project root -- fully
  personalized and reviewed. The code is also pushed and live on GitHub.
