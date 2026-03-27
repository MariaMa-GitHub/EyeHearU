# Eye Hear U — User Guide

For **people using the mobile app** (demo viewers, testers, and end users).

## What the app does

Eye Hear U turns **recorded ASL video** into **English-like gloss labels** (the same kind of labels the model was trained on) and **spoken output** (text-to-speech). You can work in two ways:

| Mode | What you record | What you get |
|------|-----------------|--------------|
| **Single sign** | One short clip per translation | One **gloss** (top prediction), confidence, optional alternate signs |
| **Multi-sign** | Several clips **in order** (one sign per clip), then **Translate** | A **sequence** of glosses chosen jointly (beam search + language model), shown as one line of text |

**Multi-sign** does **not** produce guaranteed full grammatical English sentences — it combines the model’s gloss picks into one line with simple formatting (capital letter, period, etc.). Think of it as an ordered **gloss phrase**, useful together with the classifier, not a certified translation.

See [ASL translation pipeline](ASL_TRANSLATION_PIPELINE.md) for technical detail.

## Requirements

- **iPhone** (or Android with Expo Go, if a build targets that platform)
- **Expo Go** from the App Store (or Google Play)
- **Camera and microphone** permission (video recording may use the microphone on some devices)
- **Network access** to the inference API (same Wi‑Fi as the development machine, a tunnel URL, or a deployed API)

## How to use the app

1. **Open the app** — scan the Expo QR code from the machine running the dev server, or open a development / store build.
2. **Home screen** — you will see the app title, a brief description, and two action buttons:
   - **Start Translating** — opens the camera so you can record ASL.
   - **View History** — browses past translations stored on the device.
3. Tap **Start Translating**.
4. Allow **camera** (and **microphone** if prompted).
5. On the camera screen, choose a mode at the top:
   - **Single sign** — one clip → one prediction (default).
   - **Multi-sign** — add one clip per sign, in order; when done, tap **Translate** to run the full sequence.
6. Position the signer so the **sign is visible** in the camera.
7. **Single sign:** tap **Record Sign** (records a short clip) and wait for **Processing…**.
8. **Multi-sign:** tap **Add sign** for each gloss in order; use **Clear clips** if you need to start over; tap **Translate** when ready.
9. The result area shows:
   - **Single:** top gloss, confidence, sometimes **other likely signs** as chips
   - **Multi:** a **Sentence** line (gloss sequence), and the beam’s chosen gloss list summary
10. The app may **read the text aloud** when confidence / content allows.
11. Tap **Speak Again** to repeat the audio.
12. Use **View History** for **past translations** on the device (local storage by default; cloud only if configured in the project).

You can also use the **upload** control (cloud icon) to pick a video from the library instead of recording, in either mode.

## Tips for better accuracy

- Sign **clearly** and hold the sign **steady** for the full recording (**Multi-sign:** one sign per clip).
- Use **good lighting** and a plain background when possible.
- The model only knows its **trained vocabulary** (hundreds of gloss classes — see project docs).
- If the result is wrong, **record again** with a slower, clearer sign; **Multi-sign** errors in one clip affect the whole sequence.

## Privacy
  
- Video is sent to the **configured backend** for inference (not arbitrary third-party inference services unless the deployment is changed).
- History is stored **on the device** by default (AsyncStorage).

## Troubleshooting

| Problem | What to try |
|--------|-------------|
| “Backend offline” | API not running, wrong base URL, or the phone cannot reach the server (tunnel or same network as the API host). |
| “Could not reach backend” | Same as above; confirm FastAPI is running and tunnel URL is current if used. |
| Expo shows “offline” / bundle won’t load | **iOS:** Settings → Privacy & Security → **Local Network** → enable **Expo Go**. Use LAN mode (`npm run start:lan`) when phone and dev machine share Wi‑Fi; tunnel mode only when LAN is blocked. |
| Camera won’t open | Settings → Eye Hear U (or Expo Go) → Camera. |
| No sound | Check volume and silent switch; TTS uses system speech. |
| **Translate** disabled (Multi-sign) | Add at least one clip with **Add sign** or upload before translating. |

For technical setup, see the [Developer guide](DEVELOPER_GUIDE.md).
