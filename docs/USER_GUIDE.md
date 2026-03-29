# Eye Hear U — User Guide

For **people using the mobile app** (demo viewers, testers, and end users).

## What the app does

Eye Hear U translates **isolated ASL signs** (one sign at a time) into **English text** and **spoken English**. It is not a full sentence or conversation translator.

## Requirements

- **iPhone** (or Android with Expo Go, if a build targets that platform)
- **Expo Go** from the App Store (or Google Play)
- **Camera and microphone** permission (video recording may use the microphone on some devices)
- **Network access** to the inference API (same Wi‑Fi as the development machine, a tunnel URL, or a deployed API)

## How to use the app

1. **Open the app** — scan the Expo QR code from the machine running the dev server, or open a development / store build.
2. **Home screen** — you will see the app title, a brief description, and two action buttons:
   - **Start Translating** — opens the camera so you can record an ASL sign.
   - **View History** — browses past translations stored on the device.
3. Tap **Start Translating**.
4. Allow **camera** (and **microphone** if prompted).
5. Position the signer so the **sign is visible** in the front camera.
6. Tap **Record Sign**. The app records a **short clip** (about 3 seconds).
7. Wait for **Processing…**. The screen shows:
   - The **top prediction** (English gloss) — tap it to watch a **reference ASL video**
   - **Confidence**
   - Sometimes **other likely signs** as small chips — tap any chip to see its video too
8. The app may **read the word aloud** (text-to-speech) if confidence is high enough.
9. Tap **Speak Again** to repeat the audio.
10. You can also **upload a video** from your photo library instead of recording live — tap the upload icon in the top right corner.
10. Use **View History** for **past translations** stored on the device (local storage by default; cloud sync only if configured in the project).

## Tips for better accuracy

- Sign **clearly** and hold the sign **steady** for the full recording.
- Use **good lighting** and a plain background when possible.
- The model only recognizes the **trained vocabulary** (MVP gloss set — see project documentation for the list).
- If the result is wrong, **record again** with a slower, clearer sign.

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

For technical setup, see the [Developer guide](DEVELOPER_GUIDE.md).
