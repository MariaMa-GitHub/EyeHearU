## Mobile app (`mobile/`) — Eye Hear U (React Native / Expo)

React Native (Expo) client for Eye Hear U. It:

- Records short **video** clips of isolated ASL signs via the device camera  
- Sends them to the FastAPI backend for prediction  
- Shows the predicted English gloss, confidence, and optional top‑k alternatives  
- Uses **text-to-speech** for high-confidence results  
- Persists a local **history** (AsyncStorage)

---

### Entry points & routing

- `app/_layout.tsx` — `expo-router` stack: Home, Camera, History.  
- `app/index.tsx` — Home: animated hero icon, **Start Translating**, **View History**.  
- `app/camera.tsx` — Records ~3 s video, calls `predictSign`, TTS, errors.  
- `app/history.tsx` — Reads/writes translation history from AsyncStorage.

---

### Camera & prediction flow (`app/camera.tsx`)

1. Request camera permission (`useCameraPermissions` from `expo-camera`).  
2. Show `CameraView` (front camera, video mode).  
3. On **Record Sign**, call `recordAsync` with max duration ~3 s.  
4. POST the file via `predictSign(uri)` in `services/api.ts`.  
5. Update UI with `sign`, `confidence`, `top_k`; optionally speak the label.  
6. Append successful predictions to history.

The backend base URL is **not** hard-coded in screens; it is resolved in `services/api.ts` from `EXPO_PUBLIC_API_URL` (development) or the production constant.

---

### API client (`services/api.ts`)

- `predictSign(videoUri)` — `FormData` field `file`, `POST /api/v1/predict`.  
- `checkHealth()` / readiness helpers (available for future use; not currently shown in the UI).  
- Tunnel / 503 messaging helpers for clearer errors.

Configure development URL in **`mobile/.env`** (see **`mobile/.env.example`**):

- iOS Simulator on the same Mac as the API: `EXPO_PUBLIC_API_URL=http://127.0.0.1:8000`  
- Physical device on the same LAN: `EXPO_PUBLIC_API_URL=http://<host-LAN-IP>:8000` (uvicorn `--host 0.0.0.0`)

Restart Expo after changing `.env`.

---

### How to run

```bash
cd mobile
npm install
# Same Wi‑Fi as the API host (recommended for Expo Go):
npm run start:lan
# Or: npx expo start
```

Open in **Expo Go** (QR) or press `i` for the iOS Simulator. Ensure the backend is running (see repository root **README** or `docs/DEVELOPER_GUIDE.md`).

**iOS:** If the JavaScript bundle fails to load, enable **Local Network** for **Expo Go** under Settings → Privacy & Security → Local Network.

---

### Typical development tasks

- UI / navigation: `app/*.tsx`, shared styles.  
- Networking and env handling: `services/api.ts`.  
- Error copy and tunnel hints: `services/api.ts`, `app/camera.tsx`.
