# Eye Hear U — Developer Guide

Repository layout, how to run each component, and where to change behavior.

## Repository layout

| Path | Role |
|------|------|
| `mobile/` | Expo (React Native) app — UI, camera, API client, local history |
| `backend/` | FastAPI inference API — loads I3D, `/predict`, health checks |
| `ml/i3d_msft/` | **Inception I3D** model code (must match the training branch) |
| `ml/i3d_label_map_mvp-sft-full-v1.json` | Class index ↔ gloss (856 signs, v4) |
| `infrastructure/` | Terraform (S3, ECS, Batch, etc.) |
| `data/scripts/` | Data pipeline scripts |
| `docs/` | Documentation (this file, testing, production, preprocessing) |

## Prerequisites

- **Python 3.11+** (3.12 is fine locally; CI uses 3.11)
- **Node.js 20+** and npm
- **AWS CLI** (optional) — to download `best_model.pt` from S3
- **Expo CLI** via `npx expo`

## Backend (FastAPI)

### Install

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Model files

- **Label map:** `../ml/i3d_label_map_mvp-sft-full-v1.json` (default in `Settings`)
- **Weights:** `backend/model_cache/best_model.pt`  
  Download once:
  ```bash
  mkdir -p model_cache
  aws s3 cp s3://eye-hear-u-public-data-ca1/models/i3d/modal/candidate-ac-eval-v2/mvp-sft-full-v1/best_model.pt model_cache/
  ```
  If the file is missing, the server attempts S3 download (requires AWS credentials).

### Environment

```bash
cp .env.example .env
# Edit MODEL_PATH, LABEL_MAP_PATH, MODEL_DEVICE if needed
```

### Run (device / tunnel access)

Bind to all interfaces so phones and other hosts can connect:

```bash
cd backend
export PYTHONPATH=..    # parent repo root so `import ml` works
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Key endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness |
| GET | `/ready` | Readiness + `model_loaded` |
| POST | `/api/v1/predict` | Multipart file field `file` — **mp4/mov** video |
| POST | `/api/v1/predict/sentence` | Multipart field `files` (repeat per clip) — batched I3D + beam + gloss LM; query `beam_size`, `lm_weight`, `top_k` |

### Important code paths

- `app/main.py` — FastAPI app; **lifespan** loads the model and gloss LM on startup
- `app/services/model_service.py` — S3 download (if needed), `torch.load`, I3D, `predict()`, `predict_batch()`
- `app/services/gloss_lm.py`, `app/services/beam_search.py` — bigram LM + beam over per-clip top-k
- `app/services/preprocessing.py` — bytes → tensor `(1, 3, 64, 224, 224)` — **must match training**
- `app/routers/predict.py` — validates upload, calls preprocess + predict / sentence pipeline

## Mobile (Expo)

```bash
cd mobile
npm install --legacy-peer-deps   # if peer conflicts
```

### Two different URLs (do not mix them)

| What | Purpose | Typical value (same Wi‑Fi as dev machine) |
|------|---------|-------------------------------------------|
| **Metro / Expo** | Loads the **JavaScript bundle** into Expo Go | `http://<LAN_IP>:8081` — use **`npm run start:lan`** |
| **Backend API** | FastAPI `/predict` | `EXPO_PUBLIC_API_URL=http://<LAN_IP>:8000` in **`mobile/.env`** |

If **`expo start --tunnel`** is used while the device cannot reach Expo’s tunnel servers, Expo Go may show **“Internet connection appears to be offline”** — that is a **bundle load** failure (Metro), not the inference API.

**Recommended when the API runs on LAN and the phone shares Wi‑Fi:**

```bash
cd mobile
npm run start:lan
```

The QR code should resolve to something like `exp://192.168.x.x:8081`. Phone and development machine must be on the **same Wi‑Fi** (avoid guest / client-isolation networks).

**iOS:** If the bundle still fails to load, open **Settings → Privacy & Security → Local Network** and enable **Expo Go**.

**macOS firewall:** If LAN mode fails, allow **Node** incoming connections, or temporarily relax the firewall for testing.

**Simulator:** On macOS, **`npm run ios`** or **`i`** in the Expo terminal opens the **iOS Simulator** (localhost for Metro; set `EXPO_PUBLIC_API_URL=http://127.0.0.1:8000` if the API listens on the same machine).

Use **`npm run start:tunnel`** only when the device **cannot** reach the dev machine on the local network (e.g. restrictive campus Wi‑Fi); tunnel availability then depends on Expo’s infrastructure.

### API base URL

Use **`mobile/.env`** (copy from **`mobile/.env.example`**):

```bash
EXPO_PUBLIC_API_URL=http://192.168.x.x:8000
# or after: npx localtunnel --port 8000
EXPO_PUBLIC_API_URL=https://new-subdomain.loca.lt
```

Restart **`npx expo start`** after any `.env` change (variables are inlined at bundle time).

**LocalTunnel:** URLs **expire** when the tunnel process stops or loca.lt returns **503 tunnel unavailable**. Start a **new** tunnel and update `.env`. **Same Wi‑Fi + LAN IP** is more stable for demos.

The app sends `bypass-tunnel-reminder: true` when the URL contains `loca.lt`.

### Native modules

- `expo-camera` — video recording
- `expo-speech` — TTS
- `@react-native-async-storage/async-storage` — history

## Docker (optional)

From repo root:

```bash
docker compose up --build
```

Ensure `ml/i3d_msft` is included in the image (see root `Dockerfile`). Model cache uses a volume under `model_cache`.

## Firebase (optional)

`backend/app/services/firebase_service.py` is wired for Firestore history. Not required for the default app (history is local on device). To enable:

1. Create a Firebase project; download a service account JSON.
2. Place credentials and set `FIREBASE_*` in `.env`.
3. Call `init_firebase()` from startup if new routes depend on it.

## Syncing with training code

Training lives on branch **`freya-a5-training`**. Inference must stay aligned:

- `ml/i3d_msft/pytorch_i3d.py` — I3D definition
- Training-side preprocessing (on that branch) must stay aligned with `backend/app/services/preprocessing.py` — see `docs/PREPROCESSING.md`

## Git workflow

- Run **tests** before pushing: see [TESTING.md](TESTING.md).
- CI runs on **push/PR** to `main`/`master` (`.github/workflows/ci.yml`).
