# Eye Hear U — Developer Guide

Repository layout, how to run each component, and where to change behavior.

## Repository layout

| Path | Role |
|------|------|
| `mobile/` | Expo (React Native) app — UI, camera, API client, local history |
| `backend/` | FastAPI inference API — loads I3D + gloss LM, `/predict`, `/predict/sentence`, health |
| `ml/i3d_msft/` | **Inception I3D** model code (must match the training branch) |
| `ml/i3d_label_map_mvp-sft-full-v1.json` | Class index ↔ gloss (856 signs, v4) |
| `infrastructure/` | Terraform (S3, ECS, Batch, etc.) |
| `data/scripts/` | Data pipeline scripts |
| `docs/` | Documentation — start at [docs/README.md](README.md) |

## Prerequisites

- **Python 3.11+** (3.12 is fine locally; CI uses 3.11)
- **Node.js 20+** (22 LTS recommended) and npm
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

If that file is missing at startup, the API downloads it from S3 using **`AWS_S3_BUCKET`**, **`AWS_S3_MODEL_KEY`**, and **`AWS_S3_REGION`** (see defaults in `app/config.py` and overrides in `.env`). You need **AWS credentials** (or an IAM role on the host) for that path.

Manual download example (bucket/key match the current `Settings` defaults unless you changed them):

```bash
cd backend
mkdir -p model_cache
aws s3 cp s3://eye-hear-u-public-data-ca1/models/i3d/modal/candidate-ac-eval-v4/20260325T001403Z/best_model.pt model_cache/best_model.pt
```

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
| POST | `/api/v1/predict/sentence` | Multipart field **`files`** repeated once per clip **in order** — batched I3D → beam + gloss LM → `best_glosses` + `english`; query `beam_size`, `lm_weight`, `top_k` (max 12 clips) |

### Important code paths

- `app/main.py` — FastAPI app; **lifespan** loads the I3D model and `GlossBeamLM` from `data/gloss_lm.json` (via `config.gloss_lm_path`)
- `app/services/model_service.py` — S3 download (if needed), `torch.load`, I3D, `predict()` (single tensor), `predict_batch()` (multi-clip)
- `app/services/gloss_lm.py` — `GlossBeamLM` (trigram + bigram backoff), `load_gloss_lm`
- `app/services/beam_search.py` — beam search over per-clip top‑k using `log_p_step(prev2, prev1, next)`
- `app/services/gloss_to_english.py` — joins `best_glosses` for `english` with light polish (not full MT)
- `app/services/gloss_to_english_t5.py`, `gloss_to_english_bedrock.py` — optional rewriters when `GLOSS_ENGLISH_MODE` is `t5` or `bedrock` (see `app/config.py` / `.env.example`)
- `app/services/lm_builder.py` — offline construction of LM JSON; CLI: `scripts/build_gloss_lm.py`
- `app/services/preprocessing.py` — bytes → tensor `(1, 3, 64, 224, 224)` — **must match training**
- `app/routers/predict.py` — `/predict` (one file), `/predict/sentence` (repeated `files`)

**Pipeline reference:** [ASL translation pipeline](ASL_TRANSLATION_PIPELINE.md) (accuracy scope, rebuilding `gloss_lm.json`).

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

### Camera: single sign vs multi-sign

In `mobile/app/camera.tsx`, **Single sign** calls `predictSign` (`POST /api/v1/predict`) after each recording. **Multi-sign** queues URIs, then calls `predictSentence` (`POST /api/v1/predict/sentence` with repeated `files`). Both modes share upload-from-library; multi-sign uses **Add sign** + **Translate** in the bottom bar.

### Native modules

- `expo-camera` — video recording
- `expo-speech` — TTS
- `expo-video` — in-app ASL reference video playback (video dictionary)
- `expo-web-browser` — fallback browser for sign video lookup (SignASL.org)
- `expo-image-picker` — video upload from gallery
- `@react-native-async-storage/async-storage` — history

## Docker (optional)

From repo root:

```bash
docker compose up --build
```

Ensure `ml/i3d_msft` is included in the image (see root `Dockerfile`). Model cache uses a volume under `model_cache`.

## Firebase (optional, not wired in the default app)

`backend/app/services/firebase_service.py` contains **Firestore helpers** for a possible future “cloud history” feature. The **shipped FastAPI app does not call `init_firebase()`** in `main.py`, and the **mobile app** only persists history with **AsyncStorage**. To experiment with Firestore:

1. Create a Firebase project; download a service account JSON (do not commit it).
2. Place credentials and set `FIREBASE_*` in `.env` (see `app/config.py`).
3. Call `init_firebase()` from application startup and add routes that use it.

## Syncing with training code

If your team maintains I3D training in a **separate branch or repo**, keep **inference** here aligned with it:

- `ml/i3d_msft/pytorch_i3d.py` — I3D definition shipped with this API
- Training-side preprocessing must match `backend/app/services/preprocessing.py` — see `docs/PREPROCESSING.md`
- End-to-end S3 splits and training reproduction: [I3D training — S3 reproduction](I3D_TRAINING_S3_REPRODUCTION.md)
- New AWS account, Modal GPU, and SFT warm-start: [Modal & AWS SFT migration](MODAL_AWS_SFT_MIGRATION.md)

## Git workflow

- Run **tests** before pushing: see [TESTING.md](TESTING.md).
- CI runs on **push/PR** to `main`/`master` (`.github/workflows/ci.yml`).
