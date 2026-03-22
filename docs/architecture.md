## Eye Hear U – System Architecture & Use Cases

**Version:** 2026-03-21

---

### 1. Problem & High-Level Idea

**Problem:** There is no simple, reliable tool for translating **single ASL signs into English text/speech in real time** using only a phone camera. This blocks everyday communication (e.g., restaurant orders, quick medical questions) and makes it hard for ASL learners to verify their signing.

**Solution:** Eye Hear U is an **iOS-focused mobile app** where a user:
- Opens the app and points the camera at a signer (themselves or someone else),
- Taps **Record Sign** (~3 s video),
- Sees the **predicted English gloss and confidence**, and
- Optionally hears it spoken via **text-to-speech**.

The MVP recognizes **single, isolated signs** from a **scenario-focused vocabulary** (greetings, basic needs, restaurant, medical, letters A–Z, numbers 1–10).

---

### 2. System Architecture Overview

```text
┌──────────────────────┐        ┌──────────────────────┐
│   Mobile App         │  HTTP  │   Backend API        │
│   (React Native /    │───────▶│   (FastAPI / Python) │
│    Expo, iOS first)  │        │                      │
│                      │        │  POST /api/v1/predict│
│  - Camera capture    │◀───────│  → sign label +      │
│  - Display results   │  JSON  │    confidence score  │
│  - Text-to-speech    │        │                      │
│  - Translation       │        │  /health, /ready     │
│    history (future)  │        └──────────┬───────────┘
└──────────────────────┘                   │
                                           │ loads
                                ┌──────────▼───────────┐
                                │   ML Model           │
                                │   (PyTorch)          │
                                │                      │
                                │  Inception I3D       │
                                │  (3D CNN, 64-frame)  │
                                │       ↓              │
                                │  Temporal max-pool   │
                                │       ↓              │
                                │  Softmax + top-k     │
                                │                      │
                                │       ↓              │
                                │  Classification      │
                                │  (48 MVP glosses)    │
                                └──────────┬───────────┘
                                           │
                                           │ writes/reads
                                ┌──────────▼───────────┐
                                │   Firebase           │
                                │   (Firestore)        │
                                │                      │
                                │  - Translation       │
                                │    history           │
                                │  - Session tracking  │
                                │  - Usage analytics   │
                                └──────────────────────┘
```

**Key repos & paths:**
- Mobile app: `mobile/`
- Backend API: `backend/`
- ML model & training: `ml/`
- Data pipeline & schemas: `data/`, `docs/data_schema.md`

---

### 3. Components in Detail

#### 3.1 Mobile App (`mobile/`)

**Tech:** React Native with **Expo SDK 54**, TypeScript, `expo-router` for navigation.

**Main responsibilities:**
- Request and manage **camera permission**.
- Show a **live camera preview**.
- Record a **~3 s video clip** (when user taps **Record Sign**).
- Upload the video to the backend (`/api/v1/predict`).
- Display predicted sign + confidence.
- Use **TTS** (`expo-speech`) to read out the predicted word.
- (Future) Show per-session **translation history** and collect “correct/wrong” feedback.

**Important files:**
- `mobile/app/_layout.tsx` – navigation shell using `expo-router`.
- `mobile/app/index.tsx` – **Home screen** (start translating / view history).
- `mobile/app/camera.tsx` – **Camera + prediction screen**.
- `mobile/app/history.tsx` – **History UI** (reads from AsyncStorage).
- `mobile/services/api.ts` – typed API client for the backend.

**Camera flow (step-by-step):**
1. **Permissions** – `useCameraPermissions()` from `expo-camera`:
   - If status is unknown, the app shows a screen explaining why camera is needed with a “Grant Permission” button.
   - If permission denied, the same screen is shown until the user grants in settings.

2. **Live preview** – `CameraView` component:
   - Props: `facing="front"`, `mode="video"`, full-screen style.
   - Shows the live feed from the front-facing camera.

3. **Record & upload** (in `recordAndPredict`):
   - Call `cameraRef.current.recordAsync({ maxDuration: 3 })`.
   - Returns a `video` object with a local `uri` to the mp4 file.
   - Wrap in `FormData` under the `file` key:
     - `formData.append("file", { uri, name: "clip.mp4", type: "video/mp4" })`.
   - `fetch(API_BASE_URL + "/api/v1/predict", { method: "POST", body: formData })`.

4. **Handle response:**
   - Backend returns JSON matching `PredictionResult`:

     ```ts
     type PredictionResult = {
       sign: string;
       confidence: number;   // 0.0 – 1.0
       top_k: { sign: string; confidence: number }[];
       message?: string;
     };
     ```

   - App sets local state: `prediction`, `confidence`.
   - If `confidence > 0.5`, call `Speech.speak(prediction)`. This provides **audio output**.

5. **Display:**
   - UI shows:
     - “Detected Sign:” label
     - Big blue `prediction` text
     - Confidence as a percentage
     - Button: **Speak Again** → `Speech.speak(prediction)`.

6. **Error handling:**
   - If network fails or backend returns non-200, app sets `prediction = "Error — check connection"` and `confidence = 0`.

**Implementation checklist (mobile):**
- [x] Set up Expo SDK 54 and base app.
- [x] Implement home screen + navigation (`expo-router`).
- [x] Implement camera screen with **video recording** (~3 s clips) and upload logic.
- [x] Implement TTS via `expo-speech`.
- [x] Configurable backend URL via `EXPO_PUBLIC_API_URL` in `mobile/.env` (LAN IP, tunnel, or production).
- [x] Persist predictions locally (AsyncStorage). Firebase history is optional.

---

#### 3.2 Backend API (`backend/`)

**Tech:** Python, **FastAPI**, Uvicorn.

**Responsibilities:**
- Provide **HTTP endpoints** for:
  - Health/liveness checks (`/health`, `/ready`).
  - **Prediction** (`POST /api/v1/predict`).
- Load the **trained PyTorch model** on startup and keep it in memory.
- Run the **preprocessing + inference pipeline** per request.
- (Future) Write translation logs to **Firestore**.

**Important files:**
- `backend/app/main.py` – FastAPI app, startup hook, router registration.
- `backend/app/routers/health.py` – `/health`, `/ready`.
- `backend/app/routers/predict.py` – `/api/v1/predict`.
- `backend/app/services/preprocessing.py` – video preprocessing (see `docs/PREPROCESSING.md`).
- `backend/app/services/model_service.py` – I3D model loading + inference.
- `backend/app/services/firebase_service.py` – Firestore integration.
- `backend/app/schemas/prediction.py` – response schema.

**Prediction request flow:**
1. **Client** sends `multipart/form-data` POST:
   - `file`: **mp4/mov** video clip (~3 s recording from the mobile camera).

2. **FastAPI** parses it into `UploadFile`:

   ```py
   @router.post("/predict", response_model=PredictionResponse)
   async def predict_sign(request: Request, file: UploadFile = File(...)):
   ```

3. **Validation:**
   - Check `content_type` is `video/mp4`, `video/quicktime`, or the filename ends in `.mp4`/`.mov`.
   - Read file bytes and verify non-empty.

4. **Preprocessing** (`preprocess_video` — see `docs/PREPROCESSING.md`):
   - Decode with OpenCV, adaptive temporal sampling (center-biased, frame skip).
   - Per-frame spatial resize (mobile 4K cap → min-side 226 → max-side 256 → center-crop 224).
   - Normalize to **`[-1, 1]`** (not ImageNet μ/σ).
   - Pad or trim to **64 frames**; ensure both sides ≥ 224 before crop.
   - Output tensor shape: **`(1, 3, 64, 224, 224)`**.

5. **Model inference** (`model_service.predict`):
   - With `torch.no_grad()`:
     - `logits = model(video_tensor)` → shape `(1, num_classes, T')` for I3D.
     - Max-pool over temporal dim → `(1, num_classes)`.
     - `probs = softmax(logits)`.
     - `top_probs, top_indices = topk(probs, k=5)`.
   - Map indices to glosses using the label map JSON.

6. **Response:**
   - Return `PredictionResponse` with:
     - `sign` – top-1 label.
     - `confidence` – top-1 probability.
     - `top_k` – up to 5 `{sign, confidence}` entries.
     - `message` – optional status text.

7. **Logging (optional):**
   - `firebase_service.save_translation(session_id, data)` to Firestore (not enabled by default).

**Implementation checklist (backend):**
- [x] FastAPI app with lifespan model loading.
- [x] I3D model + label map loading (`model_service.py`), optional S3 download.
- [x] Video preprocessing aligned with training (`preprocessing.py`).
- [x] `/health`, `/ready`, `POST /api/v1/predict` endpoints.
- [x] 100% test coverage on `app/` (see `docs/TESTING.md`).
- [ ] Firebase integration (optional — wired but not enabled by default).

---

#### 3.3 ML Model (`ml/`)

**Tech:** PyTorch, Torchvision.

**Goal:** Classify a **short video clip** of an isolated ASL sign into one of the trained gloss classes (48 MVP glosses for the deployed I3D model; broader vocabulary possible with other backbones).

**Key files:**
- `ml/i3d_msft/pytorch_i3d.py` – **Inception I3D** (deployed MVP model).
- `ml/i3d_label_map_mvp-sft-full-v1.json` – 48-class MVP label map.
- `ml/models/classifier.py` – `ASLVideoClassifier` (in-repo torchvision baseline).
- `ml/config.py` – central configuration (data, model, training).
- `ml/training/dataset.py` – `ASLVideoDataset` class.
- `ml/training/train.py` – training script.
- `ml/evaluation/evaluate.py` – evaluation and confusion analysis.

**Deployed architecture (Inception I3D):**
- Input: `(B, 3, 64, 224, 224)` video clips, normalized to `[-1, 1]`.
- 3D Inception modules with temporal convolutions.
- Output: logits `(B, num_classes, T')`, max-pooled over time.
- Trained on ASL Citizen with signer-disjoint splits.
- See `docs/PREPROCESSING.md` for inference preprocessing.

**In-repo baseline (ASLVideoClassifier) — `ml/models/classifier.py`:**
- Wraps torchvision 3D backbones (R3D-18, MC3-18, R(2+1)D-18) pretrained on Kinetics-400.
- Input: `(B, 3, 16, 224, 224)` — 16-frame video clips.
- Dropout + linear classification head.

**NOTE:** The old steps 2–5 below are from an earlier ResNet18+Transformer design and do not reflect the current model. They are retained for historical reference only.

2. **Classification head (legacy)**:
   - 1×1 conv reduces channels: `(B, C, 7, 7)` → `(B, d_model, 7, 7)` (e.g., `d_model = 256`).
   - Flatten spatial dims: `(B, d_model, 7*7)` → `(B, 49, d_model)`.

3. **CLS token + positional encoding**:
   - Prepend a learnable `[CLS]` token → sequence length 50.
   - Add sinusoidal positional encoding to each position.

4. **Transformer encoder** (2 layers, 4 heads):
   - Multi-head self-attention over the 50 tokens.
   - Feed-forward MLP, layernorm, residuals.
   - Output: `(B, 50, d_model)`.

5. **Classification head**:
   - Take the `[CLS]` token output: `(B, d_model)`.
   - `LayerNorm + Linear(d_model → num_classes)`.
   - Output logits: `(B, num_classes)`.

**Training strategy:**
- **Backbone freeze** for first N epochs (e.g., 2–3): only the classification head trains.
- Then **unfreeze** backbone for fine-tuning.
- Optimizer: `AdamW(lr=1e-3, weight_decay=1e-4)`.
- Scheduler: `CosineAnnealingLR` over 30 epochs.
- Early stopping: if validation accuracy doesn’t improve for 5 epochs.
- Checkpoints: `ml/checkpoints/best_model.pt` + periodic `epoch_X.pt`.

**Dataset (`ASLVideoDataset`):**
- Expected layout (see `docs/data_schema.md`):

  ```text
  data/processed/
    clips/
      train/hello/*.mp4
      train/goodbye/*.mp4
      ...
      val/hello/*.mp4
      test/hello/*.mp4
    label_map.json   # {"hello": 0, "goodbye": 1, ...}
  ```

- Augmentations (train only): random crop, color jitter, small rotations.
- **No horizontal flip** (ASL hands are not mirror-symmetric).

**Training steps:**
1. Prepare processed video clips + `label_map.json` via the data pipeline.
2. `cd ml && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`.
3. Run `python -m training.train`.
4. Evaluate: `python -m evaluation.evaluate --checkpoint checkpoints/best_model.pt`.
5. Point backend `MODEL_PATH` / `LABEL_MAP_PATH` at the artifacts.

---

#### 3.4 Data Pipeline & Datasets (`data/`)

**Goal:** Turn raw ASL video datasets into labeled **video clips** for training.

**Key files:**
- `data/scripts/ingest_asl_citizen.py` – ingests ASL Citizen metadata.
- `data/scripts/ingest_wlasl.py` – ingests WLASL metadata.
- `data/scripts/ingest_msasl.py` – ingests MS-ASL metadata.
- `data/scripts/preprocess_clips.py` – produces uniform-length mp4 clips.
- `data/scripts/build_unified_dataset.py` – merges sources and writes label maps.
- `data/scripts/validate.py` – sanity checks.
- `data/scripts/build_mvp_dataset.py` – MVP-filtered subset.
- `docs/data_pipeline.md` – detailed pipeline documentation.
- `docs/data_schema.md` – data layout and schemas.

**Datasets used:**
- **ASL Citizen** – 2.7K glosses, ~83K videos, 52 signers (primary train/val/test, signer-disjoint).
- **WLASL** – 2K glosses, ~21K videos (supplementary training).
- **MS-ASL** – 1K glosses, ~25K videos (supplementary training).

**Pipeline steps (WLASL example):**
1. `download_wlasl.py`:
   - Downloads metadata JSON from GitHub.
   - Filters entries to the configured `target_vocab` from `ml/config.py` (greetings, restaurant, medical, etc.).
   - Builds `label_map.json` for those glosses.

2. Video acquisition:
   - Download WLASL videos (e.g., from official repo) into `data/raw/wlasl/videos/`.

3. Frame extraction (`extract_frames`):
   - For each video in the target gloss set:
     - Open with OpenCV.
     - Compute evenly spaced frame indices (up to 5 per video).
     - Save frames under `data/processed/images/train/<gloss>/frame_xxxxx.jpg`.

4. (Optional) Hand cropping (`preprocess.py` + MediaPipe Hands):
   - Run hand detection.
   - Crop to bounding box + padding.
   - Replace/augment frames with cropped images.

5. Train/Val/Test split:
   - `split_dataset(images_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)`.
   - Copies files into `train/`, `val/`, `test/` subdirs per class.

6. Stats and sanity check (`compute_dataset_stats`):
   - Prints images per class and total counts.

---

#### 3.5 Firebase / Database (`docs/data_schema.md`)

**Tech:** Firebase **Firestore** (NoSQL).

**Collections (MVP + future):**
- `translations` – each prediction made:
  - `session_id`, `predicted_sign`, `confidence`, `top_k`, `timestamp`, `image_hash`, `feedback`.
- `sessions` (future) – basic device/session info.
- `vocabulary` (future) – metadata for each sign used in learning mode.

**Why Firestore?**
- Managed service with a free tier suitable for early-scale workloads.
- Easy to integrate via `firebase-admin` SDK in the backend.
- Good fit for simple document-style data (no heavy relational requirements).

Implementation steps are in **Section 3.2 Backend API** above.

---

### 4. End-to-End Data Flow (From Camera to Prediction)

**Online (runtime) path:**

1. The app opens on iOS (or Android with Expo Go) and navigates to the **Camera** screen.
2. The app requests camera permission if needed.
3. The signer performs a word (e.g., "water") in front of the camera.
4. The user taps **Record Sign** (~3 s video clip).
5. The app:
   - Records a short **video** (e.g. MP4) from the camera.
   - Wraps it in `FormData` under `file`.
   - Sends it to the backend: `POST /api/v1/predict`.

6. FastAPI backend:
   - Validates the uploaded file.
   - Preprocesses video into an I3D-ready tensor `(1, 3, 64, 224, 224)` (see `docs/PREPROCESSING.md`).
   - Runs **Inception I3D** inference.
   - Returns logits → top-1 and optional top-k over the label map.

7. Backend returns JSON with `sign`, `confidence`, `top_k`.
8. The app updates the UI with the prediction.
9. If confidence is high enough, the app uses TTS so the device **speaks** the English gloss.
10. (Optional) Backend writes a `translations` document to Firestore; the **History** screen can sync from there.

**Offline (training) path:**
- Separate from the runtime flow:
  1. Download and process datasets (e.g. ASL Citizen, WLASL, MS-ASL) via `data/scripts/`.
  2. Train models (see `ml/README.md` and training branch notes in `docs/DEVELOPER_GUIDE.md`).
  3. Evaluate and iterate.
  4. Export `best_model.pt` and the versioned label map JSON used by inference.
  5. Point backend `MODEL_PATH` / `LABEL_MAP_PATH` at the new artifacts (or S3).

---

### 5. Use Cases & How the System Supports Them

#### 5.1 Restaurant Ordering

**User story:**  
A Deaf customer signs "water" to a waiter who doesn’t know ASL. The waiter opens Eye Hear U, points the camera at the customer, taps **Record Sign**, sees "water" on the screen and hears it spoken aloud.

**Technical flow:**
- Mobile camera (short video) → `POST /predict` → backend model predicts "water" with high confidence → app displays and speaks "water".
- In future, the waiter can tap a “correct” button; backend stores this as positive feedback.

#### 5.2 Medical Triage

**User story:**  
In an urgent care clinic, a Deaf patient signs "pain" and "medicine". The nurse uses Eye Hear U as a quick helper to understand which basic needs the patient is signaling.

**Technical changes for this use case:**
- Vocabulary includes **medical-focused signs**: `pain`, `hurt`, `emergency`, `doctor`, `medicine`, `allergic`, `sick`.
- Same flow: video recording, `/predict`, TTS.
- Evaluation plan: design a **scenario script** (simulate nurse–patient interactions) and measure accuracy on that subset.

#### 5.3 ASL Learner Self-Check

**User story:**  
An ASL student practicing at home signs "thank you" into the front camera and uses Eye Hear U to check whether the model recognizes the sign.

**Technical flow:**
- Front-facing camera captures the signer themselves.
- The model recognizes the sign.
- If the app often misclassifies it as a similar sign (e.g., "sorry"), the learner knows to adjust their form.
- Future enhancement: show top-2 predictions and highlight common confusions.

#### 5.4 Fingerspelling Fallback (Names, Unknown Words)

**User story:**  
Someone fingerspells "J-O-H-N" because the app doesn’t know that full word as one sign.

**Technical flow:**
- Each letter is a separate capture → prediction.
- History shows a recent sequence: J, O, H, N.
- Future enhancement: client-side grouping to display the aggregated word.

---

### 6. Implementation checklist (by area)

#### 6.1 Mobile (Expo)

1. **Environment:** Node 20+, Xcode + Simulator and/or a device with Expo Go.
2. **Run:** `cd mobile && npm install`; prefer `npm run start:lan` when the API is on the same Wi‑Fi.
3. **API URL:** set `EXPO_PUBLIC_API_URL` in `mobile/.env` (not hard-coded in screens). Simulator on the same Mac: `http://127.0.0.1:8000`. Physical device: `http://<API-host-LAN-IP>:8000`.
4. **iOS:** enable **Local Network** for Expo Go if the bundle fails to load (Settings → Privacy & Security → Local Network).
5. **UX:** loading and error states during inference; optional onboarding for framing.
6. **History:** AsyncStorage (current default); optional Firestore sync and feedback UI.

#### 6.2 Backend (FastAPI)

1. **Setup:** `cd backend`, venv, `pip install -r requirements.txt`, `cp .env.example .env`.
2. **Run:** `export PYTHONPATH=..` (repo root) and `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`.
3. **Health:** verify `GET /health` and `GET /ready`.
4. **Model:** `model_service.py` loads I3D weights and the JSON label map; see `docs/DEVELOPER_GUIDE.md` for paths and S3.
5. **Optional Firestore:** service account + `FIREBASE_*` env; wire routes if cloud history is required.

#### 6.3 ML & data

1. **Data:** run or extend `data/scripts/` pipeline; maintain signer-disjoint evaluation where applicable.
2. **Training:** `ml/training/train.py` (R3D-style stack in-repo) and/or the **I3D** training branch referenced in `docs/DEVELOPER_GUIDE.md` — inference must match the exported checkpoint.
3. **Evaluate:** `ml/evaluation/evaluate.py` (and any I3D eval scripts on the training branch).
4. **Deploy artifacts:** versioned `best_model.pt` + label map JSON; update backend env or S3 layout.

---

### 7. Summary

- Eye Hear U is a **three-tier system**: React Native (Expo) mobile app, FastAPI backend, and a **video** classifier (**Inception I3D** at inference for the MVP checkpoint).
- The app’s **core promise** is isolated-sign ASL→English output with optional speech.
- This document summarizes how components fit together; for preprocessing details see `docs/PREPROCESSING.md`, and for day-to-day commands see `docs/DEVELOPER_GUIDE.md`.

For low-level data schemas (Firestore fields, dataset layouts), see `docs/data_schema.md`. EOF