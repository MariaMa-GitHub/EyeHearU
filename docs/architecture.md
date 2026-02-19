## Eye Hear U – System Architecture & Use Cases

**Version:** 2026-02-10  
**Owners:** Maria (iOS/frontend), Zhixiao (backend/db), Siyi & Chloe (ML)

---

### 1. Problem & High-Level Idea

**Problem:** There is no simple, reliable tool for translating **single ASL signs into English text/speech in real time** using only a phone camera. This blocks everyday communication (e.g., restaurant orders, quick medical questions) and makes it hard for ASL learners to verify their signing.

**Solution:** Eye Hear U is an **iOS-focused mobile app** where a user:
- Opens the app and points the camera at a signer (themselves or someone else),
- Taps **Capture**,
- Sees the **predicted English word/letter and confidence**, and
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
                                │  CNN backbone        │
                                │  (ResNet18)          │
                                │       ↓              │
                                │  Patch projection    │
                                │       ↓              │
                                │  Transformer encoder │
                                │  (2 layers, 4 heads) │
                                │       ↓              │
                                │  Classification head │
                                │  → ~62 classes       │
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
- Capture a **single frame** on demand (when user taps **Capture**).
- Upload the image to the backend (`/api/v1/predict`).
- Display predicted sign + confidence.
- Use **TTS** (`expo-speech`) to read out the predicted word.
- (Future) Show per-session **translation history** and collect “correct/wrong” feedback.

**Important files:**
- `mobile/app/_layout.tsx` – navigation shell using `expo-router`.
- `mobile/app/index.tsx` – **Home screen** (start translating / view history).
- `mobile/app/camera.tsx` – **Camera + prediction screen**.
- `mobile/app/history.tsx` – **History UI** (currently uses placeholder data).
- `mobile/services/api.ts` – typed API client for the backend.

**Camera flow (step-by-step):**
1. **Permissions** – `useCameraPermissions()` from `expo-camera`:
   - If status is unknown, app shows a screen explaining why camera is needed with a “Grant Permission” button.
   - If permission denied, user sees the same screen until they grant it in settings.

2. **Live preview** – `CameraView` component:
   - Props: `facing="front"`, full-screen style.
   - This shows the live feed from the front-facing camera.

3. **Capture & upload** (in `captureAndPredict`):
   - Call `cameraRef.current.takePictureAsync({ quality: 0.8, base64: false })`.
   - This returns a `photo` object with a local `uri` to the JPEG file.
   - Wrap that in `FormData` under the `file` key:
     - `formData.append("file", { uri, name: "sign.jpg", type: "image/jpeg" })`.
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
- [x] Implement camera screen with capture & upload logic.
- [x] Implement TTS via `expo-speech`.
- [ ] Wire real backend URL (for physical devices, use LAN IP instead of `localhost`).
- [ ] Persist predictions locally (AsyncStorage) + pull from Firebase history.

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
- `backend/app/services/preprocessing.py` – image preprocessing.
- `backend/app/services/model_service.py` – model loading + inference skeleton.
- `backend/app/services/firebase_service.py` – Firestore integration.
- `backend/app/schemas/prediction.py` – response schema.

**Prediction request flow:**
1. **Client** sends `multipart/form-data` POST:
   - `file`: JPEG/PNG/WebP image.

2. **FastAPI** parses it into `UploadFile`:

   ```py
   @router.post("/predict", response_model=PredictionResponse)
   async def predict_sign(file: UploadFile = File(...)):
   ```

3. **Validation:**
   - Check `content_type` is one of `{"image/jpeg", "image/png", "image/webp"}`.
   - Read file bytes and verify non-empty.

4. **Preprocessing** (`preprocess_image`):
   - `PIL.Image.open` on bytes → RGB image.
   - Resize to **224x224**.
   - Convert to `float32` array in `[0,1]`.
   - Normalize with **ImageNet** mean/std.
   - Reorder channels HWC → CHW, add batch dimension: shape `(1, 3, 224, 224)`.

5. **Model inference** (`model_service.predict`):
   - With `torch.no_grad()`:
     - `logits = model(image_tensor)` → shape `(1, num_classes)`.
     - `probs = softmax(logits)`.
     - `top_probs, top_indices = topk(probs, k=5)`.
   - Map indices to labels using `label_map.json` (e.g., index 26 → "hello").

6. **Response:**
   - Return `PredictionResponse` with:
     - `sign` – top-1 label.
     - `confidence` – top-1 probability.
     - `top_k` – up to 5 `{sign, confidence}` entries.
     - `message` – for debug/placeholder text.

7. **Logging (future):**
   - Call `firebase_service.save_translation(session_id, data)` to store in `translations` collection.

**Implementation steps (backend):**
1. **Install & run locally:**
   - `cd backend`
   - `python -m venv venv && source venv/bin/activate`
   - `pip install -r requirements.txt`
   - `cp .env.example .env` and fill in `MODEL_PATH`, `FIREBASE_*` later.
   - `uvicorn app.main:app --reload --port 8000`.

2. **Wire model loading:**
   - After model is trained and saved to `ml/checkpoints/best_model.pt`:
     - Implement `load_model` in `model_service.py` using `ASLClassifier`.
     - In `main.py` startup event:

       ```py
       from app.services.model_service import load_model
       app.state.model = load_model(settings.model_path, settings.model_device)
       ```

   - In `predict_sign`, import `preprocess_image` + `predict` and use `app.state.model`.

3. **Enable Firebase:**
   - Create Firebase project and Firestore DB.
   - Add `firebase-credentials.json` to `backend/`.
   - Set `FIREBASE_CREDENTIALS_PATH` and `FIREBASE_PROJECT_ID` in `.env`.
   - Call `init_firebase()` on startup and `save_translation` for each prediction.

---

#### 3.3 ML Model (`ml/`)

**Tech:** PyTorch, Torchvision.

**Goal:** Classify a **single 224x224 RGB image** into one of ~62 classes (scenario-specific vocabulary + letters A–Z + numbers 1–10).

**Key files:**
- `ml/models/classifier.py` – **ASLClassifier**.
- `ml/config.py` – central configuration (data, model, training).
- `ml/training/dataset.py` – `ASLImageDataset` class.
- `ml/training/train.py` – training script.
- `ml/evaluation/evaluate.py` – evaluation and confusion analysis.

**Architecture (ASLClassifier):**
1. **CNN Backbone** – e.g., `resnet18`:
   - Input: `(B, 3, 224, 224)`.
   - Output: `(B, C, 7, 7)` feature maps (`C = 512` for ResNet18).

2. **Patch projection**:
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
- **Backbone freeze** for first N epochs (e.g., 2): only Transformer + head train.
- Then **unfreeze** backbone for fine-tuning.
- Optimizer: `AdamW(lr=1e-3, weight_decay=1e-4)`.
- Scheduler: `CosineAnnealingLR` over 30 epochs.
- Early stopping: if validation accuracy doesn’t improve for 5 epochs.
- Checkpoints: `ml/checkpoints/best_model.pt` + periodic `epoch_X.pt`.

**Dataset (`ASLImageDataset`):**
- Expected layout (see `docs/data_schema.md`):

  ```text
  data/processed/
    images/
      train/hello/*.jpg
      train/goodbye/*.jpg
      ...
      val/hello/*.jpg
      test/hello/*.jpg
    label_map.json   # {"hello": 0, "goodbye": 1, ...}
  ```

- Augmentations (train only):
  - Resize slightly larger then random crop to `image_size`.
  - Color jitter.
  - Small rotations.
  - **No horizontal flip** (ASL hands are not mirror-symmetric).

**Training steps:**
1. Prepare processed images + `label_map.json` via the data pipeline (below).
2. `cd ml && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`.
3. Run `python -m training.train`.
4. Evaluate: `python -m evaluation.evaluate --checkpoint checkpoints/best_model.pt`.
5. Copy `best_model.pt` + `label_map.json` to a location readable by the backend (`MODEL_PATH`).

---

#### 3.4 Data Pipeline & Datasets (`data/`)

**Goal:** Turn raw ASL video datasets (e.g., **WLASL**) into many labeled image frames for training.

**Key files:**
- `data/scripts/download_wlasl.py` – downloads metadata + builds label map + extracts frames.
- `data/scripts/preprocess.py` – optional MediaPipe hand cropping + train/val/test split + stats.
- `docs/data_schema.md` – describes data layout and aspirational vs actual datasets.

**Datasets considered:**
- **WLASL** – 2K glosses, ~21K videos (primary training set).
- **ASL Citizen** – 2.7K glosses, 84K videos (robustness / varied background).
- **MS-ASL**, **ASL-LEX** – supplementary.

**Pipeline steps (WLASL example):**
1. `download_wlasl.py`:
   - Downloads metadata JSON from GitHub.
   - Filters entries to only your `target_vocab` from `ml/config.py` (greetings, restaurant, medical, etc.).
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
- Managed, free at your scale.
- Easy to integrate via `firebase-admin` SDK in the backend.
- Good fit for simple document-style data (no heavy relational requirements).

Implementation steps are in **Section 3.2 Backend API** above.

---

### 4. End-to-End Data Flow (From Camera to Prediction)

**Online (runtime) path:**

1. User opens the app on iOS and navigates to the **Camera** screen.
2. App requests camera permission if needed.
3. User signs a word (e.g., "water") in front of the camera.
4. User taps **Capture**.
5. The app:
   - Takes a JPEG photo from the camera preview.
   - Wraps it in `FormData` under `file`.
   - Sends it to the backend: `POST /api/v1/predict`.

6. FastAPI backend:
   - Validates the uploaded file.
   - Preprocesses the image into a model-ready tensor.
   - Runs the CNN+Transformer model for inference.
   - Gets probabilities over all sign classes.
   - Picks top-1 and top-5 predictions.

7. Backend returns JSON with `sign`, `confidence`, `top_k`.
8. App updates UI with the prediction.
9. If the confidence is high enough, app calls TTS so the phone **speaks** the English word.
10. (Future) Backend writes a `translations` document to Firestore; app pulls this into the **History** screen.

**Offline (training) path:**
- Completely separate from user-facing flow, run by ML engineers:
  1. Download + process WLASL and other datasets.
  2. Train the ASLClassifier on images.
  3. Evaluate and iterate model.
  4. Export `best_model.pt` and `label_map.json`.
  5. Deploy/model file path updated in backend.

---

### 5. Use Cases & How the System Supports Them

#### 5.1 Restaurant Ordering

**User story:**  
A Deaf customer signs "water" to a waiter who doesn’t know ASL. The waiter opens Eye Hear U, points the camera at the customer, taps **Capture**, sees "water" on the screen and hears it spoken aloud.

**Technical flow:**
- Mobile camera → `POST /predict` → backend model predicts "water" with high confidence → app displays and speaks "water".
- In future, the waiter can tap a “correct” button; backend stores this as positive feedback.

#### 5.2 Medical Triage

**User story:**  
In an urgent care clinic, a Deaf patient signs "pain" and "medicine". The nurse uses Eye Hear U as a quick helper to understand which basic needs the patient is signaling.

**Technical changes for this use case:**
- Vocabulary includes **medical-focused signs**: `pain`, `hurt`, `emergency`, `doctor`, `medicine`, `allergic`, `sick`.
- Same flow: camera capture, `/predict`, TTS.
- Evaluation plan: design a **scenario script** (simulate nurse–patient interactions) and measure accuracy on that subset.

#### 5.3 ASL Learner Self-Check

**User story:**  
An ASL student practicing at home signs "thank you" into the front camera and uses Eye Hear U to verify they’re signing it correctly.

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

### 6. Implementation Roadmap by Role

#### 6.1 Maria – Mobile / iOS Frontend

1. **Confirm mobile environment:**
   - Node 20 (>= 20.19.4 ideally).
   - Xcode + iOS simulator or physical iPhone with Expo Go.

2. **Run the app:**
   - `cd mobile && npm install` (already done once, re-run if deps change).
   - `npx expo start`.

3. **Connect to backend:**
   - In `app/camera.tsx`, set `API_BASE_URL`:
     - **Simulator (same machine):** `http://localhost:8000`.
     - **Physical device:** `http://<your-laptop-LAN-IP>:8000`.

4. **Polish UX:**
   - Nice loading state while model is predicting (`Processing...`).
   - Clear error states if backend is unreachable.
   - Simple onboarding screen explaining best framing for signs.

5. **History & feedback:**
   - Persist predictions with timestamps to AsyncStorage.
   - Sync with Firestore once `save_translation` is live.
   - Add optional **"correct" / "wrong"** buttons per history item.

#### 6.2 Zhixiao – Backend & Database

1. **Backend setup:**
   - `cd backend && python -m venv venv && source venv/bin/activate`.
   - `pip install -r requirements.txt`.
   - `cp .env.example .env` and configure values.

2. **Health endpoints:**
   - Confirm `GET /health` and `GET /ready` work via browser or curl.

3. **Model integration:**
   - Once `best_model.pt` and `label_map.json` exist:
     - Implement `load_model` + `predict` in `model_service.py` using `ASLClassifier` from `ml/models`.
     - Store `label_map` in memory for index→label mapping.

4. **Logging to Firestore:**
   - Configure `firebase-credentials.json` and `.env`.
   - Call `init_firebase` on startup; use `save_translation` for each prediction.
   - Expose a simple `GET /api/v1/history?session_id=...` endpoint to power the mobile history screen.

#### 6.3 Siyi & Chloe – ML / Data

1. **Data audit:**
   - Run `download_wlasl.py` to inspect coverage of target vocab.
   - Identify missing glosses → plan ASL Citizen/custom recordings.

2. **Pipeline execution:**
   - Extract frames, optionally crop hands, create train/val/test splits.
   - Ensure balanced-ish class counts (or at least understand imbalance).

3. **Train baseline model:**
   - Use default config in `ml/config.py`.
   - Run `python -m training.train` and monitor logs.
   - Iterate hyperparameters, backbones, and augmentations.

4. **Evaluate & error analyze:**
   - Use `evaluation/evaluate.py` to get per-class accuracy and confusion pairs.
   - Focus on classes used in restaurant/medical scenarios first.

5. **Deploy model:**
   - Export `best_model.pt` and final `label_map.json`.
   - Work with Zhixiao to update backend `MODEL_PATH` and reload.

---

### 7. Summary

- Eye Hear U is a **three-tier system**: React Native mobile app, FastAPI backend, and a CNN+Transformer ASL classifier trained on WLASL-style data.
- The app’s **core promise** is single-sign, scenario-focused ASL→English translation with optional audio.
- This document should be the go-to reference for:
  - How components fit together.
  - Which files to touch for which feature.
  - The step-by-step flows for both runtime use and ML training.

For low-level data schemas (Firestore fields, dataset layouts), see `docs/data_schema.md`. EOF