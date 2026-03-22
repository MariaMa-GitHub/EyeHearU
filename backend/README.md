## Backend (`backend/`) — FastAPI ASL inference API

FastAPI service for Eye Hear U: **health/readiness**, **video upload**, and **I3D inference**.

---

### Key files

| Path | Purpose |
|------|---------|
| `app/main.py` | App factory, CORS, router registration, **lifespan** loads the model at startup |
| `app/config.py` | Pydantic settings from `.env` (`MODEL_PATH`, `LABEL_MAP_PATH`, `MODEL_DEVICE`, AWS S3, optional Firebase) |
| `app/routers/health.py` | `GET /health`, `GET /ready` |
| `app/routers/predict.py` | `POST /api/v1/predict` — multipart field `file` (**mp4/mov** video) |
| `app/services/preprocessing.py` | Video bytes → tensor `(1, 3, 64, 224, 224)` — must match training (see `docs/PREPROCESSING.md`) |
| `app/services/model_service.py` | Load I3D checkpoint + label map, optional S3 download, `predict()` |
| `app/services/firebase_service.py` | Optional Firestore helpers |
| `tests/` | Pytest suite; **100%** line/branch coverage on `app/` (see `docs/TESTING.md`) |

---

### Run locally

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # set MODEL_PATH, LABEL_MAP_PATH, etc.

export PYTHONPATH=..        # repo root — required for `ml` imports
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- `http://127.0.0.1:8000/health` → `{"status":"ok"}`  
- `http://127.0.0.1:8000/docs` → Swagger UI  

Physical devices on the LAN need `--host 0.0.0.0` and the host’s LAN IP in `mobile/.env` as `EXPO_PUBLIC_API_URL`.

---

### Model artifacts

- Default label map: `../ml/i3d_label_map_mvp-sft-full-v1.json`
- Weights: `model_cache/best_model.pt` or path in `.env`; S3 download if configured

See **`docs/DEVELOPER_GUIDE.md`** for the full download command and environment variables.

---

### Integration

- **Mobile:** `POST /api/v1/predict` with a **video** file; optional `/health` / `/ready` for status UI.
- **ML:** Checkpoint and label map must match the **I3D** architecture in `ml/i3d_msft/`.
