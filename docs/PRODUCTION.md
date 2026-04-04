# Eye Hear U — Production Deployment

Moving from **local or tunnel demos** to a **stable production** setup. For local development, see the [Developer guide](DEVELOPER_GUIDE.md) and root [README](../README.md).

## Target architecture (AWS — matches Terraform)

1. **Container image** — Backend Dockerfile builds FastAPI + `ml/i3d_msft` + PyTorch CPU (or GPU if the base image is changed).  
2. **Amazon ECR** — Image registry.  
3. **ECS Fargate** — Service behind an **Application Load Balancer**.  
4. **S3** — `best_model.pt` (and optional label map) — task downloads at startup or mount from EFS.  
5. **IAM task role** — Grant `s3:GetObject` on the model prefix (**no long-lived keys in the container**).  
6. **Secrets** — If Firebase is used, inject JSON via **Secrets Manager** or SSM, not git.  
7. **Mobile app** — Point the production API base URL at **`https://api.eyehearu.app`** (or the ALB DNS name) in the **production** build (`__DEV__ === false` in `api.ts`).

## Backend checklist

| Step | Action |
|------|--------|
| 1 | Build and push image: `docker build -t eyehearu-api .` from repo root (see root `Dockerfile`). |
| 2 | Set env vars on ECS task: `MODEL_PATH`, `LABEL_MAP_PATH` (or bake label map into image), `MODEL_DEVICE=cpu`, `AWS_S3_*` if downloading weights. |
| 3 | Configure **health checks** on `/health` (liveness) and optionally `/ready` (readiness after model load). |
| 4 | Scale: start with **1 task**; increase CPU/memory if inference latency is high (I3D is heavy on CPU). |
| 5 | Logging: CloudWatch log group for the service; alert on elevated 5xx rate. |

## Model artifacts

- **Weights:** `s3://eye-hear-u-public-data-ca1/.../best_model.pt` (or a dedicated production bucket).  
- **Label map:** Prefer **versioned** JSON in the image or S3 next to the checkpoint so class indices never drift.  
- **Gloss LM:** Ship `backend/data/gloss_lm.json` with the image (or bake via `scripts/build_gloss_lm.py` in the build). Override path via **`gloss_lm_path`** in settings / env if you relocate it. Multi-clip **`/predict/sentence`** reads this at startup (if missing or corrupt, a **uniform** LM over the loaded label map is used).  
- **Cold start:** First request after deploy may be slow while the model loads — consider **min capacity 1** and a **warmup** request to `/ready`.

## Mobile app (production)

1. **EAS Build** or **expo prebuild** → App Store / TestFlight.  
2. Set **production API URL** in `api.ts` (non-`__DEV__` branch).  
3. Optionally remove **dev-only** UI (e.g. backend status dot).  
4. **ATS / HTTPS:** Production API must use **valid TLS** (ALB + ACM certificate).

## Security

- **Never commit** AWS keys, Firebase JSON, or `.env` files containing secrets.  
- Rotate credentials if they were pasted into chat or committed.  
- **CORS:** Replace `cors_origins: ["*"]` in `Settings` with the app’s web origin or bundle ID scheme if a web client is added.  
- **Rate limiting** — Add API Gateway, CloudFront, or middleware before public launch.

## Cost / ops notes

- **Fargate + ALB** incurs baseline monthly cost even at low traffic.  
- **S3 egress** — Model download once per task start; use a same-region bucket.  
- For low-traffic demos, **single EC2 + Docker** or **Railway/Render** with an uploaded `best_model.pt` can be cheaper than full ECS at the cost of operational polish.

## Demo vs production summary

| Concern | Local / class demo | Production |
|---------|-------------------|------------|
| API URL | Tunnel or LAN IP | HTTPS + DNS |
| TLS | Often HTTP | ACM cert on ALB |
| Model | Local `model_cache/` | S3 + IAM role |
| Secrets | `.env` on dev machine | Secrets Manager / SSM |
| Mobile | Expo Go + tunnel | Store build |
