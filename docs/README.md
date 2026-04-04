# Eye Hear U — Documentation

All guides live in this folder. The [root README](../README.md) is the entry point for cloning, architecture, and “getting started.”

## Guides by audience

| Document | Who it’s for |
|----------|----------------|
| [USER_GUIDE.md](USER_GUIDE.md) | People using the mobile app (Expo Go or a build) |
| [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) | Engineers running API + app, changing code, LAN/tunnel setup |
| [ASL_TRANSLATION_PIPELINE.md](ASL_TRANSLATION_PIPELINE.md) | Anyone tuning single vs multi-clip behavior, beam/LM, `GLOSS_ENGLISH_MODE` |
| [TESTING.md](TESTING.md) | Running pytest/Jest locally, understanding CI and coverage |
| [PRODUCTION.md](PRODUCTION.md) | Deploying the API (AWS, Docker, security checklist) |
| [PREPROCESSING.md](PREPROCESSING.md) | Why inference preprocessing matches training (I3D tensor path) |
| [EVALUATION.md](EVALUATION.md) | Step-by-step metrics, plots, and API-based evaluation |
| [BENCHMARKING.md](BENCHMARKING.md) | Shorter reference for reproducing classifier benchmarks |
| [I3D_TRAINING_S3_REPRODUCTION.md](I3D_TRAINING_S3_REPRODUCTION.md) | S3 split plans, preparing data, training reproduction |
| [MODAL_AWS_SFT_MIGRATION.md](MODAL_AWS_SFT_MIGRATION.md) | AWS account migration, Modal GPU training, SFT warm-start |

## How these relate

- **Pipeline** vs **preprocessing:** [ASL_TRANSLATION_PIPELINE.md](ASL_TRANSLATION_PIPELINE.md) describes *what* the API does with clips and glosses; [PREPROCESSING.md](PREPROCESSING.md) describes *how* raw video becomes the I3D tensor.
- **Evaluation** vs **benchmarking:** [EVALUATION.md](EVALUATION.md) is the long-form tutorial; [BENCHMARKING.md](BENCHMARKING.md) is a compact reproduction summary — start with one and use the other for detail.
