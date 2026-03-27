"""
Application configuration.
Loads settings from environment variables / .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # App
    app_name: str = "Eye Hear U API"
    debug: bool = False

    # Firebase (optional — history feature)
    firebase_credentials_path: str = "firebase-credentials.json"
    firebase_project_id: str = ""

    # ML Model
    model_path: str = "model_cache/best_model.pt"
    label_map_path: str = "../ml/i3d_label_map_mvp-sft-full-v1.json"
    model_device: str = "cpu"  # "cpu" or "cuda" or "mps"
    # Gloss n-gram LM for multi-clip sentence decoding (relative to backend root)
    gloss_lm_path: str = "data/gloss_lm.json"

    # S3 model source
    aws_s3_bucket: str = "eye-hear-u-public-data-ca1"
    aws_s3_region: str = "ca-central-1"
    aws_s3_model_key: str = "models/i3d/modal/candidate-ac-eval-v4/20260325T001403Z/best_model.pt"
    aws_s3_train_csv_key: str = "processed/mvp/i3d/split_plans/candidate-ac-eval-v4/splits/train.csv"

    # CORS — allowed origins for the mobile app
    cors_origins: list[str] = ["*"]


@lru_cache()
def get_settings() -> Settings:
    return Settings()
