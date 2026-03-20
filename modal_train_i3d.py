"""
Run EyeHearU vendored Microsoft I3D training on Modal GPU.

Usage:
  # 1) Create a Modal secret named "aws-credentials" with:
  #    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
  #
  # 2) Run:
  #    modal run modal_train_i3d.py --bucket eye-hear-u-dev-data --plan-id candidate-ac-eval-v2 --epochs 20
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import modal


def _parse_run_name() -> str:
    for i, arg in enumerate(sys.argv):
        if arg == "--run-name" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return ""


_run_name = _parse_run_name()
APP_NAME = f"eyehearu-i3d-{_run_name}" if _run_name else "eyehearu-i3d-train"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1")
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "opencv-python-headless>=4.9.0",
        "numpy>=1.26.0",
        "boto3>=1.28.0",
    )
    .add_local_dir("ml", remote_path="/root/ml")
)


@app.function(
    image=image,
    gpu="T4",
    timeout=60 * 60 * 12,  # 12h
    secrets=[modal.Secret.from_name("aws-credentials")],
)
def train_i3d_modal(
    bucket: str,
    region: str = "ca-central-1",
    plan_id: str | None = None,
    epochs: int = 20,
    batch_size: int = 6,
    num_workers: int = 2,
    clip_limit: int | None = None,
    init_checkpoint_s3_key: str | None = None,
    init_strict: bool = False,
    output_prefix: str = "models/i3d/modal",
    run_name: str | None = None,
) -> dict:
    import boto3

    run_id = run_name or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    workdir = Path("/root/ml")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(workdir)
    env["OPENCV_LOG_LEVEL"] = "ERROR"

    cmd = [
        sys.executable,
        "-m",
        "i3d_msft.train",
        "--bucket",
        bucket,
        "--region",
        region,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--device",
        "cuda",
        "--local-root",
        "workdir/i3d_msft",
    ]
    if plan_id:
        cmd.extend(["--plan-id", plan_id])
    if clip_limit is not None:
        cmd.extend(["--clip-limit", str(clip_limit)])
    if init_checkpoint_s3_key:
        cmd.extend(["--init-checkpoint-s3-key", init_checkpoint_s3_key])
    if init_strict:
        cmd.append("--init-strict")

    effective_plan = plan_id or "default"
    s3_ckpt_prefix = f"{output_prefix}/{effective_plan}/{run_id}"
    cmd.extend(["--s3-checkpoint-prefix", s3_ckpt_prefix])

    print("[modal] running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(workdir), check=True, env=env)

    # Resolve effective plan id (explicit or ACTIVE).
    effective_plan = plan_id
    if not effective_plan:
        s3 = boto3.client("s3", region_name=region)
        key = "processed/mvp/i3d/split_plans/ACTIVE_PLAN.json"
        obj = s3.get_object(Bucket=bucket, Key=key)
        effective_plan = __import__("json").loads(obj["Body"].read().decode("utf-8"))[
            "active_plan_id"
        ]

    ckpt_dir = workdir / "workdir" / "i3d_msft" / "checkpoints" / str(effective_plan)
    if not ckpt_dir.exists():
        raise RuntimeError(f"Checkpoint directory not found: {ckpt_dir}")

    s3 = boto3.client("s3", region_name=region)
    uploaded = []
    for path in ckpt_dir.rglob("*.pt"):
        rel = path.relative_to(ckpt_dir).as_posix()
        s3_key = f"{output_prefix}/{effective_plan}/{run_id}/{rel}"
        s3.upload_file(str(path), bucket, s3_key)
        uploaded.append(f"s3://{bucket}/{s3_key}")

    meta = {
        "run_id": run_id,
        "plan_id": effective_plan,
        "bucket": bucket,
        "region": region,
        "epochs": epochs,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "clip_limit": clip_limit,
        "init_checkpoint_s3_key": init_checkpoint_s3_key,
        "init_strict": init_strict,
        "uploaded_checkpoints": uploaded,
    }
    meta_key = f"{output_prefix}/{effective_plan}/{run_id}/run_metadata.json"
    s3.put_object(
        Bucket=bucket,
        Key=meta_key,
        Body=__import__("json").dumps(meta, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

    return {
        "run_id": run_id,
        "plan_id": effective_plan,
        "checkpoints": uploaded,
        "metadata": f"s3://{bucket}/{meta_key}",
    }


@app.local_entrypoint()
def main(
    bucket: str = "eye-hear-u-public-data-ca1",
    region: str = "ca-central-1",
    plan_id: str = "candidate-ac-eval-v2",
    epochs: int = 20,
    batch_size: int = 6,
    num_workers: int = 2,
    clip_limit: int = 0,
    init_checkpoint_s3_key: str = "",
    init_strict: bool = False,
    run_name: str = "",
):
    result = train_i3d_modal.remote(
        bucket=bucket,
        region=region,
        plan_id=plan_id,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        clip_limit=(None if clip_limit <= 0 else clip_limit),
        init_checkpoint_s3_key=(None if not init_checkpoint_s3_key else init_checkpoint_s3_key),
        init_strict=init_strict,
        run_name=(None if not run_name else run_name),
    )
    print(result)

