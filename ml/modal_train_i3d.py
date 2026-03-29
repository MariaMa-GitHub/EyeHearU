"""
Run EyeHearU vendored Microsoft I3D training on Modal GPU.

Usage:
  # 1) Create a Modal secret named "aws-credentials" with:
  #    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
  #
  # 2) Run:
  #    modal run ml/modal_train_i3d.py --bucket eye-hear-u-dev-data --plan-id candidate-ac-eval-v4 --epochs 20
"""

from __future__ import annotations

import json
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


def _build_train_cmd(
    *,
    bucket: str,
    region: str,
    epochs: int,
    batch_size: int,
    num_workers: int,
    head_only_epochs: int,
    head_lr: float,
    backbone_lr: float,
    plan_id: str | None,
    clip_limit: int | None,
    init_checkpoint_s3_key: str | None,
    init_strict: bool,
    s3_ckpt_prefix: str,
) -> list[str]:
    cmd = [
        sys.executable, "-m", "i3d_msft.train",
        "--bucket", bucket,
        "--region", region,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--num-workers", str(num_workers),
        "--head-only-epochs", str(head_only_epochs),
        "--head-lr", str(head_lr),
        "--backbone-lr", str(backbone_lr),
        "--device", "cuda",
        "--local-root", "workdir/i3d_msft",
    ]
    if plan_id:
        cmd.extend(["--plan-id", plan_id])
    if clip_limit is not None:
        cmd.extend(["--clip-limit", str(clip_limit)])
    if init_checkpoint_s3_key:
        cmd.extend(["--init-checkpoint-s3-key", init_checkpoint_s3_key])
    if init_strict:
        cmd.append("--init-strict")
    cmd.extend(["--s3-checkpoint-prefix", s3_ckpt_prefix])
    return cmd


def _build_eval_cmd(
    *,
    bucket: str,
    region: str,
    plan_id: str,
    split: str,
    checkpoint_arg: str,
    checkpoint_is_local: bool,
    batch_size: int = 6,
    num_workers: int = 2,
    clip_limit: int | None = None,
    local_root: str = "workdir/i3d_msft_eval",
    output_json: str | None = None,
    gloss_dict_csv: str | None = None,
) -> list[str]:
    cmd = [
        sys.executable, "-m", "i3d_msft.evaluate",
        "--bucket", bucket,
        "--region", region,
        "--plan-id", str(plan_id),
        "--split", split,
        "--device", "cuda",
        "--local-root", local_root,
    ]
    if checkpoint_is_local:
        cmd.extend(["--checkpoint-local", checkpoint_arg])
    else:
        cmd.extend(["--checkpoint-s3-key", checkpoint_arg])
    cmd.extend(["--batch-size", str(batch_size)])
    cmd.extend(["--num-workers", str(num_workers)])
    if clip_limit is not None:
        cmd.extend(["--clip-limit", str(clip_limit)])
    if output_json:
        cmd.extend(["--output-json", output_json])
    if gloss_dict_csv:
        cmd.extend(["--gloss-dict-csv", gloss_dict_csv])
    return cmd


def _resolve_active_plan(s3, bucket: str) -> str:
    key = "processed/mvp/i3d/split_plans/ACTIVE_PLAN.json"
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))["active_plan_id"]


def _upload_checkpoints(
    s3, bucket: str, ckpt_dir: Path, output_prefix: str, plan_id: str, run_id: str
) -> list[str]:
    uploaded = []
    for path in ckpt_dir.rglob("*.pt"):
        rel = path.relative_to(ckpt_dir).as_posix()
        s3_key = f"{output_prefix}/{plan_id}/{run_id}/{rel}"
        s3.upload_file(str(path), bucket, s3_key)
        uploaded.append(f"s3://{bucket}/{s3_key}")
    return uploaded


def _upload_run_metadata(s3, bucket: str, meta: dict, meta_key: str) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=meta_key,
        Body=json.dumps(meta, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


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
    .add_local_dir(str(Path(__file__).resolve().parent), remote_path="/root/ml")
)


@app.function(
    image=image,
    gpu="T4",
    timeout=60 * 60 * 12,  # 12h
    secrets=[modal.Secret.from_name("aws-credentials")],
)
def train_i3d_modal(  # pragma: no cover – requires Modal GPU infrastructure
    bucket: str,
    region: str = "ca-central-1",
    plan_id: str | None = None,
    epochs: int = 20,
    batch_size: int = 6,
    num_workers: int = 2,
    clip_limit: int | None = None,
    head_only_epochs: int = 2,
    head_lr: float = 1e-3,
    backbone_lr: float = 1e-4,
    init_checkpoint_s3_key: str | None = None,
    init_strict: bool = False,
    output_prefix: str = "models/i3d/modal",
    run_name: str | None = None,
    run_eval: bool = True,
    eval_splits: str = "val,test",
) -> dict:
    import boto3

    run_id = run_name or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    workdir = Path("/root/ml")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(workdir)
    env["OPENCV_LOG_LEVEL"] = "ERROR"

    effective_plan = plan_id or "default"
    s3_ckpt_prefix = f"{output_prefix}/{effective_plan}/{run_id}"

    cmd = _build_train_cmd(
        bucket=bucket, region=region, epochs=epochs, batch_size=batch_size,
        num_workers=num_workers, head_only_epochs=head_only_epochs,
        head_lr=head_lr, backbone_lr=backbone_lr, plan_id=plan_id,
        clip_limit=clip_limit, init_checkpoint_s3_key=init_checkpoint_s3_key,
        init_strict=init_strict, s3_ckpt_prefix=s3_ckpt_prefix,
    )

    print("[modal] running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(workdir), check=True, env=env)

    effective_plan = plan_id
    if not effective_plan:
        s3 = boto3.client("s3", region_name=region)
        effective_plan = _resolve_active_plan(s3, bucket)

    ckpt_dir = workdir / "workdir" / "i3d_msft" / "checkpoints" / str(effective_plan)
    if not ckpt_dir.exists():
        raise RuntimeError(f"Checkpoint directory not found: {ckpt_dir}")

    s3 = boto3.client("s3", region_name=region)
    uploaded = _upload_checkpoints(s3, bucket, ckpt_dir, output_prefix, effective_plan, run_id)

    eval_outputs: dict[str, str] = {}
    if run_eval:
        split_items = [s.strip() for s in eval_splits.split(",") if s.strip()]
        best_model = ckpt_dir / "best_model.pt"
        if not best_model.exists():
            print("[modal] run_eval requested but best_model.pt not found, skipping eval.")
        else:
            gloss_dict_csv = workdir / "workdir" / "i3d_msft" / "filtered_splits" / str(effective_plan) / "train.csv"
            for split in split_items:
                if split not in {"val", "test"}:
                    print(f"[modal] skipping unsupported eval split: {split}")
                    continue
                out_json = workdir / "workdir" / "i3d_msft_eval" / str(effective_plan) / f"{split}_metrics.json"
                eval_cmd = _build_eval_cmd(
                    bucket=bucket, region=region, plan_id=effective_plan, split=split,
                    checkpoint_arg=str(best_model), checkpoint_is_local=True,
                    output_json=str(out_json), gloss_dict_csv=str(gloss_dict_csv),
                )
                print("[modal] running eval:", " ".join(eval_cmd))
                subprocess.run(eval_cmd, cwd=str(workdir), check=True, env=env)

                s3_metrics_key = f"{output_prefix}/{effective_plan}/{run_id}/eval/{split}_metrics.json"
                s3.upload_file(str(out_json), bucket, s3_metrics_key)
                eval_outputs[split] = f"s3://{bucket}/{s3_metrics_key}"

    meta = {
        "run_id": run_id, "plan_id": effective_plan, "bucket": bucket,
        "region": region, "epochs": epochs, "batch_size": batch_size,
        "num_workers": num_workers, "clip_limit": clip_limit,
        "head_only_epochs": head_only_epochs, "head_lr": head_lr,
        "backbone_lr": backbone_lr, "init_checkpoint_s3_key": init_checkpoint_s3_key,
        "init_strict": init_strict, "uploaded_checkpoints": uploaded,
        "run_eval": run_eval, "eval_splits": eval_splits, "eval_outputs": eval_outputs,
    }
    meta_key = f"{output_prefix}/{effective_plan}/{run_id}/run_metadata.json"
    _upload_run_metadata(s3, bucket, meta, meta_key)

    return {
        "run_id": run_id, "plan_id": effective_plan,
        "checkpoints": uploaded, "eval_outputs": eval_outputs,
        "metadata": f"s3://{bucket}/{meta_key}",
    }


@app.function(
    image=image,
    gpu="T4",
    timeout=60 * 60 * 12,  # 12h
    secrets=[modal.Secret.from_name("aws-credentials")],
)
def eval_i3d_modal(  # pragma: no cover – requires Modal GPU infrastructure
    bucket: str,
    region: str = "ca-central-1",
    plan_id: str | None = None,
    checkpoint_s3_key: str = "models/i3d/pretrained/ASL_citizen_I3D_weights.pt",
    batch_size: int = 6,
    num_workers: int = 2,
    clip_limit: int | None = None,
    output_prefix: str = "models/i3d/eval",
    run_name: str | None = None,
    eval_splits: str = "val,test",
) -> dict:
    import boto3

    run_id = run_name or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    workdir = Path("/root/ml")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(workdir)
    env["OPENCV_LOG_LEVEL"] = "ERROR"

    s3 = boto3.client("s3", region_name=region)
    effective_plan = plan_id
    if not effective_plan:
        effective_plan = _resolve_active_plan(s3, bucket)

    split_items = [s.strip() for s in eval_splits.split(",") if s.strip()]
    eval_outputs: dict[str, str] = {}
    for split in split_items:
        if split not in {"val", "test"}:
            print(f"[modal] skipping unsupported eval split: {split}")
            continue
        out_json = workdir / "workdir" / "i3d_msft_eval" / str(effective_plan) / f"{split}_metrics.json"
        eval_cmd = _build_eval_cmd(
            bucket=bucket, region=region, plan_id=effective_plan, split=split,
            checkpoint_arg=checkpoint_s3_key, checkpoint_is_local=False,
            batch_size=batch_size, num_workers=num_workers, clip_limit=clip_limit,
            output_json=str(out_json),
        )
        print("[modal] running eval-only:", " ".join(eval_cmd))
        subprocess.run(eval_cmd, cwd=str(workdir), check=True, env=env)

        s3_metrics_key = f"{output_prefix}/{effective_plan}/{run_id}/{split}_metrics.json"
        s3.upload_file(str(out_json), bucket, s3_metrics_key)
        eval_outputs[split] = f"s3://{bucket}/{s3_metrics_key}"

    meta = {
        "run_id": run_id, "plan_id": effective_plan, "bucket": bucket,
        "region": region, "checkpoint_s3_key": checkpoint_s3_key,
        "batch_size": batch_size, "num_workers": num_workers,
        "clip_limit": clip_limit, "eval_splits": eval_splits,
        "eval_outputs": eval_outputs,
    }
    meta_key = f"{output_prefix}/{effective_plan}/{run_id}/run_metadata.json"
    _upload_run_metadata(s3, bucket, meta, meta_key)
    return {
        "run_id": run_id, "plan_id": effective_plan,
        "eval_outputs": eval_outputs,
        "metadata": f"s3://{bucket}/{meta_key}",
    }


@app.local_entrypoint()
def main(  # pragma: no cover – Modal local entrypoint, dispatches to remote
    bucket: str = "eye-hear-u-public-data-ca1",
    region: str = "ca-central-1",
    plan_id: str = "candidate-ac-eval-v4",
    epochs: int = 20,
    batch_size: int = 6,
    num_workers: int = 2,
    clip_limit: int = 0,
    head_only_epochs: int = 2,
    head_lr: float = 1e-3,
    backbone_lr: float = 1e-4,
    init_checkpoint_s3_key: str = "",
    init_strict: bool = False,
    run_name: str = "",
    run_eval: bool = True,
    eval_splits: str = "val,test",
    eval_only: bool = False,
    checkpoint_s3_key: str = "models/i3d/pretrained/ASL_citizen_I3D_weights.pt",
):
    if eval_only:
        result = eval_i3d_modal.remote(
            bucket=bucket,
            region=region,
            plan_id=(None if not plan_id else plan_id),
            checkpoint_s3_key=checkpoint_s3_key,
            batch_size=batch_size,
            num_workers=num_workers,
            clip_limit=(None if clip_limit <= 0 else clip_limit),
            run_name=(None if not run_name else run_name),
            eval_splits=eval_splits,
        )
        print(result)
        return

    result = train_i3d_modal.remote(
        bucket=bucket,
        region=region,
        plan_id=plan_id,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        clip_limit=(None if clip_limit <= 0 else clip_limit),
        head_only_epochs=head_only_epochs,
        head_lr=head_lr,
        backbone_lr=backbone_lr,
        init_checkpoint_s3_key=(None if not init_checkpoint_s3_key else init_checkpoint_s3_key),
        init_strict=init_strict,
        run_name=(None if not run_name else run_name),
        run_eval=run_eval,
        eval_splits=eval_splits,
    )
    print(result)
