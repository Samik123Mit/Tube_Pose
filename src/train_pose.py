"""Train YOLOv8s-pose on the prepared dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

from augmentations import yolov8_train_aug_kwargs
from utils import CONFIGS_DIR, MODELS_DIR, OUTPUTS_DIR, device_str, get_logger, set_seed

LOG = get_logger("train_pose")


def train_one(
    data_yaml: str,
    weights: str = "yolov8s-pose.pt",
    epochs: int = 150,
    imgsz: int = 640,
    batch: int = 16,
    project: str = "outputs/runs",
    name: str = "tube_pose",
    seed: int = 42,
    patience: int = 30,
    lr0: float = 0.01,
    lrf: float = 0.01,
    cos_lr: bool = True,
    optimizer: str = "AdamW",
    amp: bool = True,
    device: str | None = None,
) -> dict:
    from ultralytics import YOLO

    set_seed(seed)
    dev = device or device_str()
    LOG.info(f"training on device={dev} weights={weights} data={data_yaml}")

    model = YOLO(weights)
    aug = yolov8_train_aug_kwargs()
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        seed=seed,
        deterministic=True,
        patience=patience,
        lr0=lr0,
        lrf=lrf,
        cos_lr=cos_lr,
        optimizer=optimizer,
        amp=amp,
        device=dev,
        verbose=True,
        plots=True,
        save=True,
        save_period=-1,
        **aug,
    )

    # Locate best weights and copy to models dir for stable downstream paths
    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        target = Path(MODELS_DIR) / f"{name}_best.pt"
        target.write_bytes(best.read_bytes())
        LOG.info(f"copied best weights to {target}")

    return {"save_dir": str(results.save_dir), "best": str(best)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=str(CONFIGS_DIR / "data.yaml"))
    parser.add_argument("--weights", type=str, default="yolov8s-pose.pt")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", type=str, default="tube_pose")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--cv", action="store_true", help="train all 5 CV folds")
    args = parser.parse_args()

    if args.cv:
        for fold_i in range(5):
            data_yaml = CONFIGS_DIR / f"data_fold_{fold_i}.yaml"
            train_one(
                data_yaml=str(data_yaml),
                weights=args.weights,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                project=str(OUTPUTS_DIR / "runs"),
                name=f"tube_pose_fold_{fold_i}",
                seed=args.seed,
                patience=args.patience,
                device=args.device,
            )
    else:
        train_one(
            data_yaml=args.data,
            weights=args.weights,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=str(OUTPUTS_DIR / "runs"),
            name=args.name,
            seed=args.seed,
            patience=args.patience,
            device=args.device,
        )


if __name__ == "__main__":
    main()
