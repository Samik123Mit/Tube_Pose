"""Build 5-fold cross-validation splits, write per-fold yaml + labels."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold

from prepare_dataset import build_label_lines, load_annotations, read_image_size, write_yaml
from utils import CONFIGS_DIR, DATA_DIR, get_logger, set_seed

LOG = get_logger("cross_validation")


def write_fold(
    df: pd.DataFrame,
    image_names: list[str],
    images_src: Path,
    out_root: Path,
    split: str,
) -> None:
    img_out = out_root / "images" / split
    lbl_out = out_root / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    grouped = df.groupby("image")
    for img_name in image_names:
        src = images_src / img_name
        if not src.exists():
            continue
        dst = img_out / img_name
        if not dst.exists():
            shutil.copy2(src, dst)
        try:
            rows = grouped.get_group(img_name)
        except KeyError:
            rows = df.iloc[0:0]
        w, h = read_image_size(src)
        lines = build_label_lines(rows, w, h)
        (lbl_out / (Path(img_name).stem + ".txt")).write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=str(DATA_DIR / "annotations.csv"))
    parser.add_argument("--images", type=str, default=str(DATA_DIR / "images"))
    parser.add_argument("--out", type=str, default=str(DATA_DIR / "cv"))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    df = load_annotations(Path(args.csv))
    images_dir = Path(args.images)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    all_images = sorted(df["image"].unique().tolist())
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    fold_yamls: list[Path] = []
    for fold_i, (train_idx, val_idx) in enumerate(kf.split(all_images)):
        train_imgs = [all_images[i] for i in train_idx]
        val_imgs = [all_images[i] for i in val_idx]
        fold_root = out_root / f"fold_{fold_i}"
        fold_root.mkdir(parents=True, exist_ok=True)

        LOG.info(f"fold {fold_i}: train={len(train_imgs)} val={len(val_imgs)}")
        write_fold(df, train_imgs, images_dir, fold_root, "train")
        write_fold(df, val_imgs, images_dir, fold_root, "val")

        yaml_path = CONFIGS_DIR / f"data_fold_{fold_i}.yaml"
        write_yaml(fold_root, yaml_path)
        fold_yamls.append(yaml_path)

    LOG.info(f"created {len(fold_yamls)} folds: {[str(p) for p in fold_yamls]}")


if __name__ == "__main__":
    main()
