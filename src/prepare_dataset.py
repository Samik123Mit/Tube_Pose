"""Parse annotations.csv, build YOLO pose labels, train/val split, write data.yaml."""
from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from geometry import angle_to_keypoints, normalize_bbox_xywh
from utils import CONFIGS_DIR, DATA_DIR, get_logger, save_yaml, set_seed

LOG = get_logger("prepare_dataset")

REQUIRED_COLS = [
    "image",
    "center_x",
    "center_y",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "bbox_rotation",
    "angle_deg",
]


def load_annotations(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in annotations.csv: {missing}")
    df["image"] = df["image"].astype(str)
    return df


def read_image_size(p: Path) -> tuple[int, int]:
    img = cv2.imread(str(p))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {p}")
    h, w = img.shape[:2]
    return w, h


def build_label_lines(rows: pd.DataFrame, img_w: int, img_h: int) -> list[str]:
    lines: list[str] = []
    for _, r in rows.iterrows():
        cx = float(r["center_x"])
        cy = float(r["center_y"])
        bw = float(r["bbox_w"])
        bh = float(r["bbox_h"])
        rot = float(r["bbox_rotation"])
        angle = float(r["angle_deg"])

        cxn, cyn, wn, hn = normalize_bbox_xywh(
            float(r["bbox_x"]), float(r["bbox_y"]), bw, bh, rot, cx, cy, img_w, img_h
        )

        joint, tab = angle_to_keypoints(cx, cy, bw, bh, angle)
        jx, jy = joint
        tx, ty = tab
        # Clip to image bounds, keep visibility=2 (visible) regardless to preserve gradient
        jxn = float(np.clip(jx / img_w, 0.0, 1.0))
        jyn = float(np.clip(jy / img_h, 0.0, 1.0))
        txn = float(np.clip(tx / img_w, 0.0, 1.0))
        tyn = float(np.clip(ty / img_h, 0.0, 1.0))

        # YOLO pose: class cx cy w h kpt1_x kpt1_y kpt1_v kpt2_x kpt2_y kpt2_v
        line = (
            f"0 {cxn:.6f} {cyn:.6f} {wn:.6f} {hn:.6f} "
            f"{jxn:.6f} {jyn:.6f} 2 {txn:.6f} {tyn:.6f} 2"
        )
        lines.append(line)
    return lines


def write_split(
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
            LOG.warning(f"missing image referenced in csv: {img_name}")
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
        label_path = lbl_out / (Path(img_name).stem + ".txt")
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def write_yaml(out_root: Path, yaml_path: Path) -> None:
    data = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "tube"},
        "kpt_shape": [2, 3],  # 2 keypoints with (x, y, v)
        "flip_idx": [0, 1],   # swap joint <-> tab on horizontal flip (180 deg ambiguity is handled by metric)
    }
    save_yaml(data, yaml_path)
    LOG.info(f"wrote data yaml: {yaml_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=str(DATA_DIR / "annotations.csv"))
    parser.add_argument("--images", type=str, default=str(DATA_DIR / "images"))
    parser.add_argument("--out", type=str, default=str(DATA_DIR / "yolo"))
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--yaml", type=str, default=str(CONFIGS_DIR / "data.yaml"))
    args = parser.parse_args()

    set_seed(args.seed)
    csv_path = Path(args.csv)
    images_dir = Path(args.images)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    df = load_annotations(csv_path)
    all_images = sorted(df["image"].unique().tolist())
    LOG.info(f"loaded {len(df)} rows across {len(all_images)} images")

    train_imgs, val_imgs = train_test_split(
        all_images, test_size=args.val_frac, random_state=args.seed, shuffle=True
    )
    LOG.info(f"split: train={len(train_imgs)} val={len(val_imgs)}")

    write_split(df, train_imgs, images_dir, out_root, "train")
    write_split(df, val_imgs, images_dir, out_root, "val")

    write_yaml(out_root, Path(args.yaml))
    LOG.info("dataset preparation complete")


if __name__ == "__main__":
    main()
