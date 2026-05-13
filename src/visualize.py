"""Render visualizations: predicted overlays, side-by-side, error histograms, metric plots."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geometry import aabb_xyxy_from_rotated, angle_to_keypoints
from prepare_dataset import load_annotations
from utils import DATA_DIR, OUTPUTS_DIR, get_logger, load_json

LOG = get_logger("visualize")

COLOR_PRED = (0, 220, 0)
COLOR_GT = (0, 0, 220)
COLOR_JOINT = (0, 200, 255)
COLOR_TAB = (255, 80, 80)
COLOR_TEXT = (255, 255, 255)


def draw_arrow(img: np.ndarray, joint, tab, color, thickness=2):
    j = (int(round(joint[0])), int(round(joint[1])))
    t = (int(round(tab[0])), int(round(tab[1])))
    cv2.arrowedLine(img, j, t, color, thickness, tipLength=0.25)
    cv2.circle(img, j, 4, COLOR_JOINT, -1)
    cv2.circle(img, t, 4, COLOR_TAB, -1)


def draw_box(img, xyxy, color, label: str | None = None, thickness=2):
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            img,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )


def render_predictions_on_image(img: np.ndarray, dets: list[dict]) -> np.ndarray:
    out = img.copy()
    for d in dets:
        draw_box(out, d["bbox_xyxy"], COLOR_PRED, f"{d['score']:.2f}|{d['angle_deg']:.0f}°")
        draw_arrow(out, d["joint"], d["tab"], COLOR_PRED)
    return out


def render_gt_on_image(img: np.ndarray, df_img: pd.DataFrame) -> np.ndarray:
    out = img.copy()
    for _, r in df_img.iterrows():
        cx, cy = float(r["center_x"]), float(r["center_y"])
        bw, bh = float(r["bbox_w"]), float(r["bbox_h"])
        rot = float(r["bbox_rotation"])
        angle = float(r["angle_deg"])
        xyxy = aabb_xyxy_from_rotated(cx, cy, bw, bh, rot)
        draw_box(out, xyxy, COLOR_GT, f"gt|{angle:.0f}°")
        j, t = angle_to_keypoints(cx, cy, bw, bh, angle)
        draw_arrow(out, j, t, COLOR_GT)
    return out


def side_by_side(img_gt: np.ndarray, img_pred: np.ndarray) -> np.ndarray:
    h = max(img_gt.shape[0], img_pred.shape[0])
    pad_gt = np.zeros((h, img_gt.shape[1], 3), dtype=img_gt.dtype)
    pad_pr = np.zeros((h, img_pred.shape[1], 3), dtype=img_pred.dtype)
    pad_gt[: img_gt.shape[0], :, :] = img_gt
    pad_pr[: img_pred.shape[0], :, :] = img_pred
    sep = np.full((h, 8, 3), 255, dtype=img_gt.dtype)
    return np.concatenate([pad_gt, sep, pad_pr], axis=1)


def angle_error_histogram(errors: list[float], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=36, range=(0, 180), color="#3a86ff", edgecolor="black")
    ax.set_xlabel("Angle error (degrees)")
    ax.set_ylabel("Count")
    ax.set_title("Circular angle error distribution")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def metrics_bar_plot(metrics: dict, out_path: Path) -> None:
    keys = ["precision", "recall", "f1", "map50", "map50_95"]
    vals = [float(metrics.get(k, 0.0)) for k in keys]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(keys, vals, color=["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Detection metrics")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--csv", type=str, default=str(DATA_DIR / "annotations.csv"))
    parser.add_argument("--images", type=str, default=str(DATA_DIR / "images"))
    parser.add_argument("--metrics", type=str, default=str(OUTPUTS_DIR / "evaluation" / "metrics.json"))
    parser.add_argument("--out", type=str, default=str(OUTPUTS_DIR / "viz"))
    parser.add_argument("--max-images", type=int, default=20)
    args = parser.parse_args()

    out_p = Path(args.out)
    (out_p / "overlays").mkdir(parents=True, exist_ok=True)

    df = load_annotations(Path(args.csv))
    grouped = df.groupby("image")
    preds = load_json(args.predictions)

    count = 0
    for img_name, det_list in preds.items():
        img_path = Path(args.images) / img_name
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        try:
            df_img = grouped.get_group(img_name)
        except KeyError:
            df_img = df.iloc[0:0]
        gt_img = render_gt_on_image(img, df_img)
        pred_img = render_predictions_on_image(img, det_list)
        sbs = side_by_side(gt_img, pred_img)
        cv2.imwrite(str(out_p / "overlays" / f"{Path(img_name).stem}_sbs.jpg"), sbs)
        count += 1
        if count >= args.max_images:
            break
    LOG.info(f"wrote {count} side-by-side overlays")

    metrics_path = Path(args.metrics)
    if metrics_path.exists():
        m = load_json(metrics_path)
        if m.get("angle_errors"):
            angle_error_histogram(m["angle_errors"], out_p / "angle_error_histogram.png")
        metrics_bar_plot(m, out_p / "metrics_bar.png")
        LOG.info("wrote metric plots")
    else:
        LOG.warning(f"metrics file not found: {metrics_path}")


if __name__ == "__main__":
    main()
