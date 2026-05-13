"""Compare predictions against ground truth, compute metrics, write a report."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from geometry import angle_to_keypoints, aabb_xyxy_from_rotated
from metrics import aggregate, evaluate_image_multi_iou
from prepare_dataset import load_annotations
from utils import DATA_DIR, OUTPUTS_DIR, get_logger, load_json, save_json

LOG = get_logger("evaluate")


def gt_boxes_and_angles_for_image(df_img: pd.DataFrame, img_w: int, img_h: int) -> tuple[np.ndarray, np.ndarray]:
    boxes = []
    angles = []
    for _, r in df_img.iterrows():
        cx = float(r["center_x"])
        cy = float(r["center_y"])
        bw = float(r["bbox_w"])
        bh = float(r["bbox_h"])
        rot = float(r["bbox_rotation"])
        x1, y1, x2, y2 = aabb_xyxy_from_rotated(cx, cy, bw, bh, rot)
        x1 = max(0.0, x1)
        y1 = max(0.0, y1)
        x2 = min(float(img_w), x2)
        y2 = min(float(img_h), y2)
        boxes.append([x1, y1, x2, y2])
        angles.append(float(r["angle_deg"]))
    return np.array(boxes, dtype=np.float32), np.array(angles, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True, help="path to predictions.json")
    parser.add_argument("--csv", type=str, default=str(DATA_DIR / "annotations.csv"))
    parser.add_argument("--images", type=str, default=str(DATA_DIR / "images"))
    parser.add_argument("--out", type=str, default=str(OUTPUTS_DIR / "evaluation"))
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--use-refined-angle", action="store_true")
    args = parser.parse_args()

    out_p = Path(args.out)
    out_p.mkdir(parents=True, exist_ok=True)

    df = load_annotations(Path(args.csv))
    grouped = df.groupby("image")
    preds = load_json(args.predictions)

    per_image = []
    for img_name, det_list in preds.items():
        try:
            df_img = grouped.get_group(img_name)
        except KeyError:
            df_img = df.iloc[0:0]
        img_path = Path(args.images) / img_name
        img = cv2.imread(str(img_path))
        if img is None:
            LOG.warning(f"missing image during eval: {img_name}")
            continue
        img_h, img_w = img.shape[:2]
        gt_boxes, gt_angles = gt_boxes_and_angles_for_image(df_img, img_w, img_h)

        if det_list:
            pred_boxes = np.array([d["bbox_xyxy"] for d in det_list], dtype=np.float32)
            pred_scores = np.array([d["score"] for d in det_list], dtype=np.float32)
            if args.use_refined_angle:
                pred_angles = np.array(
                    [
                        d["refined_angle_deg"] if d.get("refined_angle_deg") is not None else d["angle_deg"]
                        for d in det_list
                    ],
                    dtype=np.float32,
                )
            else:
                pred_angles = np.array([d["angle_deg"] for d in det_list], dtype=np.float32)
        else:
            pred_boxes = np.zeros((0, 4), dtype=np.float32)
            pred_scores = np.zeros((0,), dtype=np.float32)
            pred_angles = np.zeros((0,), dtype=np.float32)

        per_image.append(
            evaluate_image_multi_iou(pred_boxes, pred_scores, pred_angles, gt_boxes, gt_angles)
        )

    metrics = aggregate(per_image)
    save_json(metrics, out_p / "metrics.json")

    summary = {k: v for k, v in metrics.items() if k != "angle_errors"}
    LOG.info("metrics:")
    for k, v in summary.items():
        LOG.info(f"  {k}: {v}")

    # Save flat CSV summary
    pd.DataFrame([summary]).to_csv(out_p / "metrics_summary.csv", index=False)
    pd.DataFrame({"angle_error_deg": metrics["angle_errors"]}).to_csv(
        out_p / "angle_errors.csv", index=False
    )

    LOG.info(f"wrote results to {out_p}")


if __name__ == "__main__":
    main()
