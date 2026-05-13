"""Inference: run YOLOv8 pose, reconstruct angles, optional TTA + ellipse refinement."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from geometry import keypoints_to_angle
from utils import MODELS_DIR, OUTPUTS_DIR, device_str, get_logger, list_images, save_json

LOG = get_logger("inference")


@dataclass
class TubePrediction:
    bbox_xyxy: tuple[float, float, float, float]
    score: float
    joint: tuple[float, float]
    tab: tuple[float, float]
    angle_deg: float
    refined_angle_deg: float | None = None
    angle_uncertainty_deg: float = 0.0
    extras: dict = field(default_factory=dict)


def _ultralytics_predict(model, image: np.ndarray, imgsz: int, conf: float, iou: float, device: str):
    return model.predict(
        source=image,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
    )[0]


def _parse_result(result) -> list[TubePrediction]:
    preds: list[TubePrediction] = []
    if result.boxes is None or result.boxes.shape[0] == 0:
        return preds
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    kpts = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else None
    for i in range(boxes.shape[0]):
        x1, y1, x2, y2 = boxes[i].tolist()
        score = float(scores[i])
        if kpts is None or kpts.shape[1] < 2:
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            joint = (cx, cy)
            tab = (cx + 1.0, cy)
        else:
            joint = (float(kpts[i, 0, 0]), float(kpts[i, 0, 1]))
            tab = (float(kpts[i, 1, 0]), float(kpts[i, 1, 1]))
        angle = keypoints_to_angle(joint, tab)
        preds.append(TubePrediction((x1, y1, x2, y2), score, joint, tab, angle))
    return preds


def tta_predict(model, image: np.ndarray, imgsz: int, conf: float, iou: float, device: str) -> list[TubePrediction]:
    """Light TTA: original + horizontal flip + 90 degree rotation, average angles via vectors."""
    h, w = image.shape[:2]
    variants = []

    # original
    variants.append(("orig", image, None))
    # hflip
    variants.append(("hflip", cv2.flip(image, 1), "hflip"))
    # rot90 ccw
    variants.append(("rot90", cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), "rot90"))

    all_preds: list[list[TubePrediction]] = []
    for tag, img, transform in variants:
        r = _ultralytics_predict(model, img, imgsz, conf, iou, device)
        preds = _parse_result(r)
        # invert the transform to bring boxes/keypoints back to original frame
        if transform == "hflip":
            for p in preds:
                x1, y1, x2, y2 = p.bbox_xyxy
                p.bbox_xyxy = (w - x2, y1, w - x1, y2)
                p.joint = (w - p.joint[0], p.joint[1])
                p.tab = (w - p.tab[0], p.tab[1])
                p.angle_deg = keypoints_to_angle(p.joint, p.tab)
        elif transform == "rot90":
            for p in preds:
                # ccw rotation: new (x', y') = (y, W' - x) where W' = original width of rotated image = h
                x1, y1, x2, y2 = p.bbox_xyxy
                # rotated image had width = h, height = w
                # invert: (x_orig, y_orig) = (W'_rot - y_rot, x_rot) where W'_rot = h? Use direct math:
                # The rotation we applied is ROTATE_90_COUNTERCLOCKWISE: maps (x, y) on original to (y, W - 1 - x) on rotated.
                # Inverse: original = (W - 1 - y_rot, x_rot)
                def inv(x, y):
                    return (w - 1 - y, x)

                nx1, ny1 = inv(x1, y1)
                nx2, ny2 = inv(x2, y2)
                xmin, xmax = sorted([nx1, nx2])
                ymin, ymax = sorted([ny1, ny2])
                p.bbox_xyxy = (xmin, ymin, xmax, ymax)
                p.joint = inv(*p.joint)
                p.tab = inv(*p.tab)
                p.angle_deg = keypoints_to_angle(p.joint, p.tab)
        all_preds.append(preds)

    # Merge across variants using greedy IoU
    merged: list[TubePrediction] = []
    base = all_preds[0]
    for bp in base:
        bx = np.array([bp.bbox_xyxy], dtype=np.float32)
        sumv = np.array([np.cos(np.radians(bp.angle_deg)), np.sin(np.radians(bp.angle_deg))]) * bp.score
        wsum = bp.score
        for other in all_preds[1:]:
            if not other:
                continue
            ob = np.array([o.bbox_xyxy for o in other], dtype=np.float32)
            from geometry import iou_xyxy

            iou = iou_xyxy(bx, ob)[0]
            if iou.size and iou.max() > 0.5:
                k = int(iou.argmax())
                op = other[k]
                sumv += np.array([np.cos(np.radians(op.angle_deg)), np.sin(np.radians(op.angle_deg))]) * op.score
                wsum += op.score
        avg = sumv / max(wsum, 1e-6)
        bp.angle_deg = (np.degrees(np.arctan2(avg[1], avg[0])) + 360.0) % 360.0
        # crude uncertainty: 1 - resultant length
        r_len = float(np.linalg.norm(sumv) / max(wsum, 1e-6))
        bp.angle_uncertainty_deg = float(np.degrees(np.arccos(np.clip(r_len, -1.0, 1.0))))
        merged.append(bp)
    return merged


def refine_with_ellipse(image: np.ndarray, pred: TubePrediction) -> float | None:
    """Fit an ellipse inside the predicted bbox; if elongated, use its major axis to disambiguate angle.

    Returns refined angle in [0, 360) consistent with the joint->tab direction, or None
    if refinement is not confident.
    """
    x1, y1, x2, y2 = [int(round(v)) for v in pred.bbox_xyxy]
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 + 4 or y2 <= y1 + 4:
        return None
    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 5 or cv2.contourArea(cnt) < 25:
        return None
    (ex, ey), (MA, ma), eangle = cv2.fitEllipse(cnt)
    # cv2.fitEllipse returns angle in degrees CW from vertical axis. Convert to standard CCW from +x with y-down.
    # The major axis direction (in degrees CCW from +x, y-down image space):
    axis_angle = (eangle - 90.0) % 180.0  # [0, 180)
    # Disambiguate using current predicted angle
    cur = pred.angle_deg
    candidates = [axis_angle, (axis_angle + 180.0) % 360.0]
    diffs = [min(abs(c - cur) % 360.0, 360.0 - (abs(c - cur) % 360.0)) for c in candidates]
    return float(candidates[int(np.argmin(diffs))])


def run_inference(
    weights: str,
    images_dir: str,
    out_dir: str,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.5,
    use_tta: bool = False,
    use_ellipse: bool = False,
    device: str | None = None,
) -> dict:
    from ultralytics import YOLO

    dev = device or device_str()
    LOG.info(f"loading weights: {weights}")
    model = YOLO(weights)

    images = list_images(images_dir)
    LOG.info(f"running inference on {len(images)} images")
    results: dict[str, list[dict]] = {}

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            LOG.warning(f"unreadable image: {img_path}")
            continue
        if use_tta:
            preds = tta_predict(model, image, imgsz, conf, iou, dev)
        else:
            r = _ultralytics_predict(model, image, imgsz, conf, iou, dev)
            preds = _parse_result(r)

        if use_ellipse:
            for p in preds:
                refined = refine_with_ellipse(image, p)
                if refined is not None:
                    p.refined_angle_deg = float(refined)

        results[img_path.name] = [
            {
                "bbox_xyxy": list(p.bbox_xyxy),
                "score": p.score,
                "joint": list(p.joint),
                "tab": list(p.tab),
                "angle_deg": p.angle_deg,
                "refined_angle_deg": p.refined_angle_deg,
                "angle_uncertainty_deg": p.angle_uncertainty_deg,
            }
            for p in preds
        ]

    save_json(results, out_dir_p / "predictions.json")
    LOG.info(f"wrote predictions: {out_dir_p / 'predictions.json'}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=str(MODELS_DIR / "tube_pose_best.pt"))
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--out", type=str, default=str(OUTPUTS_DIR / "inference"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--ellipse-refine", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    run_inference(
        weights=args.weights,
        images_dir=args.images,
        out_dir=args.out,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        use_tta=args.tta,
        use_ellipse=args.ellipse_refine,
        device=args.device,
    )


if __name__ == "__main__":
    main()
