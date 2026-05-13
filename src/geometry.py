"""Geometry: angle <-> keypoints, rotated boxes, IoU, AABB."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

KEYPOINT_RADIUS_FRAC = 0.35  # r = 0.35 * max(bbox_w, bbox_h)


@dataclass
class TubeAnnotation:
    image: str
    center_x: float
    center_y: float
    bbox_x: float  # top-left x of axis-aligned bbox (or center; see normalize_bbox)
    bbox_y: float
    bbox_w: float
    bbox_h: float
    bbox_rotation: float  # degrees
    angle_deg: float  # joint -> tab direction (CCW, image coords y-down)


def angle_to_keypoints(
    cx: float,
    cy: float,
    bbox_w: float,
    bbox_h: float,
    angle_deg: float,
    r_frac: float = KEYPOINT_RADIUS_FRAC,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return (joint, tab) image-space coordinates.

    angle is CCW from +x with y-down image axes, matching the dataset convention.
    The "CCW with y-down" wording in the brief is treated as the conventional
    image-space angle so atan2(dy, dx) recovers it.
    """
    r = r_frac * max(bbox_w, bbox_h)
    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    joint = (cx - r * cos_t, cy - r * sin_t)
    tab = (cx + r * cos_t, cy + r * sin_t)
    return joint, tab


def keypoints_to_angle(joint: tuple[float, float], tab: tuple[float, float]) -> float:
    """Recover angle in [0, 360) from joint and tab keypoints."""
    dx = tab[0] - joint[0]
    dy = tab[1] - joint[1]
    a = math.degrees(math.atan2(dy, dx))
    return (a + 360.0) % 360.0


def normalize_bbox_xywh(
    bbox_x: float,
    bbox_y: float,
    bbox_w: float,
    bbox_h: float,
    bbox_rotation: float,
    cx: float,
    cy: float,
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float]:
    """Convert dataset bbox + rotation to an axis-aligned YOLO bbox (cx,cy,w,h) normalized.

    Strategy:
    - Build the 4 corners of the rotated rectangle whose center is (cx, cy),
      width = bbox_w, height = bbox_h, rotated by bbox_rotation degrees (CCW).
    - Take the axis-aligned bounding box of those corners.
    - Normalize by image size.
    """
    theta = math.radians(bbox_rotation)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    hw, hh = bbox_w / 2.0, bbox_h / 2.0
    local = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=np.float64)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
    rotated = local @ R.T + np.array([cx, cy], dtype=np.float64)
    xmin, ymin = rotated.min(axis=0)
    xmax, ymax = rotated.max(axis=0)
    xmin = max(0.0, xmin)
    ymin = max(0.0, ymin)
    xmax = min(float(img_w), xmax)
    ymax = min(float(img_h), ymax)
    w = max(1.0, xmax - xmin)
    h = max(1.0, ymax - ymin)
    cxn = (xmin + xmax) / 2.0 / img_w
    cyn = (ymin + ymax) / 2.0 / img_h
    wn = w / img_w
    hn = h / img_h
    return cxn, cyn, wn, hn


def aabb_xyxy_from_rotated(
    cx: float, cy: float, w: float, h: float, rot_deg: float
) -> tuple[float, float, float, float]:
    theta = math.radians(rot_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    hw, hh = w / 2.0, h / 2.0
    pts = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=np.float64)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
    rot = pts @ R.T + np.array([cx, cy])
    return float(rot[:, 0].min()), float(rot[:, 1].min()), float(rot[:, 0].max()), float(rot[:, 1].max())


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between [N,4] and [M,4] in xyxy. Returns [N,M]."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)
    iw = np.clip(inter_x2 - inter_x1, 0, None)
    ih = np.clip(inter_y2 - inter_y1, 0, None)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return (inter / union).astype(np.float32)


def circular_angle_error(pred_deg: np.ndarray | float, gt_deg: np.ndarray | float) -> np.ndarray:
    """Smallest absolute circular angle difference in [0,180]."""
    diff = np.abs(np.asarray(pred_deg) - np.asarray(gt_deg)) % 360.0
    return np.minimum(diff, 360.0 - diff)


def rotate_point(x: float, y: float, cx: float, cy: float, deg: float) -> tuple[float, float]:
    theta = math.radians(deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    dx, dy = x - cx, y - cy
    return cx + dx * cos_t - dy * sin_t, cy + dx * sin_t + dy * cos_t
