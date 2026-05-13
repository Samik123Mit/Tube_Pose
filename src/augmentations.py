"""Offline / online augmentation helpers.

YOLOv8 applies mosaic/mixup/HSV/perspective/affine during training automatically when
configured via train kwargs. This module provides:

1) Albumentations pipelines for OFFLINE expansion of the small (70-image) dataset.
2) Angle-aware rotation helpers that update both keypoints AND the per-tube angle_deg
   field consistently, so generated samples remain correctly labeled.

A rotation of the image by `R_deg` CCW (image-space y-down) maps each pixel (x, y)
about the image center; the joint->tab vector rotates by R_deg, so the new angle is
(angle_deg + R_deg) mod 360 when using the same y-down CCW convention used to derive
the keypoints in geometry.angle_to_keypoints.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import albumentations as A
import cv2
import numpy as np

from geometry import angle_to_keypoints, keypoints_to_angle


@dataclass
class TubeInstance:
    cx: float
    cy: float
    w: float  # bbox width (axis-aligned in image pixels for label)
    h: float
    angle_deg: float


def build_basic_augmenter(image_size: int = 640) -> A.Compose:
    """Albumentations pipeline used for offline dataset expansion."""
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=180, p=0.7, border_mode=cv2.BORDER_REFLECT_101),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),
            A.Perspective(scale=(0.02, 0.05), keep_size=True, p=0.3),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False, label_fields=["kp_ids"]),
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_ids"], min_visibility=0.2),
    )


def rotate_instance(inst: TubeInstance, img_w: int, img_h: int, deg: float) -> TubeInstance:
    """Rotate a single tube label about the image center by `deg` (CCW, y-down).

    Bbox width/height are preserved (axis-aligned approximation after rotation is
    handled at write-time via AABB recomputation from keypoints elsewhere).
    """
    cx_img, cy_img = img_w / 2.0, img_h / 2.0
    theta = math.radians(deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    dx, dy = inst.cx - cx_img, inst.cy - cy_img
    new_cx = cx_img + cos_t * dx - sin_t * dy
    new_cy = cy_img + sin_t * dx + cos_t * dy
    new_angle = (inst.angle_deg + deg) % 360.0
    return TubeInstance(new_cx, new_cy, inst.w, inst.h, new_angle)


def flip_instance_horizontal(inst: TubeInstance, img_w: int) -> TubeInstance:
    new_cx = img_w - inst.cx
    # Horizontal flip negates the x-component of the joint->tab vector:
    # original angle a -> new angle = 180 - a (mod 360)
    new_angle = (180.0 - inst.angle_deg) % 360.0
    return TubeInstance(new_cx, inst.cy, inst.w, inst.h, new_angle)


def instances_to_keypoints(
    instances: Iterable[TubeInstance],
) -> tuple[list[tuple[float, float]], list[int]]:
    kps: list[tuple[float, float]] = []
    ids: list[int] = []
    for i, t in enumerate(instances):
        j, tb = angle_to_keypoints(t.cx, t.cy, t.w, t.h, t.angle_deg)
        kps.append(j)
        kps.append(tb)
        ids.append(2 * i)      # joint id
        ids.append(2 * i + 1)  # tab id
    return kps, ids


def keypoints_to_instances(
    kps: list[tuple[float, float]],
    ids: list[int],
    base: list[TubeInstance],
) -> list[TubeInstance]:
    paired: dict[int, dict[str, tuple[float, float]]] = {}
    for kp, kid in zip(kps, ids):
        tube_i = kid // 2
        role = "joint" if (kid % 2 == 0) else "tab"
        paired.setdefault(tube_i, {})[role] = (float(kp[0]), float(kp[1]))

    out: list[TubeInstance] = []
    for i, b in enumerate(base):
        if i not in paired or "joint" not in paired[i] or "tab" not in paired[i]:
            continue
        j = paired[i]["joint"]
        t = paired[i]["tab"]
        cx = (j[0] + t[0]) / 2.0
        cy = (j[1] + t[1]) / 2.0
        angle = keypoints_to_angle(j, t)
        out.append(TubeInstance(cx, cy, b.w, b.h, angle))
    return out


def yolov8_train_aug_kwargs() -> dict:
    """Augmentation kwargs forwarded to ultralytics Model.train().

    All listed augmentations in the spec are turned on with conservative magnitudes
    appropriate for a small dataset of top-down tube images.
    """
    return dict(
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=180.0,   # full rotation; angle target is updated by ultralytics for keypoints
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0005,
        flipud=0.0,      # avoid 180 ambiguity with our 2-keypoint orientation
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.0,
        erasing=0.1,
    )
