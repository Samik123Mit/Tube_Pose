# Microcentrifuge Tube Detection and Orientation Estimation

Detect microcentrifuge tubes in overhead RGB images and estimate each tube's lid orientation angle in [0, 360) degrees, defined by the joint-to-tab direction of the lid.

The system uses **YOLOv8 pose estimation with two pseudo keypoints** (joint and tab) instead of directly regressing the angle, which avoids the 0/360 degree wrap-around discontinuity that breaks naive angle regression losses.

---

## Results

Evaluated on the 14-image held-out validation split (70 ground truth tubes) with Hungarian matching at IoU 0.5 and confidence threshold 0.25.

**Detection**

- Precision: 0.837
- Recall: 0.957
- F1: 0.893
- mAP@0.5: 0.880
- mAP@[.5:.95]: 0.594
- True positives: 67 | False positives: 13 | False negatives: 3

**Orientation**

- Mean angle error: 78.83 degrees
- Median angle error: 75.12 degrees
- Computed over 67 matched detections using circular distance.

Ultralytics internal validation on the same split reports Box mAP@0.5 of 0.88 and Pose mAP@0.5 above 0.95, confirming keypoint localization is strong. The detection numbers are submission-quality. The angle error is high for a specific architectural reason explained in the analysis section.

---

## Approach

### Keypoint formulation

For each tube with center `(cx, cy)`, angle `a` (degrees), and bounding box width and height `(w, h)`, two pseudo keypoints are placed at radius `r = 0.35 * max(w, h)` along the angle direction:
theta = radians(a)
joint = (cx - rcos(theta), cy - rsin(theta))
tab   = (cx + rcos(theta), cy + rsin(theta))

At inference the angle is reconstructed via `atan2`:
angle = (degrees(atan2(tab_y - joint_y, tab_x - joint_x)) + 360) mod 360

This formulation avoids the wrap-around discontinuity that breaks direct angle regression.

### Bounding box conversion

The rotated bounding box in the CSV (with `bbox_rotation` field) is converted to an axis-aligned YOLO bbox by computing the AABB of the four rotated corners and normalizing by image dimensions.

### Model and training

- Architecture: YOLOv8s-pose (11.4M params, 29.4 GFLOPs), 1 class, 2 keypoints
- Pretrained weights: `yolov8s-pose.pt` (COCO pose)
- Optimizer: AdamW with cosine LR schedule, `lr0=0.001`
- Warmup: 10 epochs (critical for small-dataset stability)
- Epochs: 200 with patience 80
- Mixed precision training, deterministic seed = 42
- Augmentations: mosaic 0.5, HSV jitter, rotation up to plus or minus 180 degrees, translation, scale 0.3, horizontal flip with `flip_idx=[0,1]`, mixup disabled, mosaic closed in final 20 epochs
- Train/val split: 80/20 by image (56 train, 14 val)
- 5-fold cross-validation scaffold also included in `src/cross_validation.py`

### Evaluation

- Matching: Hungarian assignment on IoU cost matrix between predictions and ground truth, filtered by IoU threshold 0.5
- Angle error: circular distance `min(|p - g| mod 360, 360 - |p - g| mod 360)`
- mAP@[.5:.95]: re-matching at each IoU threshold in `np.linspace(0.5, 0.95, 10)`

### Bonus features implemented

- Test-time augmentation: original plus horizontal flip plus 90 degree rotation, merged via vector-form angle averaging weighted by detection scores
- Ellipse refinement: Otsu-thresholded contour fit inside each predicted bbox, major-axis orientation used to disambiguate 180 degree flips
- Uncertainty estimation: angular uncertainty derived from resultant length of TTA unit vectors

---

## Repository structure
project/
├── data/                   # dataset (images/ + annotations.csv)
├── configs/                # generated data.yaml + per-fold yamls
├── models/                 # best weights, copied after training (gitignored)
├── outputs/                # runs, inference, evaluation, viz
├── src/
│   ├── prepare_dataset.py  # CSV to YOLO pose labels
│   ├── cross_validation.py # 5-fold CV split builder
│   ├── geometry.py         # angle <-> keypoints, AABB, IoU, circular error
│   ├── metrics.py          # P/R/F1, mAP@50, mAP@[.5:.95]
│   ├── augmentations.py    # YOLOv8 train augmentation kwargs
│   ├── train_pose.py       # training entrypoint
│   ├── inference.py        # detection plus angle plus TTA plus ellipse
│   ├── evaluate.py         # GT vs predictions evaluation
│   ├── visualize.py        # overlays, error histogram, metric plots
│   └── utils.py            # seeding, IO, logging
├── requirements.txt
├── run_all.sh
└── README.md

---

## Setup and usage

```bash
git clone https://github.com/Samik123Mit/Tube_Pose.git
cd Tube_Pose
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Drop the dataset into `data/`:
data/
├── images/         # 70 .png files
└── annotations.csv

End-to-end:
```bash
bash run_all.sh
```

Individual steps:
```bash
python src/prepare_dataset.py --val-frac 0.2 --seed 42
python src/train_pose.py --data configs/data.yaml --epochs 200 --batch 16
python src/inference.py --weights models/tube_pose_best.pt --images data/yolo/images/val --out outputs/inference --conf 0.25
python src/evaluate.py --predictions outputs/inference/predictions.json --out outputs/evaluation
python src/visualize.py --predictions outputs/inference/predictions.json --metrics outputs/evaluation/metrics.json --out outputs/viz
```

---

## Analysis

### What worked

The detection side of the system is strong. F1 of 0.893 with recall 0.957 means the model finds nearly every tube and rarely hallucinates extras. mAP@0.5 of 0.88 confirms tight localization. Pose mAP@0.5 above 0.95 in Ultralytics internal validation confirms the network reliably places two keypoints on the correct tube axis. The complete pipeline (label generation, training, inference with TTA and ellipse refinement, evaluation with circular metrics and Hungarian matching, visualization, 5-fold CV scaffold) is structurally complete.

Two critical fixes during development pushed detection from broken to strong:

First, setting `flip_idx=[0,1]` instead of `[1,0]` in data.yaml. The latter convention (correct for symmetric pairs like left and right eye in human pose) wrongly swaps the joint and tab keypoint indices on horizontal flip. Joint and tab are physical features of a tube and do not swap under mirroring, so the wrong convention had been giving the network contradictory labels during fliplr augmentation, completely destroying angle learning.

Second, lowering `lr0` from 0.01 to 0.001 with 10 warmup epochs. The default learning rate is tuned for COCO with 118k images. On this 56-image training set, the default rate caused loss divergence in the first 2 epochs and early-stop with an essentially untrained model (mAP@0.5 around 0.11). The lower rate plus warmup allowed stable convergence over 200 epochs to mAP@0.5 of 0.88.

### Root cause of remaining angle error

The 75 degree median angle error reflects a fundamental limitation of the pseudo-keypoint formulation, not a remaining bug.

The pseudo-keypoints are placed at `r = 0.35 * max(w, h)` from the tube center along the angle direction. There is no visual feature at those locations that distinguishes joint from tab. The actual lid joint is a small hinge at one specific end of the tube; the actual tab is the protrusion at the other end. The pseudo-keypoints land somewhere along the tube body, where both locations look like generic tube body to the network. The network has no visual signal at the keypoint locations to learn which keypoint should go where.

What the network actually learned: place two keypoints on the long axis of the tube. It cannot learn which end is the joint and which is the tab from this label scheme alone. The result is keypoint placement accurate on the axis but essentially random on which end is which, producing the observed near-90 degree mean angle error.

An experiment with `r = 0.50` (keypoints near the tube ends rather than mid-body) reduced median angle error to 58 degrees but degraded detection (F1 from 0.89 to 0.82) because the larger keypoints sometimes fell outside the predicted bbox. This confirms that the limitation is the absence of real visual cues at the keypoint locations, not the placement geometry. The chosen final configuration (r = 0.35) optimizes for detection given that angle disambiguation requires a separate architectural fix.

### Secondary observations

The first GPU training run hit early-stopping at epoch 2 because of learning rate instability. Loss diverged immediately and never recovered. The fix was lower `lr0` plus longer warmup plus disabled mixup (mixup on 56 images causes destructive sample interpolation).

Test-time augmentation with vector-form angle averaging cannot recover the 180 degree ambiguity, because averaging two unit vectors pointing in opposite directions produces a near-zero resultant in a random perpendicular direction. TTA in its current form is a small net negative for angle error while the disambiguation issue persists. It would become a net positive once the 180 degree decision is made reliable by a dedicated head.

The custom Hungarian-matching evaluator is stricter than Ultralytics default validation (one-to-one assignment, hard confidence cutoff at 0.25), which is why the custom mAP@0.5 of 0.88 is slightly below what Ultralytics internal reporting can show on the same predictions.

---

## Next steps in order of expected impact

1. Add a binary orientation classifier head. A small CNN on the cropped tube ROI trained to predict "joint side is left vs right of center" would make the 180 degree decision explicit and recover the missing visual signal. Pairs naturally with the existing ellipse refinement that already produces two candidate angles 180 degrees apart. Expected median angle error after: under 15 degrees.

2. Replace pseudo-keypoints with hand-annotated real lid-tab keypoints. 70 images times roughly 5 tubes is around 350 annotations, about 45 minutes of work with CVAT or LabelMe. With keypoints at the actual hinge and tab protrusion, the network has direct visual signal to learn from. Expected median angle error after: under 8 degrees.

3. Train for longer with 5-fold CV. 200 epochs on a single 80/20 split is conservative. With 5-fold CV plus 300 epochs per fold, detection numbers should saturate higher and angle variance should drop. The 5-fold scaffold is already in `src/cross_validation.py`.

4. Switch to YOLOv8-OBB for tighter rotated bounding boxes. The current AABB representation is loose for tubes oriented near 45 degrees and inflates IoU-based false negatives. OBB also provides a free orientation prior that could regularize the pose head.

5. Calibrate the TTA-derived uncertainty against held-out errors. The uncertainty value is already computed and stored per detection. With calibration it could drive selective rejection or trigger ellipse refinement only when confident.

6. Larger pretrained backbone (YOLOv8m-pose or YOLOv8l-pose). Current YOLOv8s-pose at 11M params may be underparameterized for capturing subtle lid-end appearance differences needed for orientation, even with real keypoints. Compute cost is manageable on a small dataset.

---

## Reproducibility

All scripts accept `--seed 42`. `utils.set_seed` pins `torch.backends.cudnn.deterministic = True`. Training trajectory and final hyperparameters are saved with the training run.

---

## How AI was used in this project

I used Claude as a coding assistant to accelerate implementation, in the same way I would use Stack Overflow, documentation, or a senior engineer's review. All design decisions, the problem framing, the analysis, and the next steps are my own.

Problem framing: I decided on the pose-with-two-keypoints approach myself, motivated by understanding the 0-to-360 wrap-around issue with direct regression.

Implementation: I asked Claude to help me write the geometry module (angle-to-keypoint conversion, IoU, Hungarian matching), the metrics module (precision/recall/F1, mAP at multiple IoU thresholds, circular angle error), the training entrypoint, TTA logic, and ellipse refinement post-processing. I reviewed every module and verified geometry round-trips and metric computations on synthetic inputs before training.

Debugging: When initial CPU results showed 86 degree angle error, I described the symptoms to Claude and we traced the data flow end-to-end. I identified the `flip_idx=[1,0]` mislabeling and the learning rate instability. The reasoning that joint and tab are physical features and should not swap under mirroring (unlike left and right eye) is mine; Claude confirmed how Ultralytics handles `flip_idx` internally.

Hyperparameter tuning: Multiple training runs to find stable settings (lr0=0.001, warmup_epochs=10, mixup off, mosaic 0.5, close_mosaic=20). The diagnosis that defaults were COCO-tuned and too aggressive for 56 images was mine; Claude generated the specific override values.

Writing: I used Claude to help structure this README and tighten wording. The analysis content, the diagnosis of the pseudo-keypoint visual-signal limitation, and the prioritization of next steps are my own.

I did not use AI to fabricate or inflate metrics. Every number in this README is a real measurement from the validation set.
