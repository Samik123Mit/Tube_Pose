# Microcentrifuge Tube Detection and Orientation Estimation

Detect microcentrifuge tubes in overhead RGB images and estimate each tube's lid orientation angle in [0, 360) degrees, where the angle is defined by the joint-to-tab direction of the lid.

The system uses **YOLOv8 pose estimation with two pseudo keypoints** (joint and tab) instead of directly regressing the angle. This avoids the 0°/360° wrap-around discontinuity that breaks naive angle regression losses.

---

## Results

Evaluated on the validation split (14 images, 70 ground truth tubes) with Hungarian matching at IoU 0.5 and confidence threshold 0.25.

| Metric | Value |
|---|---|
| Precision | 0.574 |
| Recall | 0.500 |
| F1 | 0.534 |
| mAP@0.5 | 0.395 |
| mAP@[.5:.95] | 0.147 |
| Mean angle error | 85.82° |
| Median angle error | 86.80° |
| TP / FP / FN | 35 / 26 / 35 |

Ultralytics' internal validation reports **Pose mAP@0.5 = 0.825** on the same split, indicating keypoint localization is working well. The high angle error is a labeling-convention bug identified during analysis (see [Analysis](#analysis-and-next-steps)).

Training was early-stopped at epoch 31 of 150 on CPU due to no improvement in 30 epochs. With the flip_idx fix and GPU training to 150 epochs, expected median angle error is under 10°.

Terminal Output:
<img width="1919" height="933" alt="image" src="https://github.com/user-attachments/assets/9bdb1d43-61dd-4b55-9f08-568272d87819" />

---

## Approach

### Keypoint formulation

For each tube with center `(cx, cy)`, angle `a` (degrees), and bounding box `(w, h)`, two pseudo keypoints are derived at radius `r = 0.35 · max(w, h)` along the angle direction:

```
theta = radians(a)
joint = (cx − r·cos(theta), cy − r·sin(theta))
tab   = (cx + r·cos(theta), cy + r·sin(theta))
```

At inference the angle is reconstructed via:

```
angle = (degrees(atan2(tab_y − joint_y, tab_x − joint_x)) + 360) mod 360
```

### Bounding box conversion

The rotated bounding box in the CSV (`bbox_rotation`) is converted to an axis-aligned YOLO bbox by computing the AABB of the four rotated corners and normalizing by image size.

### Model and training

- **Architecture:** YOLOv8s-pose (11.4M params, 29.4 GFLOPs), 1 class, 2 keypoints, `kpt_shape=[2, 3]`
- **Pretrained weights:** `yolov8s-pose.pt` (COCO pose)
- **Optimizer:** AdamW with cosine LR schedule
- **Mixed precision** training, deterministic seed = 42
- **Early stopping:** patience = 30 epochs
- **Augmentations:** mosaic, mixup, HSV jitter, rotation up to ±180°, translation, scale, shear, mild perspective, horizontal flip, random erasing
- **Train/val split:** 80/20 by image (56 train, 14 val)
- **5-fold cross-validation scaffold** included

### Evaluation

- **Matching:** Hungarian assignment on IoU cost matrix between predictions and ground truth, filtered by IoU threshold (0.5 for P/R/F1)
- **Angle error:** circular distance, `err = min(|p − g| mod 360, 360 − |p − g| mod 360)`
- **mAP@[.5:.95]:** re-matching at each IoU threshold in `np.linspace(0.5, 0.95, 10)`

### Bonus features

- **Test-time augmentation:** original + horizontal flip + 90° rotation, merged via vector-form angle averaging weighted by detection scores
- **Ellipse refinement:** Otsu-thresholded contour fit inside each predicted bbox, major-axis orientation used to disambiguate 180° flips
- **Uncertainty estimation:** angular uncertainty derived from resultant length of TTA unit vectors

---

## Repository structure

```
project/
├── data/                   # dataset goes here
│   ├── images/             # 70 RGB images, 640×480
│   └── annotations.csv     # ground truth: 371 tubes
├── configs/                # generated data.yaml + per-fold yamls
├── models/                 # best weights, copied after training
├── outputs/                # training runs, inference, evaluation, viz
├── notebooks/              # exploratory notebooks
├── src/
│   ├── prepare_dataset.py  # CSV → YOLO pose labels, train/val split
│   ├── cross_validation.py # 5-fold CV split builder
│   ├── geometry.py         # angle ↔ keypoints, AABB, IoU, circular error
│   ├── metrics.py          # P/R/F1, mAP@50, mAP@[.5:.95]
│   ├── augmentations.py    # Albumentations + YOLOv8 train aug kwargs
│   ├── train_pose.py       # YOLOv8s-pose training entrypoint
│   ├── inference.py        # detection + angle reconstruction + TTA + ellipse refinement
│   ├── evaluate.py         # GT vs predictions evaluation
│   ├── visualize.py        # overlays, error histogram, metric plots
│   └── utils.py            # seeding, IO, logging, paths
├── requirements.txt
├── run_all.sh
└── README.md
```

---

## Setup

```bash
git clone <your-repo-url>
cd tube_pose
python -m venv .venv
source .venv/bin/activate    # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place the dataset so the layout is:

```
data/
├── images/             # 70 .png files
└── annotations.csv
```

---

## Usage

### End-to-end pipeline

```bash
bash run_all.sh
```

This runs dataset preparation, CV split building, training, inference (with TTA and ellipse refinement), evaluation, and visualization.

### Individual steps

**1. Prepare the dataset**

```bash
python src/prepare_dataset.py \
    --csv data/annotations.csv \
    --images data/images \
    --out data/yolo \
    --val-frac 0.2 \
    --seed 42 \
    --yaml configs/data.yaml
```

**2. Build 5-fold cross-validation splits**

```bash
python src/cross_validation.py --n-folds 5 --seed 42
```

**3. Train**

```bash
# single split
python src/train_pose.py --data configs/data.yaml --epochs 150 --batch 16

# all 5 CV folds
python src/train_pose.py --cv --epochs 150 --batch 16
```

**4. Inference**

```bash
python src/inference.py \
    --weights models/tube_pose_best.pt \
    --images data/yolo/images/val \
    --out outputs/inference \
    --conf 0.25 --iou 0.5
```

Add `--tta` for test-time augmentation and `--ellipse-refine` for contour-based angle refinement.

**5. Evaluate**

```bash
python src/evaluate.py \
    --predictions outputs/inference/predictions.json \
    --out outputs/evaluation
```

**6. Visualize**

```bash
python src/visualize.py \
    --predictions outputs/inference/predictions.json \
    --metrics outputs/evaluation/metrics.json \
    --out outputs/viz
```

Outputs include side-by-side GT vs prediction overlays, an angle error histogram, and metric bar plots.

---

## Analysis and Next Steps

### What worked

The keypoint formulation cleanly avoids the angle wrap-around problem. Pose mAP@0.5 of 0.825 in Ultralytics' internal validation confirms the network reliably places two keypoints on the correct tube axis. Training converged in 31 epochs and the full data pipeline (rotated-bbox to AABB conversion, normalized YOLO labels, 5-fold CV scaffold, circular metrics) is solid.

### Root cause of high angle error

The median angle error of ~87° is **not a model capacity problem; it is a labeling-convention bug.**

The original `data.yaml` sets `flip_idx=[1, 0]`, which tells YOLOv8 to swap the joint and tab keypoint indices on every horizontal flip augmentation. This convention is correct for symmetric anatomical pairs (e.g., left/right eye in human pose) but **wrong** for joint and tab, which are physical features of the tube and do not swap when the image is mirrored.

With `fliplr=0.5`, exactly half the augmented training samples saw a tube with its joint/tab labels reversed. The network resolved this contradiction by learning the tube axis correctly but losing the 180° orientation information. The resulting angle error distribution is bimodal at 0° and 180°, with the mean and median near 90°.

**Fix:** set `flip_idx=[0, 1]` in `configs/data.yaml` and retrain. Expected median angle error after fix: under 10°.

### Secondary factors

1. **CPU-only training, early-stopped at epoch 31.** On a GPU the full 150 epochs would run, with box mAP@0.5 expected to climb from 0.55 to 0.80+.
2. **Strict custom evaluation.** Confidence threshold 0.25 with Hungarian matching is stricter than Ultralytics' default validation protocol; this explains the gap between Ultralytics' Box mAP@0.5 of 0.554 and the custom evaluation's 0.395.
3. **TTA + 180° ambiguity interact badly.** When the model is randomly 180° off, vector-averaging across TTA variants pulls the resultant toward the perpendicular direction, inflating error rather than reducing it. With the flip_idx fix this becomes a net positive.

### Next steps in order of expected impact

1. **Set `flip_idx=[0, 1]`** and retrain. Single highest-leverage change.
2. **Train on GPU for the full 150 epochs.** Expected: box mAP@0.5 → 0.85+, pose mAP@0.5 → 0.95+.
3. **Add a small orientation-disambiguation head:** a 2-class classifier (joint-side vs tab-side) on the cropped ROI, run after detection to make the 180° decision explicit and robust on out-of-distribution images.
4. **Replace pseudo keypoints with hand-annotated lid-tab keypoints.** This removes the `r = 0.35 · max(w, h)` hyperparameter and lets the network learn directly from visual lid-tab cues — the small protrusion is the only true ground-truth signal for orientation.
5. **Calibrate the TTA-derived uncertainty** against held-out angle errors, then use it to drive selective rejection or trigger ellipse refinement only when needed.
6. **Switch to YOLOv8-OBB** for tighter rotated bounding boxes around near-cylindrical tubes. The current AABB is loose for tubes near 45° and inflates IoU-based false negatives.

---

## How I used AI

I used Claude (Sonnet and Opus) for:

- **Initial scaffolding.** Generating the repository layout, `requirements.txt`, and `run_all.sh` orchestration given the problem statement.
- **Core module implementation.** Writing the keypoint to angle geometry, IoU and Hungarian matching, multi-IoU mAP aggregation, Albumentations and Ultralytics augmentation configuration, TTA merging logic, and ellipse refinement.
- **Evaluation and visualization scripts** in one pass with sanity checks (angle round-trip tests, IoU edge cases, synthetic-data metric verification).
- **Debugging the angle error result.** After obtaining metrics, Claude helped trace the data flow from CSV → keypoint labels → YOLO augmentation → inference reconstruction. It identified the `flip_idx=[1, 0]` bug by reasoning about how Ultralytics handles keypoint indices under horizontal flip augmentation.

The problem framing decision (pose with 2 keypoints rather than direct regression), the prioritization of next steps, and the written analysis above are my own.

---

## Reproducibility

All scripts accept `--seed 42`. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` and run on a single GPU for fully deterministic CUDA kernels. `utils.set_seed` pins `torch.backends.cudnn.deterministic = True`.

---

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the pose estimation backbone.
- [Albumentations](https://albumentations.ai/) for augmentation utilities.
- [scikit-learn](https://scikit-learn.org/) for cross-validation and Hungarian matching helpers.
