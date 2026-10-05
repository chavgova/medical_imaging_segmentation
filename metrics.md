# Metrics

[`metrics.py`](metrics.py) provides dataset-independent segmentation evaluation
metrics: IoU, 3D Dice, 95th-percentile Hausdorff Distance (HD95), Average
Surface Distance (ASD), and a voxel-wise confusion matrix. They operate on
plain numpy label maps (not one-hot, not torch tensors), take no
dataset-specific assumptions (no class-value scaling; `confusion_matrix`'s
`n_classes=5` is only a SegTHOR default), and work for binary or multi-class
volumes.

`iou`, `hausdorff_distance_95` and `average_surface_distance` score a single
class `c` at a time; `dice` loops over classes internally, and
`confusion_matrix` covers all classes at once. For multi-class HD95/ASD, loop
over class values on the caller side (see examples below).

## `iou(pred, gt, c=1)`

Intersection-over-Union (Jaccard index) for one class. `pred`/`gt` are
integer (or boolean) label maps of identical shape; any dimensionality is
supported since IoU is purely voxel-counting.

```python
import numpy as np
from metrics import iou

pred = np.zeros((256, 256), dtype=np.uint8)
gt = np.zeros((256, 256), dtype=np.uint8)
pred[50:150, 50:150] = 1
gt[60:160, 60:160] = 1

score = iou(pred, gt, c=1)
```

**Edge case:** if class `c` is absent from both `pred` and `gt` (their union
is empty), `iou` returns `1.0` by convention â pred and gt trivially agree
there is nothing of that class, so this is treated as a perfect match rather
than an undefined `0/0`. This avoids NaNs silently poisoning any downstream
average over patients/classes.

## `dice(pred, gt, classes=None)`

Sorensen-Dice coefficient, derived algebraically from `iou` via
`Dice = 2*IoU / (1 + IoU)`, rather than reimplementing overlap counting.
Returns one score per class as a `np.ndarray`.

```python
from metrics import dice

# Score classes 1..4 of a 5-class (0=background) label volume:
scores = dice(pred_volume, gt_volume, classes=[1, 2, 3, 4])

# Or let it infer classes from the data:
scores = dice(pred_volume, gt_volume)
```

**Edge case:** a class absent from both `pred` and `gt` yields `Dice = 1.0`,
inherited directly from `iou`'s convention above (the algebra propagates it:
`IoU = 1.0` implies `Dice = 2*1/(1+1) = 1.0`).

## `hausdorff_distance_95(pred, gt, spacing, c=1)`

95th-percentile symmetric Hausdorff Distance, in mm, for one class of a 3D
volume. `spacing` is the physical voxel size `(sx, sy, sz)` in mm, in the
same axis order as `pred`/`gt` â e.g. the per-patient tuples saved to
`spacing.pkl` by [`slice_segthor.py`](slice_segthor.py).

```python
from metrics import hausdorff_distance_95

hd95 = hausdorff_distance_95(pred_volume, gt_volume, spacing=(0.98, 0.98, 2.5), c=1)

# Multi-class: loop over class values
hd95_per_class = [
    hausdorff_distance_95(pred_volume, gt_volume, spacing, c=k)
    for k in range(1, num_classes)
]
```

**Edge case:** a boundary distance is undefined when one side has no surface
to measure against.
- Class `c` absent from **both** `pred` and `gt`: returns `0.0` (no boundary
  disagreement is possible).
- Class `c` present in **exactly one** of `pred`/`gt`: returns `float("nan")`
  (there is no reference surface on the empty side).

## `average_surface_distance(pred, gt, spacing, c=1)`

Average Symmetric Surface Distance (ASD/ASSD), in mm: the mean of the pooled
pred-to-gt and gt-to-pred surface distances (one mean over every surface
voxel on both sides, not a mean of two per-direction means). Same input
convention as `hausdorff_distance_95`.

```python
from metrics import average_surface_distance

asd = average_surface_distance(pred_volume, gt_volume, spacing=(0.98, 0.98, 2.5), c=1)
```

**Edge case:** identical to `hausdorff_distance_95` â `0.0` when the class is
absent from both volumes, `float("nan")` when it's present in only one.

## `confusion_matrix(gt, pred, n_classes=5, spacing=None)`

Voxel-wise confusion matrix of shape `(n_classes, n_classes)`: **rows are
ground-truth labels, columns are predicted labels**, so entry `[i, j]` is the
number of voxels of true class `i` predicted as class `j`. Note the
`(gt, pred)` argument order, the reverse of the other functions. Computed in
one pass with `np.bincount` on `n_classes * gt + pred`.

```python
from metrics import confusion_matrix

cm = confusion_matrix(gt_volume, pred_volume, n_classes=5)
missed_esophagus_as_aorta = cm[1, 4]

# Volumes in mm³ (float) instead of voxel counts (int)
cm_mm3 = confusion_matrix(gt_volume, pred_volume, n_classes=5, spacing=(0.98, 0.98, 2.5))
```

Row sums are the ground-truth label counts and column sums the predicted
label counts. Raises `ValueError` if `gt`/`pred` shapes differ, if a label is
negative or `>= n_classes`, or if `spacing` doesn't have one value per axis.

## Implementation notes

`hausdorff_distance_95` and `average_surface_distance` delegate the distance
computation to [MONAI](https://docs.monai.io/en/stable/metrics.html):
`monai.metrics.compute_hausdorff_distance(percentile=95, directed=False)` and
`monai.metrics.compute_average_surface_distance(symmetric=True)`, with
`spacing` passed through so distances are in mm. MONAI finds each mask's
boundary voxels by binary erosion, then computes exact Euclidean
nearest-boundary distances in each direction (a KD-tree on CPU; cuCIM's GPU
distance transform for CUDA tensors when cuCIM is installed).

Both functions take an optional `device` argument (default `"cpu"`). With
`device="cuda"` the masks are moved to the GPU and MONAI uses cuCIM's GPU
erosion and distance transform. This needs cuCIM and CuPy installed: without
them MONAI computes the distances of CUDA tensors with SciPy on CPU, which is
slower than the CPU path.

The empty-mask rules above (`0.0` / `NaN`) are applied by our wrappers before
MONAI is called; MONAI on its own would return `NaN` when both masks are empty
and `NaN`/`inf` when only one is.

## eval.py

[`eval.py`](eval.py) scores a folder of merged prediction volumes (the
output of [`stitch.py`](stitch.py)) against ground-truth volumes, using
`dice`, `hausdorff_distance_95`, and `average_surface_distance` from
`metrics.py`.
`--pred_folder` must hold one `<patient_id>.nii.gz` file per patient (the
`stitch.py` convention); that file listing is what determines which
patients get scored. Voxel spacing is read straight from each patient's
ground-truth `.nii.gz` header (`nib.load(...).header.get_zooms()`), so no
separate `spacing.pkl` is needed.

`--gt_pattern` locates each patient's ground truth via a `{id_}`
placeholder — the same convention `stitch.py` already uses for
`--source_scan_pattern` — so it works equally well against a flat folder or
the raw nested SegTHOR layout, with no copying step needed:

```
# Flat folder of one <patient_id>.nii.gz per patient:
$ python eval.py --pred_folder volumes/segthor/ce \
    --gt_pattern "data/segthor_gt_val/{id_}.nii.gz" \
    --dest results/segthor/ce/eval_metrics.csv

# Directly against the raw nested layout (data/segthor_part1/train/<id>/GT.nii.gz):
$ python eval.py --pred_folder volumes/segthor/ce \
    --gt_pattern "data/segthor_part1/train/{id_}/GT.nii.gz" \
    --dest results/segthor/ce/eval_metrics.csv
```

If a predicted patient has no matching ground-truth file under
`gt_pattern`, `eval.py` fails with an error message and names exactly which patient ids
are missing.

- `--num_classes` is optional; if omitted, the set of classes is inferred
  from the ground-truth volumes. Class `0` is always treated as background
  and excluded from scoring.
- `--class_names` is optional and gives the organ name for each class index,
  starting with background (same convention as `viewer/viewer.py`). It
  defaults to the SegTHOR labels: `0` background, `1` esophagus, `2` heart,
  `3` trachea, `4` aorta. A class index beyond the list is reported by its
  number.
- `--gpu` computes HD95/ASSD on the GPU (Dice always runs on CPU). It is
  only used when CUDA is available and cuCIM and CuPy are installed
  (e.g. `pip install cupy-cuda12x cucim-cu12` for CUDA 12); otherwise
  `eval.py` prints why and evaluates on CPU. The device used is printed at
  the start.
- `--confusion_matrix` also saves voxel-count confusion matrices (rows =
  ground truth, columns = prediction, size `--num_classes`, or the number of
  `--class_names` if omitted): `eval_metrics_confusion_matrix.npz`, mapping
  each patient id to its `(K, K)` matrix (the readme's `.npz` submission
  format; load with `np.load(path)["Patient_01"]`), and
  `eval_metrics_confusion_matrix.png`, a heatmap of the matrix summed over
  all patients. Each cell is colored by its share of the ground-truth row
  (raw counts are dominated by background) and annotated with that share and
  the voxel count; organs absent from the ground truth show "no GT".
- Produces two CSV files: `eval_metrics.csv` (one row per
  `(patient_id, class)`, with an `organ` column and
  `dice`/`hausdorff_distance_95`/`average_surface_distance` columns — every
  individual score), and `eval_metrics_summary.csv` (one row per organ, with
  `..._mean`/`..._std` columns, aggregated across patients with
  `np.nanmean`/`np.nanstd`, and a `..._nan_count` column giving how many
  patients were NaN for that metric and therefore left out of the mean/std —
  for HD95/ASSD, the patients where the organ was missed or falsely
  predicted). The summary is also printed to stdout, one block per organ,
  with the NaN count next to each metric.
- Error messages if a patient's ground truth
  can't be found via `gt_pattern`, or if a patient's prediction and
  ground-truth volumes have different shapes.
