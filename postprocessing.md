# Segmentation post-processing

Post-processing refines predicted segmentation masks before evaluation by
removing components, filling enclosed holes, or applying morphological opening
and closing, denoising salt-and-pepper artifacts, or applying class-specific
anatomical component rules. The functions operate on full 3D NumPy label maps,
not individual slices or probability maps. Label `0` is background.

[`eval.py`](eval.py) loads compressed NIfTI files directly from a folder such as
`volumes/segthor/ce`, with one `<patient_id>.nii.gz` per patient. Filtering runs
in memory after loading and before computing metrics. Each filter returns a
copy with the original shape, dtype, and surviving class labels; discarded
voxels become background for component removal and opening, while hole filling
and closing can assign background voxels to a class. Salt-and-pepper denoising
can both remove and add voxels. Ground truth and source files are not overwritten.
By default, only metric CSVs are saved; `eval.py --save` also saves the evaluated
prediction volumes after optional post-processing.

### Spatial alignment assumption

Evaluation compares predictions and ground truth at matching array indices.
It checks that their shapes match, but does not resample or align images using
their NIfTI affine matrices. Predictions must therefore already use the same
voxel ordering and grid as ground truth. Distance metrics use ground-truth
voxel spacing.

In the five prediction files used for the initial comparison, the shapes match
ground truth, but the predictions have identity affines and 1 mm spacing while
ground truth has different spatial metadata. The Dice comparison assumes those
arrays are aligned despite this metadata discrepancy; matching shapes alone
do not verify alignment. The post-processing functions do not repair metadata
or establish spatial correspondence.

## Methods

### No filtering (baseline)

Evaluate the original prediction labels. This is the default in `eval.py`
(`--postprocessing none`) and the `baseline` entry in the comparison script.
It has no filtering hyperparameters.

### Top-k connected components per class

Implemented as `keep_largest_connected_components` in
[`postprocessing.py`](postprocessing.py).

For each selected foreground class, create a binary mask and label its 3D
connected components using SciPy's `ndimage.label`. Rank components by voxel
count, keep the largest k, and replace all other components with background.
Different classes never connect to each other for this operation.

For example, if one class has components of 500, 100, and 10 voxels, `k=2`
keeps the 500- and 100-voxel regions. If a class has fewer than k components,
all its components survive. Equal-size ties favor the component encountered
first in array traversal order. Absent classes have no effect.

| Python parameter | CLI option in `eval.py` | Default | Meaning |
|---|---|---|---|
| `k` | `--top_k` | `1` | Positive integer; maximum number of components retained **for each selected class**. The same k applies to all selected classes. |
| `classes` | `--postprocessing_classes` | `None` | Process all nonzero prediction labels by default. Supply labels such as `1 2` to filter only those classes. Unselected classes are unchanged; background is always skipped. |
| `connectivity` | `--connectivity` | `26` | Which neighboring voxels count as connected: 6, 18, or 26 (see below). |

Enable this method with `--postprocessing largest_connected_components`.
This ranks regions by relative size; it does not apply a minimum-size threshold
or determine anatomical plausibility. A valid disconnected organ fragment can
be removed even if it is large.

### Top-k connected components across combined foreground

Implemented experimentally as `keep_foreground_components(volume, k)` in
[`postprocessing.py`](postprocessing.py).

Create a single binary mask from `volume != 0`, label its connected components,
and keep the largest k components across that entire mask. Restore the original
class labels for surviving voxels. Touching regions of different classes count
as one component, but their labels are not merged in the returned segmentation.

| Parameter | Setting | Meaning |
|---|---|---|
| `k` | Required; comparison uses `1`, `2`, `3` | Maximum number of components across **all foreground classes together**. |
| Connectivity | Fixed at `26` in this helper | Includes face, edge, and corner neighbors. |
| Class selection | All nonzero labels | No class-selection argument is exposed. |

A disconnected correctly predicted organ may be removed. Conversely, a
false-positive region touching another organ may survive as part of a larger
component. This method is available in the comparison script and through the
Python callable interface, but is not an `eval.py` CLI choice.

### Remove small connected components per class

Implemented experimentally as `remove_small_components(volume, min_size)` in
[`postprocessing.py`](postprocessing.py).

Label components independently for every foreground class and remove only
components with fewer than `min_size` voxels. Any number of components can
survive, provided each meets the threshold. A component exactly at the threshold
is retained; a class can disappear entirely if all its components are too small.

For components of 500, 100, and 10 voxels, `min_size=100` keeps the first two.

| Parameter | Setting | Meaning |
|---|---|---|
| `min_size` | Required; comparison uses `10`, `100`, `1000` | Minimum retained component size in **voxels**, not mm³. Use a positive integer. |
| Connectivity | Fixed at `26` in this helper | Includes face, edge, and corner neighbors. |
| Class selection | All nonzero labels, independently | No class-selection argument is exposed. |

Because the threshold is a voxel count, its physical volume depends on scan
spacing. This method can remove small valid structures as well as false positives.
It is available in the comparison script and through the Python callable
interface, but is not an `eval.py` CLI choice.

### Anatomy-aware component filtering

Implemented as `anatomy_aware_filtering(volume, spacing, config)` in
[`postprocessing.py`](postprocessing.py). Enable it with
`--postprocessing anatomy_aware_filtering`.

The method labels 3D components independently for each configured organ and
measures their physical volume in mm³ using the NIfTI spacing. A component is
retained only if it meets both its class-specific minimum physical volume and
minimum fraction of that class's largest component. Each class can also cap the number of retained components, while `always_keep_largest` prevents an organ
from disappearing. Unconfigured labels are unchanged and removed voxels become
background; the method never adds or relabels foreground voxels.

All method parameters are stored under `anatomy_aware_filtering` in
[`postprocessing_config.json`](postprocessing_config.json):

| JSON setting | Meaning |
|---|---|
| `connectivity` | 3D component connectivity: 6, 18, or 26. |
| `classes` | Per-label rules; keys are positive integer label strings. |
| `min_volume_mm3` | Smallest retained physical component volume; `0` disables this threshold. |
| `min_relative_volume` | Smallest retained fraction of the largest same-class component, in `[0, 1]`. |
| `max_components` | Maximum retained components, or `null` for no cap. |
| `always_keep_largest` | Keep the largest component even when it fails a threshold. |

The SegTHOR config uses the rule of removing regions
smaller than 5% of the largest same-organ region. It conservatively permits
multiple qualifying esophagus fragments, while heart, trachea, and aorta are
capped at one component. The 5% rule was used by Feng et al. in
[Multi-organ Segmentation using Simplified Dense V-net with Post-processing](https://ceur-ws.org/Vol-2349/SegTHOR2019_paper_4.pdf).
Component rules should be retained only when validation improves, following
the validation-selected approach described by Isensee et al. in
[nnU-Net](https://doi.org/10.1038/s41592-020-01008-z).

```sh
python eval.py \
    --pred_folder results/segthor_seed43/exp_L6_balance_normalized/results/pred_volumes \
    --gt_pattern 'results/segthor_seed43/exp_L6_balance_normalized/results/gt_volumes/{id_}.nii.gz' \
    --num_classes 5 --postprocessing anatomy_aware_filtering \
    --postprocessing_config postprocessing_config.json \
    --dest results/segthor_seed43/exp_L6_balance_normalized/results/anatomy_aware.csv \
    --save --save_folder results/segthor_seed43/exp_L6_balance_normalized/results/anatomy_aware_volumes
```

### Fill holes per class

Implemented as `fill_holes(volume, classes=None, connectivity=6)` in
[`postprocessing.py`](postprocessing.py), using `scipy.ndimage.binary_fill_holes`.
Enable it with `--postprocessing fill_holes`.

For each selected class, treat its voxels as a binary mask and identify cavities
that cannot reach the boundary of the volume through the complement of that
mask. Fill those cavities only where the original segmentation is background
(`0`). Existing labels, including other organs inside a cavity, are preserved.
The function returns a copy with the original shape and dtype.

This is a **3D** operation: an apparent hole in one slice is not filled if it
has a path to the outside through another slice. It does not close open gaps,
smooth boundaries, or remove disconnected blobs. There is no hole-size limit;
all unambiguous enclosed cavities are filled, including large ones.

| Python parameter | CLI option | Default | Meaning |
|---|---|---|---|
| `classes` | `--postprocessing_classes` | `None` | Fill holes for all nonzero prediction labels, or only the supplied labels. Background and absent labels are skipped. |
| `connectivity` | `--connectivity` | `6` | Neighborhood used for background paths to the boundary: 6, 18, or 26. This matches SciPy's default face-connected background. |

With 26-connectivity, even a corner-connected escape path prevents filling;
with 6-connectivity, only face-connected paths prevent filling. Thus increasing
this parameter can leave more cavities unfilled. `--top_k` applies only to
component filtering and has no effect on hole filling.

If two selected classes both enclose the same background voxel (for example,
nested shells), it remains background. This avoids assigning ambiguous voxels
according to class-processing order. Duplicate labels in `classes` have no effect.
Real anatomical cavities can also be filled, so compare validation scores before
choosing this method.

```sh
python eval.py --pred_folder volumes/segthor/ce \
    --gt_pattern 'data/segthor_part1/train/{id_}/GT.nii.gz' \
    --postprocessing fill_holes --connectivity 6 \
    --dest results/segthor/ce/fill_holes.csv \
    --save --save_folder volumes/segthor/ce_filled
```

Omit `--save` and `--save_folder` to evaluate without writing volumes. For Python
callers, use `postprocess=partial(fill_holes, classes=[1, 2], connectivity=6)`
with `evaluate_patient` or `evaluate_dataset`.

### Morphological opening per class

Implemented as `opening(volume, iterations=1, classes=None, connectivity=6)`
using `scipy.ndimage.binary_opening`. Select it with `--postprocessing opening`.

Opening first erodes each selected class mask, then dilates the remaining mask.
It can remove isolated specks, thin protrusions, and narrow connections.
Only original voxels of the selected class can be removed; they become
background. Other classes are preserved. Thin valid structures can disappear,
and even large objects can lose boundary details.

### Morphological closing per class

Implemented as `closing(volume, iterations=1, classes=None, connectivity=6)`
using `scipy.ndimage.binary_closing`. Select it with `--postprocessing closing`.

Closing first dilates each selected class mask, then erodes it. It can bridge
narrow gaps and fill small cavities or indentations, including gaps open to the
outside. Unlike hole filling, it does not fill every enclosed cavity regardless
of size. It may also merge nearby regions that should remain separate.

Only original background voxels can be assigned a class; all existing labels
are preserved. If multiple classes propose the same background voxel, it stays
background, independent of class order. This conservative multi-class rule can
leave gaps that separate binary closings would each fill.

Both operations treat space outside the image as background. Closing pads the
mask by `iterations` voxels before processing, then crops back to the input
shape. This prevents the artificial boundary erosion that an unpadded closing
can cause. Both return a copy with the original shape and dtype.

#### Opening and closing hyperparameters

| Python parameter | CLI option | Default | Meaning |
|---|---|---|---|
| `iterations` | `--iterations` | `1` | Positive integer; number of steps in **each stage**. Opening with 2 means two erosions followed by two dilations, not two complete opening operations. Closing reverses that order. Larger values act over a larger neighborhood. |
| `connectivity` | `--connectivity` | `6` | Structuring element: center plus 6 face neighbors, 18 face/edge neighbors, or all 26 neighbors (a full 3×3×3 cube). |
| `classes` | `--postprocessing_classes` | `None` | Process all nonzero labels, or only those supplied. Background, absent labels, and duplicate selections have no additional effect. |

The neighborhood is measured in voxels, not millimeters; anisotropic voxel
spacing makes its physical extent different along different axes. `--top_k`
has no effect on either operation. Multiple methods can be listed after
`--postprocessing`; they run from left to right.

```sh
python eval.py --pred_folder volumes/segthor/ce \
    --gt_pattern 'data/segthor_part1/train/{id_}/GT.nii.gz' \
    --postprocessing opening --iterations 1 --connectivity 6 \
    --dest results/segthor/ce/opening.csv --save

python eval.py --pred_folder volumes/segthor/ce \
    --gt_pattern 'data/segthor_part1/train/{id_}/GT.nii.gz' \
    --postprocessing closing --iterations 1 --connectivity 6 \
    --dest results/segthor/ce/closing.csv --save
```

Python callers can use `postprocess=partial(opening, iterations=1)` or
`postprocess=partial(closing, iterations=1)` with the evaluation functions.

### Salt-and-pepper denoising per class

Implemented as `salt_and_pepper(volume, kernel_size=3, classes=None)` using
`scipy.ndimage.median_filter`. Select it with `--postprocessing salt_and_pepper`.

Apply a 3D median filter to each selected class's **binary mask**, rather than
to numeric class IDs (whose ordering has no anatomical meaning). In a 3×3×3
window, the center is assigned foreground only if at least 14 of its 27 samples
belong to that class. This can remove isolated foreground specks (salt) and
fill small background gaps inside foreground regions (pepper).

Rejected voxels of a selected class become background. Accepted voxels are
added only where the original volume was background. Existing foreground labels
are never directly reassigned to a different class; a selected class can still
lose its own voxels. Unselected classes are unchanged. Every mask is computed
from the original input, so class order and duplicate selections have no effect.
With the same odd-sized window, two classes cannot both have a strict majority
at the same voxel.

| Python parameter | CLI option | Default | Meaning |
|---|---|---|---|
| `kernel_size` | `--kernel_size` | `3` | Positive odd side length in voxels of the cubic window. `3` gives 3×3×3; `5` gives 5×5×5. `1` returns an unchanged copy. Larger windows smooth more aggressively and cost more computation. |
| `classes` | `--postprocessing_classes` | `None` | Process all nonzero labels, or only those supplied. Background and absent labels are skipped. |

The filter uses nearest-edge padding: locations outside the image repeat the
nearest boundary voxel. Window size is measured in voxels, not millimeters.
Thin valid structures and small organs can be removed, and boundaries can
shrink. A wrong foreground label inside another organ may become background
rather than being reassigned to that organ under the conservative label rule.

`--connectivity` is not applicable and is rejected for this method; `--top_k`
and `--iterations` do not affect it. The function returns a copy with the input
shape and dtype.

```sh
python eval.py --pred_folder volumes/segthor/ce \
    --gt_pattern 'data/segthor_part1/train/{id_}/GT.nii.gz' \
    --postprocessing salt_and_pepper --kernel_size 3 \
    --dest results/segthor/ce/salt_and_pepper.csv --save
```

Python callers can use `postprocess=partial(salt_and_pepper, kernel_size=3)`
with either evaluation function. Saved `.nii.gz` volumes retain prediction
metadata, following the same `--save` / `--save_folder` behavior as other methods.

## Connectivity

| Value | Neighbor relationship in 3D |
|---|---|
| `6` | Voxels sharing a face. |
| `18` | Voxels sharing a face or edge. |
| `26` | Voxels sharing a face, edge, or corner. |

Larger neighborhoods can join regions that would be separate under a smaller
neighborhood. Connectivity is defined on the voxel grid, not by a physical
distance in millimeters. Component filtering uses this neighborhood to connect
foreground voxels; hole filling uses it to connect the background of each class
mask. Opening and closing use it to construct the erosion/dilation neighborhood.
The existing component comparison uses `26`; hole filling, opening, and closing
default to `6`.

## Evaluate a method

Run from the project root, adjusting `--gt_pattern` to your ground-truth files.
Use distinct output paths to retain results from different settings:

```sh
python eval.py --pred_folder volumes/segthor/ce \
    --gt_pattern 'data/segthor_part1/train/{id_}/GT.nii.gz' \
    --dest results/segthor/ce/baseline.csv

python eval.py --pred_folder volumes/segthor/ce \
    --gt_pattern 'data/segthor_part1/train/{id_}/GT.nii.gz' \
    --postprocessing largest_connected_components --top_k 2 --connectivity 26 \
    --dest results/segthor/ce/components_k2.csv
```

Add `--postprocessing_classes 1 2` to filter only those labels. This controls
filtering, not which classes are scored. By default evaluation computes Dice,
HD95, and average surface distance; use `--metrics dice` to evaluate Dice only.
Each run writes per-patient/per-class scores and a corresponding `_summary.csv`.
See [`metrics.md`](metrics.md) for metric definitions and other evaluation options.

### Save the processed volumes

Add `--save` to write one `<patient_id>.nii.gz` per patient. By default, a CSV
destination of `results/segthor/ce/components_k2.csv` produces volumes in
`results/segthor/ce/components_k2_volumes/`. Set `--save_folder` to choose another
folder; this option requires `--save`:

```sh
python eval.py --pred_folder volumes/segthor/ce \
    --gt_pattern 'data/segthor_part1/train/{id_}/GT.nii.gz' \
    --postprocessing largest_connected_components --top_k 2 \
    --dest results/segthor/ce/components_k2.csv \
    --save --save_folder volumes/segthor/ce_k2
```

The saved labels are the predictions used to compute the metrics. Saving
preserves the prediction's affine, qform/sform codes, voxel spacing, and header
metadata; it does not copy ground-truth geometry or repair existing metadata
discrepancies. Output paths matching the patient's prediction or ground truth
are rejected. Existing files in the output folder are replaced, so use a
different folder for each configuration you want to keep. With
`--postprocessing none`, `--save` saves the unfiltered predictions.

Python callers can pass `save_folder=Path(...)` to `evaluate_patient` or
`evaluate_dataset`. The comparison script continues to save CSVs only.

Python callers can pass any callable taking and returning a 3D label map to
`evaluate_patient` or `evaluate_dataset`:

```python
from functools import partial
from pathlib import Path
from eval import evaluate_dataset
from postprocessing import keep_largest_connected_components

rows = evaluate_dataset(
    Path('volumes/segthor/ce'),
    'data/segthor_part1/train/{id_}/GT.nii.gz',
    metrics=['dice'],
    postprocess=partial(keep_largest_connected_components, k=2, connectivity=26),
)
```

For the experimental methods, import their helpers from `postprocessing`
and use `partial(keep_foreground_components, k=2)` or
`partial(remove_small_components, min_size=100)` as the callable.

## Reproduce the comparison

```sh
python compare_postprocessing.py \
    --pred_folder volumes/segthor/ce \
    --gt_pattern 'data/segthor_part1/train/{id_}/GT.nii.gz' \
    --dest_folder results/postprocessing_comparison
```

The script evaluates the baseline, both top-k methods at k=1, 2, and 3, and
per-class minimum sizes of 10, 100, and 1,000 voxels. These settings are defined
in the script's `methods` construction, not exposed as CLI flags. Its three CLI
options select the prediction folder, ground-truth pattern, and output folder;
the defaults match the command above.

Hole filling, opening, closing, and salt-and-pepper denoising are available in `eval.py` but are not
included in this comparison script.

It computes **Dice only** and writes `per_patient.csv` and `summary.csv`.
The summary includes mean Dice per class, the mean over all patient/class pairs,
the mean change versus baseline, and counts of improved, worsened, or unchanged
patient/class pairs (changes within `1e-12` count as unchanged).

In the initial comparison on five volumes and foreground labels 1, 2, and 3,
no tested filter improved overall mean Dice over the baseline (`0.589462`).
Removing components smaller than 10 voxels was nearly neutral (`0.589447`).
Combined-foreground top-k outperformed per-class top-k at each tested k, but
still scored below baseline. These findings concern those predictions and
settings; HD95 and average surface distance were not compared.
