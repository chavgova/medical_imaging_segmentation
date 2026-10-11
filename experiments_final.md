# stage 1

| ID | Parent | Exact change | Why this candidate / selection | Macro Dice ↑ | HD95 mm ↓ | ASD mm ↓ |
|---|---|---|---|---:|---:|---:|
| P0 | R1 | Existing `--hu_min -1000 --hu_max 300` | Wide-window control; no new run. Retains low-HU airway context. | 0.7765 | 14.884 | 3.407 |
| P1 | R1 | `--hu_min -310 --hu_max 400` | Higher soft-tissue contrast per PNG level for mediastinal organs; check trachea performance after low-HU clipping. Core. | 0.7895 | 14.102 | 2.909 |
| P2 | R1 | `--clahe --hu_min -310 --hu_max 400` | Compare to P1 to isolate adaptive contrast. Existing implementation fixes clip limit 0.01; do not invent a CLI parameter sweep. Core. | 0.7955 | 13.270 | 2.840 |
| P3 | R1 | `--hu_windows -1000 300 -310 400` | Complementary wide/narrow information. Five slices x two windows = 10 channels, supported by U-Net. Wide window first also preserves DINO's established input. Core. | 0.7891 | 18.984 | 3.224 |


Values are the unweighted mean of the four organ means in `results_new/*.csv`: `macro = (esophagus + heart + trachea + aorta) / 4`; background excluded. Each organ mean averages validation patients. 

**P2 wins stage 1** on macro metrics

# stage 2


| ID | Parent | Exact change | Why / stop rule | Status | Macro Dice ↑ | HD95 mm ↓ | ASD mm ↓ |
|---|---|---|---|---|---:|---:|---:|
| A0 | P1 | Existing rotation/noise `--augment`, scale 0 | Existing control. Rotation is +/-5 degrees, probability 0.5; Gaussian noise probability 0.25. | Reuse | 0.7895 | 14.102 | 2.909 |
| A1 | A0 (P1) | `--augment --augment_scale 0.15` | Modest shared zoom 0.85-1.15, probability 0.5. Check clipping after zoom; a scale sweep is unjustified. | Completed | 0.8086 | 14.476 | 2.731 |
| A2 | A1 (P1) | Preprocessing `--crop_body` | Crop improves all three macro metrics; retain inverse geometry. | Completed; selected stage 2 | 0.8343 | 9.461 | 2.116 |
| A2-P2 | A2 | Add preprocessing `--clahe`; retain crop/window/scale | Test P2 contrast on the best complete recipe. | Completed; retain A2 | 0.8308 | 10.447 | 2.243 |
| A3 | A2 | Set `--augment_scale 0`, keep old augmentation and crop | Check whether scale still helps after cropping. | Completed; retain A2 | 0.8102 | 12.132 | 2.659 |
| G1-resolution | Best available A configuration | Preprocessing `--shape 512 512`, no other change | Run only if esophagus/contour errors suggest information loss at 256 and memory profiling fits batch 8. This adds real image detail, unlike enlarging DINO's already-downsampled image. | Conditional: 1 | Pending | - | - |



# stage 3

| ID | Parent | Change | Why / decision | Status | Macro Dice ↑ | HD95 mm ↓ | ASD mm ↓ |
|---|---|---|---|---|---:|---:|---:|
| C0 | A2 (P1, crop, scale 0.15) | Existing `--context_slices 2` (five slices) | Incoming winner. Physical support is patient-dependent because z spacing is unchanged. | Reuse | 0.8343 | 9.461 | 2.116 |
| C1 | C0 | `--context_slices 0` | Single-slice comparison on cropped A2. | Completed; retain C0 | 0.8288 | 10.638 | 2.388 |
| C2 | C0 | `--context_slices 1` (three slices) | Intermediate context; compare directly with C0 and C1 on A2. | Completed; retain C0 | 0.8273 | 9.686 | 2.279 |



# stage 4

| ID | Parent | Exact change | Why / decision | Status | Macro Dice ↑ | HD95 mm ↓ | ASD mm ↓ |
|---|---|---|---|---|---:|---:|---:|
| L0 | A2 = C0 | Binary Balance, normalization none, alpha 0.5, t 0.9, fallback 12 | Lowest small-model HD95/ASD; carry Balance forward. | Selected loss | 0.8343 | 9.461 | 2.116 |
| L2 | L0 | `--loss_fn dicece --dicece_lambda 0.5`, no CE weights | Higher Dice; worse HD95/ASD than L0. | Completed; trade-off | 0.8554 | 12.266 | 2.194 |
| L3 | L2 | `--ce_weights invfreq --ce_weights_alpha 0.5` | Versus L2: lower Dice, slightly lower HD95, higher ASD; retain Balance. | Completed; not selected | 0.8503 | 11.965 | 2.379 |


# stage 5

| ID | Parent | Change | Why / decision | Status | Macro Dice ↑ | HD95 mm ↓ | ASD mm ↓ |
|---|---|---|---|---|---:|---:|---:|
| M0 | A2 = L0 | Existing `--model unet-small`, Balance loss | Matched control for M1. | Reuse | 0.8343 | 9.461 | 2.116 |
| M1 | M0 | `--model unet-large` | Improves all metrics versus M0; best surface distances, selected compromise versus M2. | Completed; selected | 0.8689 | 7.020 | 1.653 |
| M2 | M1 | `--loss_fn dicece --dicece_lambda 0.5`, no CE weights | Highest Dice; versus M1: +0.0069 Dice, +2.855 mm HD95, +0.438 mm ASD. | Completed; Dice alternative | 0.8758 | 9.875 | 2.092 |


Best config so far: linear HU [-310, 400], body cropping, rotation/noise + scale 0.15, five-slice context, large U-Net, Balance loss; no foundation (M1). M2 remains the Dice-leading alternative.


# stage 6

| ID | Parent | Exact change | Why / decision | Status | Macro Dice ↑ | HD95 mm ↓ | ASD mm ↓ |
|---|---|---|---|---|---:|---:|---:|
| F0 | M1 | No foundation; large U-Net + Balance | Selected stage-5 control; reuse existing metrics. | Reuse | 0.8689 | 7.020 | 1.653 |
| F1 | F0 | `--foundation_model meddinov3-vitb16 --foundation_fusion encoder --foundation_upsample 1` | Worse than M1 on all macro metrics; heart improves, other organs worsen. | Completed; not selected | 0.8644 | 9.864 | 1.808 |
| F2 | F1, if beneficial | Change `--foundation_upsample 2`; Tests a finer feature/fusion config. At 256 input: DINO input 512 and patch grid 32 x 32, fusion level 3 instead of 4 | Conditional on beneficial F1. | Skip; F1 not beneficial | - | - | - |
| F3 | F1 settings; M1 promotion control | Change only `--foundation_fusion decoder`; retain upsample 1 | One fusion-placement test; adopt only if better than M1. | Completed; not selected | 0.8595 | 7.987 | 1.729 |


Stage 6 selects M1 (no foundation model)

# stage 7

| ID | Parent | One controlled change | Trigger | Macro Dice ↑ | HD95 mm ↓ | ASD mm ↓ |
|---|---|---|---|---:|---:|---:|
| X1 | M1 | Alternative linear-HU representation | Skip: no foundation selected | - | - | - |
| X2 | M1 | Alternative loss | Covered by M2: retain Balance for lower surface errors; no duplicate run. | - | - | - |
| X3 | M1 | `--context_slices 0`; all else fixed | Completed; M1 wins all three macro metrics. | 0.8664 | 7.826 | 1.755 |


# stage 8

M1 remains selected: large U-Net, Balance, five slices, linear HU [-310, 400], crop, scale 0.15, no foundation

| ID | Input | Candidate settings | Status / purpose | Macro Dice ↑ | HD95 mm ↓ | ASD mm ↓ |
|---|---|---|---|---:|---:|---:|
| Q0 | M1 raw | No filtering | Completed; matches M1 exactly. | 0.8689 | 7.020 | 1.653 |
| Q1 | Q0 | Largest component per class; k=1, connectivity 26, classes 2/3/4 | Completed; helps aorta, harms trachea. | 0.8691 | 7.153 | 1.652 |
| Q2 | Q0 | Anatomy-aware config: relative minimum 0.05; esophagus may retain fragments; classes 2/3/4 capped at 1 | Completed; trachea HD95 worsens; superseded by Q2a. | 0.8693 | 7.100 | 1.636 |
| Q3 | Q0 | Q2 with esophagus relative minimum 0.01 | Deferred; current esophagus filtering improves all metrics. | - | - | - |
| Q4 | Q0 | Fill holes; heart only, connectivity 6 | Completed; tiny heart improvement on all metrics. | 0.8690 | 7.004 | 1.652 |
| Q5 | Q0 | Closing; heart only, 1 iteration, connectivity 6 | Completed; heart HD95/ASD worsen. | 0.8690 | 7.022 | 1.658 |
| Q6 | Q0 | Opening; heart only, 1 iteration, connectivity 6 | Completed; heart HD95/ASD worsen. | 0.8690 | 7.026 | 1.660 |
| Q6b | Q0 | Salt-and-pepper median; heart only, kernel 3 | Completed; heart ASD worsens. | 0.8691 | 7.016 | 1.659 |
| Q7 | M1 probabilities + CT | Dense CRF; configured defaults, 5 iterations, spatial/bilateral weights 3/5 | Completed; harms overall result, helps aorta only. | 0.8396 | 12.810 | 2.319 |
| Q8 | Q7 inputs | Spatial/bilateral weights 1.5/2.5 | Deferred; test saved aorta-only result first. | - | - | - |
| Q9 | Best individual methods | Combine best 2-3; CRF first, then component filtering, then accepted morphology | See Q9a/Q9b; no full-organ CRF combination. | - | - | - |
| Q10 | Q0 | Combined-foreground components; k=4, connectivity 26 | Completed; negligible improvement. | 0.8689 | 7.020 | 1.653 |
| Q11 | Q0 | Remove per-class components smaller than 100 voxels, connectivity 26 | Completed; trachea surface errors worsen. | 0.8691 | 7.531 | 1.675 |
| Q2a | Q0 | Q2 anatomy rules without class 3 | Completed; improves all macro metrics versus Q0. | 0.8694 | 6.350 | 1.581 |
| Q9a | Q0 | Q2a then Q4 heart hole-filling | Completed; lowest HD95; non-CRF shortlist. | 0.8695 | 6.328 | 1.579 |
| Q7a | Q0 + saved Q7 | Transfer Q7 aorta only into raw background/aorta voxels; preserve classes 1/2/3 | Completed; improves all macro metrics versus Q0. | 0.8754 | 6.815 | 1.549 |
| Q9b | Q0 + saved Q7 | Q7a, then Q2a, then Q4 | Completed; superseded by Q9c; aorta filtering worsens HD95. | 0.8757 | 7.029 | 1.532 |
| Q9c | Q0 + saved Q7 | Q7a, anatomy rules for classes 1/2 only, then heart hole-filling; preserve CRF aorta and raw trachea | Completed; Dice/ASD shortlist, improves all metrics versus Q0. | 0.8757 | 6.740 | 1.531 |



# stage 9

| ID | Configuration / runs | Purpose and decision |
|---|---|---|
| V1 | M1 seeds 43/44/45 on the same fold-0 split; raw, Q9a and Q9c evaluated per model | Seed stability confirmed on this split; see results below. |
| V2 | M1 with AdamW, LR 3e-4, decay 1e-4, cosine to 3e-6; maximum 60 epochs, Dice early stopping minimum 40, patience 15, delta 1e-4; seeds **43/44/45**, same 30/10 split. | Final optimization experiment: Q9c wins Dice/ASD; V1 + Q9a retains lowest HD95. |



**V1 results** (seed 43: M1/Q9a/Q9c/Q7 above).

| Seed | Recipe | Macro Dice ↑ | HD95 mm ↓ | ASD mm ↓ |
|---|---|---:|---:|---:|
| 44 | Raw | 0.8646 | 7.613 | 1.703 |
| 44 | Q9a | 0.8664 | 6.407 | 1.599 |
| 44 | Q9c | 0.8724 | 6.698 | 1.556 |
| 44 | Full CRF | 0.8198 | 12.594 | 2.547 |
| 45 | Raw | 0.8620 | 6.994 | 1.707 |
| 45 | Q9a | 0.8632 | 6.012 | 1.616 |
| 45 | Q9c | 0.8682 | 6.206 | 1.567 |
| 45 | Full CRF | 0.8621 | 9.373 | 1.859 |

| Recipe | Dice mean ± SD | HD95 mm mean ± SD | ASD mm mean ± SD |
|---|---:|---:|---:|
| Raw | 0.8652 ± 0.0035 | 7.209 ± 0.350 | 1.688 ± 0.030 |
| Q9a | 0.8664 ± 0.0031 | 6.249 ± 0.209 | 1.598 ± 0.019 |
| Q9c | 0.8721 ± 0.0038 | 6.548 ± 0.297 | 1.552 ± 0.018 |
| Full CRF | 0.8405 ± 0.0212 | 11.592 ± 1.925 | 2.242 ± 0.350 |


**V2 final results** (`V2*.csv` without a seed suffix denotes seed 43).

| Seed | Recipe | Macro Dice ↑ | HD95 mm ↓ | ASD mm ↓ |
|---|---|---:|---:|---:|
| 43 | Raw | 0.8745 | 7.413 | 1.594 |
| 43 | Q9a | 0.8756 | 6.462 | 1.505 |
| 43 | Q9c | 0.8802 | 6.947 | 1.481 |
| 43 | Full CRF | 0.8546 | 9.367 | 1.840 |
| 44 | Raw | 0.8715 | 7.220 | 1.602 |
| 44 | Q9a | 0.8719 | 6.877 | 1.562 |
| 44 | Q9c | 0.8774 | 7.133 | 1.520 |
| 44 | Full CRF | 0.8500 | 10.920 | 2.019 |
| 45 | Raw | 0.8725 | 7.578 | 1.605 |
| 45 | Q9a | 0.8731 | 6.913 | 1.565 |
| 45 | Q9c | 0.8784 | 6.935 | 1.512 |
| 45 | Full CRF | 0.8592 | 9.519 | 1.795 |

| Recipe | Dice mean ± SD | HD95 mm mean ± SD | ASD mm mean ± SD |
|---|---:|---:|---:|
| Raw | 0.8729 ± 0.0015 | 7.404 ± 0.179 | 1.600 ± 0.006 |
| Q9a | 0.8735 ± 0.0019 | 6.751 ± 0.250 | 1.544 ± 0.034 |
| Q9c | 0.8787 ± 0.0014 | 7.005 ± 0.111 | 1.504 ± 0.021 |
| Full CRF | 0.8546 ± 0.0046 | 9.935 ± 0.856 | 1.885 ± 0.118 |


- **Final selection:** V2 + Q9c for highest Dice and lowest ASD; V1 + Q9a for lowest HD95 (6.249 mm). Best individual Dice/ASD run: V2 seed 43 + Q9c, **0.8802 / 6.947 mm / 1.481 mm**; use the three-seed summary to compare configurations. Full-organ CRF remains worse than raw V2 on all macro metrics.

### Findings

- **Stage 1 - preprocessing:** CLAHE wins initially but worsens trachea versus P1. Multiwindow improves trachea but sharply worsens aorta HD95; narrow linear HU is retained after stage 2.
- **Stage 2 - augmentation/crop:** Scale 0.15 improves Dice/ASD; cropping further improves all macro metrics, especially esophagus/aorta. Cropping slightly worsens trachea HD95. Removing scale or adding CLAHE to A2 worsens all macro metrics.
- **Stage 3 - context:** Five slices outperform one or three on all macro metrics with small U-Net.
- **Stage 4 - loss:** DiceCE improves overlap, particularly esophagus Dice, but worsens macro surface distances. CE weighting lowers Dice and raises ASD versus unweighted DiceCE; retain Balance.
- **Stage 5 - model:** Large U-Net + Balance improves every organ's Dice/HD95/ASD over small U-Net. Large + DiceCE raises esophagus Dice (0.7580 -> 0.7813), but worsens macro HD95/ASD.
- **Stage 6 - foundation:** Both tested MedDINOv3 fusion placements improve heart metrics but worsen all macro metrics; retain no foundation. 
- **Stage 7 - final context check:** Five slices also beat one slice on all macro metrics with large U-Net.
- **Stage 8 - postprocessing:**
  - Anatomy filtering helps esophagus/aorta but harms tracheal surface accuracy; so we need to preserve raw trachea. Esophagus Dice gains are small.
  - Heart hole-filling helps slightly; closing/opening/median worsen heart ASD. Full CRF harms overall results but helps aorta.
  - Q9c combines CRF aorta, anatomy filtering of esophagus/heart, and heart hole-filling. Filtering the CRF aorta again worsens its metrics; Q9a is the cheaper, lower-HD95 alternative.
- **Stage 9 - optimization/best configuration:** V2 + Q9c has the highest mean Dice and lowest ASD, with lower Dice variation across seeds; tracheal surface errors worsen. Retain V1 + Q9a when prioritizing HD95. Same-split validation only.
  - **Selected Dice/ASD recipe:** linear HU [-310, 400], body crop, rotation/noise + scale 0.15, five slices, large U-Net, Balance, no foundation; AdamW LR 3e-4, weight decay 1e-4, cosine to 3e-6, up to 60 epochs with early stopping; Q9c postprocessing. Mean Dice/HD95/ASD: **0.8787 / 7.005 mm / 1.504 mm**.
