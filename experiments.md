# Experiment plan and results

Completed experiments use SegTHOR fold 0 (15 training patients, 5 validation
patients). Training uses deterministic seed 43 except V1/V2, which reuse the
same patient split with seeds 44/45. Reported results are unweighted macro
averages over foreground classes 1-4. Higher Dice is better; lower HD95 and
ASD are better.

| Stage | ID | Change from current winner | Macro Dice | Macro HD95 (mm) | Macro ASD (mm) | Status and future use |
|---|---|---|---:|---:|---:|---|
| Legacy reference | B0 | Original per-volume min-max and uncorrected merged ground truth | - | - | - | Completed, but four-organ results were not recorded here and are not directly comparable to fixed-GT runs |
| Fixed-GT reference | R0 | Corrected labels, original per-volume min-max, 2D, CE, Adam, LR `5e-4`, no augmentation/sampling/scheduler/post-processing | 0.515 | 29.330 | 6.225 | Completed; scientific fixed-GT reference |
| Preprocessing | P1 | Fixed HU window `[-1000, 300]` | **0.595** | **18.454** | 4.906 | **Selected preprocessing**; used by D1-D3 and all later stages |
| Preprocessing | P2 | CLAHE after clipping to `[-1000, 300]` | 0.591 | 23.018 | **4.823** | Completed; not selected because P1 has better Dice and HD95 |
| Data handling | D1 | P1 plus `--augment` | **0.631** | 25.242 | **4.669** | **Selected data method**; carry P1 plus augmentation into the context stage |
| Data handling | D2 | P1 plus `--drop_empty_slices 0.9` | 0.590 | 21.931 | 5.263 | Completed; rejected because it does not improve P1 overall |
| Data handling | D3 | P1 plus `--oversample_foreground --oversample_foreground_percent 0.5` | 0.541 | 29.290* | 10.489* | Completed; rejected after near-total esophagus failure |
| Data handling | D4 | P1 plus augmentation and `--drop_empty_slices 0.9` | 0.515 | 23.406 | 6.151 | Completed; rejected because combining augmentation and slice dropping substantially harms three of four classes |
| Context | C1 | P1 plus D1 augmentation and `--context_slices 1` | 0.589 | 26.047 | 5.367 | Completed; rejected because three-slice context is worse than D1 on every macro metric |
| Context | C2 | P1 plus D1 augmentation and `--context_slices 2` | **0.640** | **17.419** | **4.315** | **Selected context method**; five-slice 2.5D input is carried into the loss stage |
| Loss reference | L0 | Selected P1 + D1 + C2 pipeline with plain CE | 0.640 | 17.419 | 4.315 | Completed by C2; no duplicate L0 run is needed |
| Loss | L1 | L0 with CE and `--ce_weights invfreq --ce_weights_alpha 0.5` | 0.656 | 26.908 | 5.013 | Completed; modest Dice gain, substantially worse surface distances |
| Loss | L2 | L0 with Dice | 0.501 | 140.135 | 47.757 | Completed; rejected due to severe segmentation and boundary degradation |
| Loss | L3 | L0 with DiceCE and `--dicece_lambda 0.5` | 0.688 | 20.471 | 4.116 | Completed; better Dice than L4, but not selected after L6 |
| Loss | L4 | L0 with Balance, `alpha=0.5`, `t=0.9`, and automatic halfway fallback | 0.683 | 15.631 | 3.788 | Completed; retained as the unnormalized Balance reference |
| Loss follow-up | L5 | Weighted DiceCE | - | - | - | Deferred; weighted CE did not improve the overall trade-off enough to justify combining it with DiceCE yet |
| Loss follow-up | L6 | L4 with `--balance_normalized` | **0.713** | 16.324 | **3.484** | **Selected loss**; best Dice and ASD, with HD95 `0.693 mm` above L4 |
| Learning rate | T1 | L6 pipeline with Adam and LR `1e-4` | 0.668 | 18.255 | 4.270 | Completed; rejected, worse than L6 on all three metrics |
| Learning rate | T2 | L6 pipeline with Adam and LR `1e-3` | 0.701 | **14.563** | 3.629 | Completed; best HD95, but lower Dice and worse ASD than L6 |
| Optimizer | T3 | L6 pipeline at LR `5e-4` with AdamW instead of Adam | 0.706 | 17.531 | 3.566 | Completed; worse than L6 on all three metrics |
| Scheduler | T4 | L6 pipeline with StepLR (`step_size=10`, `gamma=0.1`) | 0.691 | 17.806 | 3.925 | Completed; worse than L6 on all three metrics |
| Scheduler | T5 | L6 pipeline with ReduceLROnPlateau (`patience=5`, `gamma=0.1`) | 0.696 | 14.967 | 3.717 | Completed; improves HD95 over L6, but T2 is better on all three metrics |
| Scheduler | T6 | L6 pipeline with CosineAnnealingLR (`T_max=25`) | 0.700 | 16.521 | 3.751 | Completed; worse than L6 on all three metrics |
| Early stopping | T7 | L6 with `--early_stopping_patience 10` | 0.713 | 16.324 | 3.484 | Completed; same reported metrics as L6|
| Stability | V1 | L6 on the same fold-0 patient split with training seed 44 | 0.684 | 17.239 | 3.940 | Completed; lower Dice and worse distances than seed 43 |
| Stability | V2 | L6 on the same fold-0 patient split with training seed 45 | 0.701 | 17.131 | 3.750 | Completed;lower Dice and worse distances than seed 43 |

\* D3 failed to predict the esophagus for Patient 03. Its class-1 HD95 and ASD
were `NaN` so those values are ommited when computing the
class means  (D3's reported distance averages are therefore optimistic)

## Stage decisions

- Preprocessing winner: P1 fixed HU windowing
- Data-handling winner: D1 augmentation. It has the best macro Dice and ASD,
  but its trachea HD95 regression must remain visible in later comparisons.
- D4 confirms that adding aggressive empty-slice dropping to augmentation is
  not beneficial. D1 remains the selected data-handling method.
- Context winner: C2 with two neighboring slices on each side. Relative to D1,
  it improves macro Dice by `0.009`, HD95 by `7.823 mm`, and ASD by `0.354 mm`.
- Loss-stage reference: P1 HU windowing, D1 augmentation, C2 five-slice 2.5D
  input, Adam, LR `5e-4`, no scheduler, and seed 43. C2 is plain-CE L0.
- Loss-stage choice: L6 normalized Balance improves macro Dice by `0.030` and
  ASD by `0.303 mm` over L4, while HD95 worsens by `0.693 mm`. L4 remains the
  best-HD95 alternative. Pure Dice (L2) is clearly unsuitable in this setup.
- Learning-rate choice: keep `5e-4` from L6. T1 (`1e-4`) is worse on all three
  metrics. T2 (`1e-3`) lowers HD95 by `1.761 mm` relative to L6, but lowers
  Dice by `0.013` and raises ASD by `0.144 mm`. Retain T2 as an HD95-focused
  alternative rather than combining it with other changes now.
- Current reference: P1 HU windowing, D1 augmentation, C2 five-slice 2.5D,
  L6 normalized Balance, Adam, LR `5e-4`, no scheduler, deterministic seed 43.
- optimizer and schedulr choice: we retain Adam and no scheduler. T3, T4, and T6
  are worse than L6 on all three metrics
- Early stopping: T7 produced the same per-class results as L6
- 3 seeds tested for stability on the fixed fold-0 split (L6, V1, V2): mean and
  sample standard deviation of run-level macro metrics are Dice
  `0.699 +/- 0.015`, HD95 `16.898 +/- 0.500 mm`, and ASD
  `3.725 +/- 0.229 mm`
- this selection is provisional before implementation of other model architectures 

## U-Net, normalization and interclass-term

- **Model size:** UNet-large is best, followed by UNet-medium and ENet.
- **Inter-class term:** InterMCBL is not better than InterCBL.
- **Normalization:** none and v2 perform the same.

| ID | Model | Inter-class term | Normalization | Seeds | Macro Dice | Macro HD95 (mm) | Macro ASD (mm) |
|---|---|---|---|---|---:|---:|---:|
| *L4* | *ENet* | *InterCBL* | *none* | *43* | *0.683* | *15.631* | *3.788* |
| *L6/V1/V2* | *ENet* | *InterCBL* | *v1* | *43, 44, 45* | *0.699 +/- 0.015* | *16.898 +/- 0.500* | *3.725 +/- 0.229* |
| M1 | ENet | InterCBL | none | 43, 44, 45 | 0.694 +/- 0.009 | 30.860 +/- 4.332 | 5.757 +/- 0.151 |
| M2 | ENet | InterCBL | v2 | 43, 44, 45 | 0.705 +/- 0.008 | 27.070 +/- 3.462 | 5.309 +/- 0.471 |
| M3 | ENet | InterMCBL | v2 | 43, 44, 45 | 0.699 +/- 0.002 | 34.275 +/- 4.641 | 5.870 +/- 0.405 |
| M4 | UNet-large | InterCBL | none | 43, 44, 45 | 0.750 +/- 0.003 | **19.341 +/- 1.144** | 4.448 +/- 0.226 |
| M5 | UNet-large | InterCBL | v2 | 43, 44, 45 | **0.757 +/- 0.015** | 20.294 +/- 2.669 | 4.275 +/- 0.430 |
| M6 | UNet-large | InterMCBL | v2 | 43, 44, 45 | 0.748 +/- 0.018 | 19.835 +/- 0.998 | **4.231 +/- 0.318** |
| M7 | UNet-medium | InterCBL | v2 | 43, 44 | 0.732 +/- 0.004 | 25.293 +/- 4.448 | 5.015 +/- 0.291 |


| ID | Esophagus Dice | Heart Dice | Trachea Dice | Aorta Dice | Esophagus HD95 (mm) | Heart HD95 (mm) | Trachea HD95 (mm) | Aorta HD95 (mm) | Esophagus ASD (mm) | Heart ASD (mm) | Trachea ASD (mm) | Aorta ASD (mm) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M1 | 0.475 +/- 0.014 | 0.859 +/- 0.029 | 0.730 +/- 0.044 | 0.712 +/- 0.021 | 25.923 +/- 5.081 | 18.426 +/- 1.954 | 53.123 +/- 19.101 | 25.969 +/- 3.993 | 5.551 +/- 0.651 | 4.716 +/- 0.387 | 6.391 +/- 0.604 | 6.371 +/- 1.002 |
| M2 | 0.495 +/- 0.009 | 0.866 +/- 0.016 | 0.754 +/- 0.011 | 0.704 +/- 0.018 | 20.362 +/- 1.608 | 19.990 +/- 4.633 | 41.352 +/- 5.788 | 26.576 +/- 3.118 | 4.975 +/- 0.440 | 4.770 +/- 0.842 | 4.765 +/- 0.222 | 6.728 +/- 0.828 |
| M3 | 0.478 +/- 0.023 | 0.874 +/- 0.023 | 0.748 +/- 0.030 | 0.696 +/- 0.006 | 20.919 +/- 0.647 | 17.368 +/- 2.019 | 73.960 +/- 22.992 | 24.850 +/- 2.229 | 5.252 +/- 0.799 | 4.343 +/- 0.713 | 7.077 +/- 1.848 | 6.810 +/- 0.191 |
| M4 | 0.547 +/- 0.010 | 0.881 +/- 0.020 | 0.805 +/- 0.007 | **0.766 +/- 0.008** | **19.121 +/- 1.712** | 21.434 +/- 7.847 | **12.148 +/- 0.538** | **24.661 +/- 1.618** | 4.623 +/- 0.452 | 4.758 +/- 1.710 | **1.995 +/- 0.189** | 6.417 +/- 0.445 |
| M5 | **0.555 +/- 0.008** | **0.887 +/- 0.005** | **0.824 +/- 0.061** | 0.760 +/- 0.014 | 19.883 +/- 3.612 | **16.958 +/- 1.689** | 18.668 +/- 8.717 | 25.668 +/- 0.567 | **4.509 +/- 0.583** | **3.952 +/- 0.403** | 2.442 +/- 0.893 | 6.196 +/- 0.265 |
| M6 | 0.538 +/- 0.021 | 0.882 +/- 0.008 | 0.818 +/- 0.054 | 0.755 +/- 0.011 | 20.856 +/- 3.093 | 17.268 +/- 1.971 | 13.843 +/- 9.530 | 27.375 +/- 1.080 | 4.698 +/- 0.438 | 4.061 +/- 0.279 | 2.043 +/- 0.637 | **6.124 +/- 0.437** |
| M7 | 0.519 +/- 0.006 | 0.873 +/- 0.002 | 0.792 +/- 0.003 | 0.744 +/- 0.020 | 28.499 +/- 16.746 | 17.252 +/- 0.700 | 30.618 +/- 0.386 | 24.802 +/- 2.130 | 4.891 +/- 0.278 | 4.192 +/- 0.013 | 4.697 +/- 2.128 | 6.280 +/- 0.673 |

```bash
python slice_segthor.py \
    --source_dir data/segthor_part1 \
    --dest_dir data/segthor_seed43/exp_P1_HU \
    --shape 256 256 \
    --retains 5 \
    --seed 43 \
    --fold 0 \
    --fix_aorta_esophagus \
    --hu_min -1000 \
    --hu_max 300 \
    --process 6

# DATA: data/segthor_seed43/exp_P1_HU
# MODEL: enet | unet-medium | unet-large
# INTER: binary | multiclass
# NORM: none | v2
# SEED: 43 | 44 | 45
python main.py \
    --dataset SEGTHOR \
    --mode full \
    --epochs 25 \
    --gpu \
    --data_dir $DATA \
    --augment \
    --context_slices 2 \
    --loss_fn balance \
    --balance_alpha 0.5 \
    --balance_t 0.9 \
    --balance_fallback_epoch -1 \
    --opt adam \
    --lr 0.0005 \
    --scheduler none \
    --deterministic \
    --seed $SEED \
    --model $MODEL \
    --balance_inter $INTER \
    --balance_normalized $NORM
```

## U-Net size, normalization and interclass-term on the new data

- **Model size:** UNet-large is best, followed by UNet-medium and UNet-small.
- **Normalization:** none, v1 and v2 perform the same.
- **Inter-class term:** InterMCBL is worse than InterCBL.

| ID | Model | Inter-class term | Normalization | Seeds | Macro Dice | Macro HD95 (mm) | Macro ASD (mm) |
|---|---|---|---|---|---:|---:|---:|
| N1 | UNet-small | InterCBL | none | 43, 44 | 0.789 +/- 0.004 | 14.706 +/- 0.288 | 3.025 +/- 0.282 |
| N2 | UNet-small | InterCBL | v1 | 43, 44 | 0.797 +/- 0.001 | 14.903 +/- 0.514 | 3.062 +/- 0.091 |
| N3 | UNet-small | InterCBL | v2 | 43, 44 | 0.790 +/- 0.006 | 17.982 +/- 5.595 | 3.323 +/- 0.535 |
| N4 | UNet-medium | InterCBL | none | 43, 44 | 0.834 +/- 0.008 | 12.523 +/- 2.311 | 2.425 +/- 0.042 |
| N5 | UNet-medium | InterCBL | v1 | 43, 44 | 0.828 +/- 0.011 | 17.019 +/- 9.477 | 2.685 +/- 0.664 |
| N6 | UNet-medium | InterCBL | v2 | 43, 44 | 0.829 +/- 0.002 | 13.459 +/- 3.074 | 2.475 +/- 0.141 |
| N7 | UNet-medium | InterMCBL | v2 | 43, 44 | 0.817 +/- 0.001 | 13.172 +/- 1.633 | 2.627 +/- 0.205 |
| N8 | UNet-large | InterCBL | none | 43, 44 | **0.846 +/- 0.007** | **12.211 +/- 1.521** | **2.258 +/- 0.136** |


| ID | Esophagus Dice | Heart Dice | Trachea Dice | Aorta Dice | Esophagus HD95 (mm) | Heart HD95 (mm) | Trachea HD95 (mm) | Aorta HD95 (mm) | Esophagus ASD (mm) | Heart ASD (mm) | Trachea ASD (mm) | Aorta ASD (mm) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| N1 | 0.567 +/- 0.002 | 0.887 +/- 0.012 | 0.869 +/- 0.005 | 0.832 +/- 0.011 | 15.113 +/- 1.045 | 15.404 +/- 6.355 | 9.291 +/- 1.568 | 19.015 +/- 5.727 | 3.535 +/- 0.067 | 3.777 +/- 0.028 | 1.604 +/- 0.246 | 3.183 +/- 0.844 |
| N2 | 0.584 +/- 0.017 | 0.878 +/- 0.016 | 0.882 +/- 0.001 | 0.845 +/- 0.003 | 14.288 +/- 3.760 | 10.615 +/- 0.253 | 9.133 +/- 1.228 | 25.577 +/- 4.843 | 3.455 +/- 0.319 | 3.803 +/- 0.187 | 1.238 +/- 0.132 | 3.751 +/- 0.362 |
| N3 | 0.569 +/- 0.019 | 0.886 +/- 0.009 | 0.884 +/- 0.001 | 0.823 +/- 0.004 | 14.724 +/- 0.535 | 11.209 +/- 1.428 | 11.934 +/- 5.930 | 34.059 +/- 14.485 | 3.527 +/- 0.076 | 3.722 +/- 0.409 | 1.585 +/- 0.539 | 4.457 +/- 1.267 |
| N4 | 0.657 +/- 0.001 | **0.911 +/- 0.011** | **0.890 +/- 0.019** | 0.878 +/- 0.005 | 15.280 +/- 6.377 | 9.732 +/- 1.151 | **7.004 +/- 0.387** | 18.075 +/- 4.404 | 2.958 +/- 0.272 | 3.091 +/- 0.363 | 1.197 +/- 0.108 | 2.453 +/- 0.030 |
| N5 | 0.645 +/- 0.020 | 0.900 +/- 0.006 | 0.885 +/- 0.011 | 0.883 +/- 0.005 | 30.027 +/- 27.000 | 9.714 +/- 1.445 | 8.279 +/- 0.743 | 20.055 +/- 10.208 | 3.567 +/- 1.543 | 3.340 +/- 0.332 | 1.191 +/- 0.038 | 2.640 +/- 0.744 |
| N6 | 0.644 +/- 0.006 | 0.900 +/- 0.005 | 0.887 +/- 0.016 | 0.883 +/- 0.004 | **11.545 +/- 2.999** | 15.411 +/- 8.299 | 9.128 +/- 1.210 | 17.751 +/- 5.788 | **2.658 +/- 0.427** | 3.701 +/- 0.567 | 1.237 +/- 0.161 | 2.302 +/- 0.264 |
| N7 | 0.634 +/- 0.027 | 0.906 +/- 0.019 | 0.871 +/- 0.013 | 0.859 +/- 0.016 | 11.861 +/- 1.426 | 9.913 +/- 1.699 | 9.180 +/- 2.640 | 21.734 +/- 8.900 | 2.998 +/- 0.579 | 3.155 +/- 0.551 | 1.297 +/- 0.276 | 3.059 +/- 1.124 |
| N8 | **0.699 +/- 0.024** | 0.908 +/- 0.001 | 0.889 +/- 0.000 | **0.889 +/- 0.004** | 16.247 +/- 7.010 | **9.466 +/- 0.718** | 7.100 +/- 0.541 | **16.030 +/- 0.749** | 2.812 +/- 0.491 | **2.951 +/- 0.065** | **1.105 +/- 0.053** | **2.165 +/- 0.067** |

```bash
python slice_segthor.py \
    --source_dir data/segthor_train \
    --dest_dir data/segthor_train_seed43/exp_P1_HU \
    --shape 256 256 \
    --retains 10 \
    --seed 43 \
    --fold 0 \
    --hu_min -1000 \
    --hu_max 300 \
    --process 6

# DATA: data/segthor_train_seed43/exp_P1_HU
# MODEL: unet-small | unet-medium | unet-large
# INTER: binary | multiclass
# NORM: none | v1 | v2
# SEED: 43 | 44
python main.py \
    --dataset SEGTHOR \
    --mode full \
    --epochs 25 \
    --gpu \
    --data_dir $DATA \
    --augment \
    --context_slices 2 \
    --loss_fn balance \
    --balance_alpha 0.5 \
    --balance_t 0.9 \
    --balance_fallback_epoch -1 \
    --opt adam \
    --lr 0.0005 \
    --scheduler none \
    --deterministic \
    --seed $SEED \
    --model $MODEL \
    --balance_inter $INTER \
    --balance_normalized $NORM
```

## DINOv3 versions, feature resolution and fusion location

| Stage | ID | Compared with | Question |
|---|---|---|---|
| 0 | F0 | N8 | Control without DINO |
| 1 | F1 | F0 | Does DINO help? |
| 1 | F2 | F1 | Encoder or decoder fusion? |
| 2 | F3 | F2 | Does CT pretraining (MedDINOv3) help? |
| 2 | F4 | F3 | Do finer DINO features help? |
| 2 | F5 | N4, N8 | Does DINO help UNet-medium, and can it match UNet-large? |

- **Foundation model:** DINO improves the U-Net.
- **Fusion location:** encoder fusion is better than decoder fusion.
- **DINO version:** MedDINOv3 is not better than DINOv3 ViT-S.
- **Feature resolution:** finer DINO features improve HD95 and ASD, not Dice.
- **U-Net size:** with DINO, UNet-medium beats UNet-large without DINO.

| ID | Model | Foundation model | DINO input (px) | Feature resolution | Fusion | Fusion level | 1x1 conv channels | Seeds | Macro Dice | Macro HD95 (mm) | Macro ASD (mm) |
|---|---|---|---|---|---|---|---|---|---:|---:|---:|
| F0 | UNet-large | - | - | - | - | - | - | 43 (= N8), 44 | 0.828 +/- 0.020 | 14.644 +/- 4.963 | 2.608 +/- 0.630 |
| F1 | UNet-large | DINOv3 ViT-S/16 | 256 | 16x16 | decoder | 4 | 256 | 43, 44 | 0.847 +/- 0.014 | 11.280 +/- 0.224 | 2.221 +/- 0.131 |
| F2 | UNet-large | DINOv3 ViT-S/16 | 256 | 16x16 | encoder | 4 | 256 | 43, 44 | 0.851 +/- 0.025 | 9.682 +/- 2.606 | 2.084 +/- 0.322 |
| F3 | UNet-large | MedDINOv3 ViT-B/16 | 256 | 16x16 | encoder | 4 | 256 | 43 | 0.863 | 10.724 | 1.926 |
| F4 | UNet-large | MedDINOv3 ViT-B/16 | 512 | 32x32 | encoder | 3 | 128 | 43 | **0.864** | **8.562** | **1.740** |
| F5 | UNet-medium | MedDINOv3 ViT-B/16 | 256 | 16x16 | encoder | 4 | 128 | 43 | 0.850 | 10.246 | 1.853 |


| ID | Esophagus Dice | Heart Dice | Trachea Dice | Aorta Dice | Esophagus HD95 (mm) | Heart HD95 (mm) | Trachea HD95 (mm) | Aorta HD95 (mm) | Esophagus ASD (mm) | Heart ASD (mm) | Trachea ASD (mm) | Aorta ASD (mm) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F0 | 0.672 +/- 0.013 | 0.902 +/- 0.010 | 0.866 +/- 0.032 | 0.870 +/- 0.023 | 12.679 +/- 1.963 | 13.753 +/- 5.346 | 7.937 +/- 1.725 | 24.208 +/- 10.816 | 2.773 +/- 0.435 | 3.546 +/- 0.906 | 1.458 +/- 0.552 | 2.654 +/- 0.625 |
| F1 | 0.712 +/- 0.028 | 0.918 +/- 0.011 | 0.877 +/- 0.011 | 0.880 +/- 0.026 | 10.966 +/- 1.262 | 11.777 +/- 0.408 | 9.189 +/- 2.112 | 13.187 +/- 1.339 | 2.334 +/- 0.324 | 2.952 +/- 0.214 | 1.335 +/- 0.052 | 2.265 +/- 0.360 |
| F2 | 0.718 +/- 0.019 | 0.930 +/- 0.006 | 0.872 +/- 0.041 | 0.885 +/- 0.033 | 9.398 +/- 2.871 | 10.353 +/- 1.158 | 8.305 +/- 4.069 | **10.671 +/- 2.329** | 2.295 +/- 0.430 | 2.715 +/- 0.027 | 1.456 +/- 0.300 | 1.869 +/- 0.530 |
| F3 | **0.719** | 0.940 | 0.891 | 0.901 | 10.955 | 6.500 | 8.495 | 16.945 | 2.348 | 2.160 | 1.113 | 2.083 |
| F4 | 0.711 | **0.943** | **0.899** | **0.903** | **9.185** | **5.843** | 8.382 | 10.837 | **2.023** | **2.035** | 1.085 | **1.817** |
| F5 | 0.668 | 0.937 | 0.898 | 0.897 | 10.401 | 10.736 | **6.427** | 13.418 | 2.407 | 2.196 | **0.921** | 1.889 |

```bash
python slice_segthor.py \
    --source_dir data/segthor_train \
    --dest_dir data/segthor_train_seed$SEED/exp_P1_HU \
    --shape 256 256 \
    --retains 10 \
    --seed $SEED \
    --fold 0 \
    --hu_min -1000 \
    --hu_max 300 \
    --process 6

# DATA: data/segthor_train_seed$SEED/exp_P1_HU
# MODEL: unet-large | unet-medium
# FOUNDATION: dinov3-vits16 | meddinov3-vitb16
# FUSION: decoder | encoder
# UPSAMPLE: 1 | 2
# SEED: 43 | 44
python main.py \
    --dataset SEGTHOR \
    --mode full \
    --epochs 25 \
    --gpu \
    --data_dir $DATA \
    --augment \
    --context_slices 2 \
    --loss_fn balance \
    --balance_alpha 0.5 \
    --balance_t 0.9 \
    --balance_fallback_epoch -1 \
    --opt adam \
    --lr 0.0005 \
    --scheduler none \
    --deterministic \
    --seed $SEED \
    --model $MODEL \
    --balance_inter binary \
    --balance_normalized none \
    --foundation_model $FOUNDATION \
    --foundation_fusion $FUSION \
    --foundation_upsample $UPSAMPLE
```
