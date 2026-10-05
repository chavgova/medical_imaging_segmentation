# stage 1

| ID | Parent | Exact change | Why this candidate / selection |
|---|---|---|---|
| P0 | R1 | Existing `--hu_min -1000 --hu_max 300` | Wide-window control; no new run. Retains low-HU airway context. |
| P1 | R1 | `--hu_min -310 --hu_max 400` | Higher soft-tissue contrast per PNG level for mediastinal organs; check trachea performance after low-HU clipping. Core. |
| P2 | R1 | `--clahe --hu_min -310 --hu_max 400` | Compare to P1 to isolate adaptive contrast. Existing implementation fixes clip limit 0.01; do not invent a CLI parameter sweep. Core. |
| P3 | R1 | `--hu_windows -1000 300 -310 400` | Complementary wide/narrow information. Five slices x two windows = 10 channels, supported by U-Net. Wide window first also preserves DINO's established input. Core. |
| P4 | R1 | `--hu_percentile` alone | Learn the pooled foreground 0.5th/99.5th HU percentiles from **training patients only**, then apply unchanged to validation. Skip if the quantized representation is effectively identical to an existing candidate. Core subject to this deduplication. |


# stage 2


| ID | Parent | Exact change | Why / stop rule | Status |
|---|---|---|---|---|
| A0 | P_best | Existing rotation/noise `--augment`, scale 0 | Existing control. Rotation is +/-5 degrees, probability 0.5; Gaussian noise probability 0.25. | Reuse |
| A1 | A0 | `--augment --augment_scale 0.15` | Modest shared zoom 0.85-1.15, probability 0.5. Check clipping after zoom; a scale sweep is unjustified. | Core: 1 |
| A2 | Best(A0, A1) | Preprocessing `--crop_body` | CT-derived, one bounding box per volume. Potentially devotes more pixels to anatomy; preserve all organs and inverse geometry. If A1 failed, A2 uses A0, not A1. | Gated G2: 1 |
| A3 | A2, only if both A1 and A2 were accepted | Set `--augment_scale 0`, keep old augmentation and crop | Tests whether zoom ceases to help or truncates anatomy after tighter cropping. | Conditional: 1 |
| G1-resolution | Best available A configuration | Preprocessing `--shape 512 512`, no other change | Run only if esophagus/contour errors suggest information loss at 256 and memory profiling fits batch 8. This adds real image detail, unlike enlarging DINO's already-downsampled image. | Conditional: 1 |
| G2-spacing | G1-resolution | Add `--target_spacing s`, where s is median native in-plane spacing of the training patients | Isolates physical scale at the **same 512 x 512 shape**. Before training, verify the fixed FOV retains every target organ. Skip if center cropping loses anatomy; do not try an unsafe 256 x 256 / ~1 mm field. Compare against G1, not a differently sized control. | Conditional + G2: 1 |


# stage 3

| ID | Parent | Change | Why / decision | Status |
|---|---|---|---|---|
| C0 | G_best | Existing `--context_slices 2` (five slices) | Incoming winner. Physical support is patient-dependent because z spacing is unchanged. | Reuse |
| C1 | C0 | `--context_slices 0` | Recheck whether context still improves the stronger U-Net/updated inputs and warrants extra channels. | Core: 1 |
| C2 | C0 | `--context_slices 1` (three slices) | Only if C0 and C1 are close, or five-slice input harms thin-organ endpoints. Earlier ENet failure alone does not prove it cannot work here. | Conditional: 1 |


# stage 4

| ID | Parent | Exact change | Why / decision | Status |
|---|---|---|---|---|
| L0 | C_best | Binary Balance, normalization none, alpha 0.5, t 0.9, fallback 12 | Supported by larger-data N8; incoming control. | Reuse |
| L1 | L0 | `--balance_normalized v2` | Best-motivated normalization alternative: corrected InterCBL weight sum for empty/foreground-heavy images. v1 remains historical only. | Core: 1 |
| L2 | L0 | `--loss_fn dicece --dicece_lambda 0.5`, no CE weights | Strong simpler alternative to Balance on a new architecture/data representation. Clear inherited Balance flags in the recorded config. | Core: 1 |
| L3 | L2 | `--ce_weights invfreq --ce_weights_alpha 0.5` | Only if L2 is competitive but persistently undersegments esophagus. Moderate training-only inverse-frequency weighting; reject if false positives/surface errors grow as in old weighted CE. | Conditional: 1 |
| L4 | Best binary Balance row | `--balance_inter multiclass --balance_normalized v2` | Only if a persistent organ-imbalance problem remains after other choices and L3 fails. If parent used none, compare to L1 with v2 to isolate the inter-term. Existing negative evidence makes this low priority. | Conditional: 0-1 |


# stage 5

| ID | Parent | Change | Why / decision | Status |
|---|---|---|---|---|
| M0 | L_best | Existing `--model unet-large` | Best-supported accuracy control. | Reuse |
| M1 | M0 | `--model unet-medium` | One matched cost/accuracy comparison on updated inputs. Promote medium if accuracy/boundary performance is effectively tied and measured cost is lower. | Core: 1 |
| M2 | M1 | `--model unet-small` | Only under a real runtime/memory constraint and if medium already matches large. Historical small-model accuracy does not justify another default run. | Conditional: 1 |


# stage 6

| ID | Parent | Exact change | Why / decision | Status |
|---|---|---|---|---|
| F0 | M_best | No foundation | Existing control. If P_best is CLAHE, use P_linear and create one matched no-foundation control for this branch; also retain the CLAHE winner in the overall shortlist. | Reuse or 1 bridge |
| F1 | F0 | `--foundation_model dinov3-vits16 --foundation_fusion encoder --foundation_upsample 1` | Cheapest supported feature baseline; compare with no foundation. | Core: 1 |
| F2 | F0 | `--foundation_model dinov3-vitb16 --foundation_fusion encoder --foundation_upsample 1` | Natural-image ViT-B control separates larger backbone from CT adaptation. Keep projection width equal to F1. | Core: 1 |
| F3 | F2 configuration | Replace only foundation identifier with `meddinov3-vitb16` | Matched ViT-B architecture comparison. MedDINO needs the saved linear HU mapping; CLAHE/min-max are invalid. | Core: 1 |
| F4 | Best beneficial F1/F2/F3 | Change only `--foundation_upsample 2`; keep projection width fixed | Tests a finer feature/fusion configuration. At 256 input: DINO input 512 and patch grid 32 x 32, fusion level 3 instead of 4. This changes both feature resolution and insertion level by design; do not claim a pure resolution ablation. | Conditional: 1 |
| F5 | Best beneficial foundation row | Change only `--foundation_fusion decoder` | Only if encoder advantage is unclear or the preprocessing/backbone change reverses earlier behavior. | Conditional: 1 |
| F6 | Best beneficial foundation row on U-Net-large | Change to `--model unet-medium`, retaining projection width | Only if plain medium lost but was close (within 0.02 Dice); test whether foundation features close its gap. | Conditional: 1 |

# stage 7

| ID | Parent | One controlled change | Trigger |
|---|---|---|---|
| X1 | Best foundation pipeline | Swap P_best/P_linear for the strongest complementary earlier **linear-HU** representation, keeping geometry fixed | Multiwindow versus single-window may change once DINO already supplies central-slice features; or narrow/percentile clipping may hurt MedDINO. Compare with the exact same foundation/model/loss. |
| X2 | X1 winner or best foundation pipeline | Replace loss with the strongest competitive alternative from Stage 4 | Only when earlier Balance versus DiceCE/normalization differences were close or organ-specific errors changed after DINO. |
| X3 | Current winner | Toggle context 0 versus 2, keeping all else fixed | Substitute for X2 if both context rows were close and DINO plausibly supplies enough context already. Not a third automatic run. |


# stage 8

| ID | Input | Candidate settings | Reason and promotion rule |
|---|---|---|---|
| Q0 | Raw predictions | `--postprocessing none` | Always retain as control; postprocessing is optional, not a compulsory improvement. |
| Q1 | Q0 | `--postprocessing largest_connected_components --top_k 1 --connectivity 26 --postprocessing_classes 2 3 4` | Conservative first component filter; leave esophagus untouched. Inspect whether valid tracheal/aortic fragments are removed. |
| Q2 | Q0 | `--postprocessing anatomy_aware_filtering`; current JSON rules: connectivity 26; class 1 retains components >= 0.05 of its largest; classes 2/3/4 retain largest; absolute minimum 0; always keep largest | Direct comparison with Q1 isolates esophagus fragment removal. This is a size heuristic, not a complete anatomical model. |
| Q3 | Q2 settings | Lower esophagus relative minimum from 0.05 to **0.01**, other settings unchanged | Conditional only if Q2 removes real esophageal fragments or is close to Q1. This is the single conservative threshold follow-up, not a large threshold grid. |
| Q4 | Best Q0-Q3 | Append `fill_holes` for **heart only**, connectivity 6 | Only if inspection shows enclosed false-negative cavities consistent with the annotation definition. Avoid blanket filling of tubular organs. |
| Q5 | Q0 | `--postprocessing closing --iterations 1 --connectivity 6 --postprocessing_classes 2` | Conditional alternative to Q4 for small heart-boundary gaps. Do not automatically stack hole filling and closing. |
| Q6 | Q0 | `--postprocessing opening --iterations 1 --connectivity 6 --postprocessing_classes 2` **or** `--postprocessing salt_and_pepper --kernel_size 3 --postprocessing_classes 2` | Choose at most one from observed heart-boundary spurs/isolated noise that component filtering did not address. Otherwise skip; both can remove valid thin structures. |
| Q7 | Native probabilities + original CT | `--postprocessing dense_crf`, current parameter values listed below | Gated G3. One candidate for persistent CT-aligned boundary errors. Reject if it erodes low-contrast esophagus or gives poor runtime/accuracy trade-off. |
| Q8 | Same as Q7 | Spatial/bilateral weights **1.5/2.5** instead of 3/5, all else fixed | Conditional single follow-up if Q7 oversmooths but improves some boundaries. No arbitrary sigma/iteration grid. |
| Q9 | Best useful CRF probabilities/labels | Winning CRF **first**, then accepted component policy, then heart-only Q4 if independently useful | Test only a justified combination. Compare with both its individual parents and Q0; do not combine every method. |


# stage 10 

| ID | Configuration / runs | Purpose and decision |
|---|---|---|
| V1 | Frozen best two complete pipelines; existing training seed 43 plus seeds **44 and 45** on the **same fold-0 manifest**. Up to 4 new training runs at the selected full budget. Apply each pipeline's fixed postprocessing. | Compare mean/sample SD and paired patient effects. Do not choose a favorable seed as the final “method.” A marginal expensive feature that fails repeats is removed. |
| V2 | Same two frozen recipes on **folds 1,2,3**, preprocessing seed 43, retains 10, training seed 43. Up to 6 new runs. Fit percentile bounds/CE weights on each fold's training patients only. | With verified 40 patients, this completes four 30/10 folds. Compare paired out-of-fold predictions across all patients; report fold variation separately from training-seed variation. Keep postprocessing fixed. |
| V3 | R0 baseline-method recipe on the same remaining folds (3 runs), if no valid matched runs exist | Supports a robust baseline-improvement claim with the same native-GT metric route. If omitted for budget reasons, explicitly limit the baseline claim to fold 0. |
| E1 | Optional equal-probability average of the accepted configuration's three same-split seed models; no new training | Gated G3 and a later ensemble wrapper (not currently a CLI feature). Evaluate on their common unseen validation patients; apply fixed postprocessing **after averaging**. Keep only if confirmed improvement warrants 3x inference/storage. |
| Z1 | Freeze winning preprocessing, context, model/foundation, loss, training duration, and postprocessing | Record exact resolved configuration. Generate final test predictions only after freezing. Never score an “ensemble validation” prediction using a model that trained on that patient. |
