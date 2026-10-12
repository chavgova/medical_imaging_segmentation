#!/usr/bin/env python3
"""Analyze saved ensemble uncertainty; no models or inference are required.

Preferred usage (matching processed PNGs, including ground truth):
  python analyze_ensemble_uncertainty.py --results_dir results/v2_ensemble \
    --data_dir data/segthor_full_seed43/final_exp_A2_A1_crop

Writes PNG/PDF plots, CSV/JSON metrics, an offline HTML gallery, and ranked
slice montages. Rankings use predicted foreground plus a small neighbourhood,
without using GT. Error analysis separately uses all pixels and GT/prediction
foreground union. Native GT/CT can also be supplied with {id_} path patterns.
Add --per_class_only for organ-specific plots and galleries, and optionally
--class_binary_entropy to use saved averaged class probabilities.
"""

import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.stats import spearmanr
from skimage.transform import resize


METRICS = ("entropy", "mutual_information", "total_variance")
ORGANS = ("esophagus", "heart", "trachea", "aorta")
COLORS = np.array([[0, 0, 0], [71, 196, 234], [247, 178, 76],
                   [212, 111, 222], [104, 210, 138]], dtype=np.uint8)


def read_labels(path):
    labels = np.asarray(Image.open(path))
    if labels.ndim != 2 or not np.isin(labels, [0, 63, 126, 189, 252]).all():
        raise ValueError(f"Expected SegTHOR PNG labels 0/63/126/189/252: {path}")
    return labels // 63


def read_uncertainty(path):
    with np.load(path) as saved:
        entropy = saved["entropy"]
        mi = saved["mutual_information"]
        variance = saved["variance"]
    if entropy.ndim != 2 or mi.shape != entropy.shape or variance.shape != (5, *entropy.shape):
        raise ValueError(f"Invalid uncertainty shapes: {path}")
    maps = {"entropy": entropy, "mutual_information": mi, "total_variance": variance.sum(0)}
    if any(not np.isfinite(values).all() or (values < 0).any() for values in maps.values()):
        raise ValueError(f"Nonfinite or negative uncertainty: {path}")
    maps["class_variance"] = variance
    return maps


def error_metrics(scores, errors):
    """Tie-aware error-detection ROC/AP, plus selective risk (low U retained)."""
    scores, errors = np.asarray(scores), np.asarray(errors, dtype=bool)
    n = len(errors)
    order = np.argsort(-scores, kind="stable")
    ordered_errors = errors[order]
    positive, negative = int(errors.sum()), int((~errors).sum())
    ends = np.r_[np.flatnonzero(np.diff(scores[order]) != 0), n - 1]
    tp = np.cumsum(ordered_errors)[ends]
    fp = ends + 1 - tp
    recall = tp / positive if positive else np.zeros(len(ends))
    precision = tp / (ends + 1)
    tpr = np.r_[0, recall]
    fpr = np.r_[0, fp / negative] if negative else np.zeros(len(ends) + 1)
    auc = float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2)) if positive and negative else None
    ap = float(np.sum(np.diff(np.r_[0, recall]) * precision)) if positive else None
    coverage = np.linspace(0.05, 1, 40)
    counts = np.maximum(1, np.ceil(n * coverage).astype(int))
    risk = np.cumsum(ordered_errors[::-1])[counts - 1] / counts
    prevalence = positive / n
    top = max(1, int(np.ceil(n * .1)))
    metrics = {"sample_pixels": n, "error_prevalence": prevalence, "error_auroc": auc,
               "error_average_precision": ap,
               "error_at_50pct_coverage": float(ordered_errors[::-1][:max(1, int(np.ceil(n * .5)))].mean()),
               "top_10pct_error_enrichment": float(ordered_errors[:top].mean() / prevalence) if positive else None}
    return metrics, dict(fpr=fpr, tpr=tpr, recall=recall, precision=precision,
                         coverage=coverage, risk=risk)


class PixelReservoir:
    """Uniform pixel sampling with random priorities and bounded memory."""
    def __init__(self, size, seed, names=METRICS):
        self.size = size
        self.names = tuple(names)
        self.rng = np.random.default_rng(seed)
        self.records = np.empty((0, len(self.names) + 2))

    def add(self, maps, errors, mask):
        if not mask.any():
            return
        values = np.column_stack([self.rng.random(int(mask.sum())), errors[mask],
                                  *[maps[name][mask] for name in self.names]])
        self.records = np.concatenate([self.records, values])
        if len(self.records) > self.size * 2:
            self.trim()

    def trim(self):
        if len(self.records) > self.size:
            indices = np.argpartition(self.records[:, 0], -self.size)[-self.size:]
            self.records = self.records[indices]
        return self.records[np.argsort(self.records[:, 0])]  # Random order for tied U.


def load_patient(args, patient, manifest):
    """Load optional native data and establish one shared crop for every layer."""
    data = {"gt": None, "ct": None, "q9c": None, "bounds": None}
    gt_path = Path(args.gt_pattern.format(id_=patient)) if args.gt_pattern else None
    ct_pattern = args.ct_pattern or (manifest.get("source_scan_pattern") if args.data_dir is None else None)
    ct_path = Path(ct_pattern.format(id_=patient)) if ct_pattern else None
    images = {}
    for key, path in (("gt", gt_path), ("ct", ct_path)):
        if path and path.exists():
            images[key] = nib.load(path)
            data[key] = np.asarray(images[key].dataobj)
    if data["gt"] is not None:
        if not np.isin(data["gt"], np.arange(5)).all():
            raise ValueError(f"Invalid native GT labels: {patient}")
        if args.gt_preprocess == "split_merged":
            if np.any(data["gt"] == 4):
                raise ValueError(f"{patient}: GT already has aorta label 4; do not split it again")
            from slice_segthor import split_merged_aorta_esophagus
            data["gt"] = split_merged_aorta_esophagus(data["gt"])
        elif not np.any(data["gt"] == 4):
            raise ValueError(f"{patient}: GT has no aorta label 4. For the project's merged-label GT, explicitly use --gt_preprocess split_merged.")
    q9c_path = args.results_dir / "postprocessing/q9c" / args.subset / "pred_volumes" / f"{patient}.nii.gz"
    geometry_dir = args.geometry_dir
    if geometry_dir is None and args.data_dir:
        geometry_dir = args.data_dir / args.subset / "geometry"
    if geometry_dir is None and manifest.get("geometry_dir"):
        geometry_dir = Path(manifest["geometry_dir"])
    geometry_path = geometry_dir / f"{patient}.json" if geometry_dir else None
    if geometry_path and geometry_path.exists():
        geometry = json.loads(geometry_path.read_text())
        if geometry.get("target_spacing") is not None:
            raise ValueError("Spacing-resampled geometry is not supported")
        for image in images.values():
            if list(image.shape) != geometry["original_shape"]:
                raise ValueError(f"Geometry/image shape mismatch: {patient}")
        data["bounds"] = geometry["crop_bbox"]
        data["shape"] = tuple(geometry["original_shape"])
    elif args.recompute_crop and data["ct"] is not None:
        from slice_segthor import compute_body_bbox
        row, col = compute_body_bbox(data["ct"])
        data["bounds"] = [int(row.start), int(row.stop), int(col.start), int(col.stop)]
        data["shape"] = data["ct"].shape
        path = args.results_dir / "pred_volumes" / args.subset / f"{patient}.nii.gz"
        images["raw"] = nib.load(path)
        data["verify_raw"] = np.asarray(images["raw"].dataobj)
        r0, r1, c0, c1 = data["bounds"]
        if np.count_nonzero(data["verify_raw"]) != np.count_nonzero(data["verify_raw"][r0:r1, c0:c1]):
            raise ValueError(f"Recomputed crop excludes predictions: {patient}")
    if data["gt"] is not None and data["bounds"] is None:
        raise ValueError(f"Native GT needs --geometry_dir (or --recompute_crop and CT): {patient}")
    if data["bounds"] is not None:
        r0, r1, c0, c1 = data["bounds"]
        if not (0 <= r0 < r1 <= data["shape"][0] and 0 <= c0 < c1 <= data["shape"][1]):
            raise ValueError(f"Invalid crop geometry: {patient}")
        if q9c_path.exists():
            images["q9c"] = nib.load(q9c_path)
            data["q9c"] = np.asarray(images["q9c"].dataobj)
            if not np.isin(data["q9c"], np.arange(5)).all():
                raise ValueError(f"Invalid Q9c labels: {patient}")
        reference = next(iter(images.values()), None)
        for image in images.values():
            if image.shape != data["shape"] or not np.allclose(image.affine, reference.affine, atol=1e-4):
                raise ValueError(f"Native image grids disagree: {patient}")
    return data


def slice_data(args, patient_data, stem, z, shape):
    raw = read_labels(args.results_dir / "predictions" / args.subset / f"{stem}.png")
    if raw.shape != shape:
        raise ValueError(f"Prediction/uncertainty shape mismatch: {stem}")
    result = {"raw": raw, "gt": None, "ct": None, "q9c": None}
    if args.data_dir:
        for key, folder in (("gt", "gt"), ("ct", "img")):
            path = args.data_dir / args.subset / folder / f"{stem}.png"
            if path.exists():
                result[key] = read_labels(path) if key == "gt" else np.asarray(Image.open(path).convert("L"))
                if result[key].shape != shape:
                    raise ValueError(f"Processed data shape mismatch: {stem}")
    if patient_data["bounds"] is not None:
        r0, r1, c0, c1 = patient_data["bounds"]
        if not 0 <= z < patient_data["shape"][2]:
            raise ValueError(f"Invalid slice index: {stem}")
        for key in ("gt", "ct", "q9c"):
            volume = patient_data[key]
            if volume is not None and result[key] is None:
                values = resize(volume[r0:r1, c0:c1, z], shape, order=1 if key == "ct" else 0,
                                preserve_range=True, anti_aliasing=False)
                result[key] = (np.clip((values + 310) / 710, 0, 1) * 255).astype(np.uint8) if key == "ct" else values.astype(np.uint8)
        if "verify_raw" in patient_data:
            reconstructed = resize(raw, (r1-r0, c1-c0), order=0, mode="constant",
                                   preserve_range=True, anti_aliasing=False).astype(np.uint8)
            if not np.array_equal(reconstructed, patient_data["verify_raw"][r0:r1, c0:c1, z]):
                raise ValueError(f"Recomputed crop does not reproduce native predictions: {stem}")
    return result


def write_csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def diverse_selection(rows, score, top_k, gap, descending=True, limit=3):
    selected = []
    candidates = [row for row in rows if row.get(score) is not None]
    for row in sorted(candidates, key=lambda r: r[score], reverse=descending):
        same_patient = [r for r in selected if r["patient"] == row["patient"]]
        if len(same_patient) >= limit or any(abs(r["slice"] - row["slice"]) < gap for r in same_patient):
            continue
        selected.append(row)
        if len(selected) >= top_k:
            break
    return selected


def class_scores(maps, label, probabilities=None):
    """Global scores evaluated per organ plus genuinely class-specific measures."""
    scores = {"global_entropy": maps["entropy"],
              "global_mutual_information": maps["mutual_information"],
              "class_variance": maps["class_variance"][label]}
    if probabilities is not None:
        from scipy.special import xlogy
        p = probabilities[label].astype(np.float64)
        scores["binary_entropy"] = (-xlogy(p, p) - xlogy(1-p, 1-p)) / np.log(2)
    return scores


def read_probabilities(args, stem, shape):
    if not args.class_binary_entropy:
        return None
    path = args.results_dir / "probabilities" / args.subset / f"{stem}.npz"
    if not path.exists():
        raise ValueError(f"Binary entropy needs averaged class probabilities: {path}")
    with np.load(path) as saved:
        probabilities = saved["probabilities"]
    if (probabilities.shape != (5, *shape) or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(0), 1, atol=1e-5)):
        raise ValueError(f"Invalid probabilities: {path}")
    return probabilities


def per_class_analysis(args):
    """One-vs-rest organ errors, FP/FN distributions, and organ-specific rankings."""
    root = args.output_dir or args.results_dir / "uncertainty_insights"
    output = root / "per_class"
    output.mkdir(parents=True, exist_ok=True)
    (output / "samples").mkdir(exist_ok=True)
    manifest_path = args.results_dir / f"ensemble_config_{args.subset}.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if args.data_dir is None and manifest.get("data_dir") and Path(manifest["data_dir"]).is_dir():
        args.data_dir = Path(manifest["data_dir"])
    maps_dir = args.uncertainty_dir or args.results_dir / "uncertainty" / args.subset
    patients = {}
    for path in sorted(maps_dir.glob("*.npz")):
        patient, z = path.stem.rsplit("_", 1)
        patients.setdefault(patient, []).append((int(z), path))
    if not patients:
        raise ValueError(f"No uncertainty maps in {maps_dir}")
    names = ("global_entropy", "global_mutual_information", "class_variance")
    if args.class_binary_entropy:
        names += ("binary_entropy",)
    reservoirs, rows, totals, coverage = {}, [], {}, {}
    for patient, slices in patients.items():
        print(f">> Per-class analysis: {patient}", flush=True)
        data = load_patient(args, patient, manifest)
        annotated = 0
        for z, path in sorted(slices):
            maps = read_uncertainty(path)
            shape = maps["entropy"].shape
            probabilities = read_probabilities(args, path.stem, shape)
            layers = slice_data(args, data, path.stem, z, shape)
            gt = layers["gt"]
            if gt is None and not args.allow_missing_gt:
                raise ValueError(f"Missing GT: {path.stem}; provide matching GT or use --allow_missing_gt")
            annotated += gt is not None
            predictions = {"raw": layers["raw"]}
            if gt is not None and layers["q9c"] is not None:
                predictions["q9c"] = layers["q9c"]
            for label, organ in enumerate(ORGANS, 1):
                scores = class_scores(maps, label, probabilities)
                predicted_raw = layers["raw"] == label
                ranking_region = ndimage.binary_dilation(predicted_raw, iterations=args.roi_margin) if args.roi_margin else predicted_raw
                for recipe, pred in predictions.items():
                    row = dict(patient=patient, slice=z, stem=path.stem, recipe=recipe,
                               organ=organ, class_id=label, has_gt=gt is not None)
                    for name in names:
                        row[f"ranking_{name}"] = float(scores[name][ranking_region].mean()) if ranking_region.any() else None
                    if gt is not None:
                        predicted, truth = pred == label, gt == label
                        tp, fp, fn = predicted & truth, predicted & ~truth, ~predicted & truth
                        union = predicted | truth
                        errors = predicted != truth
                        row.update(tp_pixels=int(tp.sum()), fp_pixels=int(fp.sum()), fn_pixels=int(fn.sum()),
                                   gt_pixels=int(truth.sum()), predicted_pixels=int(predicted.sum()),
                                   foreground_error_rate=float(errors[union].mean()) if union.any() else None,
                                   dice=float(2 * tp.sum() / (predicted.sum() + truth.sum())) if union.any() else None)
                        total = totals.setdefault((recipe, organ), np.zeros(3, np.int64))
                        total += [row["tp_pixels"], row["fp_pixels"], row["fn_pixels"]]
                        for name in names:
                            for event, mask in (("tp", tp), ("fp", fp), ("fn", fn)):
                                row[f"{event}_{name}"] = float(scores[name][mask].mean()) if mask.any() else None
                        if union.any():
                            neighbourhood = ndimage.binary_dilation(union, iterations=args.roi_margin) if args.roi_margin else union
                            events = np.where(fp, 1, np.where(fn, 2, np.where(tp, 0, 3)))
                            values = dict(scores, event_code=events)
                            for scope, mask in (("class_union", union), ("class_neighbourhood", neighbourhood)):
                                key = (recipe, organ, scope)
                                reservoir = reservoirs.setdefault(key, PixelReservoir(args.max_samples, 43, names + ("event_code",)))
                                reservoir.add(values, errors, mask)
                    rows.append(row)
        coverage[patient] = dict(slices=len(slices), slices_with_gt=annotated)
    write_csv(output / "slice_metrics.csv", rows)
    class_totals = []
    for (recipe, organ), (tp, fp, fn) in totals.items():
        class_totals.append(dict(recipe=recipe, organ=organ, tp_pixels=int(tp), fp_pixels=int(fp), fn_pixels=int(fn),
                                 pooled_dice=float(2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else None,
                                 precision=float(tp / (tp + fp)) if tp + fp else None,
                                 recall=float(tp / (tp + fn)) if tp + fn else None))
    write_csv(output / "class_counts.csv", class_totals)
    metrics_rows, plots, plot_data = [], [], {}
    for (recipe, organ, scope), reservoir in reservoirs.items():
        records = reservoir.trim()
        if not len(records):
            continue
        for index, name in enumerate(names):
            metrics, curves = error_metrics(records[:, index + 2], records[:, 1])
            metrics_rows.append(dict(recipe=recipe, organ=organ, region=scope, uncertainty=name, **metrics))
            plot_data[(recipe, organ, scope, name)] = (records[:, index+2], records[:, -1], curves, metrics)
        if scope != "class_union":
            continue
        fig, axes = plt.subplots(2, len(names), figsize=(5 * len(names), 8), constrained_layout=True)
        for index, name in enumerate(names):
            scores, events, curves, metrics = plot_data[(recipe, organ, scope, name)]
            for event, label, color in ((0, "Correct organ", "#377eb8"), (1, "False positive", "#b455ca"), (2, "False negative", "#e45756")):
                values = scores[events == event]
                if len(values):
                    axes[0, index].hist(values, bins=40, density=True, alpha=.45, label=label, color=color)
            axes[0, index].set(title=name, xlabel="Uncertainty", ylabel="Density")
            axes[0, index].legend(fontsize=8)
            order = np.argsort(scores)
            groups = [b for b in np.array_split(order, 10) if len(b)]
            axes[1, index].plot([scores[b].mean() for b in groups], [(events[b] != 0).mean() for b in groups], "o-")
            axes[1, index].axhline(metrics["error_prevalence"], color="gray", linestyle="--")
            axes[1, index].set(xlabel="Mean uncertainty (equal-count bins)", ylabel="One-vs-rest error rate")
        fig.suptitle(f"{organ} / {recipe}: correct pixels vs false positives vs missed pixels")
        save_plot(fig, output, f"{recipe}_{organ}_error_types", plots)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        for name in names:
            _, _, curves, metrics = plot_data[(recipe, organ, scope, name)]
            auc = metrics["error_auroc"]
            axes[0].plot(curves["fpr"], curves["tpr"], label=f"{name}: {auc:.3f}" if auc is not None else f"{name}: undefined")
            axes[1].plot(curves["recall"], curves["precision"], label=name)
            axes[2].plot(curves["coverage"], curves["risk"], label=name)
        axes[0].plot([0, 1], [0, 1], "k--", alpha=.4)
        for ax in axes[1:]:
            ax.axhline(metrics["error_prevalence"], color="gray", linestyle="--", label="Baseline")
        for ax, title, xlabel, ylabel in zip(axes, ("Error detection ROC", "Precision–recall", "Selective risk–coverage"),
                                            ("False positive rate", "Recall of organ errors", "Fraction retained"),
                                            ("True positive rate", "Precision of organ errors", "Error among retained pixels")):
            ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
            ax.legend(fontsize=7)
        fig.suptitle(f"{organ} / {recipe}: one-vs-rest error in GT/predicted class union")
        save_plot(fig, output, f"{recipe}_{organ}_error_detection", plots)
    write_csv(output / "error_detection_metrics.csv", metrics_rows)
    correlations = []
    for organ in ORGANS:
        for recipe in ("raw", "q9c"):
            for name in names:
                selected = [r for r in rows if r["organ"] == organ and r["recipe"] == recipe
                            and r.get("dice") is not None and r[f"ranking_{name}"] is not None]
                if not selected:
                    continue
                x = np.array([r[f"ranking_{name}"] for r in selected])
                y = np.array([1-r["dice"] for r in selected])
                rho = float(spearmanr(x, y).statistic) if len(x) > 1 and np.ptp(x) and np.ptp(y) else None
                correlations.append(dict(organ=organ, recipe=recipe, uncertainty=name, slices=len(x),
                                         spearman_rho_with_one_minus_dice=rho))
    write_csv(output / "slice_correlations.csv", correlations)
    for recipe in ("raw", "q9c"):
        if not any(r["recipe"] == recipe and r.get("dice") is not None for r in rows):
            continue
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
        for ax, organ in zip(axes.flat, ORGANS):
            selected = [r for r in rows if r["organ"] == organ and r["recipe"] == recipe
                        and r.get("dice") is not None and r["ranking_class_variance"] is not None]
            ax.scatter([r["ranking_class_variance"] for r in selected],
                       [1-r["dice"] for r in selected], s=12, alpha=.35)
            ax.set(title=f"{organ} ({len(selected)} slices)",
                   xlabel="Mean class variance near raw predicted organ", ylabel="1 − class Dice")
        fig.suptitle(f"{recipe}: GT-independent organ uncertainty vs slice error")
        save_plot(fig, output, f"{recipe}_slice_uncertainty_vs_error", plots)
    groups = {}
    for organ in ORGANS:
        selected = [r for r in rows if r["organ"] == organ and r["recipe"] == "raw"]
        groups[f"{organ}: uncertain predictions (class variance)"] = diverse_selection(selected, "ranking_class_variance", args.top_k, args.min_slice_gap)
        missed = [r for r in selected if r.get("fn_pixels", 0) > 0]
        groups[f"{organ}: confident missed pixels (low class variance)"] = diverse_selection(missed, "fn_class_variance", args.top_k, args.min_slice_gap, descending=False)
    ranked = [dict(group=group, rank=index+1, **row) for group, values in groups.items() for index, row in enumerate(values)]
    write_csv(output / "ranked_samples.csv", ranked)
    scales = {}
    for name in names:
        values = [value[0] for key, value in plot_data.items() if key[0] == "raw" and key[2] == "class_union" and key[3] == name]
        scales[name] = max(float(np.quantile(np.concatenate(values), .99)), 1e-8) if values else (0.25 if name == "class_variance" else 1)
    chosen = {(r["stem"], r["class_id"]) for r in ranked}
    for patient, slices in patients.items():
        selected = [(z, path) for z, path in slices if any((path.stem, label) in chosen for label in range(1, 5))]
        if not selected:
            continue
        data = load_patient(args, patient, manifest)
        for z, path in selected:
            maps = read_uncertainty(path)
            shape = maps["entropy"].shape
            layers = slice_data(args, data, path.stem, z, shape)
            probabilities = read_probabilities(args, path.stem, shape)
            for label, organ in enumerate(ORGANS, 1):
                if (path.stem, label) in chosen:
                    make_class_montage(output / "samples" / f"{path.stem}_{organ}.png", organ, label, z, patient,
                                       layers, class_scores(maps, label, probabilities), scales)
    summary = dict(patients=coverage, error_detection=metrics_rows, class_counts=class_totals,
                   gt_preprocessing=args.gt_preprocess, slice_correlations=correlations,
                   grid="cropped model grid", error_definition="(prediction == class) XOR (GT == class)",
                   regions="class_union = GT class OR predicted class; class_neighbourhood = its dilation by roi_margin",
                   metric_definitions={"global_entropy": "shared predictive entropy evaluated against each class's errors",
                                       "global_mutual_information": "shared model disagreement; NOT class-specific MI",
                                       "class_variance": "population variance of this class's probability across models",
                                       "binary_entropy": "entropy of averaged probability of this class vs all other classes, normalized by log(2)"},
                   sampling=f"uniform pixel reservoir, max {args.max_samples} per recipe/organ/region, seed 43",
                   heatmap_upper_limits=scales,
                   limitations="GT unions require GT for evaluation. Ranking uses predicted class neighbourhood; entirely missed classes have no ranking score. Slice correlations exclude those missing predictions, while FN analysis includes them. Pixel observations are correlated. Pooled Dice in class_counts is not patient-macro Dice. Class-specific MI cannot be recovered from the cached global MI or averaged probabilities.")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    note = f"Per-organ uncertainty analysis; {sum(v['slices_with_gt'] for v in coverage.values())} GT slices across {len(coverage)} patients. GT preprocessing: {args.gt_preprocess}."
    if args.gt_preprocess == "split_merged":
        note += " PROVISIONAL: esophagus/aorta labels reconstructed."
    gallery = {group: [dict(src=f"samples/{r['stem']}_{r['organ']}.png", patient=r["patient"],
                           slice=r["slice"], dice=r.get("dice")) for r in values] for group, values in groups.items() if values}
    page = '''<!doctype html><html><meta charset="utf-8"><title>Per-organ uncertainty</title>
<style>body{font:16px system-ui;background:#101922;color:#eee;margin:24px;max-width:1500px}img{width:100%;margin:12px 0}a{color:#8bd}select,input{margin:8px}</style>
<h1>Per-organ uncertainty</h1><p>__NOTE__</p><p>Each organ is evaluated as organ vs everything else. False positives are predicted organ pixels outside GT; false negatives are missed GT organ pixels. The shared entropy/MI maps are evaluated separately for each organ; class variance and optional binary entropy are class-specific.</p>
<p>Primary plots use GT/predicted organ union. Neighbourhood metrics include nearby correct background. Compare average precision with each organ's error prevalence, not just with another organ's AP. High uncertainty is useful for prioritization; low uncertainty can coexist with missed organs. Raw uncertainty is reused for Q9c.</p>
<label>Organ/case <select id="group"></select></label><label>Rank <input id="rank" type="range" min="0" value="0"></label><p id="details"></p><img id="sample">
<h2>Plots</h2>__PLOTS__<p><a href="error_detection_metrics.csv">Per-class pixel metrics</a> · <a href="slice_metrics.csv">Slice metrics and FP/FN uncertainty</a> · <a href="class_counts.csv">Class TP/FP/FN counts</a> · <a href="slice_correlations.csv">GT-independent ranking correlations</a> · <a href="summary.json">Definitions and coverage</a></p>
<script>const gallery=__GALLERY__,group=document.getElementById('group'),rank=document.getElementById('rank');Object.keys(gallery).forEach(g=>group.add(new Option(g,g)));function draw(){const rows=gallery[group.value];if(!rows)return;rank.max=rows.length-1;const r=rows[+rank.value];document.getElementById('sample').src=r.src;document.getElementById('details').textContent=`${r.patient}, slice ${r.slice}, rank ${+rank.value+1}/${rows.length}; Dice: ${r.dice===null?'GT unavailable':r.dice.toFixed(3)}`;}group.onchange=()=>{rank.value=0;draw()};rank.oninput=draw;draw();</script></html>'''
    page = page.replace("__NOTE__", html.escape(note)).replace("__GALLERY__", json.dumps(gallery))
    page = page.replace("__PLOTS__", ''.join(f'<a href="{p}"><img src="{p}"></a>' for p in plots))
    (output / "report.html").write_text(page)
    print(f">> Per-class report: {(output / 'report.html').resolve()}", flush=True)


def make_class_montage(path, organ, label, z, patient, layers, scores, scales):
    gray = layers["ct"]
    base = np.repeat(gray[..., None], 3, axis=-1) if gray is not None else np.zeros((*layers["raw"].shape, 3), np.uint8)
    panels = [("CT" if gray is not None else "CT unavailable", base)]
    for name in ("gt", "raw", "q9c"):
        values = layers[name]
        image = None
        if values is not None:
            mask = values == label
            image = base.copy()
            image[mask] = (.4 * image[mask] + .6 * COLORS[label]).astype(np.uint8)
        panels.append((name + f": {organ}" + (" unavailable" if values is None else ""), image))
    error_image = None
    if layers["gt"] is not None:
        pred, gt = layers["raw"] == label, layers["gt"] == label
        error_image = base.copy()
        error_image[pred & ~gt] = [190, 65, 210]
        error_image[~pred & gt] = [245, 65, 65]
    panels.append(("FP purple / FN red" if error_image is not None else "GT unavailable", error_image))
    panels.extend(scores.items())
    rows = int(np.ceil(len(panels) / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(15, rows * 4), constrained_layout=True)
    for ax, (name, image) in zip(axes.flat, panels):
        if image is not None:
            image = np.swapaxes(image, 0, 1)[::-1]
            if name in scores:
                artist = ax.imshow(image, cmap="magma", vmin=0, vmax=scales[name])
                fig.colorbar(artist, ax=ax, shrink=.6)
            else:
                ax.imshow(image)
        ax.set_title(name)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"{patient}, axial slice {z}: {organ} uncertainty and one-vs-rest errors")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(args):
    if args.per_class or args.per_class_only:
        per_class_analysis(args)
        if args.per_class_only:
            return
    output = args.output_dir or args.results_dir / "uncertainty_insights"
    output.mkdir(parents=True, exist_ok=True)
    (output / "samples").mkdir(exist_ok=True)
    manifest_path = args.results_dir / f"ensemble_config_{args.subset}.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if args.data_dir is None and manifest.get("data_dir") and Path(manifest["data_dir"]).is_dir():
        args.data_dir = Path(manifest["data_dir"])
    maps_dir = args.uncertainty_dir or args.results_dir / "uncertainty" / args.subset
    paths = sorted(maps_dir.glob("*.npz"))
    if not paths:
        raise ValueError(f"No uncertainty files in {maps_dir}")
    patients = {}
    for path in paths:
        patient, z = path.stem.rsplit("_", 1)
        patients.setdefault(patient, []).append((int(z), path))
    rows, reservoirs, counts, coverage = [], {}, {}, {}
    for patient, slices in patients.items():
        print(f">> Analyze {patient}: {len(slices)} slices", flush=True)
        data = load_patient(args, patient, manifest)
        annotated = 0
        for z, path in sorted(slices):
            maps = read_uncertainty(path)
            shape = maps["entropy"].shape
            layers = slice_data(args, data, path.stem, z, shape)
            gt = layers["gt"]
            raw = layers["raw"]
            if gt is None and not args.allow_missing_gt:
                raise ValueError(f"Missing GT: {path.stem}. Provide matching --data_dir or --gt_pattern, or explicitly allow partial analysis with --allow_missing_gt.")
            annotated += gt is not None
            region = ndimage.binary_dilation(raw > 0, iterations=args.roi_margin) if args.roi_margin else raw > 0
            if not region.any():
                region = np.ones(shape, bool)
            predictions = {"raw": raw}
            if gt is not None and layers["q9c"] is not None:
                predictions["q9c"] = layers["q9c"]
            for recipe, pred in predictions.items():
                row = dict(patient=patient, slice=z, stem=path.stem, recipe=recipe, has_gt=gt is not None,
                           ranking_region_pixels=int(region.sum()), predicted_foreground_pixels=int((raw > 0).sum()))
                for name in METRICS:
                    row[f"ranking_{name}"] = float(maps[name][region].mean())
                    row[f"p95_{name}"] = float(np.quantile(maps[name][region], .95))
                if gt is not None:
                    errors = pred != gt
                    foreground = (pred > 0) | (gt > 0)
                    row["foreground_error_rate"] = float(errors[foreground].mean()) if foreground.any() else None
                    row["all_pixel_error_rate"] = float(errors.mean())
                    for name in METRICS:
                        row[f"foreground_{name}"] = float(maps[name][foreground].mean()) if foreground.any() else None
                        row[f"error_{name}"] = float(maps[name][errors].mean()) if errors.any() else None
                    scores = []
                    counter = counts.setdefault((patient, recipe), np.zeros((4, 3), np.int64))
                    for label in range(1, 5):
                        intersection = int(((pred == label) & (gt == label)).sum())
                        predicted, truth = int((pred == label).sum()), int((gt == label).sum())
                        counter[label-1] += [intersection, predicted, truth]
                        if predicted + truth:
                            scores.append(2 * intersection / (predicted + truth))
                    row["macro_dice_present_organs"] = float(np.mean(scores)) if scores else None
                    for scope, mask in (("all_pixels", np.ones(shape, bool)), ("foreground_union", foreground)):
                        key = (recipe, scope)
                        reservoir = reservoirs.setdefault(key, PixelReservoir(args.max_samples, 43))
                        reservoir.add(maps, errors, mask)
                rows.append(row)
        coverage[patient] = {"slices": len(slices), "slices_with_gt": annotated}
    write_csv(output / "slice_metrics.csv", rows)
    patient_rows = []
    for (patient, recipe), counter in counts.items():
        denominator = counter[:, 1] + counter[:, 2]
        dice = np.divide(2 * counter[:, 0], denominator, out=np.full(4, np.nan), where=denominator > 0)
        patient_rows.append(dict(patient=patient, recipe=recipe,
                                 macro_dice_model_grid=float(np.nanmean(dice)),
                                 **{f"{organ}_dice": float(value) if np.isfinite(value) else None for organ, value in zip(ORGANS, dice)}))
    write_csv(output / "patient_metrics.csv", patient_rows)
    metric_rows, plots, scales = [], [], {}
    for (recipe, scope), reservoir in reservoirs.items():
        records = reservoir.trim()
        if not len(records):
            continue
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        curves_by_metric = {}
        for index, name in enumerate(METRICS):
            scores, errors = records[:, index+2], records[:, 1].astype(bool)
            metrics, curves = error_metrics(scores, errors)
            metric_rows.append(dict(recipe=recipe, region=scope, uncertainty=name, **metrics))
            curves_by_metric[name] = (metrics, curves)
            if recipe == "raw" and scope == "foreground_union":
                scales[name] = max(float(np.quantile(scores, .99)), 1e-8)
            for error, label, color in ((False, "Correct", "#377eb8"), (True, "Error", "#e45756")):
                values = scores[errors == error]
                if len(values):
                    axes[0, index].hist(values, bins=40, density=True, alpha=.5, color=color, label=label)
            axes[0, index].set(title=name, xlabel="Uncertainty", ylabel="Density")
            axes[0, index].legend()
            bins = [b for b in np.array_split(np.argsort(scores), 10) if len(b)]
            axes[1, index].plot([scores[b].mean() for b in bins], [errors[b].mean() for b in bins], "o-")
            axes[1, index].axhline(errors.mean(), color="gray", linestyle="--", label="Baseline error")
            axes[1, index].set(xlabel="Mean uncertainty (equal-count bins)", ylabel="Observed error rate")
            axes[1, index].legend()
        fig.suptitle(f"{recipe} / {scope}: uncertainty vs segmentation error")
        save_plot(fig, output, f"{recipe}_{scope}_uncertainty_vs_error", plots)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        for name, (metrics, curve) in curves_by_metric.items():
            auc = metrics["error_auroc"]
            axes[0].plot(curve["fpr"], curve["tpr"], label=f"{name}: {auc:.3f}" if auc is not None else f"{name}: undefined")
            axes[1].plot(curve["recall"], curve["precision"], label=name)
            axes[2].plot(curve["coverage"], curve["risk"], label=name)
        axes[0].plot([0, 1], [0, 1], "k--", alpha=.4)
        axes[1].axhline(metrics["error_prevalence"], color="gray", linestyle="--", label="Error prevalence")
        axes[2].axhline(metrics["error_prevalence"], color="gray", linestyle="--", label="Random retention")
        for ax, title, x, y in zip(axes, ("Error detection ROC", "Error detection precision–recall", "Selective risk–coverage"),
                                  ("False positive rate", "Recall of errors", "Fraction retained (least uncertain first)"),
                                  ("True positive rate", "Precision of errors", "Error rate among retained pixels")):
            ax.set(title=title, xlabel=x, ylabel=y)
            ax.legend(fontsize=7)
        fig.suptitle(f"{recipe} / {scope}: can uncertainty identify errors?")
        save_plot(fig, output, f"{recipe}_{scope}_error_detection", plots)
    write_csv(output / "error_detection_metrics.csv", metric_rows)
    correlations = []
    for recipe in sorted(set(row["recipe"] for row in rows)):
        selected = [r for r in rows if r["recipe"] == recipe and r.get("foreground_error_rate") is not None]
        if not selected:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        for ax, name in zip(axes, METRICS):
            x = np.array([r[f"foreground_{name}"] for r in selected])
            y = np.array([r["foreground_error_rate"] for r in selected])
            rho = float(spearmanr(x, y).statistic) if len(x) > 1 and np.ptp(x) and np.ptp(y) else None
            correlations.append(dict(recipe=recipe, uncertainty=name, slices=len(x), spearman_rho=rho))
            for patient in patients:
                mask = np.array([r["patient"] == patient for r in selected])
                if mask.any():
                    ax.scatter(x[mask], y[mask], s=12, alpha=.5, label=patient)
            ax.set(title=f"{name}; Spearman rho={rho:.3f}" if rho is not None else name,
                   xlabel="Mean foreground-union uncertainty", ylabel="Slice foreground error rate")
            ax.legend(fontsize=6)
        save_plot(fig, output, f"{recipe}_slice_scatter", plots)
    write_csv(output / "slice_correlations.csv", correlations)
    raw_rows = [row for row in rows if row["recipe"] == "raw"]
    groups = {}
    for name in METRICS:
        groups[f"Most uncertain: {name}"] = diverse_selection(raw_rows, f"ranking_{name}", args.top_k, args.min_slice_gap)
    groups["Largest errors"] = diverse_selection(raw_rows, "foreground_error_rate", args.top_k, args.min_slice_gap)
    candidates = [r for r in raw_rows if r.get("foreground_error_rate", 0) is not None and r.get("foreground_error_rate", 0) >= .05]
    groups["Confident errors (low entropy at erroneous pixels)"] = diverse_selection(
        candidates, "error_entropy", args.top_k, args.min_slice_gap, descending=False)
    mostly_correct = [r for r in raw_rows if r.get("foreground_error_rate") is not None
                      and r["foreground_error_rate"] <= .05]
    groups["Uncertain but mostly correct (foreground error <=5%)"] = diverse_selection(
        mostly_correct, "ranking_entropy", args.top_k, args.min_slice_gap)
    ranked_rows = [dict(group=group, rank=index+1, **row) for group, selected in groups.items() for index, row in enumerate(selected)]
    write_csv(output / "ranked_samples.csv", ranked_rows)
    for name in METRICS:
        if name not in scales:
            scales[name] = max(float(np.quantile([r[f"p95_{name}"] for r in raw_rows], .99)), 1e-8)
    selected_stems = {row["stem"] for row in ranked_rows}
    for patient, slices in patients.items():
        chosen = [(z, path) for z, path in slices if path.stem in selected_stems]
        if not chosen:
            continue
        data = load_patient(args, patient, manifest)
        for z, path in chosen:
            maps = read_uncertainty(path)
            layers = slice_data(args, data, path.stem, z, maps["entropy"].shape)
            make_montage(output / "samples" / f"{path.stem}.png", patient, z, layers, maps, scales)
    summary = dict(patients=coverage, slices_ranked=len(raw_rows), error_detection=metric_rows,
                   slice_correlations=correlations, gt_preprocessing=args.gt_preprocess,
                   grid="cropped model grid; native GT/Q9c resized with nearest-neighbor",
                   ranking="mean uncertainty in predicted foreground dilated by roi_margin pixels; full image if no foreground; no GT used",
                   roi_margin_pixels=args.roi_margin, sampling=f"uniform pixel reservoirs, max {args.max_samples} per recipe/region, seed 43",
                   heatmap_upper_limits_99th_percentile=scales,
                   limitations="GT coverage listed per patient. Reconstructed GT is provisional when split_merged is used. Pixel/slice observations are correlated. Raw uncertainty is reused for Q9c. Foreground unions differ between raw/Q9c. Error detection is not probability calibration.")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    write_report(output, groups, plots, summary)
    print(f">> Report: {(output / 'report.html').resolve()}", flush=True)


def save_plot(fig, output, name, plots):
    fig.savefig(output / f"{name}.png", dpi=150)
    fig.savefig(output / f"{name}.pdf")
    plots.append(f"{name}.png")
    plt.close(fig)


def make_montage(path, patient, z, layers, maps, scales):
    view = lambda values: np.swapaxes(values, 0, 1)[::-1]
    gray = layers["ct"]
    gray_rgb = np.repeat(gray[..., None], 3, axis=-1) if gray is not None else np.zeros((*layers["raw"].shape, 3), np.uint8)
    panels = [("CT" if gray is not None else "CT unavailable", gray_rgb)]
    for key, label in (("gt", "Ground truth"), ("raw", "Raw prediction"), ("q9c", "Q9c prediction")):
        mask = layers[key]
        if mask is None:
            panels.append((label + " unavailable", None))
        else:
            overlay = gray_rgb.copy()
            foreground = mask > 0
            overlay[foreground] = (.45 * gray_rgb[foreground] + .55 * COLORS[mask[foreground]]).astype(np.uint8)
            panels.append((label, overlay))
    error = (layers["raw"] != layers["gt"]) if layers["gt"] is not None else None
    error_image = gray_rgb.copy()
    if error is not None:
        error_image[error] = [245, 65, 65]
    panels.append(("Raw error" if error is not None else "Error unavailable", error_image if error is not None else None))
    panels.extend((name, maps[name]) for name in METRICS)
    fig, axes = plt.subplots(2, 4, figsize=(14, 8), constrained_layout=True)
    for ax, (name, values) in zip(axes.flat, panels):
        if values is not None:
            if name in METRICS:
                image = ax.imshow(view(values), cmap="magma", vmin=0, vmax=scales[name])
                fig.colorbar(image, ax=ax, shrink=.6)
            else:
                ax.imshow(view(values))
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(f"{patient} — axial slice {z} (zero-based)")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_report(output, groups, plots, summary):
    coverage = summary["patients"]
    annotated = sum(value["slices_with_gt"] for value in coverage.values())
    complete = sum(value["slices_with_gt"] == value["slices"] for value in coverage.values())
    note = f"Ranked {summary['slices_ranked']} slices across {len(coverage)} patients. Error analysis: {annotated} slices; {complete} patients have complete GT coverage."
    if summary["gt_preprocessing"] == "split_merged":
        note += " PROVISIONAL: esophagus/aorta GT reconstructed with the project's morphological split; not original separate annotations."
    image_links = ''.join(f'<a href="{name}"><img class="plot" src="{name}"></a>' for name in plots)
    gallery = {group: [dict(patient=row["patient"], slice=row["slice"],
                           src=f"samples/{row['stem']}.png",
                           uncertainty={name: row[f"ranking_{name}"] for name in METRICS},
                           error=row.get("foreground_error_rate")) for row in selected]
               for group, selected in groups.items() if selected}
    page = '''<!doctype html><html><meta charset="utf-8"><title>Ensemble uncertainty insights</title>
<style>body{font:16px system-ui;background:#101922;color:#eee;margin:24px;max-width:1500px}a{color:#8bd}.plot,#sample{width:100%;margin:12px 0}select,input{margin:8px}p{line-height:1.5}</style>
<h1>Ensemble uncertainty insights</h1><p>__NOTE__</p>
<p>Samples are ranked by mean uncertainty around the predicted organs; ground truth is not used in uncertainty ranking. Nearby slices and repeated patients are limited for diversity. Entropy measures predictive ambiguity; mutual information and variance measure model disagreement.</p>
<p>High AUROC means uncertainty ranks errors above correct pixels. Precision–recall should be compared with error prevalence. Risk–coverage shows error after retaining only the least uncertain pixels. Foreground-union plots exclude easy distant background; all-pixel plots are provided separately.</p>
<p>All maps and errors use the cropped model grid. Q9c is compared using the original ensemble uncertainty. Heatmaps use a shared 99th-percentile limit per metric, so colors can be compared between slices. Cyan: esophagus; gold: heart; purple: trachea; green: aorta; red: errors. Empty panels indicate unavailable data. Results are descriptive, not independent-pixel statistical evidence.</p>
<h2>Ranked samples</h2><label>Selection <select id="group"></select></label><label>Rank <input id="rank" type="range" min="0" value="0"><span id="label"></span></label>
<p id="details"></p><img id="sample"><h2>Uncertainty and error plots</h2>__PLOTS__
<p><a href="ranked_samples.csv">Ranked samples CSV</a> · <a href="error_detection_metrics.csv">Error detection CSV</a> · <a href="slice_metrics.csv">All slice metrics</a> · <a href="patient_metrics.csv">Patient Dice</a> · <a href="summary.json">Summary JSON</a></p>
<script>const gallery=__GALLERY__,group=document.getElementById('group'),rank=document.getElementById('rank');Object.keys(gallery).forEach(g=>group.add(new Option(g,g)));
function draw(){const rows=gallery[group.value];if(!rows)return;rank.max=rows.length-1;const row=rows[+rank.value];document.getElementById('label').textContent=`${+rank.value+1}/${rows.length}`;document.getElementById('sample').src=row.src;document.getElementById('details').textContent=`${row.patient}, slice ${row.slice}; `+Object.entries(row.uncertainty).map(([k,v])=>`${k}: ${v.toPrecision(4)}`).join(', ')+(row.error===null?'; GT unavailable':`; foreground error ${(100*row.error).toFixed(2)}%`)}group.onchange=()=>{rank.value=0;draw()};rank.oninput=draw;draw();</script></html>'''
    (output / "report.html").write_text(page.replace("__NOTE__", html.escape(note)).replace("__PLOTS__", image_links).replace("__GALLERY__", json.dumps(gallery)))


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", "--ensemble_dir", type=Path, default=Path("results/v2_ensemble"))
    parser.add_argument("--data_dir", type=Path, help="Exact processed dataset used by the ensemble (img/gt PNGs)")
    parser.add_argument("--subset", choices=("train", "val", "test"), default="val")
    parser.add_argument("--uncertainty_dir", type=Path, help="Directory containing slice uncertainty NPZs")
    parser.add_argument("--output_dir", type=Path, help="Default: results_dir/uncertainty_insights")
    parser.add_argument("--gt_pattern", help="Optional native GT pattern containing {id_}")
    parser.add_argument("--ct_pattern", help="Optional native CT pattern containing {id_}")
    parser.add_argument("--geometry_dir", type=Path, help="Saved crop geometry for native data/Q9c")
    parser.add_argument("--recompute_crop", action="store_true", help="Recompute A2 crop from CT and verify against raw native predictions")
    parser.add_argument("--gt_preprocess", choices=("none", "split_merged"), default="none")
    parser.add_argument("--allow_missing_gt", action="store_true", help="Allow explicitly labelled partial/no-GT analysis")
    parser.add_argument("--per_class", action="store_true", help="Also generate per-organ uncertainty analysis in output_dir/per_class")
    parser.add_argument("--per_class_only", action="store_true", help="Generate only the per-organ report")
    parser.add_argument("--class_binary_entropy", action="store_true", help="Include class-vs-rest entropy; requires saved probabilities/subset/*.npz and per-class analysis")
    parser.add_argument("--top_k", type=int, default=8, help="Maximum samples in each gallery group")
    parser.add_argument("--min_slice_gap", type=int, default=5, help="Spacing between selected slices within each patient/group")
    parser.add_argument("--roi_margin", type=int, default=3, help="Predicted-foreground dilation in pixels for GT-independent ranking")
    parser.add_argument("--max_samples", type=int, default=200000, help="Pixel reservoir size per prediction/region")
    args = parser.parse_args()
    if args.class_binary_entropy and not (args.per_class or args.per_class_only):
        parser.error("--class_binary_entropy requires --per_class or --per_class_only")
    if args.data_dir is not None and not args.data_dir.is_dir():
        parser.error(f"Processed data directory does not exist: {args.data_dir}")
    for pattern in (args.gt_pattern, args.ct_pattern):
        if pattern is not None and "{id_}" not in pattern:
            parser.error("Native file patterns must contain {id_}")
    if min(args.top_k, args.max_samples, args.min_slice_gap) < 1 or args.roi_margin < 0:
        parser.error("top_k/max_samples/min_slice_gap must be positive; roi_margin must be nonnegative")
    return args


if __name__ == "__main__":
    main(get_args())
