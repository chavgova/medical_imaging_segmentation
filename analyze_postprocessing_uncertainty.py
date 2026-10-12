"""Compare original ensemble uncertainty with pixels changed by Q9c.

Uses saved predictions, uncertainty and matching crop geometry; no inference.
All comparisons use the cropped model grid. Q9c uncertainty is not recomputed.
"""
import argparse
import html
import json
from pathlib import Path

import numpy as np
from analyze_ensemble_uncertainty import (
    ORGANS, METRICS, PixelReservoir, error_metrics, load_patient,
    slice_data, read_uncertainty, write_csv, save_plot, plt,
)


def change_masks(raw, post, label=None):
    if label is None:
        return {
            "retained": (raw > 0) & (post == raw),
            "removed": (raw > 0) & (post == 0),
            "relabeled": (raw > 0) & (post > 0) & (post != raw),
            "added": (raw == 0) & (post > 0),
        }
    return {"retained": (raw == label) & (post == label),
            "removed": (raw == label) & (post != label),
            "added": (raw != label) & (post == label)}


def change_outcomes(raw, post, gt, mask, label=None):
    if gt is None:
        return {}
    before = raw != gt if label is None else (raw == label) != (gt == label)
    after = post != gt if label is None else (post == label) != (gt == label)
    return {"corrected_pixels": int((mask & before & ~after).sum()),
            "harmed_pixels": int((mask & ~before & after).sum()),
            "wrong_before_pixels": int((mask & before).sum()),
            "wrong_after_pixels": int((mask & after).sum())}


def mean_uncertainty_table(rows):
    """Render full-pixel uncertainty means from summary dictionaries or CSV rows."""
    columns = ("mean_entropy", "mean_mutual_information", "mean_class_variance", "mean_total_variance")
    table = '<section id="mean-uncertainties"><h2>Mean uncertainty by class and change</h2>'
    table += '<p>Means use all matching pixels and the original ensemble uncertainty. Class variance measures disagreement for the named organ; total variance sums disagreement across all classes. A dash means no pixels or that the metric does not apply. Retained pixels may still be incorrect. The all-organs rows summarize pixels together, rather than averaging the class means.</p>'
    table += '<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%"><thead><tr>'
    headings = ("Class", "Change", "Pixels", "Mean entropy", "Mean mutual information", "Mean class variance", "Mean total variance")
    table += ''.join(f'<th scope="col" style="text-align:left;padding:10px;border-bottom:2px solid #888">{h}</th>' for h in headings)
    table += '</tr></thead><tbody>'
    for scope in (*ORGANS, "all_organs"):
        for row in rows:
            if row["scope"] != scope:
                continue
            values = [scope.replace("_", " ").title(), row["group"].capitalize(), f'{int(row["pixels"]):,}']
            for key in columns:
                value = row.get(key)
                values.append(f'{float(value):.4f}' if value is not None and value != "" else '—')
            table += '<tr>' + ''.join(f'<td style="padding:10px;border-bottom:1px solid #ddd">{html.escape(value)}</td>' for value in values) + '</tr>'
    return table + '</tbody></table></div><p>Source: <a href="change_summary.csv">change_summary.csv</a>.</p></section>'


def main(args):
    output = args.output_dir or args.results_dir / "postprocessing_uncertainty"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.results_dir / f"ensemble_config_{args.subset}.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if args.data_dir is None and manifest.get("data_dir") and Path(manifest["data_dir"]).is_dir():
        args.data_dir = Path(manifest["data_dir"])
    patients = {}
    for path in sorted((args.results_dir / "uncertainty" / args.subset).glob("*.npz")):
        patient, z = path.stem.rsplit("_", 1)
        patients.setdefault(patient, []).append((int(z), path))
    if not patients:
        raise ValueError("No saved slice uncertainty maps found")
    totals, reservoirs, added_reservoirs, slice_rows, coverage = {}, {}, {}, [], {}
    for patient, slices in patients.items():
        print(f">> Analyze postprocessing changes: {patient}", flush=True)
        data = load_patient(args, patient, manifest)
        if data["q9c"] is None:
            raise ValueError(f"{patient}: Q9c comparison needs its prediction volume and matching --geometry_dir (or --recompute_crop with CT)")
        annotated = 0
        for z, path in slices:
            maps = read_uncertainty(path)
            layers = slice_data(args, data, path.stem, z, maps["entropy"].shape)
            raw, post, gt = layers["raw"], layers["q9c"], layers["gt"]
            if gt is None and not args.allow_missing_gt:
                raise ValueError(f"Missing GT: {path.stem}; supply matching GT or explicitly use --allow_missing_gt")
            annotated += gt is not None
            for label, scope in [(None, "all_organs"), *enumerate(ORGANS, 1)]:
                names = METRICS if label is None else ("entropy", "mutual_information", "class_variance")
                scores = {name: maps[name] for name in names if name != "class_variance"}
                if label is not None:
                    scores["class_variance"] = maps["class_variance"][label]
                masks = change_masks(raw, post, label)
                codes = np.zeros(raw.shape, np.uint8)
                codes[masks["removed"]] = 1
                if label is None:
                    codes[masks["relabeled"]] = 2
                candidate = raw > 0 if label is None else raw == label
                values = dict(scores, event=codes)
                reservoir = reservoirs.setdefault(scope, PixelReservoir(args.max_samples, 43, names + ("event",)))
                reservoir.add(values, masks["removed"], candidate)
                added = added_reservoirs.setdefault(scope, PixelReservoir(args.max_samples, 44, names))
                added.add(scores, np.zeros(raw.shape, bool), masks["added"])
                for group, mask in masks.items():
                    n = int(mask.sum())
                    outcome = change_outcomes(raw, post, gt, mask, label)
                    row = dict(patient=patient, slice=z, scope=scope, group=group,
                               pixels=n, has_gt=gt is not None, **outcome)
                    entry = totals.setdefault((scope, group), dict(pixels=0, gt_pixels=0,
                        corrected_pixels=0, harmed_pixels=0, wrong_before_pixels=0, wrong_after_pixels=0,
                        sums={name: 0.0 for name in names}))
                    entry["pixels"] += n
                    if gt is not None:
                        entry["gt_pixels"] += n
                        for key, value in outcome.items():
                            entry[key] += value
                    for name in names:
                        score_sum = float(scores[name][mask].sum(dtype=np.float64))
                        entry["sums"][name] += score_sum
                        row[f"mean_{name}"] = score_sum / n if n else None
                    slice_rows.append(row)
        coverage[patient] = dict(slices=len(slices), slices_with_gt=annotated)
    stats = []
    for (scope, group), entry in totals.items():
        n, ngt = entry["pixels"], entry["gt_pixels"]
        stats.append(dict(scope=scope, group=group, **{k: v for k, v in entry.items() if k != "sums"},
                          corrected_fraction=entry["corrected_pixels"] / ngt if ngt else None,
                          harmed_fraction=entry["harmed_pixels"] / ngt if ngt else None,
                          net_errors_removed=entry["wrong_before_pixels"]-entry["wrong_after_pixels"] if ngt else None,
                          **{f"mean_{name}": value/n if n else None for name, value in entry["sums"].items()}))
    write_csv(output / "change_summary.csv", stats)
    write_csv(output / "slice_changes.csv", slice_rows)
    metric_rows, bins_rows, plots = [], [], []
    for scope, reservoir in reservoirs.items():
        records = reservoir.trim()
        if not len(records):
            continue
        names = reservoir.names[:-1]
        added = added_reservoirs[scope].trim()
        removed = records[:, 1].astype(bool)
        events = records[:, -1]
        fig, axes = plt.subplots(2, len(names), figsize=(5*len(names), 8), constrained_layout=True)
        for index, name in enumerate(names):
            scores = records[:, index+2]
            for code, group, color in [(0, "Retained", "#377eb8"), (1, "Removed", "#e45756"), (2, "Relabeled", "#b455ca")]:
                selected = scores[events == code]
                if len(selected):
                    axes[0, index].hist(selected, bins=40, density=True, alpha=.45, label=group, color=color)
            if len(added):
                axes[0, index].hist(added[:, index+2], bins=40, density=True, alpha=.35, label="Added", color="#50aa60")
            axes[0, index].set(title=name, xlabel="Original ensemble uncertainty", ylabel="Density")
            axes[0, index].legend(fontsize=8)
            groups = [g for g in np.array_split(np.argsort(scores), 10) if len(g)]
            x = [float(scores[g].mean()) for g in groups]
            y = [float(removed[g].mean()) for g in groups]
            axes[1, index].plot(x, y, "o-")
            axes[1, index].axhline(removed.mean(), color="gray", linestyle="--", label="Overall removal rate (sample)")
            axes[1, index].set(xlabel="Mean uncertainty (equal-count bins)", ylabel="Fraction removed by Q9c", ylim=(0, 1))
            axes[1, index].legend(fontsize=7)
            if removed.any():
                metrics, _ = error_metrics(scores, removed)
                metric_rows.append(dict(scope=scope, uncertainty=name, sample_pixels=len(records),
                    sampled_removal_prevalence=metrics["error_prevalence"],
                    removal_auroc=metrics["error_auroc"], removal_average_precision=metrics["error_average_precision"],
                    top_10pct_removal_enrichment=metrics["top_10pct_error_enrichment"]))
            for i, (g, avg, rate) in enumerate(zip(groups, x, y), 1):
                bins_rows.append(dict(scope=scope, uncertainty=name, quantile_bin=i,
                                      sample_pixels=len(g), mean_uncertainty=avg, removal_rate=rate))
        fig.suptitle(f"{scope}: does Q9c preferentially remove uncertain predictions?")
        save_plot(fig, output, f"{scope}_uncertainty_and_removal", plots)
    write_csv(output / "removal_detection_metrics.csv", metric_rows)
    write_csv(output / "removal_by_uncertainty_bin.csv", bins_rows)
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    organs = [row for row in stats if row["scope"] != "all_organs" and row["group"] == "removed"]
    x = np.arange(len(organs))
    ax.bar(x-.2, [r["corrected_pixels"] for r in organs], .4, label="Removed false positives: helpful", color="#50aa60")
    ax.bar(x+.2, [r["harmed_pixels"] for r in organs], .4, label="Removed true positives: harmful", color="#e45756")
    ax.set(xticks=x, xticklabels=[r["scope"] for r in organs], ylabel="Pixels with GT", title="What did class removal do to segmentation accuracy?")
    ax.legend()
    save_plot(fig, output, "removal_benefit_and_harm", plots)
    summary = dict(patients=coverage, gt_preprocessing=args.gt_preprocess,
        grid="cropped model grid; native Q9c predictions resized using matching geometry",
        definitions={"all_organs_removed": "raw foreground -> Q9c background",
                     "all_organs_relabeled": "raw foreground -> a different foreground class",
                     "class_removed": "raw class -> any other label (including background)",
                     "class_added": "any other raw label -> Q9c class",
                     "retained": "same foreground/class label before and after",
                     "removal_auroc": "probability a removed raw prediction has higher uncertainty than a nonremoved raw prediction; ties count half"},
        sampling=f"uniform reservoir per scope, seed 43, max {args.max_samples}; counts and means use all pixels",
        limitations="Observed association, not proof Q9c explicitly uses uncertainty. Maps are original ensemble uncertainty. Per-class changes overlap across organs for relabeling; do not sum as unique pixels. Pixel observations are correlated. Missing-GT changes are excluded from benefit/harm counts.",
        changes=stats, removal_detection=metric_rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    note = f"{sum(v['slices'] for v in coverage.values())} slices across {len(coverage)} patients; {sum(v['slices_with_gt'] for v in coverage.values())} with GT. GT preprocessing: {args.gt_preprocess}."
    page = '<!doctype html><html><meta charset="utf-8"><title>Postprocessing and uncertainty</title><style>body{font:16px system-ui;margin:24px;max-width:1500px}img{width:100%}p{line-height:1.5}</style><h1>Does Q9c remove uncertain pixels?</h1>'
    page += f'<p>{html.escape(note)}</p>'
    page += '<p>Each organ plot compares uncertainty in retained, removed and added class pixels. Removal can also mean relabeling to another organ. The all-organs plot separates deletion to background from relabeling. Heatmaps are not recomputed after Q9c.</p><p>Increasing removal rate across uncertainty bins means Q9c preferentially removes uncertain predictions. Removal AUROC >0.5 indicates the same association. This does not establish that removal improves accuracy: inspect corrected versus harmed pixels separately. Densities are normalized per group and do not show group sizes.</p>'
    page += mean_uncertainty_table(stats)
    page += ''.join(f'<a href="{p}"><img loading="lazy" src="{p}"></a>' for p in plots)
    page += '<p>' + ' · '.join(f'<a href="{name}">{name}</a>' for name in ["change_summary.csv", "removal_detection_metrics.csv", "removal_by_uncertainty_bin.csv", "slice_changes.csv", "summary.json"]) + '</p></html>'
    (output / "report.html").write_text(page)
    print(f">> Report: {(output / 'report.html').resolve()}", flush=True)


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, default=Path("results/v2_ensemble"))
    parser.add_argument("--data_dir", type=Path, help="Exact processed dataset with GT PNGs and crop geometry")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--subset", choices=("train", "val", "test"), default="val")
    parser.add_argument("--geometry_dir", type=Path)
    parser.add_argument("--gt_pattern", help="Optional native GT pattern containing {id_}")
    parser.add_argument("--ct_pattern", help="Optional native CT pattern containing {id_}")
    parser.add_argument("--recompute_crop", action="store_true")
    parser.add_argument("--gt_preprocess", choices=("none", "split_merged"), default="none")
    parser.add_argument("--allow_missing_gt", action="store_true")
    parser.add_argument("--max_samples", type=int, default=200000)
    args = parser.parse_args()
    if args.max_samples < 1:
        parser.error("max_samples must be positive")
    if args.data_dir is not None and not args.data_dir.is_dir():
        parser.error(f"Dataset directory does not exist: {args.data_dir}")
    for pattern in (args.gt_pattern, args.ct_pattern):
        if pattern is not None and "{id_}" not in pattern:
            parser.error("Native patterns must contain {id_}")
    return args


if __name__ == "__main__":
    main(get_args())
