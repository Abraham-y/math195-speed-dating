"""ROC curves of the best Part 1 model (random forest), before the date and with the in-date ratings, for the slides.

Reruns the headline event 5-fold cross-validation of part1_models.py for the random forest only (same folds, seeds,
tuning and preparation), checks that the held-out AUCs equal the stored ones in results/part1_models_metrics.csv,
then draws the two ROC curves.
Input : data/pairs.csv, data/feature_sets.json, results/part1_models_metrics.csv
Output: results/report_roc_predictions.csv  held-out predicted match probability per dyad, both feature sets
        results/report_roc_points.csv       the ROC curve points
        results/report_roc_<mode>.png        mode = figure (titled), paper (untitled), slide (untitled, fonts 1.5x);
                                             copied to report/figures/roc.png, paper_roc.png, slide_roc.png
Seeds: event folds 20261006, inner folds 20261006 + 1 + fold, forest 20261106 + fold (all from part1_models.py).
Run from the repository root: python scripts/report_roc.py [--jobs 5] [--plot-only]
"""
import sys, json, pathlib, shutil, argparse
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
sys.path.insert(0, "scripts")
import modeling as M
from part1_models import cross_validate

RES, OUT = pathlib.Path("results"), pathlib.Path("report/figures")
BLUE, ORANGE, INK, INK2, GRID, REF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
LABELS = {"pre": "Before the date (pre)", "pre+date": "With the in-date ratings (pre+date)"}


def predictions(jobs):
    pairs = pd.read_csv("data/pairs.csv"); sets = json.loads(pathlib.Path("data/feature_sets.json").read_text())
    stored = pd.read_csv(RES / "part1_models_metrics.csv")
    stored = stored[(stored.headline == True) & (stored.fold_scheme == "event") & (stored.model == "rf")].set_index("feature_set").auc
    out = []
    for fs in ("pre", "pre+date"):
        rows, _, _, (prob, y, ev, _) = cross_validate(pairs, sets, M.HEADLINE, fs, ("rf",), jobs)
        auc = roc_auc_score(y, prob["rf"])
        assert abs(auc - stored[fs]) < 1e-9, f"{fs}: rerun AUC {auc} differs from stored {stored[fs]}"
        print(f"{fs}: random forest held-out AUC {auc:.6f} (stored {stored[fs]:.6f})")
        out.append(pd.DataFrame(dict(feature_set=fs, event=ev, match=np.asarray(y), prob=prob["rf"])))
    return pd.concat(out, ignore_index=True)


def plot(pred, mode):
    sc = 1.5 if mode == "slide" else 1.0
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9 * sc, "axes.edgecolor": REF, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False})
    fig, ax = plt.subplots(figsize=(4.6, 4.2) if mode == "slide" else (4.2, 3.9))
    ax.plot([0, 1], [0, 1], color=REF, linestyle="--", linewidth=1)
    ax.text(0.74, 0.66, "chance (AUC 0.5)", rotation=45, rotation_mode="anchor", color=INK2, fontsize=7.5 * sc, ha="center", va="top")
    for fs, col in (("pre", BLUE), ("pre+date", ORANGE)):
        d = pred[pred.feature_set == fs]
        fpr, tpr, _ = roc_curve(d.match, d.prob); auc = roc_auc_score(d.match, d.prob)
        ax.plot(fpr, tpr, color=col, linewidth=2, label=f"{LABELS[fs]}: AUC {auc:.2f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01); ax.set_aspect("equal")
    ax.set_xlabel("False positive rate (non-matches called a match)"); ax.set_ylabel("True positive rate (matches caught)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=7.5 * sc, handlelength=1.5)
    if mode == "figure":
        ax.set_title("Random forest, held-out events (event 5-fold CV, 4,184 dyads)", loc="left", fontsize=9, color=INK)
    name = {"figure": "roc", "paper": "paper_roc", "slide": "slide_roc"}[mode]
    path = RES / f"report_roc_{mode}.png"
    fig.savefig(path, dpi=200 if mode == "slide" else 300, bbox_inches="tight", facecolor="white"); plt.close(fig)
    OUT.mkdir(exist_ok=True); shutil.copy(path, OUT / f"{name}.png"); print(f"wrote {path} (copy: {OUT / (name + '.png')})")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--plot-only", action="store_true", help="redraw from results/report_roc_predictions.csv")
    args = ap.parse_args()
    if args.plot_only:
        pred = pd.read_csv(RES / "report_roc_predictions.csv")
    else:
        pred = predictions(args.jobs); pred.to_csv(RES / "report_roc_predictions.csv", index=False)
    pts = []
    for fs in ("pre", "pre+date"):
        d = pred[pred.feature_set == fs]; fpr, tpr, thr = roc_curve(d.match, d.prob)
        pts.append(pd.DataFrame(dict(feature_set=fs, false_positive_rate=fpr, true_positive_rate=tpr, threshold=thr)))
    pd.concat(pts).to_csv(RES / "report_roc_points.csv", index=False)
    for mode in ("figure", "paper", "slide"):
        plot(pred, mode)


if __name__ == "__main__":
    main()
