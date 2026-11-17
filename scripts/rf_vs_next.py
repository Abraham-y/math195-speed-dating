"""Is the random forest's lead over the next-best models real? Paired AUC differences with event-clustered SEs.

Input : data/pairs.csv, data/feature_sets.json, results/report_roc_predictions.csv (forest) and
        results/report_roc_qda_predictions.csv (QDA), both from scripts/report_roc.py, results/part1_models_metrics.csv
Output: results/rf_vs_next.csv
Seeds : event folds 20261006; lasso random_state 20261206 + fold (part1_models.py, rerun here for its predictions).

The forest, QDA and lasso predictions are the held-out predictions of the headline event 5-fold cross-validation (each
checked against the stored AUC). The difference in AUC on the same dyads has the event-clustered SE of
modeling.auc_diff_clustered_var (checked by event resampling in scripts/power.py: mean SE / SD 0.97); 90% intervals
use t with 20 degrees of freedom.
Run from the repository root: python scripts/rf_vs_next.py
"""
import sys, json, pathlib
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, "scripts")
import modeling as M
from part1_models import cross_validate

RES = pathlib.Path("results")
T20 = stats.t.ppf(0.95, 20)


def main():
    pairs = pd.read_csv("data/pairs.csv"); sets = json.loads(pathlib.Path("data/feature_sets.json").read_text())
    met = pd.read_csv(RES / "part1_models_metrics.csv")
    met = met[(met.spec == M.HEADLINE.label) & (met.dyads == 4184)].set_index(["feature_set", "model"]).auc
    rf_all = pd.read_csv(RES / "report_roc_predictions.csv"); qda_all = pd.read_csv(RES / "report_roc_qda_predictions.csv")
    rows = []
    for fs in ("pre", "pre+date"):
        _, _, _, (prob, y, ev, _) = cross_validate(pairs, sets, M.HEADLINE, fs, ("lasso",), jobs=5)
        rf, qda = rf_all[rf_all.feature_set == fs], qda_all[qda_all.feature_set == fs]
        assert (rf.match.to_numpy() == y).all() and (qda.match.to_numpy() == y).all() and (rf.event.to_numpy() == ev).all()
        scores = {"rf": rf.prob.to_numpy(), "qda": qda.prob.to_numpy(), "lasso": prob["lasso"]}
        for m, s in scores.items():
            assert abs(M.auc(s, y) - met[fs, m]) < 1e-9, (fs, m, M.auc(s, y), met[fs, m])
        for other in ("qda", "lasso"):
            d = M.auc(scores["rf"], y) - M.auc(scores[other], y)
            se = np.sqrt(M.auc_diff_clustered_var(scores["rf"], scores[other], y, ev))
            rows.append(dict(feature_set=fs, comparison=f"rf minus {other}", auc_difference=d, se=se,
                             ci90_low=d - T20 * se, ci90_high=d + T20 * se, z=d / se, dyads=len(y), fold_scheme="event",
                             fold_seed=M.FOLD_SEED))
    out = pd.DataFrame(rows); out.to_csv(RES / "rf_vs_next.csv", index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
