"""Known-answer tests for Part 12: the MIT-BIH annotation reader against the published beat counts, the loaders'
published sizes, and the pieces of scripts/cross_field.py (design, training-fold preparation, grouped folds, the
Nadeau-Bengio ratio for fixed losses, the naive and clustered SEs, Stage B's study/new split).
Run from the repository root: python tests/test_cross_field.py"""
import sys, pathlib, collections
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import modeling as M
import external_data as E
import cross_field as C


def test_mit_annotations_match_the_published_counts():
    """PhysioNet's MIT-BIH tables: 75,052 normal beats, 7,130 PVCs and 106 ventricular escape beats over the 48
    records; record 208 has 992 PVCs; record 100 has 2,239 normal beats, 33 APCs and 1 PVC."""
    base = E.EXT / "mitdb"; tot = collections.Counter(); per = {}
    for r in (base / "RECORDS").read_text().split():
        per[r] = collections.Counter(a for _, a in E.read_atr(base / f"{r}.atr")); tot += per[r]
    assert (tot[1], tot[5], tot[10]) == (75052, 7130, 106)
    assert per["208"][5] == 992 and (per["100"][1], per["100"][8], per["100"][5]) == (2239, 33, 1)


def test_mitdb_loader():
    X, y, g = E.mitdb()
    assert len(np.unique(g)) == 43 and not set(E.MIT_PACED) & set(g) and "202" not in set(g)
    assert (X.rr_pre > 0).all() and (X.rr_post > 0).all() and abs(X.pre_over_local.median() - 1) < 0.05
    assert y.sum() == 7130 + 106 - _paced_or_edge_v()


def _paced_or_edge_v():
    """Ventricular beats lost to the paced records and to each record's first and last beat."""
    base = E.EXT / "mitdb"; lost = 0
    for r in (base / "RECORDS").read_text().split():
        beats = [E.MIT_BEATS[a] for _, a in E.read_atr(base / f"{r}.atr") if a in E.MIT_BEATS]
        lost += sum(b == "V" for b in beats) if r in E.MIT_PACED else (beats[0] == "V") + (beats[-1] == "V")
    return lost


def test_published_sizes():
    """folktables' ACSIncome for California 2018 has 195,665 rows; OULAD has 32,593 registrations in 22
    module-presentations; ExtraSensory has 60 users."""
    X, y, g = E.acs(); assert len(y) == 195665 and len(np.unique(g)) == 265
    assert len(pd.read_csv(E.EXT / "oulad" / "extracted" / "studentInfo.csv")) == 32593
    X, y, g = E.oulad(); assert len(np.unique(g)) == 22
    assert len(list((E.EXT / "extrasensory" / "extracted").glob("*.features_labels.csv.gz"))) == 60


def test_design_merges_rare_codes():
    X = pd.DataFrame(dict(a=["x"] * 20 + ["y"] * 15 + ["z"] * 3, b=np.arange(38.0)))
    D, names = C.design(X)
    assert names == ["a=x", "a=y", "b"]                              # z (3 rows) merged into "other", the baseline
    assert D.shape == (38, 3) and not np.isnan(D).any() and D[:, 0].sum() == 20 and D[:, 1].sum() == 15
    X2 = pd.DataFrame(dict(a=pd.Series(["u", "v"] * 10, dtype=object), b=np.ones(20)))
    assert C.design(X2)[1] == ["a=v", "b"]                           # object columns too


def test_prepare_uses_training_rows_only():
    D = np.array([[1, np.nan], [3, 2], [5, 4], [100, np.nan]], np.float32); tr = np.array([True, True, True, False])
    F = C.prepare(D, tr)
    assert F.shape == (4, 3)                                         # two columns plus one missing flag
    assert np.allclose(F[tr].mean(0), 0, atol=1e-6)                  # standardized on the training rows
    raw = np.where(np.isnan(D), np.nanmedian(D[tr], 0), D)           # training median 3 (column 2), not 3 with row 4
    assert np.isclose((F[3, 0] * D[tr, 0].std()) + D[tr, 0].mean(), 100, atol=1e-3) and raw[3, 1] == 3


def test_grouped_partition_keeps_groups_whole():
    g = np.repeat(np.arange(23), 7); f = C.grouped_partition(g, np.random.default_rng(1))
    assert all(len(np.unique(f[g == h])) == 1 for h in range(23)) and set(f) == set(range(5))


def test_naive_se_is_cluster_robust_with_singletons():
    rng = np.random.default_rng(2); u = rng.random(300)
    assert np.isclose(np.sqrt(M.cluster_robust_var_of_mean(u, np.arange(300))), u.std(ddof=1) / np.sqrt(300))


def test_nb_ratio_is_one_for_fixed_losses():
    """Proposition 6: for fixed losses on grouped folds, averaged over random partitions the Nadeau-Bengio variance
    is (J+K-1)/(J-1) times the cluster-robust variance, so the ratio's mean is about 1."""
    rng = np.random.default_rng(3); G, m = 30, 10
    g = np.repeat(np.arange(G), m); vals = []
    for _ in range(300):
        loss = rng.normal(size=G)[g] + rng.normal(size=G * m)
        fv = []
        for _ in range(20):
            f = C.grouped_partition(g, rng); fv += [loss[f == k].mean() for k in range(5)]
        vals.append(C.nb_ratio(fv, M.cluster_robust_var_of_mean(loss, g)))
    assert abs(np.mean(vals) - 1) < 0.05, np.mean(vals)


def test_boosting_survives_a_column_missing_in_the_training_fold():
    """ExtraSensory: some sensors are missing for every user in a training fold. Boosting drops such a column."""
    rng = np.random.default_rng(5); D = rng.normal(size=(200, 3)).astype(np.float32); D[:100, 2] = np.nan
    y = (rng.random(200) < 0.4).astype(float); tr = np.arange(200) < 100
    p = C.fit_predict("boosting", D, y, tr, ~tr, 1)
    assert p.shape == (100,) and np.isfinite(p).all()


def test_stage_b_scores_only_new_groups():
    """Stage B on a small synthetic population: the study has G groups, and the target uses the other groups."""
    rng = np.random.default_rng(4); P, m = 60, 40
    g = np.repeat(np.arange(P), m); x = rng.normal(size=(P * m, 2)).astype(np.float32)
    y = (rng.random(P * m) < 1 / (1 + np.exp(-(x[:, 0] + rng.normal(size=P)[g])))).astype(float)
    out = C.stage_b_rep(20, 0, x, y, g, 999)
    for row in out:
        assert row["study_rows"] == 20 * m and 0.5 < row["auc_new"] < 1 and row["wild_q"] > 1.5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
