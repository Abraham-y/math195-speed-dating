"""Stable matching within each event (PREREG.md, Part 3, exploratory; proposals/05, connections: Gale-Shapley).

Input : data/clean.csv
Output: results/stable_matching_events.csv (one row per event), results/stable_matching_summary.csv,
        results/stable_matching_pairs.csv (every stable pair, with whether it was a mutual yes)
No random operations.

Each person's preference list over the opposite sex at their event follows the liking they gave each partner on
the scorecard (ties broken by the partner's mean liking received, then iid; a missing rating ranks last). From
these lists, the men-proposing and the women-proposing Gale-Shapley algorithms give the two extreme stable
matchings (one-to-one; when one side is larger, some of its members stay unmatched). They are compared with what
happened: the mutual yeses (realized matches, which can be many per person).
Run from the repository root: python scripts/stable_matching.py
"""
import pathlib
import numpy as np, pandas as pd

RES = pathlib.Path("results")


def gale_shapley(prop_prefs, recv_prefs):
    """prop_prefs: {proposer: [receivers, best first]}; recv_prefs: {receiver: [proposers, best first]}.
    Returns {proposer: receiver} for the proposer-optimal stable matching (complete lists)."""
    rank = {r: {p: k for k, p in enumerate(lst)} for r, lst in recv_prefs.items()}
    free, nxt, held = list(prop_prefs), {p: 0 for p in prop_prefs}, {}
    while free:
        p = free.pop()
        if nxt[p] >= len(prop_prefs[p]):
            continue                                            # p has been rejected by everyone: stays unmatched
        r = prop_prefs[p][nxt[p]]; nxt[p] += 1
        cur = held.get(r)
        if cur is None:
            held[r] = p
        elif rank[r][p] < rank[r][cur]:
            held[r] = p; free.append(cur)
        else:
            free.append(p)
    return {p: r for r, p in held.items()}


def blocking_pairs(match, prefs_a, prefs_b):
    """Pairs (a, b) who both prefer each other to their partners in `match` (a -> b); unmatched prefer anyone."""
    inv = {b: a for a, b in match.items()}
    rank_a = {a: {b: k for k, b in enumerate(l)} for a, l in prefs_a.items()}
    rank_b = {b: {a: k for k, a in enumerate(l)} for b, l in prefs_b.items()}
    out = []
    for a in prefs_a:
        for b in prefs_b:
            if match.get(a) == b:
                continue
            a_pref = a not in match or rank_a[a][b] < rank_a[a][match[a]]
            b_pref = b not in inv or rank_b[b][a] < rank_b[b][inv[b]]
            if a_pref and b_pref:
                out.append((a, b))
    return out


def preferences(ev, rater_gender):
    """Preference lists of the raters of one gender over the other gender, from their liking ratings."""
    raters = sorted(ev[ev.gender == rater_gender].rater.unique())
    others = sorted(ev[ev.gender == 1 - rater_gender].rater.unique())
    received = ev[ev.gender == rater_gender].groupby("ratee").like.mean()
    prefs = {}
    for r in raters:
        mine = ev[(ev.gender == rater_gender) & (ev.rater == r)].set_index("ratee").like
        key = lambda o: (-(mine.get(o, np.nan) if pd.notna(mine.get(o, np.nan)) else -np.inf), -received.get(o, -np.inf), o)
        prefs[r] = sorted(others, key=key)
    return prefs


def main():
    clean = pd.read_csv("data/clean.csv", low_memory=False)
    rows, pairs = [], []
    for e, ev in clean.groupby("event"):
        wp, mp = preferences(ev, 0), preferences(ev, 1)
        realized = set(map(tuple, ev[(ev.gender == 0) & (ev.match == 1)][["rater", "ratee"]].to_numpy()))   # (woman, man)
        men_prop = gale_shapley(mp, wp)                          # man -> woman
        wom_prop = gale_shapley(wp, mp)                          # woman -> man
        assert not blocking_pairs(wom_prop, wp, mp) and not blocking_pairs({w: m for m, w in men_prop.items()}, wp, mp)
        stable = {"men propose": {(w, m) for m, w in men_prop.items()}, "women propose": {(w, m) for w, m in wom_prop.items()}}
        def mean_rank(prs, side):
            if side == "women":
                return np.mean([wp[w].index(m) + 1 for w, m in prs])
            return np.mean([mp[m].index(w) + 1 for w, m in prs])
        best_real = {}
        for w, m in realized:
            best_real.setdefault(("w", w), []).append(wp[w].index(m) + 1); best_real.setdefault(("m", m), []).append(mp[m].index(w) + 1)
        row = dict(event=e, women=len(wp), men=len(mp), realized_matches=len(realized),
                   people_with_a_match=len(best_real), pairs_in_both_stable=len(stable["men propose"] & stable["women propose"]))
        for name, S in stable.items():
            key = name.split()[0]
            row[f"{key}_stable_pairs"] = len(S)
            row[f"{key}_share_mutual_yes"] = np.mean([p in realized for p in S])
            row[f"{key}_women_mean_rank"] = mean_rank(S, "women"); row[f"{key}_men_mean_rank"] = mean_rank(S, "men")
            pairs += [dict(event=e, matching=name, woman=w, man=m, mutual_yes=(w, m) in realized,
                           woman_rank_of_man=wp[w].index(m) + 1, man_rank_of_woman=mp[m].index(w) + 1) for w, m in sorted(S)]
        row["share_realized_in_a_stable_matching"] = np.mean([p in (stable["men propose"] | stable["women propose"]) for p in realized]) if realized else np.nan
        row["mean_best_realized_rank"] = np.mean([min(v) for v in best_real.values()]) if best_real else np.nan
        rows.append(row)
    E = pd.DataFrame(rows); E.to_csv(RES / "stable_matching_events.csv", index=False)
    P = pd.DataFrame(pairs); P.to_csv(RES / "stable_matching_pairs.csv", index=False)
    summ = []
    for name in ("men propose", "women propose"):
        p = P[P.matching == name]
        summ.append(dict(matching=name, stable_pairs=len(p), share_mutual_yes=p.mutual_yes.mean(),
                         women_mean_rank_of_partner=p.woman_rank_of_man.mean(), men_mean_rank_of_partner=p.man_rank_of_woman.mean(),
                         women_with_top_choice=(p.woman_rank_of_man == 1).mean(), men_with_top_choice=(p.man_rank_of_woman == 1).mean()))
    w = E.realized_matches
    summ.append(dict(matching="realized (mutual yeses)", stable_pairs=int(w.sum()),
                     share_mutual_yes=1.0, women_mean_rank_of_partner=np.nan, men_mean_rank_of_partner=np.nan,
                     share_of_realized_in_a_stable_matching=np.nansum(E.share_realized_in_a_stable_matching * w) / w.sum(),
                     pairs_in_both_stable_matchings=int(E.pairs_in_both_stable.sum())))
    S = pd.DataFrame(summ); S.to_csv(RES / "stable_matching_summary.csv", index=False)
    print(S.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(E.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
