"""
Inter-rater agreement measures for fidelity calibration.

Cohen's κ   — agreement between two judges (AI vs one recruiter).
Fleiss' κ   — agreement across N judges (human-human baseline).

Both treat "left", "right", "tie" as the three nominal categories.
"""
from __future__ import annotations

from hrfidelity.fidelity.protocol import KappaResult, Vote

_CATEGORIES = ("left", "right", "tie")


def cohen_kappa(votes_a: list[Vote], votes_b: list[Vote]) -> KappaResult:
    """Cohen's κ between two judges on overlapping pairs.

    Pairs not voted by both judges are excluded (outer join drops silently).
    Returns κ=0.0 for degenerate cases (no shared pairs, p_e=1).
    """
    judge_a = votes_a[0].judge_id if votes_a else "unknown"
    judge_b = votes_b[0].judge_id if votes_b else "unknown"

    map_a = {v.pair_id: v.choice for v in votes_a}
    map_b = {v.pair_id: v.choice for v in votes_b}

    shared = [pid for pid in map_a if pid in map_b]
    n = len(shared)

    if n == 0:
        return KappaResult(
            kappa=0.0, judge_a=judge_a, judge_b=judge_b,
            n_pairs=0, observed_agreement=0.0, expected_agreement=0.0,
        )

    n_agree = sum(1 for pid in shared if map_a[pid] == map_b[pid])
    p_o = n_agree / n

    p_e = sum(
        (sum(1 for pid in shared if map_a[pid] == c) / n)
        * (sum(1 for pid in shared if map_b[pid] == c) / n)
        for c in _CATEGORIES
    )

    if p_e >= 1.0:
        kappa = 0.0
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)

    return KappaResult(
        kappa=kappa,
        judge_a=judge_a,
        judge_b=judge_b,
        n_pairs=n,
        observed_agreement=p_o,
        expected_agreement=p_e,
    )


def fleiss_kappa(votes_per_judge: list[list[Vote]]) -> float:
    """Fleiss' κ across N raters on the same subjects.

    Returns 0.0 for degenerate cases (< 2 raters, 0 subjects).
    """
    if len(votes_per_judge) < 2:
        return 0.0

    all_pair_ids = sorted({v.pair_id for votes in votes_per_judge for v in votes})
    n_subjects = len(all_pair_ids)
    n_raters = len(votes_per_judge)

    if n_subjects == 0:
        return 0.0

    pid_idx = {pid: i for i, pid in enumerate(all_pair_ids)}
    cat_idx = {c: j for j, c in enumerate(_CATEGORIES)}
    n_cats = len(_CATEGORIES)

    # n_ratings[i][j] = number of raters assigning category j to subject i
    n_ratings = [[0] * n_cats for _ in range(n_subjects)]
    for votes in votes_per_judge:
        for v in votes:
            if v.pair_id in pid_idx:
                i = pid_idx[v.pair_id]
                j = cat_idx.get(v.choice, 0)
                n_ratings[i][j] += 1

    # Marginal probability of each category
    total = n_subjects * n_raters
    p_j = [
        sum(n_ratings[i][j] for i in range(n_subjects)) / total
        for j in range(n_cats)
    ]
    p_e = sum(p ** 2 for p in p_j)

    # Per-subject agreement
    if n_raters < 2:
        return 0.0
    p_i = [
        sum(n_ratings[i][j] * (n_ratings[i][j] - 1) for j in range(n_cats))
        / (n_raters * (n_raters - 1))
        for i in range(n_subjects)
    ]
    p_bar = sum(p_i) / n_subjects

    if p_e >= 1.0:
        return 0.0

    return (p_bar - p_e) / (1.0 - p_e)
