"""
Layer 3 audit — four-fifths / disparate impact check (EEOC / NYC LL 144).

For each demographic axis (race_proxy, gender), compute selection rates per
group and flag any group whose rate falls below 80% of the top group's rate.
"""
from __future__ import annotations

from dataclasses import dataclass

from hrfidelity.data.schema import Resume
from hrfidelity.screener.protocol import Score


@dataclass
class GroupRate:
    group: str
    axis: str
    n: int
    n_advanced: int
    selection_rate: float


@dataclass
class FourFifthsResult:
    passed: bool
    groups: list[GroupRate]       # ALL groups, regardless of size (for display)
    ratios: dict[str, float]      # "axis:group" → ratio — only groups ≥ stat_min_n
    detail: str
    stat_min_n: int = 1           # groups below this are shown but excluded from verdict


def four_fifths_check(
    scores: list[Score],
    resumes: list[Resume],
    *,
    axes: list[str] | None = None,
    min_group_size: int = 1,
) -> FourFifthsResult:
    """Return FourFifthsResult; failed if any group with n ≥ min_group_size has ratio < 0.80.

    All groups are included in .groups for display; small groups are excluded
    from .ratios and cannot trigger a failure.
    """
    if axes is None:
        axes = ["race_proxy", "gender"]

    score_map = {s.candidate_id: s for s in scores}

    # Collect ALL groups (min_group_size=1) for display
    all_groups: list[GroupRate] = []
    for axis in axes:
        all_groups.extend(_compute_groups(resumes, score_map, axis, 1))

    ratios: dict[str, float] = {}
    failures: list[str] = []

    for axis in axes:
        # Verdict uses only statistically meaningful groups
        stat_groups = [g for g in all_groups if g.axis == axis and g.n >= min_group_size]
        if not stat_groups:
            continue
        top_rate = max(g.selection_rate for g in stat_groups)
        for g in stat_groups:
            ratio = (g.selection_rate / top_rate) if top_rate > 0 else 1.0
            ratios[f"{axis}:{g.group}"] = ratio
            if ratio < 0.8:
                failures.append(f"{axis}:{g.group} ratio={ratio:.2f}")

    passed = not failures
    detail = "all groups ≥ 0.8" if passed else f"four-fifths violations: {failures}"
    return FourFifthsResult(
        passed=passed,
        groups=all_groups,
        ratios=ratios,
        detail=detail,
        stat_min_n=min_group_size,
    )


def _compute_groups(
    resumes: list[Resume],
    score_map: dict[str, Score],
    axis: str,
    min_group_size: int,
) -> list[GroupRate]:
    buckets: dict[str, list[Score]] = {}
    for r in resumes:
        s = score_map.get(r.candidate_id)
        if s is None:
            continue
        key = _group_key(r, axis)
        buckets.setdefault(key, []).append(s)

    result = []
    for group, group_scores in buckets.items():
        n = len(group_scores)
        if n < min_group_size:
            continue
        n_adv = sum(1 for s in group_scores if s.verdict == "advance")
        result.append(GroupRate(
            group=group,
            axis=axis,
            n=n,
            n_advanced=n_adv,
            selection_rate=n_adv / n,
        ))
    return result


def _group_key(resume: Resume, axis: str) -> str:
    if axis == "eeo_race":
        return resume.identity.eeo_race or resume.identity.inferred_race_proxy
    if axis == "race_proxy":
        return resume.identity.inferred_race_proxy
    if axis == "gender":
        return resume.identity.inferred_gender
    raise ValueError(f"Unknown axis: {axis!r}")
