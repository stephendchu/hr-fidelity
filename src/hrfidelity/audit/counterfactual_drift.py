"""
Layer 3 audit — counterfactual drift check.

For each matched pair (base, twin), compute the absolute score difference.
If mean drift on any axis exceeds drift_threshold, the screener is BLOCKED.
A fair screener scores pairs identically (drift = 0); any non-zero drift
means the screener treats identical candidates differently based on a proxy.
"""
from __future__ import annotations

from dataclasses import dataclass

from hrfidelity.data.schema import CounterfactualPair
from hrfidelity.screener.protocol import Score


@dataclass
class DriftResult:
    passed: bool
    axis_results: dict[str, dict]  # axis → {mean_drift, max_drift, flip_rate, n_pairs}
    detail: str
    drift_threshold: float


def drift_check(
    scores: list[Score],
    pairs: list[CounterfactualPair],
    *,
    drift_threshold: float = 0.05,
) -> DriftResult:
    """Return DriftResult; failed if mean drift on any axis > drift_threshold."""
    score_map = {s.candidate_id: s for s in scores}

    by_axis: dict[str, list[tuple[float, float, bool]]] = {}
    for p in pairs:
        base_s = score_map.get(p.base.candidate_id)
        twin_s = score_map.get(p.twin.candidate_id)
        if base_s is None or twin_s is None:
            continue
        changed = base_s.verdict != twin_s.verdict
        by_axis.setdefault(p.axis, []).append(
            (base_s.raw_score, twin_s.raw_score, changed)
        )

    axis_results: dict[str, dict] = {}
    failures: list[str] = []

    for axis, entries in sorted(by_axis.items()):
        drifts = [abs(b - t) for b, t, _ in entries]
        mean_drift = sum(drifts) / len(drifts)
        max_drift = max(drifts)
        flip_rate = sum(1 for _, _, c in entries if c) / len(entries)
        axis_results[axis] = {
            "mean_drift": mean_drift,
            "max_drift": max_drift,
            "flip_rate": flip_rate,
            "n_pairs": len(entries),
        }
        if mean_drift > drift_threshold:
            failures.append(f"{axis}: mean_drift={mean_drift:.3f}")

    passed = not failures
    detail = "no significant drift" if passed else f"drift violations: {failures}"
    return DriftResult(
        passed=passed,
        axis_results=axis_results,
        detail=detail,
        drift_threshold=drift_threshold,
    )
