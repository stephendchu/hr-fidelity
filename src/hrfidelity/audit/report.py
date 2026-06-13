"""
Layer 3 audit — combined report.

run_audit() runs four_fifths_check + drift_check and returns a single
AuditReport with a CERTIFIED / BLOCKED verdict and the evidence behind it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from hrfidelity.audit.counterfactual_drift import DriftResult, drift_check
from hrfidelity.audit.disparate_impact import FourFifthsResult, four_fifths_check
from hrfidelity.data.schema import CounterfactualPair, Resume
from hrfidelity.screener.protocol import Score

AuditVerdict = Literal["CERTIFIED", "BLOCKED"]


@dataclass
class AuditReport:
    req_id: str
    verdict: AuditVerdict
    four_fifths: FourFifthsResult
    drift: DriftResult
    n_resumes: int
    n_pairs: int
    detail: str


def run_audit(
    scores: list[Score],
    resumes: list[Resume],
    pairs: list[CounterfactualPair],
    req_id: str,
    *,
    drift_threshold: float = 0.05,
    four_fifths_axes: list[str] | None = None,
) -> AuditReport:
    ff = four_fifths_check(scores, resumes, axes=four_fifths_axes)
    dr = drift_check(scores, pairs, drift_threshold=drift_threshold)

    verdict: AuditVerdict = "CERTIFIED" if (ff.passed and dr.passed) else "BLOCKED"

    reasons = []
    if not ff.passed:
        reasons.append(f"four-fifths: {ff.detail}")
    if not dr.passed:
        reasons.append(f"drift: {dr.detail}")

    return AuditReport(
        req_id=req_id,
        verdict=verdict,
        four_fifths=ff,
        drift=dr,
        n_resumes=len(resumes),
        n_pairs=len(pairs),
        detail="; ".join(reasons) if reasons else "all checks passed",
    )
