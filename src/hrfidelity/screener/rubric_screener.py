"""
Layer 1 screener: deterministic rubric-based scoring.

Scores a Resume against a Req by measuring required-skill coverage,
nice-to-have matches, and years of experience — weighted per ScreenerConfig.

Bias knobs (prestige_bonus, race_proxy_bias, gender_bias) default to zero.
With defaults the score is purely merit-based and passes the Layer 3 audit.
Setting any bias knob reproduces the Amazon failure mode and triggers BLOCKED.

No LLM calls. An LLM back-end can be swapped in later behind the same
Score / ScreenerConfig interface without breaking tests or the audit layer.
"""
from __future__ import annotations

from hrfidelity.data.req_loader import Req
from hrfidelity.data.schema import Resume
from hrfidelity.screener.protocol import Score, ScreenerConfig, ScreenerVerdict

_FAIR = ScreenerConfig()


def score(resume: Resume, req: Req, config: ScreenerConfig = _FAIR) -> Score:
    req_set = set(req.required_skills)
    resume_skills = set(resume.skills)

    req_coverage = len(req_set & resume_skills) / max(len(req_set), 1)

    nh_count = sum(1 for s in req.nice_to_have if s in resume_skills)
    nh_score = nh_count / len(req.nice_to_have) if req.nice_to_have else 0.0

    target_years = req.true_rubric.strong.min_years_exp
    total_years = _total_years(resume)
    exp_score = min(1.0, total_years / max(target_years, 1)) if target_years > 0 else 1.0

    raw = (
        config.required_skill_weight * req_coverage
        + config.nice_to_have_weight * nh_score
        + config.experience_weight * exp_score
    )

    # Bias adjustments — applied after merit score so any non-zero value
    # produces audit-detectable drift on counterfactual pairs.
    if config.prestige_bonus and resume.education:
        tier = resume.education[0].prestige_tier
        raw += config.prestige_bonus * (2 - tier)  # tier1:+1×, tier2:0, tier3:−1×

    race = resume.identity.inferred_race_proxy
    if race in config.race_proxy_bias:
        raw += config.race_proxy_bias[race]

    gender = resume.identity.inferred_gender
    if gender in config.gender_bias:
        raw += config.gender_bias[gender]

    raw = max(0.0, min(1.0, raw))
    verdict = _to_verdict(raw, config)
    rationale = (
        f"Skill coverage {req_coverage:.0%}, {nh_count} nice-to-have, "
        f"{total_years} yr exp → {verdict}"
    )

    return Score(
        candidate_id=resume.candidate_id,
        req_id=req.id,
        verdict=verdict,
        rationale=rationale,
        raw_score=raw,
    )


def _total_years(resume: Resume) -> int:
    total = 0
    for exp in resume.experience:
        start_yr = int(exp.start.split("-")[0])
        end_yr = int(exp.end.split("-")[0]) if exp.end else 2024
        total += end_yr - start_yr
    return total


def _to_verdict(raw: float, config: ScreenerConfig) -> ScreenerVerdict:
    if raw >= config.threshold_advance:
        return "advance"
    if raw >= config.threshold_borderline:
        return "borderline"
    return "reject"
