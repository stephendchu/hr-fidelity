"""
CI validation checks (§6 of data-generator.md).

Four checks must pass before the corpus is considered valid:
  1. invariant      — every counterfactual pair shares a content_hash
  2. independence   — identity ⊥ _latent_fit in the base population
  3. realism        — résumés have non-empty skills, education, valid structure
  4. reproducibility— same seed → same résumé content

Each check returns a CheckResult(passed, detail) so the manifest can embed
the evidence alongside the corpus provenance.
"""
from __future__ import annotations

from dataclasses import dataclass

from hrfidelity.data.counterfactual import content_hash
from hrfidelity.data.schema import CounterfactualPair, Resume


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


# ---------------------------------------------------------------------------
# 1. Invariant check
# ---------------------------------------------------------------------------

def check_invariant(pairs: list[CounterfactualPair]) -> CheckResult:
    """Every pair must share content_hash — proxy change only, no content leak."""
    failures: list[str] = []
    for p in pairs:
        if content_hash(p.base) != content_hash(p.twin):
            failures.append(
                f"{p.base.candidate_id[:8]}…/{p.twin.candidate_id[:8]}… axis={p.axis}"
            )
    if failures:
        return CheckResult(
            "invariant",
            False,
            f"{len(failures)} pair(s) failed hash assertion: {failures[:3]}",
        )
    return CheckResult("invariant", True, f"{len(pairs)} pairs passed")


# ---------------------------------------------------------------------------
# 2. Independence check
# ---------------------------------------------------------------------------

def check_independence(resumes: list[Resume]) -> CheckResult:
    """identity ⊥ _latent_fit: no fit level should be skewed to one gender.

    Threshold: gender ratio per fit level must be in [0.15, 0.85].
    With n ≥ 30 per level, a genuine 50/50 draw has <0.1% chance of exceeding
    these bounds; a correlated generator would routinely fail.
    """
    for fit in ("strong", "medium", "weak"):
        group = [r for r in resumes if r.latent_fit == fit]
        if not group:
            return CheckResult("independence", False, f"no résumés with latent_fit={fit!r}")
        n = len(group)
        male_frac = sum(1 for r in group if r.identity.inferred_gender == "M") / n
        if not (0.15 <= male_frac <= 0.85):
            return CheckResult(
                "independence",
                False,
                f"fit={fit}: male_frac={male_frac:.2f} (n={n}) — identity not independent of fit",
            )
    return CheckResult(
        "independence",
        True,
        "gender ratio within [0.15, 0.85] for all fit levels",
    )


# ---------------------------------------------------------------------------
# 3. Realism check
# ---------------------------------------------------------------------------

def check_realism(resumes: list[Resume]) -> CheckResult:
    """Résumés must have the structural hallmarks of real-world résumés."""
    for r in resumes:
        if not r.skills:
            return CheckResult("realism", False, f"{r.candidate_id}: empty skills list")
        if not r.education:
            return CheckResult("realism", False, f"{r.candidate_id}: no education entries")
        if not r.candidate_id:
            return CheckResult("realism", False, "résumé missing candidate_id")
        for exp in r.experience:
            if not exp.bullets:
                return CheckResult(
                    "realism",
                    False,
                    f"{r.candidate_id}: experience entry '{exp.title}' has no bullets",
                )
            if not exp.start:
                return CheckResult("realism", False, f"{r.candidate_id}: experience missing start date")
    return CheckResult("realism", True, f"{len(resumes)} résumés pass structure check")


# ---------------------------------------------------------------------------
# 4. Reproducibility check
# ---------------------------------------------------------------------------

def check_reproducibility(
    reqs: list,
    *,
    n_per_fit: int,
    seed: int,
) -> CheckResult:
    """Same seed must produce the same résumé content."""
    from hrfidelity.data.corpus_generator import generate_corpus

    a_resumes, _ = generate_corpus(reqs, n_per_fit=n_per_fit, seed=seed)
    b_resumes, _ = generate_corpus(reqs, n_per_fit=n_per_fit, seed=seed)

    if len(a_resumes) != len(b_resumes):
        return CheckResult("reproducibility", False, "corpus size differs between runs")

    mismatches = [
        i for i, (a, b) in enumerate(zip(a_resumes, b_resumes))
        if content_hash(a) != content_hash(b)
    ]
    if mismatches:
        return CheckResult(
            "reproducibility",
            False,
            f"{len(mismatches)} résumé(s) differ between runs with seed={seed}",
        )
    return CheckResult(
        "reproducibility",
        True,
        f"seed={seed} → identical corpus ({len(a_resumes)} résumés)",
    )


# ---------------------------------------------------------------------------
# Run all checks
# ---------------------------------------------------------------------------

def run_all_checks(
    resumes: list[Resume],
    pairs: list[CounterfactualPair],
    reqs: list | None = None,
    seed: int | None = None,
    n_per_fit: int | None = None,
) -> list[CheckResult]:
    results = [
        check_invariant(pairs),
        check_independence(resumes),
        check_realism(resumes),
    ]
    if reqs is not None and seed is not None and n_per_fit is not None:
        results.append(check_reproducibility(reqs, n_per_fit=n_per_fit, seed=seed))
    return results
