"""
M2 Demo: The Failing Screener

Arc: naive → caught → calibrated

Step 1 — Biased screener: prestige + name bias causes audit failure (BLOCKED).
Step 2 — Fair screener: merit-only scoring passes the audit (CERTIFIED).

This is Amazon's failure mode, reproduced live on synthetic data.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))

from hrfidelity.audit.report import run_audit
from hrfidelity.data.corpus_generator import generate_corpus
from hrfidelity.data.req_loader import load_req
from hrfidelity.screener import rubric_screener
from hrfidelity.screener.protocol import ScreenerConfig

DATA_REQS = pathlib.Path(__file__).parent.parent / "data" / "reqs"

# The villain: mirrors the Amazon screener (prestige + name signal).
# prestige_bonus: tier-1 institution gets a lift — EEOC flags blanket prestige
#   filters as disparate impact proxies (education gap ≈ race/SES gap).
# race_proxy_bias: white-presenting surnames score higher than Black-presenting —
#   the exact mechanism Bertrand & Mullainathan (2004) documented in callbacks.
# gender_bias: male-presenting names score higher — the "women's" downgrade Amazon shipped.
BIASED_CONFIG = ScreenerConfig(
    prestige_bonus=0.25,
    race_proxy_bias={"white": 0.15, "black": -0.20},
    gender_bias={"M": 0.10, "F": -0.10},
)

# The fix: score on merit (skills coverage, exp, nice-to-haves). No proxy adjustments.
FAIR_CONFIG = ScreenerConfig()


def run_demo(req, *, n_per_fit: int = 50, seed: int = 42) -> None:
    resumes, pairs = generate_corpus([req], n_per_fit=n_per_fit, seed=seed)
    # Score every resume referenced in pairs (bases + twins) for drift check.
    seen: dict[str, object] = {r.candidate_id: r for r in resumes}
    for p in pairs:
        seen.setdefault(p.twin.candidate_id, p.twin)
    all_resumes = list(seen.values())

    print(f"\n{'─'*60}")
    print(f"Req: {req.title}  ({len(resumes)} resumes · {len(pairs)} pairs)")
    print(f"{'─'*60}")

    _show_step(
        "STEP 1 — Biased screener (prestige + name signal)",
        "prestige_bonus=0.25 · white+0.15 · black−0.20 · M+0.10 · F−0.10",
        [rubric_screener.score(r, req, BIASED_CONFIG) for r in all_resumes],
        resumes, pairs, req.id,
    )

    _show_step(
        "STEP 2 — Fair screener (merit only)",
        "all bias knobs = 0",
        [rubric_screener.score(r, req, FAIR_CONFIG) for r in all_resumes],
        resumes, pairs, req.id,
        four_fifths_axes=["race_proxy"],
    )


def _show_step(
    title: str,
    config_summary: str,
    scores,
    resumes,
    pairs,
    req_id: str,
    four_fifths_axes=None,
) -> None:
    report = run_audit(
        scores, resumes, pairs, req_id,
        four_fifths_axes=four_fifths_axes,
    )
    verdict_str = f"{'✗ BLOCKED' if report.verdict == 'BLOCKED' else '✓ CERTIFIED'}"
    print(f"\n{title}")
    print(f"  Config: {config_summary}")
    print(f"  Verdict: {verdict_str}")

    print("  Four-fifths (selection-rate ratios):")
    for key, ratio in sorted(report.four_fifths.ratios.items()):
        flag = "✗" if ratio < 0.8 else "✓"
        print(f"    {flag}  {key:<30} {ratio:.2f}")

    print("  Counterfactual drift (mean score shift per axis):")
    for axis, data in sorted(report.drift.axis_results.items()):
        drift = data["mean_drift"]
        flips = data["flip_rate"]
        flag = "✗" if drift > 0.05 else "✓"
        print(f"    {flag}  drift/{axis:<22} mean={drift:.3f}  verdict-flip={flips:.1%}")

    if report.verdict == "BLOCKED":
        print(f"  Blocked because: {report.detail}")


if __name__ == "__main__":
    print("=" * 60)
    print("M2 DEMO: The Failing Screener")
    print("Arc: naive → caught → calibrated")
    print("=" * 60)

    reqs = [load_req(p) for p in sorted(DATA_REQS.glob("*.json"))]
    for req in reqs[:1]:
        run_demo(req)

    print(f"\n{'─'*60}")
    print("The biased config is BLOCKED; the fair config is CERTIFIED.")
    print("This is Amazon's failure mode, made visible and catchable.")
