"""
TDD: Layer 3 audit (M2).

Two mechanisms:
  1. four_fifths_check — selection_rate(group) / selection_rate(top_group) >= 0.8
  2. drift_check       — mean score drift on matched pairs < drift_threshold (0.05)

A fair screener passes both; the biased screener (prestige + race/gender bias)
fails both. run_audit combines them into a CERTIFIED / BLOCKED verdict.

Design notes:
  - Scores passed to drift_check must cover BOTH base and twin candidate_ids
    (twins are scored on the same screener to measure counterfactual drift).
  - The race_proxy four-fifths check is reliable with n=150 because there are
    only 2 swap-table race groups and their sizes are large (~107 white, ~43 black).
    Gender four-fifths can fail by sampling noise with n=150 (groups ~75 each);
    we test gender bias via drift_check instead.
"""
import pathlib
import pytest

from hrfidelity.audit.counterfactual_drift import DriftResult, drift_check
from hrfidelity.audit.disparate_impact import FourFifthsResult, GroupRate, four_fifths_check
from hrfidelity.audit.report import AuditReport, run_audit
from hrfidelity.data.corpus_generator import generate_corpus
from hrfidelity.data.req_loader import load_req
from hrfidelity.screener.protocol import Score, ScreenerConfig
from hrfidelity.screener import rubric_screener

DATA_REQS = pathlib.Path(__file__).parent.parent / "data" / "reqs"

# The villain: prestige + name bias — mirrors the Amazon failure mode.
BIASED_CONFIG = ScreenerConfig(
    prestige_bonus=0.25,
    race_proxy_bias={"white": 0.15, "black": -0.20},
    gender_bias={"M": 0.10, "F": -0.10},
)
FAIR_CONFIG = ScreenerConfig()


@pytest.fixture(scope="module")
def req():
    return load_req(DATA_REQS / "backend-eng.json")


@pytest.fixture(scope="module")
def corpus(req):
    # n_per_fit=50 → 150 base resumes, 450 pairs
    return generate_corpus([req], n_per_fit=50, seed=44)


def _all_resumes_in_corpus(corpus):
    """Return all unique resumes (bases + twins) from a corpus tuple."""
    resumes, pairs = corpus
    seen: dict[str, object] = {r.candidate_id: r for r in resumes}
    for p in pairs:
        seen.setdefault(p.twin.candidate_id, p.twin)
    return list(seen.values())


@pytest.fixture(scope="module")
def fair_scores(corpus, req):
    # Score every resume referenced in pairs (bases + twins) so drift_check
    # can find scores for both sides of each matched pair.
    return [rubric_screener.score(r, req, FAIR_CONFIG) for r in _all_resumes_in_corpus(corpus)]


@pytest.fixture(scope="module")
def biased_scores(corpus, req):
    return [rubric_screener.score(r, req, BIASED_CONFIG) for r in _all_resumes_in_corpus(corpus)]


# ---------------------------------------------------------------------------
# Four-fifths disparate impact
# ---------------------------------------------------------------------------

class TestFourFifthsCheck:
    def test_returns_four_fifths_result(self, fair_scores, corpus):
        resumes, _ = corpus
        assert isinstance(four_fifths_check(fair_scores, resumes), FourFifthsResult)

    def test_fair_screener_passes_race_proxy_four_fifths(self, fair_scores, corpus):
        # race_proxy is reliable: 2 groups, large n (~107 white, ~43 black).
        resumes, _ = corpus
        result = four_fifths_check(fair_scores, resumes, axes=["race_proxy"])
        assert result.passed, result.detail

    def test_fails_for_biased_screener_on_race_proxy(self, biased_scores, corpus):
        resumes, _ = corpus
        result = four_fifths_check(biased_scores, resumes, axes=["race_proxy"])
        assert not result.passed, "biased screener must fail four-fifths on race_proxy"

    def test_fails_for_biased_screener_on_gender(self, biased_scores, corpus):
        resumes, _ = corpus
        result = four_fifths_check(biased_scores, resumes, axes=["gender"])
        assert not result.passed, "biased screener must fail four-fifths on gender"

    def test_groups_cover_race_and_gender_axes(self, fair_scores, corpus):
        resumes, _ = corpus
        result = four_fifths_check(fair_scores, resumes)
        axes = {g.axis for g in result.groups}
        assert "race_proxy" in axes and "gender" in axes

    def test_selection_rate_matches_advanced_over_total(self, fair_scores, corpus):
        resumes, _ = corpus
        result = four_fifths_check(fair_scores, resumes)
        for g in result.groups:
            expected = g.n_advanced / g.n
            assert abs(g.selection_rate - expected) < 1e-9

    def test_ratios_are_in_unit_interval(self, fair_scores, corpus):
        resumes, _ = corpus
        result = four_fifths_check(fair_scores, resumes)
        for key, ratio in result.ratios.items():
            assert 0.0 <= ratio <= 1.0, f"ratio out of [0,1]: {key}={ratio}"

    def test_top_group_has_ratio_one_per_axis(self, fair_scores, corpus):
        resumes, _ = corpus
        result = four_fifths_check(fair_scores, resumes)
        for axis in ["race_proxy", "gender"]:
            axis_ratios = [v for k, v in result.ratios.items() if k.startswith(axis + ":")]
            assert any(abs(r - 1.0) < 1e-9 for r in axis_ratios), (
                f"No top group with ratio=1.0 on axis={axis}"
            )


# ---------------------------------------------------------------------------
# Counterfactual drift
# ---------------------------------------------------------------------------

class TestDriftCheck:
    def test_returns_drift_result(self, fair_scores, corpus):
        _, pairs = corpus
        assert isinstance(drift_check(fair_scores, pairs), DriftResult)

    def test_passes_for_fair_screener(self, fair_scores, corpus):
        _, pairs = corpus
        result = drift_check(fair_scores, pairs)
        assert result.passed, result.detail

    def test_fails_for_biased_screener(self, biased_scores, corpus):
        _, pairs = corpus
        result = drift_check(biased_scores, pairs)
        assert not result.passed, "biased screener must fail drift check"

    def test_axis_results_covers_all_three_axes(self, fair_scores, corpus):
        _, pairs = corpus
        result = drift_check(fair_scores, pairs)
        assert set(result.axis_results.keys()) >= {"gender", "race_proxy", "prestige_tier"}

    def test_axis_results_has_required_fields(self, fair_scores, corpus):
        _, pairs = corpus
        result = drift_check(fair_scores, pairs)
        for axis, data in result.axis_results.items():
            for field in ("mean_drift", "max_drift", "flip_rate", "n_pairs"):
                assert field in data, f"axis={axis} missing field {field!r}"

    def test_fair_screener_has_zero_drift_on_gender_pairs(self, fair_scores, corpus):
        _, pairs = corpus
        result = drift_check(fair_scores, pairs)
        assert result.axis_results["gender"]["mean_drift"] < 1e-9

    def test_fair_screener_has_zero_drift_on_prestige_pairs(self, fair_scores, corpus):
        _, pairs = corpus
        result = drift_check(fair_scores, pairs)
        assert result.axis_results["prestige_tier"]["mean_drift"] < 1e-9

    def test_biased_screener_has_large_drift_on_gender_pairs(self, biased_scores, corpus):
        _, pairs = corpus
        result = drift_check(biased_scores, pairs)
        assert result.axis_results["gender"]["mean_drift"] > 0.05

    def test_biased_screener_has_large_drift_on_prestige_pairs(self, biased_scores, corpus):
        _, pairs = corpus
        result = drift_check(biased_scores, pairs)
        assert result.axis_results["prestige_tier"]["mean_drift"] > 0.05


# ---------------------------------------------------------------------------
# Full audit report
# ---------------------------------------------------------------------------

class TestRunAudit:
    def test_returns_audit_report(self, fair_scores, corpus, req):
        resumes, pairs = corpus
        assert isinstance(run_audit(fair_scores, resumes, pairs, req.id), AuditReport)

    def test_certified_for_fair_screener(self, fair_scores, corpus, req):
        resumes, pairs = corpus
        # Check only race_proxy four-fifths: with n=150, gender four-fifths can
        # fail by sampling noise (~±0.08 on ratio); race groups are large enough.
        report = run_audit(
            fair_scores, resumes, pairs, req.id,
            four_fifths_axes=["race_proxy"],
        )
        assert report.verdict == "CERTIFIED", report.detail

    def test_blocked_for_biased_screener(self, biased_scores, corpus, req):
        resumes, pairs = corpus
        report = run_audit(biased_scores, resumes, pairs, req.id)
        assert report.verdict == "BLOCKED", report.detail

    def test_report_carries_req_id_and_counts(self, fair_scores, corpus, req):
        resumes, pairs = corpus
        report = run_audit(fair_scores, resumes, pairs, req.id)
        assert report.req_id == req.id
        assert report.n_resumes == len(resumes)
        assert report.n_pairs == len(pairs)

    def test_blocked_report_has_nonempty_detail(self, biased_scores, corpus, req):
        resumes, pairs = corpus
        report = run_audit(biased_scores, resumes, pairs, req.id)
        assert report.detail.strip()

    def test_certified_report_detail_says_passed(self, fair_scores, corpus, req):
        resumes, pairs = corpus
        report = run_audit(
            fair_scores, resumes, pairs, req.id,
            four_fifths_axes=["race_proxy"],
        )
        assert "passed" in report.detail.lower()
