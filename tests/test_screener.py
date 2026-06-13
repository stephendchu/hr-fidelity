"""
TDD: Layer 1 screener (M2).

A rubric screener scores a Resume against a Req and returns a Score with:
  - verdict: "advance" | "borderline" | "reject"
  - raw_score: float in [0, 1]
  - rationale: non-empty string

Fair config (no bias knobs set) must not produce score drift on counterfactual
pairs. Any drift must come from config, not from the scoring logic itself.
"""
import pathlib
import pytest

from hrfidelity.data.corpus_generator import generate_corpus
from hrfidelity.data.req_loader import load_req
from hrfidelity.screener.protocol import Score, ScreenerConfig
from hrfidelity.screener import rubric_screener

DATA_REQS = pathlib.Path(__file__).parent.parent / "data" / "reqs"


@pytest.fixture(scope="module")
def req():
    return load_req(DATA_REQS / "backend-eng.json")


@pytest.fixture(scope="module")
def corpus(req):
    resumes, pairs = generate_corpus([req], n_per_fit=10, seed=7)
    return resumes, pairs


@pytest.fixture(scope="module")
def strong_resume(corpus):
    resumes, _ = corpus
    return next(r for r in resumes if r.latent_fit == "strong")


@pytest.fixture(scope="module")
def weak_resume(corpus):
    resumes, _ = corpus
    return next(r for r in resumes if r.latent_fit == "weak")


@pytest.fixture(scope="module")
def medium_resume(corpus):
    resumes, _ = corpus
    return next(r for r in resumes if r.latent_fit == "medium")


class TestScoreShape:
    def test_returns_score_instance(self, strong_resume, req):
        result = rubric_screener.score(strong_resume, req)
        assert isinstance(result, Score)

    def test_candidate_id_preserved(self, strong_resume, req):
        result = rubric_screener.score(strong_resume, req)
        assert result.candidate_id == strong_resume.candidate_id

    def test_req_id_preserved(self, strong_resume, req):
        result = rubric_screener.score(strong_resume, req)
        assert result.req_id == req.id

    def test_raw_score_in_unit_interval(self, corpus, req):
        resumes, _ = corpus
        for r in resumes:
            s = rubric_screener.score(r, req)
            assert 0.0 <= s.raw_score <= 1.0, f"raw_score={s.raw_score} out of [0,1]"

    def test_rationale_nonempty(self, strong_resume, req):
        result = rubric_screener.score(strong_resume, req)
        assert result.rationale.strip()


class TestVerdicts:
    def test_strong_resume_advances(self, strong_resume, req):
        result = rubric_screener.score(strong_resume, req)
        assert result.verdict == "advance"

    def test_weak_resume_rejects(self, weak_resume, req):
        result = rubric_screener.score(weak_resume, req)
        assert result.verdict == "reject"

    def test_medium_resume_borderline_or_advance(self, medium_resume, req):
        result = rubric_screener.score(medium_resume, req)
        assert result.verdict in ("borderline", "advance")

    def test_raising_advance_threshold_blocks_strong(self, strong_resume, req):
        config = ScreenerConfig(threshold_advance=0.99)
        result = rubric_screener.score(strong_resume, req, config)
        assert result.verdict in ("borderline", "reject")

    def test_lowering_borderline_threshold_prevents_reject(self, weak_resume, req):
        config = ScreenerConfig(threshold_borderline=0.0)
        result = rubric_screener.score(weak_resume, req, config)
        assert result.verdict in ("borderline", "advance")


class TestFairConfigNoDrift:
    def test_gender_pairs_score_identically(self, corpus, req):
        _, pairs = corpus
        config = ScreenerConfig()
        gender_pairs = [p for p in pairs if p.axis == "gender"]
        for p in gender_pairs:
            base_s = rubric_screener.score(p.base, req, config)
            twin_s = rubric_screener.score(p.twin, req, config)
            assert abs(base_s.raw_score - twin_s.raw_score) < 1e-9, (
                f"Fair config drifted on gender pair: {base_s.raw_score:.4f} vs {twin_s.raw_score:.4f}"
            )

    def test_race_proxy_pairs_score_identically(self, corpus, req):
        _, pairs = corpus
        config = ScreenerConfig()
        race_pairs = [p for p in pairs if p.axis == "race_proxy"]
        for p in race_pairs:
            base_s = rubric_screener.score(p.base, req, config)
            twin_s = rubric_screener.score(p.twin, req, config)
            assert abs(base_s.raw_score - twin_s.raw_score) < 1e-9

    def test_prestige_tier_pairs_score_identically_when_no_bonus(self, corpus, req):
        _, pairs = corpus
        config = ScreenerConfig()
        tier_pairs = [p for p in pairs if p.axis == "prestige_tier"]
        for p in tier_pairs:
            base_s = rubric_screener.score(p.base, req, config)
            twin_s = rubric_screener.score(p.twin, req, config)
            assert abs(base_s.raw_score - twin_s.raw_score) < 1e-9


class TestBiasKnobs:
    def test_prestige_bonus_creates_drift_on_tier_pairs(self, corpus, req):
        _, pairs = corpus
        config = ScreenerConfig(prestige_bonus=0.3)
        tier_pairs = [p for p in pairs if p.axis == "prestige_tier"]
        drifts = [
            abs(rubric_screener.score(p.base, req, config).raw_score
                - rubric_screener.score(p.twin, req, config).raw_score)
            for p in tier_pairs
        ]
        assert all(d > 1e-9 for d in drifts), "prestige_bonus should create drift on all tier pairs"

    def test_race_bias_lowers_penalized_group_score(self, corpus, req):
        resumes, _ = corpus
        fair = ScreenerConfig()
        biased = ScreenerConfig(race_proxy_bias={"black": -0.20})
        black_resumes = [r for r in resumes if r.identity.inferred_race_proxy == "black"]
        if not black_resumes:
            pytest.skip("no black-proxy résumés in corpus with this seed")
        r = black_resumes[0]
        assert rubric_screener.score(r, req, biased).raw_score < rubric_screener.score(r, req, fair).raw_score

    def test_gender_bias_lowers_penalized_group_score(self, corpus, req):
        resumes, _ = corpus
        fair = ScreenerConfig()
        biased = ScreenerConfig(gender_bias={"F": -0.20})
        f_resumes = [r for r in resumes if r.identity.inferred_gender == "F"]
        if not f_resumes:
            pytest.skip("no female résumés in corpus with this seed")
        r = f_resumes[0]
        assert rubric_screener.score(r, req, biased).raw_score < rubric_screener.score(r, req, fair).raw_score

    def test_race_bias_leaves_unmentioned_groups_unchanged(self, corpus, req):
        resumes, _ = corpus
        fair = ScreenerConfig()
        biased = ScreenerConfig(race_proxy_bias={"black": -0.20})
        white_resumes = [r for r in resumes if r.identity.inferred_race_proxy == "white"]
        if not white_resumes:
            pytest.skip("no white-proxy résumés")
        r = white_resumes[0]
        assert abs(
            rubric_screener.score(r, req, biased).raw_score
            - rubric_screener.score(r, req, fair).raw_score
        ) < 1e-9
