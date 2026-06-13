"""
TDD: résumé generator (§3 of data-generator.md).

generate_resume(req, latent_fit, rng) must:
  - Condition skills + experience on the rubric band (latent_fit drives quality)
  - Sample identity INDEPENDENTLY of latent_fit (the independence invariant)
  - Be fully reproducible given a seed

The independence invariant is the §6.1 honesty check: in the base population,
identity ⊥ _latent_fit.  Any correlation that appears in screener output is
bias introduced by the screener, not baked into the data.
"""
import pathlib
import random

import pytest

from hrfidelity.data.req_loader import load_req
from hrfidelity.data.resume_generator import generate_resume
from hrfidelity.data.schema import Resume

DATA_REQS = pathlib.Path(__file__).parent.parent / "data" / "reqs"


def _total_years(resume: Resume) -> int:
    """Year-level total to avoid month-arithmetic flakiness."""
    total = 0
    for exp in resume.experience:
        start_y = int(exp.start.split("-")[0])
        end_y = int(exp.end.split("-")[0]) if exp.end else 2024
        total += end_y - start_y
    return total


# ---------------------------------------------------------------------------
# generate_resume contract
# ---------------------------------------------------------------------------

class TestGenerateResume:

    @pytest.fixture
    def backend_req(self):
        return load_req(DATA_REQS / "backend-eng.json")

    def test_returns_resume_instance(self, backend_req):
        r = generate_resume(backend_req, "strong", random.Random(0))
        assert isinstance(r, Resume)

    @pytest.mark.parametrize("fit", ["strong", "medium", "weak"])
    def test_latent_fit_set_correctly(self, backend_req, fit):
        r = generate_resume(backend_req, fit, random.Random(0))
        assert r.latent_fit == fit

    def test_skills_subset_of_req_pool(self, backend_req):
        pool = set(backend_req.required_skills + backend_req.nice_to_have)
        for fit in ("strong", "medium", "weak"):
            r = generate_resume(backend_req, fit, random.Random(0))
            assert set(r.skills) <= pool, f"{fit}: skills outside req pool: {set(r.skills) - pool}"

    def test_strong_has_more_skills_than_weak(self, backend_req):
        strong = generate_resume(backend_req, "strong", random.Random(42))
        weak = generate_resume(backend_req, "weak", random.Random(42))
        assert len(strong.skills) > len(weak.skills)

    def test_strong_has_more_experience_years_than_weak(self, backend_req):
        strong = generate_resume(backend_req, "strong", random.Random(42))
        weak = generate_resume(backend_req, "weak", random.Random(42))
        assert _total_years(strong) > _total_years(weak)

    def test_same_seed_same_resume(self, backend_req):
        a = generate_resume(backend_req, "strong", random.Random(7))
        b = generate_resume(backend_req, "strong", random.Random(7))
        assert a.identity.first_name == b.identity.first_name
        assert a.skills == b.skills
        assert _total_years(a) == _total_years(b)

    def test_candidate_ids_are_unique(self, backend_req):
        ids = {
            generate_resume(backend_req, "medium", random.Random(i)).candidate_id
            for i in range(30)
        }
        assert len(ids) == 30

    def test_strong_and_medium_have_experience(self, backend_req):
        for fit in ("strong", "medium"):
            r = generate_resume(backend_req, fit, random.Random(0))
            assert len(r.experience) >= 1, f"{fit}: expected ≥1 experience entry"

    def test_has_education_entry(self, backend_req):
        r = generate_resume(backend_req, "strong", random.Random(0))
        assert len(r.education) >= 1

    def test_prestige_tier_in_valid_range(self, backend_req):
        for fit in ("strong", "medium", "weak"):
            r = generate_resume(backend_req, fit, random.Random(0))
            assert r.education[0].prestige_tier in {1, 2, 3}

    def test_prestige_tier_varies_independently_of_fit(self, backend_req):
        """prestige_tier must not track latent_fit — it is a proxy axis, not a quality signal."""
        strong_tiers = {
            generate_resume(backend_req, "strong", random.Random(i)).education[0].prestige_tier
            for i in range(30)
        }
        weak_tiers = {
            generate_resume(backend_req, "weak", random.Random(i)).education[0].prestige_tier
            for i in range(30)
        }
        assert len(strong_tiers) > 1, "prestige_tier is constant for strong — not independent"
        assert len(weak_tiers) > 1, "prestige_tier is constant for weak — not independent"

    def test_experience_bullets_reference_candidate_skills(self, backend_req):
        r = generate_resume(backend_req, "strong", random.Random(0))
        skill_set = set(r.skills)
        all_bullets = " ".join(b for exp in r.experience for b in exp.bullets)
        hits = sum(1 for s in skill_set if s in all_bullets)
        assert hits >= 1, "no candidate skill mentioned in any bullet"

    @pytest.mark.parametrize("req_id", ["backend-eng", "ml-researcher", "infra-eng", "recruiter"])
    def test_all_archetypes_generate_without_error(self, req_id):
        req = load_req(DATA_REQS / f"{req_id}.json")
        for fit in ("strong", "medium", "weak"):
            generate_resume(req, fit, random.Random(0))  # must not raise


# ---------------------------------------------------------------------------
# Independence invariant: identity ⊥ _latent_fit
# ---------------------------------------------------------------------------

class TestIdentityIndependenceOfFit:
    """§6.1 honesty check: in the base population, identity must be uncorrelated
    with _latent_fit.  If this fails, bias in screener output is confounded by
    bias in the data — the audit claim collapses."""

    @pytest.fixture(scope="class")
    def corpus(self):
        req = load_req(DATA_REQS / "backend-eng.json")
        rng = random.Random(0)
        resumes = [
            generate_resume(req, fit, rng)
            for fit in ("strong", "medium", "weak")
            for _ in range(30)
        ]
        return resumes

    def test_both_genders_appear_in_strong(self, corpus):
        genders = {r.identity.inferred_gender for r in corpus if r.latent_fit == "strong"}
        assert genders == {"M", "F"}

    def test_both_genders_appear_in_weak(self, corpus):
        genders = {r.identity.inferred_gender for r in corpus if r.latent_fit == "weak"}
        assert genders == {"M", "F"}

    def test_multiple_race_proxies_in_strong(self, corpus):
        races = {r.identity.inferred_race_proxy for r in corpus if r.latent_fit == "strong"}
        assert len(races) >= 2

    def test_multiple_race_proxies_in_weak(self, corpus):
        races = {r.identity.inferred_race_proxy for r in corpus if r.latent_fit == "weak"}
        assert len(races) >= 2

    def test_gender_ratio_roughly_equal_across_fit_levels(self, corpus):
        """No fit level should be >80% one gender (would indicate a bug, not bad luck at n=30)."""
        for fit in ("strong", "medium", "weak"):
            group = [r for r in corpus if r.latent_fit == fit]
            male_frac = sum(1 for r in group if r.identity.inferred_gender == "M") / len(group)
            assert 0.15 <= male_frac <= 0.85, (
                f"{fit}: male fraction {male_frac:.2f} — identity not independent of fit"
            )
