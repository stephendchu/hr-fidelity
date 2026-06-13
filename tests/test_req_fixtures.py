"""
TDD for job req fixtures and the Req loader.

Reqs are the rubric foundation — each defines what "strong / medium / weak"
means for that role. The résumé generator is conditioned on (req, _latent_fit),
so every downstream number (κ, four-fifths, counterfactual drift) is only as
credible as these fixtures.

There must be ≥4 archetypes: ml-researcher, backend-eng, infra-eng, recruiter.
"""
import json
import pathlib

import pytest

from hrfidelity.data.req_loader import Req, TrueRubricBand, load_req

DATA_REQS = pathlib.Path(__file__).parent.parent / "data" / "reqs"
REQUIRED_IDS = {"ml-researcher", "backend-eng", "infra-eng", "recruiter"}


def _all_paths() -> list[pathlib.Path]:
    if not DATA_REQS.exists():
        return []
    return sorted(DATA_REQS.glob("*.json"))


def _all_dicts() -> list[dict]:
    return [json.loads(p.read_text()) for p in _all_paths()]


# ---------------------------------------------------------------------------
# Corpus-level
# ---------------------------------------------------------------------------

def test_data_reqs_dir_exists():
    assert DATA_REQS.exists(), "data/reqs/ must exist"


def test_at_least_four_reqs():
    assert len(_all_paths()) >= 4, f"expected ≥4 req files, got {len(_all_paths())}"


def test_all_archetype_ids_present():
    ids = {r["id"] for r in _all_dicts()}
    missing = REQUIRED_IDS - ids
    assert not missing, f"missing req fixtures: {missing}"


def test_req_ids_are_unique():
    ids = [r["id"] for r in _all_dicts()]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Per-req structure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("req", _all_dicts(), ids=[p.stem for p in _all_paths()])
class TestReqStructure:
    TOP_FIELDS = {"id", "title", "required_skills", "nice_to_have", "min_years", "true_rubric"}
    BANDS = {"strong", "medium", "weak"}
    BAND_FIELDS = {"min_years_exp", "required_skill_coverage", "nice_to_have_count"}

    def test_has_all_required_fields(self, req):
        missing = self.TOP_FIELDS - req.keys()
        assert not missing, f"{req.get('id')!r} missing: {missing}"

    def test_required_skills_at_least_three(self, req):
        assert len(req["required_skills"]) >= 3

    def test_skills_are_non_empty_strings(self, req):
        for s in req["required_skills"] + req["nice_to_have"]:
            assert isinstance(s, str) and s.strip(), f"bad skill value: {s!r}"

    def test_min_years_positive(self, req):
        assert req["min_years"] > 0

    def test_rubric_has_all_bands(self, req):
        missing = self.BANDS - req["true_rubric"].keys()
        assert not missing, f"true_rubric missing bands: {missing}"

    def test_rubric_bands_have_all_fields(self, req):
        for band, criteria in req["true_rubric"].items():
            missing = self.BAND_FIELDS - criteria.keys()
            assert not missing, f"band {band!r} missing: {missing}"

    def test_rubric_min_years_ordered(self, req):
        """strong ≥ medium ≥ weak — harder to reach going up."""
        r = req["true_rubric"]
        assert r["strong"]["min_years_exp"] >= r["medium"]["min_years_exp"]
        assert r["medium"]["min_years_exp"] >= r["weak"]["min_years_exp"]

    def test_rubric_skill_coverage_ordered(self, req):
        r = req["true_rubric"]
        assert r["strong"]["required_skill_coverage"] >= r["medium"]["required_skill_coverage"]
        assert r["medium"]["required_skill_coverage"] >= r["weak"]["required_skill_coverage"]

    def test_skill_coverage_in_unit_range(self, req):
        for band, criteria in req["true_rubric"].items():
            cov = criteria["required_skill_coverage"]
            assert 0.0 <= cov <= 1.0, f"band {band!r}: coverage {cov} outside [0, 1]"


# ---------------------------------------------------------------------------
# Loader round-trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", _all_paths(), ids=[p.stem for p in _all_paths()])
def test_load_req_returns_typed_req(path):
    req = load_req(path)
    assert isinstance(req, Req)
    assert req.id
    assert len(req.required_skills) >= 3
    assert isinstance(req.true_rubric.strong, TrueRubricBand)
    assert isinstance(req.true_rubric.medium, TrueRubricBand)
    assert isinstance(req.true_rubric.weak, TrueRubricBand)


@pytest.mark.parametrize("path", _all_paths(), ids=[p.stem for p in _all_paths()])
def test_load_req_rubric_coverage_ordered(path):
    req = load_req(path)
    assert req.true_rubric.strong.required_skill_coverage >= req.true_rubric.medium.required_skill_coverage
    assert req.true_rubric.medium.required_skill_coverage >= req.true_rubric.weak.required_skill_coverage
