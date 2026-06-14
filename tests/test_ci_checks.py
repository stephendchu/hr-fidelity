"""
TDD: CI validation checks (§6) and corpus generator (§7).

Four checks run against every generated corpus:
  1. invariant    — all counterfactual pairs pass the content-hash assertion
  2. independence — identity ⊥ _latent_fit in the base population
  3. reproducibility — same seed → same résumé content
  4. realism      — résumés have non-empty skills, education, valid dates

The corpus generator saves resumes / pairs / manifest to disk (eval-store
layout from §7) and must round-trip cleanly through JSON.
"""
import pathlib

import pytest

from hrfidelity.data.ci_checks import (
    CheckResult,
    check_independence,
    check_invariant,
    check_realism,
    check_reproducibility,
)
from hrfidelity.data.corpus_generator import generate_corpus, load_corpus, save_corpus
from hrfidelity.data.counterfactual import content_hash
from hrfidelity.data.req_loader import load_req
from hrfidelity.data.schema import CounterfactualPair

DATA_REQS = pathlib.Path(__file__).parent.parent / "data" / "reqs"


def _all_reqs():
    return [load_req(p) for p in sorted(DATA_REQS.glob("*.json"))]


# ---------------------------------------------------------------------------
# Shared fixture — small in-memory corpus (fast)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def corpus():
    reqs = _all_reqs()
    resumes, pairs = generate_corpus(reqs, n_per_fit=10, seed=0)
    return resumes, pairs


# ---------------------------------------------------------------------------
# Invariant check (§6.2)
# ---------------------------------------------------------------------------

class TestInvariantCheck:
    def test_passes_on_valid_corpus(self, corpus):
        _, pairs = corpus
        result = check_invariant(pairs)
        assert result.passed, result.detail

    def test_returns_check_result(self, corpus):
        _, pairs = corpus
        assert isinstance(check_invariant(pairs), CheckResult)

    def test_fails_on_corrupted_pair(self, corpus):
        resumes, pairs = corpus
        # Build a pair whose twin has different skills — invariant must catch it
        import copy, dataclasses
        base = resumes[0]
        bad_twin = copy.deepcopy(base)
        bad_twin.skills = ["completely_different_skill"]
        bad_pair = CounterfactualPair(base=base, twin=bad_twin, axis="gender")
        result = check_invariant([bad_pair])
        assert not result.passed

    def test_all_pairs_cover_all_axes(self, corpus):
        _, pairs = corpus
        axes = {p.axis for p in pairs}
        assert axes >= {"gender", "race_proxy", "prestige_tier"}

    def test_pair_count_matches_expected(self, corpus):
        resumes, pairs = corpus
        # Each résumé gets 3 pairs (one per axis)
        assert len(pairs) == len(resumes) * 3


# ---------------------------------------------------------------------------
# Independence check (§6.1)
# ---------------------------------------------------------------------------

class TestIndependenceCheck:
    def test_passes_on_valid_corpus(self, corpus):
        resumes, _ = corpus
        result = check_independence(resumes)
        assert result.passed, result.detail

    def test_returns_check_result(self, corpus):
        resumes, _ = corpus
        assert isinstance(check_independence(resumes), CheckResult)

    def test_both_genders_in_every_fit_level(self, corpus):
        resumes, _ = corpus
        for fit in ("strong", "medium", "weak"):
            group = [r for r in resumes if r.latent_fit == fit]
            genders = {r.identity.inferred_gender for r in group}
            assert genders == {"M", "F"}, f"fit={fit}: only genders {genders}"

    def test_multiple_race_proxies_in_every_fit_level(self, corpus):
        resumes, _ = corpus
        for fit in ("strong", "medium", "weak"):
            group = [r for r in resumes if r.latent_fit == fit]
            races = {r.identity.inferred_race_proxy for r in group}
            assert len(races) >= 2, f"fit={fit}: only race proxies {races}"


# ---------------------------------------------------------------------------
# Reproducibility check (§6.4)
# ---------------------------------------------------------------------------

class TestReproducibilityCheck:
    def test_passes_with_same_seed(self):
        reqs = _all_reqs()
        result = check_reproducibility(reqs, n_per_fit=5, seed=99)
        assert result.passed, result.detail

    def test_returns_check_result(self):
        reqs = _all_reqs()
        assert isinstance(check_reproducibility(reqs, n_per_fit=3, seed=1), CheckResult)

    def test_different_seeds_produce_different_names(self):
        reqs = _all_reqs()
        a, _ = generate_corpus(reqs, n_per_fit=5, seed=1)
        b, _ = generate_corpus(reqs, n_per_fit=5, seed=2)
        names_a = {r.identity.first_name for r in a}
        names_b = {r.identity.first_name for r in b}
        # Different seeds should produce at least some variation
        assert names_a != names_b or [r.skills for r in a] != [r.skills for r in b]


# ---------------------------------------------------------------------------
# Realism check (§6.3)
# ---------------------------------------------------------------------------

class TestRealismCheck:
    def test_passes_on_valid_corpus(self, corpus):
        resumes, _ = corpus
        result = check_realism(resumes)
        assert result.passed, result.detail

    def test_returns_check_result(self, corpus):
        resumes, _ = corpus
        assert isinstance(check_realism(resumes), CheckResult)

    def test_fails_on_resume_with_no_skills(self, corpus):
        import copy
        resumes, _ = corpus
        bad = copy.deepcopy(resumes[0])
        bad.skills = []
        result = check_realism([bad])
        assert not result.passed

    def test_fails_on_resume_with_no_education(self, corpus):
        import copy
        resumes, _ = corpus
        bad = copy.deepcopy(resumes[0])
        bad.education = []
        result = check_realism([bad])
        assert not result.passed


# ---------------------------------------------------------------------------
# Corpus generator — disk round-trip (§7)
# ---------------------------------------------------------------------------

class TestCorpusGenerator:
    def test_resume_count(self, corpus):
        resumes, _ = corpus
        # 5 reqs × 3 fit levels × 10 per fit = 150
        assert len(resumes) == 150

    def test_fit_levels_balanced(self, corpus):
        resumes, _ = corpus
        from collections import Counter
        counts = Counter(r.latent_fit for r in resumes)
        assert counts["strong"] == counts["medium"] == counts["weak"] == 50

    def test_all_req_archetypes_represented(self, corpus):
        resumes, _ = corpus
        req_ids = {r.skills[0] for r in resumes}  # skills differ per req
        # Simpler: check via experience title variety
        titles = {exp.title for r in resumes for exp in r.experience}
        assert len(titles) >= 4  # different archetypes produce different titles

    def test_save_load_round_trip(self, corpus, tmp_path):
        resumes, pairs = corpus
        save_corpus(resumes, pairs, output_dir=tmp_path, seed=0, n_per_fit=10)

        # Manifest exists
        manifest = tmp_path / "manifest.json"
        assert manifest.exists()

        # Résumé files exist (bases + twins — twins are saved alongside bases)
        resume_files = list((tmp_path / "resumes").glob("*.json"))
        assert len(resume_files) == len(resumes) + len(pairs)

        # Pair files exist
        pair_files = list((tmp_path / "counterfactual_pairs").glob("*.json"))
        assert len(pair_files) == len(pairs)

    def test_saved_manifest_has_required_keys(self, corpus, tmp_path):
        import json
        resumes, pairs = corpus
        save_corpus(resumes, pairs, output_dir=tmp_path, seed=0, n_per_fit=10)
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        for key in ("seed", "n_per_fit", "req_ids", "axes", "counts", "checks"):
            assert key in manifest, f"manifest missing key: {key!r}"

    def test_load_corpus_restores_resumes(self, corpus, tmp_path):
        resumes, pairs = corpus
        save_corpus(resumes, pairs, output_dir=tmp_path, seed=0, n_per_fit=10)
        loaded_resumes, loaded_pairs = load_corpus(tmp_path)
        assert len(loaded_resumes) == len(resumes)
        assert len(loaded_pairs) == len(pairs)
        # Content hash must survive round-trip
        for orig, loaded in zip(
            sorted(resumes, key=lambda r: r.candidate_id),
            sorted(loaded_resumes, key=lambda r: r.candidate_id),
        ):
            assert content_hash(orig) == content_hash(loaded)
