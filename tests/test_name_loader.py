"""
TDD: name / demographic loaders (§4 of data-generator.md).

Three vendored samples ground the identity axis:
  SSA first names     → inferred_gender
  Census surnames     → inferred_race_proxy (modal race probability)
  Bertrand-Mullainathan pairs → the canonical white/Black first-name pairs
                               that anchor every counterfactual swap

sample_identity is the single entry point the résumé generator calls.
Its output must be fully reproducible given a seed and must never depend
on _latent_fit — the caller enforces that independence.
"""
import pathlib
import random

import pytest

from hrfidelity.data.name_loader import (
    load_bm_pairs,
    load_census_surnames,
    load_ssa_first_names,
    sample_identity,
)
from hrfidelity.data.schema import Identity

DATA_NAMES = pathlib.Path(__file__).parent.parent / "data" / "names"


# ---------------------------------------------------------------------------
# Data file structure
# ---------------------------------------------------------------------------

class TestDataFiles:
    def test_names_dir_exists(self):
        assert DATA_NAMES.exists()

    def test_ssa_has_male_and_female_lists(self):
        data = load_ssa_first_names()
        assert "male" in data and "female" in data

    def test_ssa_male_at_least_ten_names(self):
        assert len(load_ssa_first_names()["male"]) >= 10

    def test_ssa_female_at_least_ten_names(self):
        assert len(load_ssa_first_names()["female"]) >= 10

    def test_ssa_entries_are_weighted(self):
        data = load_ssa_first_names()
        for entry in data["male"] + data["female"]:
            assert "name" in entry
            assert "weight" in entry
            assert isinstance(entry["name"], str) and entry["name"].strip()
            assert entry["weight"] > 0

    def test_census_surnames_at_least_ten(self):
        assert len(load_census_surnames()) >= 10

    def test_census_entries_have_modal_race(self):
        valid = {"white", "black", "hispanic", "asian", "other"}
        for s in load_census_surnames():
            assert "name" in s
            assert "modal_race" in s, f"{s['name']!r} missing modal_race"
            assert s["modal_race"] in valid

    def test_bm_has_male_and_female_pairs(self):
        data = load_bm_pairs()
        assert "male_pairs" in data and "female_pairs" in data

    def test_bm_male_pairs_at_least_five(self):
        assert len(load_bm_pairs()["male_pairs"]) >= 5

    def test_bm_female_pairs_at_least_five(self):
        assert len(load_bm_pairs()["female_pairs"]) >= 5

    def test_bm_pairs_have_white_and_black_keys(self):
        data = load_bm_pairs()
        for pair in data["male_pairs"] + data["female_pairs"]:
            assert "white" in pair and "black" in pair
            assert pair["white"].strip() and pair["black"].strip()


# ---------------------------------------------------------------------------
# sample_identity contract
# ---------------------------------------------------------------------------

class TestSampleIdentity:
    def test_returns_identity_instance(self):
        assert isinstance(sample_identity(random.Random(0)), Identity)

    def test_inferred_gender_is_m_or_f(self):
        rng = random.Random(1)
        for _ in range(30):
            assert sample_identity(rng).inferred_gender in {"M", "F"}

    def test_source_is_valid(self):
        valid = {"ssa", "census", "bertrand_mullainathan"}
        rng = random.Random(2)
        for _ in range(30):
            assert sample_identity(rng).source in valid

    def test_same_seed_same_result(self):
        a = sample_identity(random.Random(42))
        b = sample_identity(random.Random(42))
        assert a.first_name == b.first_name
        assert a.last_name == b.last_name
        assert a.inferred_gender == b.inferred_gender
        assert a.inferred_race_proxy == b.inferred_race_proxy

    def test_different_seeds_produce_variety_of_first_names(self):
        names = {sample_identity(random.Random(i)).first_name for i in range(40)}
        assert len(names) >= 5, "name pool too narrow or rng not being used"

    def test_different_seeds_produce_variety_of_surnames(self):
        surnames = {sample_identity(random.Random(i)).last_name for i in range(40)}
        assert len(surnames) >= 5

    def test_all_fields_non_empty(self):
        rng = random.Random(3)
        for _ in range(20):
            ident = sample_identity(rng)
            assert ident.first_name.strip()
            assert ident.last_name.strip()
            assert ident.inferred_gender
            assert ident.inferred_race_proxy.strip()
            assert ident.source

    def test_bm_names_are_reachable(self):
        """Bertrand-Mullainathan names must be in the sample pool."""
        bm = load_bm_pairs()
        bm_first = {p["white"] for p in bm["male_pairs"] + bm["female_pairs"]}
        bm_first |= {p["black"] for p in bm["male_pairs"] + bm["female_pairs"]}
        pool = {sample_identity(random.Random(i)).first_name for i in range(200)}
        overlap = bm_first & pool
        assert overlap, "no B-M names appear in sample_identity output after 200 draws"
