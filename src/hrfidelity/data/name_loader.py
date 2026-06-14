"""
Name / demographic loaders — §4 of data-generator.md.

Three vendored samples feed the identity axis:
  data/names/ssa_first_names.json          → first name + inferred_gender
  data/names/census_surnames.json          → surname + inferred_race_proxy
  data/names/bertrand_mullainathan_pairs.json → canonical audit-study pairs

sample_identity() is the single call-site for résumé generation. The caller
is responsible for independence: never pass _latent_fit into this function.
"""
import json
import pathlib
import random

from hrfidelity.data.schema import Identity

_NAMES_DIR = pathlib.Path(__file__).parents[3] / "data" / "names"


def load_ssa_first_names() -> dict:
    return json.loads((_NAMES_DIR / "ssa_first_names.json").read_text())


def load_census_surnames() -> list[dict]:
    return json.loads((_NAMES_DIR / "census_surnames.json").read_text())["surnames"]


def load_bm_pairs() -> dict:
    return json.loads((_NAMES_DIR / "bertrand_mullainathan_pairs.json").read_text())


# ---------------------------------------------------------------------------
# Build lookup structures once at import time
# ---------------------------------------------------------------------------

def _weighted_choices(entries: list[dict]) -> tuple[list[str], list[float]]:
    names = [e["name"] for e in entries]
    weights = [float(e["weight"]) for e in entries]
    return names, weights


_ssa = load_ssa_first_names()
_MALE_NAMES, _MALE_WEIGHTS = _weighted_choices(_ssa["male"])
_FEMALE_NAMES, _FEMALE_WEIGHTS = _weighted_choices(_ssa["female"])

_SURNAMES: list[dict] = load_census_surnames()
_SURNAME_NAMES = [s["name"] for s in _SURNAMES]
_SURNAME_RACES = {s["name"]: s["modal_race"] for s in _SURNAMES}

_bm = load_bm_pairs()
_BM_FIRST_NAMES: set[str] = {
    p[side]
    for pairs in (_bm["male_pairs"], _bm["female_pairs"])
    for p in pairs
    for side in ("white", "black")
}

# EEO race given name-inferred proxy.  Calibrated to reflect a FANG-tier tech
# company in New York (public EEO-1 disclosures: Google, Meta, Amazon ~2023):
#   Asian ~40%, White ~35%, Hispanic/Latino ~12%, Black ~8%, other ~5%.
#
# Key modelling insight: many Asian engineers in the US have Western first names
# (James, David, Kevin) that appear as "white" proxy in our B-M name dataset,
# but they self-report Asian on the EEO form — hence the 0.45 weight on asian
# from a "white" proxy.  Black is intentionally kept at realistic FANG levels
# (~8%).  With n ≈ 12 candidates the group is below the statistical minimum
# for four-fifths analysis, so bias detection falls to the counterfactual drift
# check — which is the real-world situation at most large tech companies.
_EEO_RACE_GIVEN_PROXY: dict[str, tuple[list[str], list[float]]] = {
    "white":    (["white", "black", "hispanic", "asian"], [0.43, 0.02, 0.10, 0.45]),
    "black":    (["white", "black", "hispanic", "asian"], [0.15, 0.22, 0.25, 0.28]),
    "hispanic": (["white", "black", "hispanic", "asian"], [0.12, 0.08, 0.70, 0.10]),
    "asian":    (["white", "black", "hispanic", "asian"], [0.10, 0.05, 0.08, 0.77]),
    "other":    (["white", "black", "hispanic", "asian"], [0.25, 0.25, 0.25, 0.25]),
}


def _sample_eeo_race(race_proxy: str, rng: random.Random) -> str:
    groups, weights = _EEO_RACE_GIVEN_PROXY.get(
        race_proxy, _EEO_RACE_GIVEN_PROXY["other"]
    )
    return rng.choices(groups, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sample_identity(rng: random.Random) -> Identity:
    """Return a randomly sampled Identity.

    Gender is drawn 50/50. First name is weighted by SSA frequency.
    Surname is drawn uniformly from the Census sample.
    source = 'bertrand_mullainathan' when the first name appears in the B-M
    pairs list; 'ssa' otherwise.
    """
    gender: str = rng.choice(["M", "F"])

    if gender == "M":
        first_name = rng.choices(_MALE_NAMES, weights=_MALE_WEIGHTS, k=1)[0]
    else:
        first_name = rng.choices(_FEMALE_NAMES, weights=_FEMALE_WEIGHTS, k=1)[0]

    surname_entry = rng.choice(_SURNAMES)
    last_name = surname_entry["name"]
    race_proxy = surname_entry["modal_race"]

    source = "bertrand_mullainathan" if first_name in _BM_FIRST_NAMES else "ssa"

    return Identity(
        first_name=first_name,
        last_name=last_name,
        inferred_gender=gender,
        inferred_race_proxy=race_proxy,
        source=source,
        eeo_race=_sample_eeo_race(race_proxy, rng),
    )


def sample_swappable_identity(rng: random.Random) -> Identity:
    """Like sample_identity but restricted to names that exist in the
    counterfactual swap table — guarantees generate_counterfactual never
    raises ValueError on gender or race_proxy axes.

    Used by the corpus generator so every résumé in the dataset has a
    valid counterfactual twin on all three axes.
    """
    from hrfidelity.data.counterfactual import (
        SWAPPABLE_FEMALE_NAMES,
        SWAPPABLE_MALE_NAMES,
        SWAPPABLE_SURNAMES,
    )

    gender: str = rng.choice(["M", "F"])
    first_name = rng.choice(SWAPPABLE_MALE_NAMES if gender == "M" else SWAPPABLE_FEMALE_NAMES)

    last_name = rng.choice(SWAPPABLE_SURNAMES)
    race_proxy = _SURNAME_RACES.get(last_name, "other")

    source = "bertrand_mullainathan" if first_name in _BM_FIRST_NAMES else "ssa"

    return Identity(
        first_name=first_name,
        last_name=last_name,
        inferred_gender=gender,
        inferred_race_proxy=race_proxy,
        source=source,
        eeo_race=_sample_eeo_race(race_proxy, rng),
    )
