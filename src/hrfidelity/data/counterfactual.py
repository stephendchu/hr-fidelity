"""
Counterfactual pair generator.

generate_counterfactual(base, axis=...) produces a twin that is identical to
*base* in every job-relevant field and differs only in the named proxy.

The content_hash enforces this mechanically: it covers skills, experience, and
education content, but excludes identity, prestige_tier, and latent_fit.
Any pair whose content_hash differs would have failed validation — but since
generate_counterfactual only touches proxy fields, the invariant holds by
construction. The tests confirm it.
"""
import copy
import hashlib
import json
import uuid

from hrfidelity.data.schema import Education, Resume

# ---------------------------------------------------------------------------
# Name swap tables — drawn from Bertrand-Mullainathan (2004) and SSA data.
# Keys are names we generate in the base population; values are their twins.
# ---------------------------------------------------------------------------

_GENDER_PAIRS: list[tuple[str, str]] = [
    ("James", "Jennifer"),
    ("John", "Jane"),
    ("Michael", "Michelle"),
    ("Robert", "Rebecca"),
    ("David", "Deborah"),
    ("William", "Whitney"),
    ("Richard", "Rachel"),
    ("Joseph", "Josephine"),
    ("Thomas", "Tamika"),
    ("Christopher", "Christina"),
    ("Kevin", "Karen"),
    ("Brian", "Brittany"),
]
_MALE_TO_FEMALE: dict[str, str] = {m: f for m, f in _GENDER_PAIRS}
_FEMALE_TO_MALE: dict[str, str] = {f: m for m, f in _GENDER_PAIRS}

# Surname pairs — Bertrand-Mullainathan "distinctively white" vs "distinctively Black".
_RACE_SURNAME_PAIRS: list[tuple[str, str]] = [
    ("Walsh", "Washington"),
    ("Sullivan", "Jefferson"),
    ("McCarthy", "Jackson"),
    ("Murray", "Williams"),
    ("Baker", "Robinson"),
    ("Kelly", "Johnson"),
    ("Quinn", "Davis"),
]
_WHITE_TO_BLACK: dict[str, str] = {w: b for w, b in _RACE_SURNAME_PAIRS}
_BLACK_TO_WHITE: dict[str, str] = {b: w for w, b in _RACE_SURNAME_PAIRS}

# prestige_tier flip: keep it simple — elite↔regional, selective↔regional.
_PRESTIGE_FLIP: dict[int, int] = {1: 3, 2: 3, 3: 1}


# ---------------------------------------------------------------------------
# content_hash
# ---------------------------------------------------------------------------

def content_hash(resume: Resume) -> str:
    """SHA-256 of job-relevant fields only.

    Excluded (proxy / label fields):
      identity.*, candidate_id, prestige_tier, latent_fit

    Included (must be identical across twins):
      skills, certifications, experience (all subfields), education
      (degree, field, institution, grad_year, gpa)
    """
    payload = {
        "skills": sorted(resume.skills),
        "certifications": sorted(resume.certifications),
        "experience": [
            {
                "title": e.title,
                "company": e.company,
                "start": e.start,
                "end": e.end,
                "bullets": sorted(e.bullets),
            }
            for e in sorted(resume.experience, key=lambda e: e.start)
        ],
        "education": [
            {
                "degree": ed.degree,
                "field": ed.field,
                "institution": ed.institution,
                "grad_year": ed.grad_year,
                "gpa": ed.gpa,
            }
            for ed in sorted(resume.education, key=lambda ed: ed.grad_year)
        ],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


# ---------------------------------------------------------------------------
# generate_counterfactual
# ---------------------------------------------------------------------------

def generate_counterfactual(base: Resume, *, axis: str) -> Resume:
    """Return a deep copy of *base* with exactly one proxy field changed.

    Axes:
      "gender"       — swap first_name and inferred_gender
      "race_proxy"   — swap last_name and inferred_race_proxy
      "prestige_tier"— flip the prestige_tier of the first education entry

    Raises ValueError if the base name is not in the built-in swap table
    (add new pairs to _GENDER_PAIRS / _RACE_SURNAME_PAIRS as needed).
    """
    twin = copy.deepcopy(base)
    twin.candidate_id = str(uuid.uuid4())

    if axis == "gender":
        fn = base.identity.first_name
        if fn in _MALE_TO_FEMALE:
            twin.identity.first_name = _MALE_TO_FEMALE[fn]
            twin.identity.inferred_gender = "F"
        elif fn in _FEMALE_TO_MALE:
            twin.identity.first_name = _FEMALE_TO_MALE[fn]
            twin.identity.inferred_gender = "M"
        else:
            raise ValueError(
                f"first_name {fn!r} not in gender swap table; "
                "add it to _GENDER_PAIRS in counterfactual.py"
            )

    elif axis == "race_proxy":
        ln = base.identity.last_name
        if ln in _WHITE_TO_BLACK:
            twin.identity.last_name = _WHITE_TO_BLACK[ln]
            twin.identity.inferred_race_proxy = "black"
        elif ln in _BLACK_TO_WHITE:
            twin.identity.last_name = _BLACK_TO_WHITE[ln]
            twin.identity.inferred_race_proxy = "white"
        else:
            raise ValueError(
                f"last_name {ln!r} not in race-proxy swap table; "
                "add it to _RACE_SURNAME_PAIRS in counterfactual.py"
            )

    elif axis == "prestige_tier":
        if not twin.education:
            raise ValueError(
                "Cannot apply prestige_tier axis: resume has no education entries"
            )
        ed = twin.education[0]
        twin.education[0] = Education(
            degree=ed.degree,
            field=ed.field,
            institution=ed.institution,
            prestige_tier=_PRESTIGE_FLIP.get(ed.prestige_tier, 3),
            grad_year=ed.grad_year,
            gpa=ed.gpa,
        )

    else:
        raise ValueError(f"Unknown axis: {axis!r}; valid: gender, race_proxy, prestige_tier")

    return twin
