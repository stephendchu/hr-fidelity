"""
Résumé generator (§3 of data-generator.md).

generate_resume(req, latent_fit, rng) produces one synthetic Resume.

Skills and experience years are conditioned on the rubric band for latent_fit.
Identity is sampled AFTER and INDEPENDENTLY — the rng stream for identity is
never seeded from latent_fit. That independence is what makes any score-
demographic correlation in the screener's output interpretable as bias.

No LLM calls: template-based generation.  An LLM back-end can be swapped in
later without changing the interface or breaking any tests.
"""
import random
import uuid

from hrfidelity.data.name_loader import sample_identity, sample_swappable_identity
from hrfidelity.data.req_loader import Req
from hrfidelity.data.schema import Education, Experience, Resume

# ---------------------------------------------------------------------------
# Lookup tables — kept small for M1; expand when realism demands it
# ---------------------------------------------------------------------------

_TITLES: dict[str, list[str]] = {
    "backend-eng":     ["Software Engineer", "Senior Software Engineer", "Backend Developer", "Staff Engineer"],
    "fang-swe-nyc":   ["Software Engineer II", "Senior Software Engineer", "Staff Software Engineer", "Senior SWE"],
    "ml-researcher":  ["ML Engineer", "Data Scientist", "Research Engineer", "ML Researcher"],
    "infra-eng":      ["DevOps Engineer", "Infrastructure Engineer", "Site Reliability Engineer", "Platform Engineer"],
    "recruiter":      ["Recruiter", "Talent Acquisition Specialist", "Senior Recruiter", "Recruiting Lead"],
}
_TITLES_DEFAULT = ["Analyst", "Specialist", "Associate", "Senior Associate"]

_COMPANIES = [
    "Acme Corp", "Globex", "Initech", "Hooli", "Pied Piper", "Massive Dynamic",
    "Bluth Company", "Stark Industries", "Wayne Enterprises", "Cyberdyne Systems",
    "Tyrell Corporation", "Aperture Science", "Soylent Corp", "InGen", "Rekall",
]

_DEGREES: dict[str, list[tuple[str, str]]] = {
    "backend-eng":   [("BS", "Computer Science"), ("BS", "Software Engineering"), ("BA", "Computer Science")],
    "fang-swe-nyc":  [("BS", "Computer Science"), ("MS", "Computer Science"), ("BS", "Electrical Engineering & CS"), ("MS", "Software Engineering")],
    "ml-researcher": [("MS", "Machine Learning"), ("PhD", "Computer Science"), ("BS", "Statistics"), ("MS", "Statistics")],
    "infra-eng":     [("BS", "Computer Science"), ("BS", "Information Technology"), ("AS", "Network Administration")],
    "recruiter":     [("BS", "Human Resources"), ("BA", "Psychology"), ("BS", "Business Administration"), ("BA", "Communications")],
}
_DEGREES_DEFAULT = [("BS", "General Studies"), ("BA", "Business")]

_INSTITUTIONS: dict[int, list[str]] = {
    1: ["MIT", "Stanford University", "Carnegie Mellon University", "UC Berkeley", "Caltech"],
    2: ["University of Michigan", "Georgia Tech", "Purdue University", "UC San Diego", "University of Washington"],
    3: ["State University", "Regional College", "City University", "Western State University", "Metro College"],
}

# SKILL is the literal placeholder; replaced via str.replace() to avoid .format()
_BULLET_TEMPLATES = [
    "Built and maintained production systems using SKILL",
    "Designed and implemented solutions with SKILL",
    "Led adoption of SKILL, improving team velocity by 20%",
    "Optimized performance using SKILL, reducing latency by 30%",
    "Delivered SKILL-based feature serving thousands of daily users",
    "Architected scalable infrastructure leveraging SKILL",
    "Mentored team members on best practices for SKILL",
    "Reduced on-call burden 40% by improving observability with SKILL",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_bullet(skill: str, rng: random.Random) -> str:
    return rng.choice(_BULLET_TEMPLATES).replace("SKILL", skill)


def _generate_experience(
    req_id: str,
    min_years: int,
    skills: list[str],
    rng: random.Random,
    current_year: int = 2024,
) -> list[Experience]:
    if min_years == 0:
        return []

    titles = _TITLES.get(req_id, _TITLES_DEFAULT)
    experiences: list[Experience] = []
    remaining = min_years
    end_year = current_year
    end_month = rng.randint(1, 12)

    while remaining > 0:
        duration = min(rng.randint(1, 3), remaining)
        start_year = end_year - duration
        start_month = rng.randint(1, 12)

        bullet_skills = rng.sample(skills, min(3, len(skills))) if skills else ["cross-functional collaboration"]
        n_bullets = rng.randint(2, min(3, len(bullet_skills)))
        bullets = [_make_bullet(s, rng) for s in bullet_skills[:n_bullets]]

        experiences.append(Experience(
            title=rng.choice(titles),
            company=rng.choice(_COMPANIES),
            start=f"{start_year}-{start_month:02d}",
            end=f"{end_year}-{end_month:02d}",
            bullets=bullets,
        ))

        end_year = start_year
        end_month = start_month
        remaining -= duration

    return experiences


def _generate_education(
    req_id: str,
    min_years_exp: int,
    rng: random.Random,
) -> list[Education]:
    prestige_tier = rng.choices([1, 2, 3], weights=[0.15, 0.40, 0.45])[0]
    degree, field = rng.choice(_DEGREES.get(req_id, _DEGREES_DEFAULT))
    institution = rng.choice(_INSTITUTIONS[prestige_tier])
    grad_year = max(1990, 2024 - min_years_exp - rng.randint(0, 3) - 4)
    return [Education(
        degree=degree,
        field=field,
        institution=institution,
        prestige_tier=prestige_tier,
        grad_year=grad_year,
    )]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_resume(
    req: Req,
    latent_fit: str,
    rng: random.Random,
    *,
    swappable_identity: bool = False,
    identity_rng: random.Random | None = None,
) -> Resume:
    """Return one synthetic Resume conditioned on (req, latent_fit).

    Content (skills, experience, education) is drawn from `rng`.
    Identity is drawn from `identity_rng` when provided, otherwise from `rng`.
    Callers that care about eeo_race being independent of latent_fit should
    pass a separately-seeded identity_rng (see corpus_generator.generate_corpus).
    """
    band = getattr(req.true_rubric, latent_fit)

    n_req = max(1, round(band.required_skill_coverage * len(req.required_skills)))
    required = rng.sample(req.required_skills, min(n_req, len(req.required_skills)))
    n_nice = min(band.nice_to_have_count, len(req.nice_to_have))
    nice = rng.sample(req.nice_to_have, n_nice) if n_nice > 0 else []
    skills = required + nice

    experience = _generate_experience(req.id, band.min_years_exp, skills, rng)
    education = _generate_education(req.id, band.min_years_exp, rng)

    _id_rng = identity_rng if identity_rng is not None else rng
    _sample = sample_swappable_identity if swappable_identity else sample_identity
    identity = _sample(_id_rng)

    return Resume(
        candidate_id=str(uuid.UUID(int=rng.getrandbits(128))),
        identity=identity,
        education=education,
        experience=experience,
        skills=skills,
        certifications=[],
        latent_fit=latent_fit,
    )
