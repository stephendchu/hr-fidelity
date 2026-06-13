from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Identity:
    first_name: str
    last_name: str
    inferred_gender: Literal["M", "F"]
    inferred_race_proxy: str
    source: Literal["ssa", "census", "bertrand_mullainathan"]


@dataclass
class Education:
    degree: str
    field: str
    institution: str
    prestige_tier: int  # 1=elite, 2=selective, 3=regional — proxy axis, excluded from content_hash
    grad_year: int
    gpa: float | None = None


@dataclass
class Experience:
    title: str
    company: str
    start: str
    end: str | None
    bullets: list[str] = field(default_factory=list)


@dataclass
class Resume:
    candidate_id: str
    identity: Identity
    education: list[Education]
    experience: list[Experience]
    skills: list[str]
    certifications: list[str]
    latent_fit: Literal["strong", "medium", "weak"]
