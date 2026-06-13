"""
Req fixtures: typed schema + loader.

TrueRubric encodes what makes a candidate strong/medium/weak for a specific
role. It is fixed and job-derived — never learned from past hires (the Amazon
rule). The résumé generator is conditioned on (req, _latent_fit) using these
criteria to set skills coverage and years of experience.
"""
import json
import pathlib
from dataclasses import dataclass


@dataclass
class TrueRubricBand:
    min_years_exp: int
    required_skill_coverage: float  # fraction of required_skills the candidate holds
    nice_to_have_count: int         # number of nice_to_have skills


@dataclass
class TrueRubric:
    strong: TrueRubricBand
    medium: TrueRubricBand
    weak: TrueRubricBand


@dataclass
class Req:
    id: str
    title: str
    required_skills: list[str]
    nice_to_have: list[str]
    min_years: int
    true_rubric: TrueRubric


def _band(d: dict) -> TrueRubricBand:
    return TrueRubricBand(
        min_years_exp=d["min_years_exp"],
        required_skill_coverage=d["required_skill_coverage"],
        nice_to_have_count=d["nice_to_have_count"],
    )


def load_req(path: str | pathlib.Path) -> Req:
    d = json.loads(pathlib.Path(path).read_text())
    rubric = d["true_rubric"]
    return Req(
        id=d["id"],
        title=d["title"],
        required_skills=d["required_skills"],
        nice_to_have=d["nice_to_have"],
        min_years=d["min_years"],
        true_rubric=TrueRubric(
            strong=_band(rubric["strong"]),
            medium=_band(rubric["medium"]),
            weak=_band(rubric["weak"]),
        ),
    )
