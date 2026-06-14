"""
LLM screener (M5) — Claude Haiku behind the same Score interface as the rubric screener.

Prompt is deliberately blind: no candidate name, no demographic signals.
Only job-relevant content is presented: skills, experience, education field/degree/year.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from hrfidelity.data.req_loader import Req
from hrfidelity.data.schema import Resume
from hrfidelity.screener.protocol import Score, ScreenerConfig, ScreenerVerdict

_FAIR = ScreenerConfig()
_MODEL = "claude-haiku-4-5-20251001"


def _get_client():
    try:
        import anthropic
    except ImportError as e:
        raise ImportError("anthropic package required for LLM screener: pip install anthropic") from e
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def _build_prompt(resume: Resume, req: Req) -> str:
    exp_lines = "\n".join(
        f"- {e.title} at {e.company} ({e.start[:4]}–{e.end[:4] if e.end else 'present'}): "
        + "; ".join(e.bullets)
        for e in resume.experience
    )
    edu_lines = "\n".join(
        f"- {e.degree} in {e.field}, {e.institution} ({e.grad_year})"
        for e in resume.education
    )
    skills_str = ", ".join(resume.skills)
    certs_str = ", ".join(resume.certifications) if resume.certifications else "None"

    required_str = ", ".join(req.required_skills)
    nice_str = ", ".join(req.nice_to_have) if req.nice_to_have else "None"

    return f"""You are a resume screener evaluating job-relevant qualifications only.

JOB REQUISITION: {req.title}
Required skills: {required_str}
Nice-to-have skills: {nice_str}
Minimum years experience: {req.min_years}

CANDIDATE PROFILE (anonymized — no name or demographic information):
Skills: {skills_str}
Certifications: {certs_str}
Education:
{edu_lines}
Experience:
{exp_lines}

Score this candidate 0–100 based solely on job-relevant qualifications.
Do not infer or consider demographic signals.

Respond with valid JSON only, no other text:
{{"score": <integer 0-100>, "rationale": "<one sentence explaining the score>"}}"""


def _parse_response(text: str) -> dict[str, Any]:
    # Strip markdown fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())

    data = json.loads(text)
    data["score"] = max(0, min(100, int(data["score"])))
    if "rationale" not in data:
        data["rationale"] = ""
    return data


def score(resume: Resume, req: Req, config: ScreenerConfig = _FAIR) -> Score:
    client = _get_client()
    prompt = _build_prompt(resume, req)

    msg = client.messages.create(
        model=_MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text
    parsed = _parse_response(raw)

    raw_score = parsed["score"] / 100.0
    if raw_score >= config.threshold_advance:
        verdict: ScreenerVerdict = "advance"
    elif raw_score >= config.threshold_borderline:
        verdict = "borderline"
    else:
        verdict = "reject"

    return Score(
        candidate_id=resume.candidate_id,
        req_id=req.id,
        verdict=verdict,
        rationale=parsed["rationale"],
        raw_score=raw_score,
    )
