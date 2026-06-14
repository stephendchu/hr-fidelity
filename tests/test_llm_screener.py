"""
M5 — LLM screener tests.

Tests that don't need the API (prompt construction, response parsing) run always.
The integration test requires ANTHROPIC_API_KEY and is skipped without it.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from hrfidelity.data.req_loader import Req, TrueRubric, TrueRubricBand
from hrfidelity.data.schema import Education, Experience, Identity, Resume
from hrfidelity.screener import llm_screener
from hrfidelity.screener.protocol import ScreenerConfig


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def req() -> Req:
    return Req(
        id="backend-eng",
        title="Senior Backend Engineer",
        required_skills=["Python", "SQL", "Docker"],
        nice_to_have=["Kubernetes", "Redis"],
        min_years=4,
        true_rubric=TrueRubric(
            strong=TrueRubricBand(min_years_exp=5, required_skill_coverage=0.85, nice_to_have_count=2),
            medium=TrueRubricBand(min_years_exp=3, required_skill_coverage=0.65, nice_to_have_count=1),
            weak=TrueRubricBand(min_years_exp=1, required_skill_coverage=0.35, nice_to_have_count=0),
        ),
    )


@pytest.fixture
def resume() -> Resume:
    return Resume(
        candidate_id="test-001",
        identity=Identity(
            first_name="Lakisha",
            last_name="Washington",
            inferred_gender="F",
            inferred_race_proxy="black",
            source="bertrand_mullainathan",
        ),
        education=[Education(degree="B.S.", field="Computer Science",
                             institution="State University", prestige_tier=3,
                             grad_year=2018, gpa=3.4)],
        experience=[Experience(title="Software Engineer", company="Acme Corp",
                               start="2018-06-01", end="2024-01-01",
                               bullets=["Built REST APIs", "Maintained SQL databases"])],
        skills=["Python", "SQL", "Docker", "Redis"],
        certifications=["AWS Certified Developer"],
        latent_fit="strong",
    )


# ── Prompt construction (no API needed) ───────────────────────────────────────

def test_prompt_excludes_candidate_name(resume, req):
    prompt = llm_screener._build_prompt(resume, req)
    assert "Lakisha" not in prompt
    assert "Washington" not in prompt


def test_prompt_excludes_demographic_signals(resume, req):
    prompt = llm_screener._build_prompt(resume, req)
    assert "black" not in prompt.lower()
    assert "race" not in prompt.lower()
    assert "gender" not in prompt.lower()
    assert "female" not in prompt.lower()


def test_prompt_includes_required_skills(resume, req):
    prompt = llm_screener._build_prompt(resume, req)
    for skill in req.required_skills:
        assert skill in prompt


def test_prompt_includes_candidate_skills(resume, req):
    prompt = llm_screener._build_prompt(resume, req)
    for skill in resume.skills:
        assert skill in prompt


def test_prompt_includes_experience(resume, req):
    prompt = llm_screener._build_prompt(resume, req)
    assert "Software Engineer" in prompt
    assert "Acme Corp" in prompt


def test_prompt_requests_json(resume, req):
    prompt = llm_screener._build_prompt(resume, req)
    assert "json" in prompt.lower() or "JSON" in prompt


# ── Response parsing (no API needed) ─────────────────────────────────────────

def test_parse_response_valid():
    result = llm_screener._parse_response('{"score": 82, "rationale": "Strong Python and SQL coverage."}')
    assert result["score"] == 82
    assert "Python" in result["rationale"]


def test_parse_response_clamps_above_100():
    result = llm_screener._parse_response('{"score": 110, "rationale": "Exceptional."}')
    assert result["score"] == 100


def test_parse_response_clamps_below_zero():
    result = llm_screener._parse_response('{"score": -5, "rationale": "Poor fit."}')
    assert result["score"] == 0


def test_parse_response_with_markdown_fences():
    text = '```json\n{"score": 75, "rationale": "Decent match."}\n```'
    result = llm_screener._parse_response(text)
    assert result["score"] == 75


def test_parse_response_missing_rationale():
    result = llm_screener._parse_response('{"score": 60}')
    assert result["score"] == 60
    assert "rationale" in result


# ── score() with mocked API ───────────────────────────────────────────────────

def _mock_client(score: int = 78, rationale: str = "Good match."):
    client = MagicMock()
    msg = MagicMock()
    msg.content = [MagicMock(text=f'{{"score": {score}, "rationale": "{rationale}"}}')]
    client.messages.create.return_value = msg
    return client


def test_score_returns_valid_score_object(resume, req):
    with patch.object(llm_screener, "_get_client", return_value=_mock_client(78)):
        s = llm_screener.score(resume, req)
    assert s.candidate_id == resume.candidate_id
    assert s.req_id == req.id
    assert s.verdict in ("advance", "borderline", "reject")
    assert 0.0 <= s.raw_score <= 1.0
    assert s.rationale


def test_score_advance_above_threshold(resume, req):
    with patch.object(llm_screener, "_get_client", return_value=_mock_client(score=85)):
        s = llm_screener.score(resume, req, ScreenerConfig(threshold_advance=0.7))
    assert s.verdict == "advance"
    assert s.raw_score == pytest.approx(0.85)


def test_score_reject_below_threshold(resume, req):
    with patch.object(llm_screener, "_get_client", return_value=_mock_client(score=30)):
        s = llm_screener.score(resume, req, ScreenerConfig(threshold_borderline=0.4))
    assert s.verdict == "reject"


def test_score_api_call_excludes_name(resume, req):
    client = _mock_client()
    with patch.object(llm_screener, "_get_client", return_value=client):
        llm_screener.score(resume, req)
    call_kwargs = client.messages.create.call_args
    prompt_text = str(call_kwargs)
    assert "Lakisha" not in prompt_text
    assert "Washington" not in prompt_text


# ── Integration (real API — skip if no key) ───────────────────────────────────

@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
def test_score_integration(resume, req):
    s = llm_screener.score(resume, req)
    assert s.candidate_id == resume.candidate_id
    assert s.verdict in ("advance", "borderline", "reject")
    assert 0.0 <= s.raw_score <= 1.0
    assert len(s.rationale) > 10
