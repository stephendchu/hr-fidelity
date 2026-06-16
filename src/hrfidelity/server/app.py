"""
FastAPI certification dashboard server (M4).

Endpoints:
  GET  /                    — serve index.html
  GET  /api/reqs            — list available req fixtures
  POST /api/audit           — run bias audit with configurable screener
  GET  /api/fidelity/{req_id} — run fidelity calibration (AI vs recruiter κ)
"""
from __future__ import annotations

import json
import os
import pathlib
import random
from dataclasses import asdict, dataclass
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from hrfidelity.audit.report import AuditReport, run_audit
from hrfidelity.data.corpus_generator import generate_corpus
from hrfidelity.data.req_loader import Req, load_req
from hrfidelity.data.schema import CounterfactualPair, Resume
from hrfidelity.fidelity.calibration import run_calibration, simulate_human_vote
from hrfidelity.fidelity.pairs import generate_fidelity_pairs
from hrfidelity.fidelity.protocol import FidelityReport
from hrfidelity.screener import llm_screener, rubric_screener
from hrfidelity.screener.protocol import Score, ScreenerConfig

_DATA_ROOT = pathlib.Path(__file__).parents[3] / "data"
_STATIC_DIR = pathlib.Path(__file__).parent / "static"

_BIAS_RACE = {"white": 0.15, "black": -0.20, "hispanic": -0.10, "asian": 0.05}
_BIAS_GENDER = {"M": 0.10, "F": -0.10}


# ── Data loading (cached at module level) ─────────────────────────────────────

@lru_cache(maxsize=None)
def _load_reqs() -> list[Req]:
    return [load_req(p) for p in sorted((_DATA_ROOT / "reqs").glob("*.json"))]


@lru_cache(maxsize=None)
def _req_by_id() -> dict[str, Req]:
    return {r.id: r for r in _load_reqs()}


@lru_cache(maxsize=None)
def _corpus_for_req(req_id: str) -> tuple[list[Resume], list[CounterfactualPair]]:
    req = _req_by_id().get(req_id)
    if req is None:
        raise KeyError(req_id)
    resumes, pairs = generate_corpus([req], n_per_fit=50, seed=44)
    return resumes, pairs


_LLM_SCORES_DIR = _DATA_ROOT / "llm_scores"


def _llm_fixture_path(req_id: str) -> pathlib.Path:
    return _LLM_SCORES_DIR / f"{req_id}.json"


@lru_cache(maxsize=None)
def _llm_raw_scores(req_id: str) -> dict[str, float]:
    """Raw LLM scores for every corpus resume (bases + twins), by candidate_id.

    Served from a committed fixture when present (data/llm_scores/<req_id>.json).
    The prompt is blind to identity, so a candidate's raw score is a stable
    constant — captured once, this lets the public demo show real Claude Haiku
    output at $0 with no API key on the server.

    When the fixture is missing (local dev, or a new req) it falls back to live
    API calls if a key is available; regenerate fixtures with
    scripts/gen_llm_scores.py. The prompt being blind, raw scores never depend on
    the bias knobs — only the verdict threshold does.
    """
    fixture = _llm_fixture_path(req_id)
    if fixture.exists():
        return {cid: float(v) for cid, v in json.loads(fixture.read_text()).items()}

    req = _req_by_id()[req_id]
    resumes, pairs = _corpus_for_req(req_id)
    all_map: dict[str, Resume] = {r.candidate_id: r for r in resumes}
    for p in pairs:
        all_map.setdefault(p.twin.candidate_id, p.twin)
    return {cid: llm_screener.score(r, req).raw_score for cid, r in all_map.items()}


def _make_scorer(req: Req, cfg: AuditConfigIn):
    """Return a callable score(resume) -> Score for the selected screener."""
    if cfg.screener == "llm":
        if not _llm_fixture_path(req.id).exists() and not os.environ.get("ANTHROPIC_API_KEY"):
            raise HTTPException(
                status_code=503,
                detail=(
                    "LLM screener needs a cached score fixture or ANTHROPIC_API_KEY; "
                    "neither is available on this server."
                ),
            )
        raw = _llm_raw_scores(req.id)
        t_adv, t_bord = cfg.threshold_advance, 0.4

        def llm_score(r: Resume) -> Score:
            rs = raw.get(r.candidate_id, 0.0)
            v = "advance" if rs >= t_adv else "borderline" if rs >= t_bord else "reject"
            return Score(candidate_id=r.candidate_id, req_id=req.id,
                         verdict=v, rationale="", raw_score=rs)
        return llm_score

    sc = ScreenerConfig(
        prestige_bonus=cfg.prestige_bonus,
        race_proxy_bias=_BIAS_RACE if cfg.name_signal else {},
        gender_bias=_BIAS_GENDER if cfg.name_signal else {},
        required_skill_weight=cfg.required_skill_weight,
        nice_to_have_weight=round((1.0 - cfg.required_skill_weight) * 0.5, 4),
        experience_weight=round((1.0 - cfg.required_skill_weight) * 0.5, 4),
        threshold_advance=cfg.threshold_advance,
    )
    return lambda r: rubric_screener.score(r, req, sc)


# ── Pydantic request/response models ─────────────────────────────────────────

class AuditConfigIn(BaseModel):
    prestige_bonus: float = 0.0
    name_signal: bool = False
    required_skill_weight: float = 0.6
    threshold_advance: float = 0.7
    screener: str = "rubric"  # "rubric" | "llm"


class AuditRequest(BaseModel):
    req_id: str
    config: AuditConfigIn


# ── Serialisation helpers ─────────────────────────────────────────────────────

def _audit_to_dict(report: AuditReport, cfg: AuditConfigIn | None = None) -> dict:
    ff = report.four_fifths
    dr = report.drift
    nice_w = round((1.0 - cfg.required_skill_weight) * 0.5, 4) if cfg else None
    exp_w  = round((1.0 - cfg.required_skill_weight) * 0.5, 4) if cfg else None
    screener_config = {
        "required_skill_weight": cfg.required_skill_weight if cfg else None,
        "nice_to_have_weight":   nice_w,
        "experience_weight":     exp_w,
        "prestige_bonus":        cfg.prestige_bonus if cfg else None,
        "threshold_advance":     cfg.threshold_advance if cfg else None,
        "name_signal":           cfg.name_signal if cfg else None,
    } if cfg else None
    return {
        "req_id": report.req_id,
        "verdict": report.verdict,
        "n_resumes": report.n_resumes,
        "n_pairs": report.n_pairs,
        "detail": report.detail,
        "screener_config": screener_config,
        "four_fifths": {
            "passed": ff.passed,
            "ratios": ff.ratios,
            "stat_min_n": ff.stat_min_n,
            "detail": ff.detail,
            "groups": [
                {
                    "group": g.group,
                    "axis": g.axis,
                    "n": g.n,
                    "n_advanced": g.n_advanced,
                    "selection_rate": round(g.selection_rate, 4),
                    "in_verdict": g.n >= ff.stat_min_n,
                }
                for g in ff.groups
            ],
        },
        "drift": {
            "passed": dr.passed,
            "drift_threshold": dr.drift_threshold,
            "detail": dr.detail,
            "axis_results": {
                axis: {
                    "mean_drift": round(r["mean_drift"], 4),
                    "max_drift": round(r["max_drift"], 4),
                    "flip_rate": round(r["flip_rate"], 4),
                    "n_pairs": r["n_pairs"],
                }
                for axis, r in dr.axis_results.items()
            },
        },
    }


def _fidelity_to_dict(report: FidelityReport) -> dict:
    return {
        "req_id": report.req_id,
        "passes": report.passes,
        "mean_kappa_ai_human": round(report.mean_kappa_ai_human, 4),
        "kappa_human_human": round(report.kappa_human_human, 4),
        "n_pairs": report.n_pairs,
        "n_gold": report.n_gold,
        "gold_pass_rate": round(report.gold_pass_rate, 4),
        "kappa_ai_human": [
            {
                "judge_a": kr.judge_a,
                "judge_b": kr.judge_b,
                "kappa": round(kr.kappa, 4),
                "n_pairs": kr.n_pairs,
                "observed_agreement": round(kr.observed_agreement, 4),
                "expected_agreement": round(kr.expected_agreement, 4),
            }
            for kr in report.kappa_ai_human
        ],
    }


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(title="HR Fidelity", version="0.1.0")
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def root():
        index_name = os.environ.get("HRFIDELITY_INDEX", "index.html")
        html_path = (_STATIC_DIR / index_name).resolve()
        if not str(html_path).startswith(str(_STATIC_DIR.resolve())):
            raise HTTPException(status_code=403)
        if not html_path.exists():
            raise HTTPException(status_code=404)
        return HTMLResponse(content=html_path.read_text(), status_code=200)

    @app.get("/preview/{name}", response_class=HTMLResponse)
    async def preview(name: str):
        # Serve design previews from static/designs/
        if not name.replace("-", "").replace("_", "").isalnum():
            raise HTTPException(status_code=400)
        html_path = _STATIC_DIR / "designs" / f"{name}.html"
        if not html_path.exists():
            raise HTTPException(status_code=404)
        return HTMLResponse(content=html_path.read_text(), status_code=200)

    @app.get("/api/reqs")
    async def list_reqs():
        return [{"id": r.id, "title": r.title} for r in _load_reqs()]

    @app.post("/api/audit")
    async def audit(body: AuditRequest):
        req = _req_by_id().get(body.req_id)
        if req is None:
            raise HTTPException(status_code=404, detail=f"req not found: {body.req_id!r}")

        scorer = _make_scorer(req, body.config)

        resumes, pairs = _corpus_for_req(body.req_id)

        # Score all unique resumes (bases + twins) for drift check
        all_resume_map: dict[str, Resume] = {r.candidate_id: r for r in resumes}
        for p in pairs:
            all_resume_map.setdefault(p.twin.candidate_id, p.twin)
        all_resumes = list(all_resume_map.values())

        scores = [scorer(r) for r in all_resumes]

        report = run_audit(
            scores,
            resumes,
            pairs,
            req.id,
            four_fifths_axes=["eeo_race"],
            min_group_size=20,
        )
        return _audit_to_dict(report, body.config)

    @app.post("/api/scores")
    async def scores(body: AuditRequest):
        """Return individual candidate scores for pool visualization."""
        req = _req_by_id().get(body.req_id)
        if req is None:
            raise HTTPException(status_code=404, detail=f"req not found: {body.req_id!r}")

        scorer = _make_scorer(req, body.config)

        resumes, _ = _corpus_for_req(body.req_id)
        scored = [scorer(r) for r in resumes]
        resume_map = {r.candidate_id: r for r in resumes}

        return [
            {
                "candidate_id": s.candidate_id,
                "raw_score": round(s.raw_score, 4),
                "verdict": s.verdict,
                "eeo_race": resume_map[s.candidate_id].identity.eeo_race
                    or resume_map[s.candidate_id].identity.inferred_race_proxy,
                "race_proxy": resume_map[s.candidate_id].identity.inferred_race_proxy,
                "gender": resume_map[s.candidate_id].identity.inferred_gender,
                "latent_fit": resume_map[s.candidate_id].latent_fit,
            }
            for s in scored
            if s.candidate_id in resume_map
        ]

    @app.post("/api/pairs")
    async def pairs(body: AuditRequest):
        """Return the top counterfactual pairs by score divergence for the current config."""
        req = _req_by_id().get(body.req_id)
        if req is None:
            raise HTTPException(status_code=404, detail=f"req not found: {body.req_id!r}")

        scorer = _make_scorer(req, body.config)

        _, cf_pairs = _corpus_for_req(body.req_id)

        AXIS_LABELS = {
            "gender":        "Gender signal (name swap)",
            "race_proxy":    "Race proxy (name swap)",
            "prestige_tier": "Prestige tier (institution swap)",
        }
        AXIS_ORDER = ["race_proxy", "gender", "prestige_tier"]

        results = []
        for p in cf_pairs:
            base_score = scorer(p.base)
            twin_score = scorer(p.twin)
            delta = twin_score.raw_score - base_score.raw_score
            results.append({
                "axis":        p.axis,
                "axis_label":  AXIS_LABELS.get(p.axis, p.axis),
                "delta":       round(delta, 4),
                "abs_delta":   round(abs(delta), 4),
                "base": {
                    "name":         f"{p.base.identity.first_name} {p.base.identity.last_name}",
                    "race_proxy":   p.base.identity.inferred_race_proxy,
                    "gender":       p.base.identity.inferred_gender,
                    "institution":  p.base.education[0].institution if p.base.education else "",
                    "prestige_tier": p.base.education[0].prestige_tier if p.base.education else 0,
                    "skills":       p.base.skills[:6],
                    "raw_score":    round(base_score.raw_score, 4),
                    "verdict":      base_score.verdict,
                },
                "twin": {
                    "name":         f"{p.twin.identity.first_name} {p.twin.identity.last_name}",
                    "race_proxy":   p.twin.identity.inferred_race_proxy,
                    "gender":       p.twin.identity.inferred_gender,
                    "institution":  p.twin.education[0].institution if p.twin.education else "",
                    "prestige_tier": p.twin.education[0].prestige_tier if p.twin.education else 0,
                    "skills":       p.twin.skills[:6],
                    "raw_score":    round(twin_score.raw_score, 4),
                    "verdict":      twin_score.verdict,
                },
            })

        # Sort: protected axes first (race_proxy, gender), then prestige; within each by |delta| desc
        results.sort(key=lambda r: (AXIS_ORDER.index(r["axis"]) if r["axis"] in AXIS_ORDER else 99, -r["abs_delta"]))

        # Return top 2 per axis (up to 6 total)
        seen: dict[str, int] = {}
        top: list[dict] = []
        for r in results:
            if seen.get(r["axis"], 0) < 2:
                top.append(r)
                seen[r["axis"]] = seen.get(r["axis"], 0) + 1
            if len(top) >= 6:
                break

        return top

    @app.get("/api/fidelity/{req_id}")
    async def fidelity(req_id: str):
        req = _req_by_id().get(req_id)
        if req is None:
            raise HTTPException(status_code=404, detail=f"req not found: {req_id!r}")

        resumes, _ = _corpus_for_req(req_id)
        pairs = generate_fidelity_pairs(resumes, req, n_pairs=20, seed=42)

        rng = random.Random(99)
        human_votes_by_judge = {
            f"recruiter_{i}": [
                simulate_human_vote(p, req, f"recruiter_{i}", agreement_rate=0.80, rng=rng)
                for p in pairs
            ]
            for i in range(3)
        }

        report = run_calibration(pairs, req, human_votes_by_judge)
        return _fidelity_to_dict(report)

    return app
