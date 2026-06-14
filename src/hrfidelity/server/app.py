"""
FastAPI certification dashboard server (M4).

Endpoints:
  GET  /                    — serve index.html
  GET  /api/reqs            — list available req fixtures
  POST /api/audit           — run bias audit with configurable screener
  GET  /api/fidelity/{req_id} — run fidelity calibration (AI vs recruiter κ)
"""
from __future__ import annotations

import os
import pathlib
import random
from dataclasses import asdict, dataclass
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from hrfidelity.audit.report import AuditReport, run_audit
from hrfidelity.data.corpus_generator import generate_corpus
from hrfidelity.data.req_loader import Req, load_req
from hrfidelity.data.schema import CounterfactualPair, Resume
from hrfidelity.fidelity.calibration import run_calibration, simulate_human_vote
from hrfidelity.fidelity.pairs import generate_fidelity_pairs
from hrfidelity.fidelity.protocol import FidelityReport
from hrfidelity.screener import rubric_screener
from hrfidelity.screener.protocol import ScreenerConfig

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
    resumes, pairs = generate_corpus([req], n_per_fit=50, seed=42)
    return resumes, pairs


# ── Pydantic request/response models ─────────────────────────────────────────

class AuditConfigIn(BaseModel):
    prestige_bonus: float = 0.0
    name_signal: bool = False
    required_skill_weight: float = 0.6
    threshold_advance: float = 0.7


class AuditRequest(BaseModel):
    req_id: str
    config: AuditConfigIn


# ── Serialisation helpers ─────────────────────────────────────────────────────

def _audit_to_dict(report: AuditReport) -> dict:
    ff = report.four_fifths
    dr = report.drift
    return {
        "req_id": report.req_id,
        "verdict": report.verdict,
        "n_resumes": report.n_resumes,
        "n_pairs": report.n_pairs,
        "detail": report.detail,
        "four_fifths": {
            "passed": ff.passed,
            "ratios": ff.ratios,
            "detail": ff.detail,
            "groups": [
                {
                    "group": g.group,
                    "axis": g.axis,
                    "n": g.n,
                    "n_advanced": g.n_advanced,
                    "selection_rate": round(g.selection_rate, 4),
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

        cfg = body.config
        screener_config = ScreenerConfig(
            prestige_bonus=cfg.prestige_bonus,
            race_proxy_bias=_BIAS_RACE if cfg.name_signal else {},
            gender_bias=_BIAS_GENDER if cfg.name_signal else {},
            required_skill_weight=cfg.required_skill_weight,
            nice_to_have_weight=round((1.0 - cfg.required_skill_weight) * 0.5, 4),
            experience_weight=round((1.0 - cfg.required_skill_weight) * 0.5, 4),
            threshold_advance=cfg.threshold_advance,
        )

        resumes, pairs = _corpus_for_req(body.req_id)

        # Score all unique resumes (bases + twins) for drift check
        all_resume_map: dict[str, Resume] = {r.candidate_id: r for r in resumes}
        for p in pairs:
            all_resume_map.setdefault(p.twin.candidate_id, p.twin)
        all_resumes = list(all_resume_map.values())

        scores = [rubric_screener.score(r, req, screener_config) for r in all_resumes]

        report = run_audit(
            scores,
            resumes,
            pairs,
            req.id,
            four_fifths_axes=["race_proxy"],
        )
        return _audit_to_dict(report)

    @app.post("/api/scores")
    async def scores(body: AuditRequest):
        """Return individual candidate scores for pool visualization."""
        req = _req_by_id().get(body.req_id)
        if req is None:
            raise HTTPException(status_code=404, detail=f"req not found: {body.req_id!r}")

        cfg = body.config
        screener_config = ScreenerConfig(
            prestige_bonus=cfg.prestige_bonus,
            race_proxy_bias=_BIAS_RACE if cfg.name_signal else {},
            gender_bias=_BIAS_GENDER if cfg.name_signal else {},
            required_skill_weight=cfg.required_skill_weight,
            nice_to_have_weight=round((1.0 - cfg.required_skill_weight) * 0.5, 4),
            experience_weight=round((1.0 - cfg.required_skill_weight) * 0.5, 4),
            threshold_advance=cfg.threshold_advance,
        )

        resumes, _ = _corpus_for_req(body.req_id)
        scored = [rubric_screener.score(r, req, screener_config) for r in resumes]
        resume_map = {r.candidate_id: r for r in resumes}

        return [
            {
                "candidate_id": s.candidate_id,
                "raw_score": round(s.raw_score, 4),
                "verdict": s.verdict,
                "race_proxy": resume_map[s.candidate_id].identity.inferred_race_proxy,
                "gender": resume_map[s.candidate_id].identity.inferred_gender,
                "latent_fit": resume_map[s.candidate_id].latent_fit,
            }
            for s in scored
            if s.candidate_id in resume_map
        ]

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
