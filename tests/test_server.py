"""
API tests for the FastAPI certification dashboard (M4).

Tests the HTTP contract before the server exists (RED phase).
"""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def app():
    from hrfidelity.server.app import create_app
    return create_app()


@pytest.fixture
async def client(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


# ── / → HTML landing page ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_root_returns_html(client):
    r = await client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "<html" in r.text.lower()


# ── /api/reqs ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_reqs_returns_list(client):
    r = await client.get("/api/reqs")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    assert len(body) >= 1


@pytest.mark.asyncio
async def test_list_reqs_shape(client):
    r = await client.get("/api/reqs")
    first = r.json()[0]
    assert "id" in first
    assert "title" in first


# ── /api/audit ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_audit_returns_200_fair(client):
    reqs = (await client.get("/api/reqs")).json()
    req_id = reqs[0]["id"]
    r = await client.post("/api/audit", json={
        "req_id": req_id,
        "config": {
            "prestige_bonus": 0.0,
            "name_signal": False,
            "required_skill_weight": 0.6,
            "threshold_advance": 0.7,
        },
    })
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_audit_fair_is_certified(client):
    reqs = (await client.get("/api/reqs")).json()
    req_id = reqs[0]["id"]
    r = await client.post("/api/audit", json={
        "req_id": req_id,
        "config": {
            "prestige_bonus": 0.0,
            "name_signal": False,
            "required_skill_weight": 0.6,
            "threshold_advance": 0.7,
        },
    })
    body = r.json()
    assert body["verdict"] == "CERTIFIED"


@pytest.mark.asyncio
async def test_audit_biased_is_blocked(client):
    reqs = (await client.get("/api/reqs")).json()
    req_id = reqs[0]["id"]
    r = await client.post("/api/audit", json={
        "req_id": req_id,
        "config": {
            "prestige_bonus": 0.25,
            "name_signal": True,
            "required_skill_weight": 0.6,
            "threshold_advance": 0.7,
        },
    })
    body = r.json()
    assert body["verdict"] == "BLOCKED"


@pytest.mark.asyncio
async def test_audit_response_shape(client):
    reqs = (await client.get("/api/reqs")).json()
    req_id = reqs[0]["id"]
    r = await client.post("/api/audit", json={
        "req_id": req_id,
        "config": {"prestige_bonus": 0.0, "name_signal": False,
                   "required_skill_weight": 0.6, "threshold_advance": 0.7},
    })
    body = r.json()
    assert "verdict" in body
    assert "four_fifths" in body
    assert "drift" in body
    assert "n_resumes" in body
    assert "n_pairs" in body


@pytest.mark.asyncio
async def test_audit_four_fifths_shape(client):
    reqs = (await client.get("/api/reqs")).json()
    req_id = reqs[0]["id"]
    r = await client.post("/api/audit", json={
        "req_id": req_id,
        "config": {"prestige_bonus": 0.0, "name_signal": False,
                   "required_skill_weight": 0.6, "threshold_advance": 0.7},
    })
    ff = r.json()["four_fifths"]
    assert "passed" in ff
    assert "ratios" in ff


@pytest.mark.asyncio
async def test_audit_drift_shape(client):
    reqs = (await client.get("/api/reqs")).json()
    req_id = reqs[0]["id"]
    r = await client.post("/api/audit", json={
        "req_id": req_id,
        "config": {"prestige_bonus": 0.0, "name_signal": False,
                   "required_skill_weight": 0.6, "threshold_advance": 0.7},
    })
    drift = r.json()["drift"]
    assert "passed" in drift
    assert "axis_results" in drift


@pytest.mark.asyncio
async def test_audit_unknown_req_returns_404(client):
    r = await client.post("/api/audit", json={
        "req_id": "does-not-exist",
        "config": {"prestige_bonus": 0.0, "name_signal": False,
                   "required_skill_weight": 0.6, "threshold_advance": 0.7},
    })
    assert r.status_code == 404


# ── /api/fidelity/{req_id} ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fidelity_returns_200(client):
    reqs = (await client.get("/api/reqs")).json()
    req_id = reqs[0]["id"]
    r = await client.get(f"/api/fidelity/{req_id}")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_fidelity_shape(client):
    reqs = (await client.get("/api/reqs")).json()
    req_id = reqs[0]["id"]
    r = await client.get(f"/api/fidelity/{req_id}")
    body = r.json()
    assert "passes" in body
    assert "mean_kappa_ai_human" in body
    assert "kappa_human_human" in body
    assert "n_pairs" in body
    assert "gold_pass_rate" in body


@pytest.mark.asyncio
async def test_fidelity_unknown_req_returns_404(client):
    r = await client.get("/api/fidelity/does-not-exist")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_fidelity_kappa_in_range(client):
    reqs = (await client.get("/api/reqs")).json()
    req_id = reqs[0]["id"]
    r = await client.get(f"/api/fidelity/{req_id}")
    body = r.json()
    assert -1.0 <= body["mean_kappa_ai_human"] <= 1.0
    assert 0.0 <= body["gold_pass_rate"] <= 1.0
