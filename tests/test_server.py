"""
API tests for the FastAPI certification dashboard (M4).

Tests the HTTP contract before the server exists (RED phase).
"""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from hrfidelity.server import app as app_module


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


# ── /api/pairs (M6 — counterfactual pair comparison) ──────────────────────────

def _fair_cfg() -> dict:
    return {"prestige_bonus": 0.0, "name_signal": False,
            "required_skill_weight": 0.6, "threshold_advance": 0.7}


@pytest.mark.asyncio
async def test_pairs_returns_200(client):
    reqs = (await client.get("/api/reqs")).json()
    req_id = reqs[0]["id"]
    r = await client.post("/api/pairs", json={"req_id": req_id, "config": _fair_cfg()})
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_pairs_returns_list(client):
    reqs = (await client.get("/api/reqs")).json()
    req_id = reqs[0]["id"]
    r = await client.post("/api/pairs", json={"req_id": req_id, "config": _fair_cfg()})
    body = r.json()
    assert isinstance(body, list)
    assert len(body) >= 1


@pytest.mark.asyncio
async def test_pairs_item_shape(client):
    reqs = (await client.get("/api/reqs")).json()
    req_id = reqs[0]["id"]
    r = await client.post("/api/pairs", json={"req_id": req_id, "config": _fair_cfg()})
    item = r.json()[0]
    for key in ("axis", "axis_label", "delta", "abs_delta", "base", "twin"):
        assert key in item, f"missing {key!r}"
    for side in ("base", "twin"):
        for key in ("name", "raw_score", "verdict", "skills"):
            assert key in item[side], f"{side} missing {key!r}"


@pytest.mark.asyncio
async def test_pairs_capped_at_six_and_two_per_axis(client):
    reqs = (await client.get("/api/reqs")).json()
    req_id = reqs[0]["id"]
    r = await client.post("/api/pairs", json={"req_id": req_id, "config": _fair_cfg()})
    body = r.json()
    assert len(body) <= 6
    by_axis: dict[str, int] = {}
    for p in body:
        by_axis[p["axis"]] = by_axis.get(p["axis"], 0) + 1
    assert all(c <= 2 for c in by_axis.values())


@pytest.mark.asyncio
async def test_pairs_abs_delta_matches_delta(client):
    reqs = (await client.get("/api/reqs")).json()
    req_id = reqs[0]["id"]
    r = await client.post("/api/pairs", json={"req_id": req_id, "config": _fair_cfg()})
    for p in r.json():
        assert p["abs_delta"] == pytest.approx(abs(p["delta"]), abs=1e-6)


@pytest.mark.asyncio
async def test_pairs_unknown_req_returns_404(client):
    r = await client.post("/api/pairs", json={"req_id": "does-not-exist", "config": _fair_cfg()})
    assert r.status_code == 404


# ── LLM screener routing ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_audit_llm_without_key_returns_503(client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    reqs = (await client.get("/api/reqs")).json()
    req_id = reqs[0]["id"]
    r = await client.post("/api/audit", json={
        "req_id": req_id,
        "config": _fair_cfg() | {"screener": "llm"},
    })
    assert r.status_code == 503


@pytest.mark.asyncio
async def test_audit_llm_routes_to_llm_scores(client, monkeypatch):
    """With a key present and the LLM cache stubbed, the audit must use the
    stubbed LLM raw scores — no real API call, no rubric scoring."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    reqs = (await client.get("/api/reqs")).json()
    req_id = reqs[0]["id"]

    resumes, pairs = app_module._corpus_for_req(req_id)
    cids = {r.candidate_id for r in resumes} | {p.twin.candidate_id for p in pairs}
    stub = {cid: 0.95 for cid in cids}  # everyone scores high under the LLM
    monkeypatch.setattr(app_module, "_llm_raw_scores", lambda _req_id: stub)

    r = await client.post("/api/audit", json={
        "req_id": req_id,
        "config": _fair_cfg() | {"screener": "llm"},
    })
    assert r.status_code == 200
    body = r.json()
    # Every group should advance at ~100% since all raw scores are 0.95 > 0.7
    for g in body["four_fifths"]["groups"]:
        assert g["selection_rate"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_pairs_name_signal_increases_protected_drift(client):
    """The mechanism: enabling name signals must raise counterfactual drift on
    the protected axes (race_proxy / gender) vs. a fair screener."""
    reqs = (await client.get("/api/reqs")).json()
    req_id = reqs[0]["id"]

    fair = (await client.post("/api/pairs", json={
        "req_id": req_id, "config": _fair_cfg()})).json()
    biased_cfg = _fair_cfg() | {"name_signal": True}
    biased = (await client.post("/api/pairs", json={
        "req_id": req_id, "config": biased_cfg})).json()

    def max_protected_drift(pairs: list[dict]) -> float:
        protected = [p["abs_delta"] for p in pairs
                     if p["axis"] in ("race_proxy", "gender")]
        return max(protected) if protected else 0.0

    # A fair screener should be near-invariant on name swaps; a name-signal
    # screener should show real drift on the protected axes.
    assert max_protected_drift(biased) > max_protected_drift(fair)


@pytest.mark.asyncio
async def test_fidelity_kappa_in_range(client):
    reqs = (await client.get("/api/reqs")).json()
    req_id = reqs[0]["id"]
    r = await client.get(f"/api/fidelity/{req_id}")
    body = r.json()
    assert -1.0 <= body["mean_kappa_ai_human"] <= 1.0
    assert 0.0 <= body["gold_pass_rate"] <= 1.0
