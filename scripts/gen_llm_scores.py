"""
Pre-compute LLM-screener raw scores into committed fixtures.

The screener prompt is blind to identity, so a candidate's raw score is a stable
constant — and counterfactual twins (name/gender/prestige swapped) produce a
byte-identical prompt to their base. We therefore score each *unique prompt* once
(~150 per req, not 600) and map the score back to every candidate that shares it.

Capturing these lets the public demo serve real Claude Haiku output at $0 with no
API key on the server (see app._llm_raw_scores). Re-run if the corpus seed,
n_per_fit, the prompt, or the model changes.

    ANTHROPIC_API_KEY=... .venv/bin/python scripts/gen_llm_scores.py [req_id ...]

Writes data/llm_scores/<req_id>.json  ({candidate_id: raw_score}).
Cost: ~150 Haiku calls per req (~$0.06). With no req_id args, does all reqs.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor

from hrfidelity.data.schema import Resume
from hrfidelity.screener import llm_screener
from hrfidelity.screener.llm_screener import _build_prompt
from hrfidelity.server import app as app_module
from hrfidelity.tracing import setup_tracing

WORKERS = 8  # modest concurrency; well under Haiku rate limits


def _score_req(req) -> dict[str, float]:
    resumes, pairs = app_module._corpus_for_req(req.id)
    all_map: dict[str, Resume] = {r.candidate_id: r for r in resumes}
    for p in pairs:
        all_map.setdefault(p.twin.candidate_id, p.twin)

    # Group candidates by their (blind) prompt so identical prompts score once.
    prompt_to_cids: dict[str, list[str]] = {}
    cid_to_resume: dict[str, Resume] = {}
    for cid, r in all_map.items():
        prompt_to_cids.setdefault(_build_prompt(r, req), []).append(cid)
        cid_to_resume[cid] = r

    unique = list(prompt_to_cids.items())
    print(f"{req.id}: {len(all_map)} candidates, {len(unique)} unique prompts to score…")

    def score_one(item):
        _, cids = item
        rep = cid_to_resume[cids[0]]  # any candidate with this prompt
        return cids, round(llm_screener.score(rep, req).raw_score, 4)

    out: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for cids, raw in ex.map(score_one, unique):
            for cid in cids:
                out[cid] = raw
    return out


def main(argv: list[str]) -> int:
    setup_tracing("hr-fidelity")
    app_module._LLM_SCORES_DIR.mkdir(parents=True, exist_ok=True)

    reqs = app_module._load_reqs()
    if argv:
        wanted = set(argv)
        reqs = [r for r in reqs if r.id in wanted]
        if not reqs:
            print(f"no matching reqs for {argv}; available: {[r.id for r in app_module._load_reqs()]}")
            return 1

    for req in reqs:
        scores = _score_req(req)
        out = app_module._llm_fixture_path(req.id)
        out.write_text(json.dumps(scores, indent=2, sort_keys=True))
        print(f"  -> {out.relative_to(app_module._DATA_ROOT.parent)}  ({len(scores)} scores)")

    try:
        from opentelemetry import trace as otel_trace
        otel_trace.get_tracer_provider().force_flush()
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
