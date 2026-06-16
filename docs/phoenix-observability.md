# Phoenix Observability — LLM Screener Trace Run

This documents a one-time, real-API validation of the M5 LLM screener (Claude
Haiku) end-to-end with [Arize Phoenix](https://github.com/Arize-ai/phoenix)
tracing via OpenTelemetry.

**Run:** `2026-06-15` · model `claude-haiku-4-5-20251001` · req `backend-eng`
(Senior Backend Engineer) · 12 real API calls (~$0.005).

Reproduce locally:

```bash
python -m phoenix.server.main serve            # collector at :6006
ANTHROPIC_API_KEY=... .venv/bin/python scripts/llm_smoke.py
```

## What Phoenix captured

All 12 `messages.create` spans landed in the `hr-fidelity` project with full
OpenInference LLM telemetry — auto-instrumented by `AnthropicInstrumentor`:

| Attribute | Example |
|---|---|
| `llm.model_name` | `claude-haiku-4-5-20251001` |
| `llm.provider` | `anthropic` |
| `llm.input_messages` | full blind prompt (skills/experience only) |
| `llm.output_messages` | `{"score": 92, "rationale": "..."}` |
| `llm.token_count.prompt` / `.completion` | e.g. prompt N / completion 98 |
| `status_code` | `OK` |

Raw exports committed for reference:
- `docs/phoenix-llm-smoke.json` — per-resume scores, verdicts, rationales, and the counterfactual-pair drift.
- `docs/phoenix-traces-export.json` — the 12 spans pulled back out of Phoenix.

The captured prompt confirms the screener is **blind**: the input message reads
*"CANDIDATE PROFILE (anonymized — no name or demographic information)"*. No name,
gender, race proxy, or institution prestige is sent to the model.

## Two findings

**1. The screener works and scores sensibly.** Across the latent-fit spectrum:
`strong → 0.92 advance`, `medium → 0.62 borderline`, `weak → 0.28 reject`.

**2. Blind-invariance holds (the honest teaching point).** Counterfactual twins
— identical job-relevant content, one proxy signal swapped — barely move:

| Axis | base | twin | Δ |
|---|---|---|---|
| race_proxy (John Jackson → John McCarthy) | 0.95 | 0.92 | −0.03 |
| gender (John → Jane) | 0.92 | 0.92 | 0.00 |
| prestige_tier | 0.92 | 0.92 | 0.00 |

Max protected-axis drift **0.03 < 0.05** → blind-invariant. A screener that never
sees identity cannot drift on identity. This is *not* a bug — it is the expected,
and desirable, behavior of a properly blinded screener.

## A real bug this run surfaced (now fixed)

The custom `hr_fidelity.*` span attributes (`req_id`, `candidate_id`,
`latent_fit`, `raw_score`, `verdict`) were **missing** from every span.

**Cause:** `llm_screener.score()` called `get_current_span()` and set attributes
on it *outside* the instrumented LLM call. The `AnthropicInstrumentor` span only
exists *during* `client.messages.create()`; before/after it the current span is
the non-recording default span, so `set_attributes` was silently dropped.

**Fix:** `score()` now opens its own recording span (`screen_resume`) via a
module tracer, with the LLM span nested beneath it. The custom attributes attach
to that parent span. Covered by
`tests/test_llm_screener.py::test_score_emits_recording_span_with_custom_attributes`
(verified with an in-memory span exporter — no API spend).

On the next real run, each `screen_resume` span will carry the business context
(which candidate, which req, the resulting verdict) alongside the nested LLM
telemetry — turning raw model calls into auditable screening decisions.

## Blind-invariance fairness eval (Phoenix experiment)

Tracing proves the wiring works. The reason Phoenix exists is the layer above it:
running an **evaluation over traced LLM calls** and surfacing a score + pass/fail
in the Experiments UI. We express the Layer-3 counterfactual drift check as a
Phoenix experiment.

**Run:** `scripts/blind_invariance_experiment.py` · model `claude-haiku-4-5-20251001`
· dataset `blind-invariance-backend-eng` · 6 counterfactual pairs · 12 real API
calls (~$0.005).

- **Dataset** — one example per counterfactual pair. Input carries a stable
  `pair_key` (`axis::candidate_id`); the task rebuilds the exact `Resume` objects
  deterministically (`seed=44`) so the experiment is reproducible. Expected output
  is the drift bound `max_drift = 0.05`.
- **Task** (`screen_pair`) — scores base and twin via the LLM screener and returns
  `{base_score, twin_score, abs_delta, base_verdict, twin_verdict}`. Each score is
  a real, auto-instrumented Anthropic call, so every eval row links to its trace.
- **Evaluators** —
  - `blind_invariant` (gate): PASS iff `|Δ| ≤ 0.05`.
  - `drift_headroom` (numeric): `1.0 − |Δ|/0.05` — how much of the drift budget is unused.

**Result:** `blind_invariant = 1.00` on all 6 pairs.

| axis | base | twin | \|Δ\| | verdicts |
|---|---|---|---|---|
| race_proxy | 0.92 | 0.92 | 0.000 | advance → advance |
| race_proxy | 0.28 | 0.28 | 0.000 | reject → reject |
| gender | 0.92 | 0.92 | 0.000 | advance → advance |
| gender | 0.28 | 0.28 | 0.000 | reject → reject |
| prestige_tier | 0.92 | 0.92 | 0.000 | advance → advance |
| prestige_tier | 0.28 | 0.28 | 0.000 | reject → reject |

Invariance holds at both score extremes — a blinded screener that never sees
identity cannot drift on identity. This is the same finding as the smoke run's
counterfactual table, now measured *inside Phoenix as a first-class eval* rather
than a side JSON artifact.

Screenshots: `docs/phoenix-experiments-list.png` (aggregate: blind_invariant 1.00,
6 runs, <$0.01) and `docs/phoenix-blind-invariance-experiment.png` (per-pair grid).

## Failure-mode taxonomy (error tracing)

`scripts/llm_error_demo.py` triggers one genuine Anthropic API rejection per
category into the `hr-fidelity-errors` project, proving the layer distinguishes
*kinds* of failure — not just "something errored":

| Code | Error type | Cause in the demo |
|---|---|---|
| 404 | `not_found_error` | retired/wrong model id |
| 401 | `authentication_error` | bad API key |
| 400 | `invalid_request_error` | malformed request (empty content) |

Each error carries a real `request_id` and is returned before generation (so $0).
The exception is recorded on the `screen_resume` span and the nested LLM span, both
of which turn red with the status description. Screenshots: `docs/phoenix-error-taxonomy.png`
plus per-category `docs/phoenix-error-{404,401,400}.png`.
