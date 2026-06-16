# HR Fidelity

> **Can you trust the AI reading your résumés?**

<p align="center">
  <a href="https://hr-fidelity-schu.fly.dev">
    <img src="https://img.shields.io/badge/▶%20%20Live%20demo-hr--fidelity--schu.fly.dev-2563eb?style=for-the-badge&logoColor=white" alt="Live demo"/>
  </a>
</p>

<p align="center"><strong><a href="https://hr-fidelity-schu.fly.dev">hr-fidelity-schu.fly.dev</a> — toggle bias knobs, watch the audit flip CERTIFIED → BLOCKED in real time</strong></p>

---

An integrated AI governance evaluation pipeline for resume screeners. The system combines disparate impact analysis, proxy-signal detection, counterfactual testing, recruiter calibration, and deployment gating into a single reproducible workflow — issuing a machine-readable CERTIFIED or BLOCKED verdict before the screener touches a live application.

**All data is 100% synthetic. No real candidates. No employer records. [See disclosure ↓](#data-disclosure)**

---

## The problem

In 2014–2017, Amazon built an ML hiring tool. Trained on past hires — who skewed male — it taught itself to penalize the word *"women's"* and downgrade graduates of two all-women's colleges. Amazon scrapped it in 2017.

**The law caught up.** NYC **Local Law 144** (in force, 2023) mandates a bias audit for any "automated employment decision tool." The **EU AI Act** classifies hiring as high-risk and requires conformity assessment. **California FEHA** and CRD guidance extend disparate-impact liability to AI employment tools statewide.

The screener is not the hard part. The evaluation layer is.

---

## The evaluation pipeline

The screener is the **subject under test**. The pipeline takes its scores as input and runs three independent checks before issuing a verdict.

![Audit pipeline](docs/audit-pipeline.png)

The audit layer is screener-agnostic — it receives scores and measures their properties. Swap the rubric for an LLM, a fine-tuned classifier, or a third-party vendor API behind the `score(resume, req, config) → Score` interface and the audit runs unchanged.

**[→ Full architecture diagram + module map](docs/ARCHITECTURE.md)**

**Four-fifths disparate impact (EEOC § 60-3.4 / NYC LL 144 / California FEHA):** any demographic group selected at less than 80% of the top group's rate triggers a violation. Measured from synthetic self-reported EEO race — the same basis real HR compliance uses, not name inference. Groups below the statistical minimum sample size are shown in the table but excluded from the verdict.

**Counterfactual drift (proxy detection):** every résumé gets a matched twin — identical skills and experience, one proxy signal swapped (name, institution tier, or gender signal). Mean score drift > 5% blocks certification. Pairs are grounded in Bertrand & Mullainathan (2004). The counterfactual invariant — content hash equality between twins — is mechanically enforced and CI-tested.

**Recruiter–AI agreement (Cohen's κ):** measures whether the AI and three synthetic human recruiters agree on A/B pairs. κ ≥ 0.60 ("substantial" agreement) is required. Gold pairs act as attention checks. The claim is never "the AI is correct" — it's "the AI agrees with calibrated human judgment."

**Deployment gating:** the CERTIFIED / BLOCKED verdict is the integration point. The screener only operates when all three layers clear. This is the governance pattern emerging in AI Act and LL 144 compliance discussions: evaluation results gate deployment, not just inform it.

### Regulatory frameworks referenced

| Framework | Jurisdiction | What it requires |
|---|---|---|
| EEOC Uniform Guidelines (four-fifths rule) | Federal (US) | Disparate impact baseline for all employment tools |
| NYC Local Law 144 | New York City | Annual third-party bias audit + public disclosure for AEDTs |
| California FEHA / CRD AI guidance | California | Disparate impact liability under state anti-discrimination law |
| EU AI Act (Annex III) | European Union | Conformity assessment for high-risk hiring systems |

---

## Why we built the rubric screener first

The screener (Layer 1) is the **subject under test**, not the product. The audit layer is the product.

We built a deterministic, rubric-based screener first — deliberately, not as a shortcut. The reasons:

**Verify the audit before auditing anything opaque.** With a transparent rubric, you can inject *known* bias — set `race_proxy_bias["black"] = -0.20` — and confirm the four-fifths check catches it. If your audit can't detect bias you manually injected, it won't detect bias a model learned implicitly. Rubric-first lets you prove the audit instrument works before pointing it at a black box.

**Separate the audit layer from the screener implementation.** The `score(resume, req, config) → Score` interface is the only contract. The audit layer never calls the screener directly — it receives scores and measures their properties. This makes the audit layer screener-agnostic: swap the rubric for an LLM, a fine-tuned classifier, or a third-party vendor API, and the audit runs unchanged. The interface was designed for this.

**Reproduce the Amazon failure on demand.** The rubric's bias knobs (`prestige_bonus`, `race_proxy_bias`, `gender_bias`) let you recreate Amazon's failure mode in a live demo — and watch the audit catch it. An LLM screener would show emergent bias, which is more realistic but harder to explain and reproduce reliably. The rubric version makes the mechanism legible.

The LLM screener (M5) plugs in behind the same interface. The audit layer doesn't change.

---

## The demo arc

1. **Default config** — screener passes all three checks. CERTIFIED.
2. **Enable name-based signals** — screener now sees race/gender proxies from applicant names. Four-fifths ratios drop. Drift spikes. BLOCKED.
3. **Toggle prestige bonus** — elite-school weighting introduces proxy correlation. Watch the verdict flip in real time.

---

## Counterfactual invariant

The audit claim — "score drift is caused by the protected proxy" — is only defensible if the twins are truly identical in job-relevant content. The generator enforces this mechanically:

```python
content_hash(skills, experience, education)  # must be equal across twins
proxy_field                                  # must differ: name, gender, prestige_tier
```

Any pair whose job-relevant hash differs is rejected before it enters the corpus. Identity independence is also enforced: in the base population, `identity ⊥ latent_fit`. Any score–demographic correlation in screener output is bias introduced by the screener, not baked into the data.

---

## Observability & evals (Arize Phoenix)

The LLM screener (M5) runs under [Arize Phoenix](https://github.com/Arize-ai/phoenix)
tracing via OpenTelemetry. Tracing alone just proves the wiring works — the point
is what runs *on top of* the traces: **evaluations**.

**Blind-invariance fairness eval, as a Phoenix experiment.** The counterfactual
drift check (Layer 3) is expressed as a Phoenix *experiment* over a dataset of
résumé twins. For each pair the screener scores base and twin via Claude Haiku,
and a code evaluator gates on `|Δ| ≤ 0.05`. Every call is auto-instrumented, so
each eval row links straight to its LLM trace.

![Phoenix experiment — blind-invariance fairness eval](docs/phoenix-experiments-list.png)

Result: `blind_invariant = 1.00` across all axes (race proxy, gender, prestige),
at both score extremes (reject 0.28 → 0.28, advance 0.92 → 0.92). A screener that
never sees identity cannot drift on identity — the expected, desirable behavior of
a properly blinded screener, now measured *inside the eval tool*.

![Phoenix experiment — per-pair results](docs/phoenix-blind-invariance-experiment.png)

**Failure-mode tracing (supporting).** The observability layer also distinguishes
*kinds* of failure with debuggable detail — each a real Anthropic API rejection
with a `request_id` (returned before generation, so $0). Spans turn red with the
exception recorded on both the `screen_resume` span and the nested LLM call. Click
through to the captured detail for each:

- [`404 not_found_error`](docs/phoenix-error-404.png) — retired/wrong model id
- [`401 authentication_error`](docs/phoenix-error-401.png) — bad API key
- [`400 invalid_request_error`](docs/phoenix-error-400.png) — malformed request (empty content)

![Phoenix — error taxonomy](docs/phoenix-error-taxonomy.png)

Reproduce locally:

```bash
python -m phoenix.server.main serve                                  # collector at :6006
ANTHROPIC_API_KEY=... .venv/bin/python scripts/blind_invariance_experiment.py  # the fairness eval
ANTHROPIC_API_KEY=... .venv/bin/python scripts/llm_error_demo.py     # the failure taxonomy
```

**[→ Full Phoenix write-up](docs/phoenix-observability.md)**

---

## Build status

| Milestone | Status | What |
|---|---|---|
| M1 — Data foundation | ✅ | Schema · req fixtures · SSA/Census/B-M name loaders · résumé generator · counterfactual invariant · CI checks |
| M2 — The failing demo | ✅ | Rubric screener · four-fifths audit · counterfactual drift probe |
| M3 — Fidelity layer | ✅ | Blind A/B pairing · Cohen's κ · Fleiss' κ · gold pair accuracy |
| M4 — Certification dashboard | ✅ | FastAPI server · live config knobs · CERTIFIED/BLOCKED verdict · deployed |
| M5 — LLM screener | ✅ | Swap rubric for Claude (Haiku) behind the same interface · blind prompt · emergent bias visible in audit |
| M6 — Pair comparison UI | ✅ | Side-by-side counterfactual pairs showing identical résumés scored differently |
| M7 — Methodology page | ✅ | GitHub Pages explainer — statistical choices, regulatory citations, Bertrand-Mullainathan grounding |

**259 tests passing, 0 failures.**

---

## Stack

- **Python 3.12** — FastAPI, uvicorn, pytest (259 tests), httpx
- **Statistics** — Cohen's κ, Fleiss' κ, EEOC four-fifths ratio, counterfactual mean drift
- **Frontend** — vanilla JS, GSAP 3 (hero animations, ScrollTrigger, mouse parallax), no framework
- **Deploy** — Docker, Fly.io (`shared-cpu-1x`, 256 MB)
- **Data** — 100% synthetic; SSA, US Census, Bertrand-Mullainathan public datasets

---

## Data disclosure

| Source | Used for |
|---|---|
| SSA baby-name data (vendored sample) | First names → inferred gender |
| US Census 2010 surname file (vendored sample) | Surnames → inferred race proxy (white/Black binary — for counterfactual drift only) |
| Bertrand & Mullainathan (2004) name pairs | Counterfactual swaps — white- and Black-signal first names |
| EEOC EEO-1 / tech-sector EEO-1 filings (public) | Calibration weights for synthetic self-reported race (eeo_race) |

**Two race fields, two purposes:**

- `inferred_race_proxy` — name-inferred signal (white/Black), used only for counterfactual drift. Grounded in B-M. Binary by design — the B-M pairs are the axis.
- `eeo_race` — synthetic self-reported EEO race (white/Black/Hispanic/Asian), assigned independently of name via a probability matrix calibrated to tech-sector EEO-1 demographics. This is what the four-fifths disparate impact check uses — the same basis real HR compliance uses. Asian engineers in tech often have Western first names (→ white proxy) but self-report Asian; the matrix models this realistically.

Names and demographic fields are proxies for inferred signal, not real identity. The generation method is committed to this repo so a reviewer can verify it is honest synthetic data, not scraped.

Hard rules: no "culture fit" scoring (discrimination vector). Screener scores against a fixed req-derived rubric only — never trained on past hires. That is the Amazon rule.

---

*Layer 2 fidelity method (blind A/B + κ) reused from [concord](https://github.com/stephendchu/agentic-test-eval).*
