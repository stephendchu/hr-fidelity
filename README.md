# hr-fidelity

> **Can you trust the AI reading your résumés?**

Not another screener. A **trust-and-audit layer** that answers the only two questions that matter before you let AI touch a req: *Does it agree with your best recruiters? And will it get you sued?*

**Status:** M1 (synthetic data foundation) in progress — data generator, counterfactual pair engine, and hash invariant complete. 108 tests passing.  
**Data:** 100% synthetic. Never real candidates. [See disclosure ↓](#data-disclosure)

---

## The problem

Hypergrowth teams get thousands of applications per role. AI screeners promise to keep up. Every vendor sells one.

In 2014–2017, Amazon built one. Trained on past hires — who skewed male — it taught itself to penalize the word *"women's"* and downgrade graduates of two all-women's colleges. Amazon scrapped it in 2017. It failed at exactly one task: screening résumés.

**The law caught up.** NYC **Local Law 144** (in force, 2023) mandates a bias audit — using the EEOC four-fifths disparate-impact rule — for any "automated employment decision tool." The **EU AI Act** classifies hiring as high-risk and requires conformity assessment. Regulators are not waiting.

The screener is not the hard part. The trust layer is.

---

## What this is

Three layers, each with a distinct job:

```
┌─────────────────────────────────────────────────────────┐
│  LAYER 3 — AUDIT       four-fifths / disparate impact   │
│                        counterfactual probe             │
├─────────────────────────────────────────────────────────┤
│  LAYER 2 — FIDELITY    blind A/B vs recruiters (κ)      │  ← concord, reused
├─────────────────────────────────────────────────────────┤
│  LAYER 1 — SCREENER    scores résumés vs a job req      │  ← subject under test
│            (the thing everyone else sells)              │
└─────────────────────────────────────────────────────────┘
```

**Layer 1 — the screener (subject under test).** An LLM scores each synthetic résumé against a job req: `advance / borderline / reject`. Simple, honest, and deliberately boring — because it's the thing being audited, not the product. Hard rule: never trained on past hires. Scoring against a fixed, req-derived rubric is the only way to avoid becoming Amazon.

**Layer 2 — fidelity** (reused from [concord](https://github.com/stephendchu/agentic-test-eval)). Blind A/B pairing: show a recruiter two résumés, ask "which advances?" Measure AI–recruiter agreement (Cohen's κ) against the recruiter–recruiter baseline (Fleiss' κ). The claim is never "the AI is right" — it's "the AI votes like a calibrated recruiter."

**Layer 3 — audit.** Two probes:
- **Counterfactual probe:** hold skills and experience fixed; swap the name or gender signal. Any score delta is attributable to the proxy alone. This is the Amazon failure, made visible and measurable.
- **Disparate impact:** compute selection-rate ratios across synthetic demographic groups. Flag any below 0.80 — the EEOC four-fifths rule, the same standard NYC LL144 requires.

---

## The counterfactual invariant

The audit claim — "score drift is caused by the protected proxy" — is only defensible if the twins are *truly* identical in job-relevant content. The generator enforces this mechanically:

```python
content_hash(skills, experience, education)  # must be equal across twins
proxy_field                                  # must differ: name, gender, prestige_tier
```

`content_hash` is computed at generation time. Any pair whose job-relevant hash differs is rejected before it enters the corpus. This makes it mechanically impossible for content to bleed into the bias measurement.

The invariant is TDD'd and will run in CI against every generated pair. Name pairs are grounded in **Bertrand & Mullainathan (2004)** — the landmark audit study whose methodology this work extends.

---

## The demo arc

1. **Here is a raw AI screener** ranking synthetic résumés for a real-ish req. Looks fast. Looks confident.
2. **Watch it fail.** Run the counterfactual probe → identical résumés score differently when the name changes. Run disparate impact → one group lands at 0.68, below four-fifths. *This is Amazon, reproduced live.*
3. **Now gate it.** Same screener, wrapped: ships its ranking only when κ-agreement clears a threshold **and** it passes four-fifths. The failing config is blocked, with the evidence shown.

The arc — *naive → caught → calibrated* — is the proof of judgment.

---

## Build status

| Milestone | Status | What |
|---|---|---|
| M0 — Planning | ✅ | Design doc, landing copy, data spec |
| M1 — Data foundation | 🔄 in progress | Schema · req fixtures · SSA/Census/B-M name loaders · résumé generator · counterfactual invariant · CI validation · corpus output |
| M2 — The failing demo | ⬜ | Layer 1 screener + Layer 3 audit (screener fails — this is the point) |
| M3 — Fidelity layer | ⬜ | Port concord A/B + κ onto résumé pairs |
| M4 — Certification dashboard | ⬜ | UX centerpiece: live tune → re-audit → verdict loop with inline education |

**108 tests passing, 0 failures.** The counterfactual hash invariant and identity ⊥ fit independence check are green.

---

## Data disclosure

Everything here is synthetic. Résumés, candidates, and job reqs were generated — never scraped, never real people.

| Source | Used for |
|---|---|
| SSA baby-name data (vendored sample) | First names → inferred gender |
| US Census 2010 surname file (vendored sample) | Surnames → inferred race proxy (modal probability) |
| Bertrand & Mullainathan (2004) name pairs | Counterfactual swaps — white- and Black-signal first names from the published audit study |
| O\*NET occupational data | Skill pools for each req archetype |

Names are proxies for inferred demographic signal, not identity. **Independence enforced:** in the base population, `identity ⊥ _latent_fit` — any score–demographic correlation in screener output is bias introduced by the screener, not baked into the data. The generation method is committed to this repo so a reviewer can verify it is honest synthetic, not scraped.

---

## Stack

- **Python 3.12**, pytest (108 tests), FastAPI + SQLite (M2+)
- **Anthropic SDK** — LLM screener + judge (M2+; no API spend in M1)
- **Eval-store discipline:** observed inputs (résumés, reqs, votes) kept separate from derived outputs (κ, impact ratios) — every metric is recomputable from captured inputs × versioned scorer

---

*The blind A/B calibration method (Layer 2) is reused from [concord](https://github.com/stephendchu/agentic-test-eval), which ships as a methods appendix in `agentic-test-eval`. This repo is the recruiter-facing showcase: full-stack + UX + responsible AI on the exact task Amazon failed.*
