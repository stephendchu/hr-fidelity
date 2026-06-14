# HR Fidelity — Architecture

A short technical reference. For design rationale see `DESIGN.md`; for the data spec see `data-generator.md`.

---

## Data and evaluation flow

```
┌─────────────────────────────────────────────────────────────────┐
│  SYNTHETIC RESUME GENERATOR                                     │
│  src/hrfidelity/data/resume_generator.py                        │
│  src/hrfidelity/data/name_loader.py                             │
│                                                                 │
│  Generates résumés at three latent fit levels (strong /         │
│  medium / weak) for each job req. Identity (name, gender,       │
│  eeo_race) is assigned independently of fit — enforced via      │
│  separate RNG stream and CI-tested. 150 résumés per req.        │
└────────────────────────┬────────────────────────────────────────┘
                         │  150 base résumés
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  CANDIDATE PAIR BUILDER                                         │
│  src/hrfidelity/data/counterfactual.py                          │
│                                                                 │
│  For each résumé, generates a matched twin per axis:            │
│    gender       — swap first name (SSA name tables)             │
│    race_proxy   — swap surname (Bertrand–Mullainathan 2004)     │
│    prestige_tier — toggle school tier (elite ↔ regional)        │
│                                                                 │
│  Invariant enforced: content_hash(twin) == content_hash(base)   │
│  Any pair where job-relevant content differs is rejected.       │
│  450 matched pairs per req (150 résumés × 3 axes).             │
└────────────────────────┬────────────────────────────────────────┘
                         │  150 bases + 450 twins
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  AI SCREENER  ·  subject under test                             │
│  src/hrfidelity/screener/rubric_screener.py                     │
│  src/hrfidelity/screener/llm_screener.py                        │
│                                                                 │
│  Interface:  score(resume, req, config) → Score                 │
│                                                                 │
│  Both screeners implement the same interface. The audit layer   │
│  never calls the screener directly — it receives Score objects  │
│  and measures their properties. Swap the implementation;        │
│  the pipeline is unchanged.                                     │
│                                                                 │
│  Config knobs (rubric screener):                                │
│    prestige_bonus       — weight added for elite-tier schools   │
│    race_proxy_bias      — score delta by inferred race signal   │
│    gender_bias          — score delta by inferred gender        │
│    required_skill_weight — share of score from req skills       │
│    threshold_advance    — minimum score to advance              │
└──────┬──────────────────┬──────────────────┬────────────────────┘
       │ 150 base scores  │ 600 pair scores  │ 20 A/B pair scores
       ▼                  ▼                  ▼
┌────────────────┐ ┌──────────────────┐ ┌──────────────────────────┐
│ DISPARATE      │ │ COUNTERFACTUAL   │ │ RECRUITER CALIBRATION    │
│ IMPACT         │ │ DRIFT ANALYSIS   │ │                          │
│                │ │                  │ │ src/hrfidelity/           │
│ audit/         │ │ audit/           │ │   fidelity/calibration.py│
│ disparate_     │ │ counterfactual_  │ │   fidelity/pairs.py      │
│ impact.py      │ │ drift.py         │ │                          │
│                │ │                  │ │ Blind A/B pairs shown to │
│ four_fifths_   │ │ drift_check()    │ │ 3 synthetic recruiters.  │
│ check()        │ │                  │ │ Measures AI–recruiter    │
│                │ │ Mean score drift │ │ agreement (Cohen's κ)    │
│ EEOC § 60-3.4  │ │ across matched   │ │ vs recruiter–recruiter   │
│ four-fifths    │ │ pairs, per axis. │ │ baseline (Fleiss' κ).    │
│ rule per EEO   │ │ Protected axes   │ │                          │
│ race group.    │ │ only (race_proxy,│ │ Gold pairs act as        │
│ Threshold:     │ │ gender). Prestige│ │ attention checks.        │
│ ratio ≥ 0.80.  │ │ is proxy warning.│ │ Threshold: κ ≥ 0.60.    │
│                │ │ Threshold: ≤0.05.│ │                          │
└───────┬────────┘ └────────┬─────────┘ └──────────────┬───────────┘
        │                   │                           │
        └───────────────────┼───────────────────────────┘
                            │  three CheckResult objects
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  CERTIFICATION ENGINE                                           │
│  src/hrfidelity/audit/report.py                                 │
│                                                                 │
│  run_audit() combines the three results into one AuditReport.  │
│  All three checks must pass. Any single failure blocks.         │
│  Report includes: verdict, n_resumes, n_pairs, four_fifths      │
│  result, drift result, plain-English detail string.             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                ┌────────┴─────────┐
                │                  │
          ✅ CERTIFIED        🚫 BLOCKED
          Screener cleared    + which check failed
          for deployment      + which proxy caused it
                              + plain-English reason
```

---

## Module map

| Module | Responsibility |
|---|---|
| `data/schema.py` | `Resume`, `Identity`, `CounterfactualPair`, `Education`, `Experience` dataclasses |
| `data/name_loader.py` | SSA / Census / B-M name tables → `sample_identity()`, `eeo_race` probability matrix |
| `data/resume_generator.py` | `generate_resume(req, latent_fit, rng)` — O\*NET-grounded skills, prestige-weighted education |
| `data/counterfactual.py` | `generate_counterfactual(resume, axis)` + `content_hash()` invariant |
| `data/corpus_generator.py` | `generate_corpus()` — orchestrates generator → pairs; `save_corpus()` / `load_corpus()` |
| `data/ci_checks.py` | `check_invariant()`, `check_independence()`, `check_realism()`, `check_reproducibility()` |
| `screener/rubric_screener.py` | Deterministic rubric scorer; bias knobs make the Amazon failure reproducible |
| `screener/llm_screener.py` | Claude (Haiku) behind the same `score()` interface; emergent bias visible in audit |
| `audit/disparate_impact.py` | `four_fifths_check()` — groups by `eeo_race`; `stat_min_n` guards small samples |
| `audit/counterfactual_drift.py` | `drift_check()` — mean drift + flip rate per axis; protected axes only |
| `fidelity/calibration.py` | `run_calibration()` — Cohen's κ per judge pair, Fleiss' κ human baseline |
| `audit/report.py` | `run_audit()` → `AuditReport` with `verdict`, `four_fifths`, `drift`, `detail` |
| `server/app.py` | FastAPI — `/api/audit`, `/api/scores`, `/api/fidelity/{req_id}`, static serving |

---

## Key invariants (CI-tested)

1. **Content hash** — `content_hash(base) == content_hash(twin)` for every pair. Any score delta between twins is attributable to the proxy alone.
2. **Identity independence** — `identity ⊥ latent_fit` in the base population. Bias must come from the screener, not the dataset.
3. **Reproducibility** — same seed → same corpus. Seeded RNG, committed manifest.
4. **Realism** — every résumé has non-empty skills, at least one education entry, valid date ranges.

---

## Interface contract

```python
# The only contract between screener and audit layer.
# Both rubric_screener and llm_screener implement this.

def score(resume: Resume, req: Req, config: ScreenerConfig) -> Score:
    ...

@dataclass
class Score:
    candidate_id: str
    raw_score: float        # 0.0–1.0
    verdict: str            # "advance" | "borderline" | "reject"
    breakdown: dict         # component scores (optional, for explainability)
```

The audit layer receives a `list[Score]` and never calls the screener. This makes the pipeline screener-agnostic.
