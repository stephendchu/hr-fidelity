# M1 — synthetic data + counterfactual generator (spec)

This is the foundation everything stands on. The screener, the config knobs, the κ, and the four-fifths numbers are only as credible as the résumés underneath — and the counterfactual matched-pair logic is the single most failure-prone piece. If twins aren't *truly* identical except for one proxy, the entire bias claim collapses. Build this airtight first.

**Hard constraint:** synthetic / generated only. Never scraped, never real people. Public reference sources (O\*NET, Census, SSA, Bertrand–Mullainathan) inform *realism and the bias axis*, never ship as data. Commit the generator + a seeded manifest, not just the output.

---

## 1. Résumé schema

```
Resume {
  candidate_id: str
  identity: {                 # the bias axis — kept SEPARATE from screener input
    first_name, last_name
    inferred_gender            # from SSA first-name → sex
    inferred_race_proxy        # name-inferred (white/Black) — B-M axis, used for counterfactual drift only
    eeo_race                   # synthetic self-reported EEO race (white/Black/Hispanic/Asian)
                               # assigned independently via EEO-1-calibrated probability matrix
                               # this is what the four-fifths check uses (same basis as real HR compliance)
    source: "ssa" | "census" | "bertrand_mullainathan"
  }
  education: [{ degree, field, institution, prestige_tier, grad_year, gpa? }]
  experience: [{ title, company, start, end, bullets[] }]   # gaps are computed, not stored
  skills: [str]               # drawn from O*NET for the target archetype
  certifications: [str]
  # hidden / not shown to screener:
  _latent_fit: "strong" | "medium" | "weak"   # ground-truth quality vs the req's rubric
}
```

**Critical separation:** `identity` (the protected proxy) is metadata used only by the **audit**. The screener sees résumé *content* (education/experience/skills/name-on-page) — but in the **base population, identity must be statistically independent of `_latent_fit`** (see §5). That independence is what makes any score–demographic correlation *bias*, not signal.

## 2. Job reqs

A handful across archetypes (ML researcher, backend, infra, recruiter). Each:
```
Req { id, title, required_skills[] (O*NET), nice_to_have[], min_years, true_rubric }
```
`true_rubric` is the fixed, job-derived scoring key (NOT learned from past hires — that's the Amazon rule). It defines what `_latent_fit` means for this req.

## 3. Generation pipeline

1. Pick a req.
2. Sample `_latent_fit` (strong / medium / weak) → controls how well skills + experience match `true_rubric`.
3. LLM-generate résumé content conditioned on `(req, _latent_fit, skills sampled from O*NET)`. **Fit is driven only by job-relevant content.**
4. Assign `identity` **independently** of `_latent_fit` (random draw from name datasets). This independence is enforced and tested.
5. Seed everything; emit a manifest (seed, req, fit, identity source) for reproducibility.

## 4. Name / demographic datasets (the bias axis)

- **SSA baby names** → first name → inferred sex.
- **US Census surname file** → surname → race/ethnicity probability.
- **Bertrand–Mullainathan (2004)** name list → the canonical "distinctively white vs Black" axis; grounds the counterfactual method in the foundational hiring-bias literature (credibility multiplier).
- Document the mapping; state plainly that names are *proxies*, not identity.

## 5. Counterfactual matched-pair generator — THE heart (must be airtight)

Given a base résumé, produce a **twin** that is identical in every job-relevant field and differs **only** in the protected proxy:

- Axes (one at a time): gender (swap first name male↔female), race-proxy (swap surname), school signal (toggle prestige_tier), keyword (inject/remove a benign group marker like "women's chess club" — the Amazon probe).
- **The invariant, enforced as code:**
  ```
  content_hash(skills, experience, education_fields_except_proxy)  MUST be equal across twins
  proxy_fields                                                     MUST differ
  ```
  Compute the hash at generation time; **reject any pair whose job-relevant hash differs.** This assertion is the airtight guarantee — it's mechanically impossible for content to leak into the "bias" measurement.
- Why it matters: any score delta between twins is attributable to the proxy *alone* → clean, defensible bias measurement.

## 6. Validation / honesty checks (run in CI)

1. **Independence check:** in the base population, `identity ⊥ _latent_fit` (no correlation). Proves the *dataset* isn't pre-biased — bias must come from the screener/config, which is the whole thesis. Enforced via a separate `rng_id` stream for identity draws (so identity never correlates with the fit-level generation order) and a shuffled fit assignment sequence.
2. **Invariant check:** every counterfactual pair passes the §5 hash assertion.
3. **Realism check:** résumé structure mirrors real ones (Kaggle datasets as *structure* reference only).
4. **Reproducibility check:** same seed → same corpus.

## 7. Outputs / storage (carry concord's eval-store discipline)

```
data/reqs/*.json
data/resumes/*.json
data/counterfactual_pairs/*.json
data/manifest.json          # seeds + provenance
```
Observed (résumés, reqs, pairs) vs derived (scores, κ, impact ratios) kept in separate stores so every metric is recomputable from captured inputs × versioned scorer — same senior-signal as concord's eval store.

---

## Audit math (Layer 3 — concrete formulas)

- **Disparate impact (four-fifths):** `selection_rate(group) / selection_rate(top group)`; **flag < 0.80** (EEOC / NYC LL144). `selection_rate(g) = advanced(g) / total(g)`.
- **Counterfactual drift:** `mean |score(twin_A) − score(twin_B)|` and **% of pairs whose advance/reject verdict flips**. Target ≈ 0; any flip is a named Amazon-style failure.
- **Fidelity (κ, reused from concord):** Cohen's κ (AI vs aggregated human) reported against Fleiss' κ (human–human baseline). Claim = `κ_AI-human ≈ κ_human-human`, never "AI is correct."

## Build order (M1)

1. Schema + req fixtures (§1–2).
2. Name/demographic loaders (§4) with a tiny vendored sample, full files documented.
3. Résumé generator + independence enforcement (§3, §5-independence).
4. **Counterfactual pair generator + the hash invariant (§5)** — the airtight core; unit-test the invariant first (TDD: write the "twins must share content hash / differ on proxy" test, watch it fail, then build).
5. CI validation checks (§6).
6. Manifest + eval-store layout (§7).

Only after M1's invariant tests are green does the screener (Layer 1) have credible ground to stand on.
