# hr-fidelity — build handoff

Fresh session? Start here. Everything you need to build M1 without re-reading the conversation.

---

## What this project is (one paragraph)

A trust-and-audit layer that sits on top of an AI resume screener and answers two questions: *does the screener agree with senior recruiters (Cohen's κ)?* and *will it get you sued (NYC LL144 four-fifths rule)?* The narrative anchor is the Amazon recruiting-AI scandal (2014–2017) — an AI trained to match candidates to past hires learned to penalize the word "women's." This tool audits that failure mode in real time. Everything runs on **synthetic/generated data only** — never scraped, never real people. Full design: `docs/DESIGN.md`.

## Repository layout

```
hr-fidelity/
  src/hrfidelity/
    data/          ← M1: schema, generator, counterfactual pairs  ← START HERE
    audit/         ← M2+: four-fifths, counterfactual drift, κ
    screener/      ← M2: LLM-rubric screener (subject under test)
  tests/           ← TDD: write failing test first, always
  data/
    reqs/          ← job req fixtures (JSON)
    resumes/       ← generated resume corpus (JSON)
    counterfactual_pairs/  ← matched pairs (JSON)
  docs/
    DESIGN.md      ← master plan
    data-generator.md  ← M1 build spec (READ THIS)
    landing-content.md ← splash copy
    HANDOFF.md     ← this file
```

## Environment

```bash
cd /mnt/c/Users/Steph/projects/hr-fidelity
source .venv/bin/activate          # Python 3.12, pytest installed
.venv/bin/pytest                   # run tests
```

Uses `uv` for package management. `pyproject.toml` is at root.

**Hard constraint: do NOT call the Anthropic API / SDK without explicit user OK.** The user's spend is on the subscription, not the API key. All M1 work is pure Python — no API calls needed.

---

## Where to start — M1, counterfactual hash invariant (TDD)

The build order from `docs/data-generator.md`:

1. Resume schema + req fixtures
2. Name/demographic loaders (SSA + Census + Bertrand–Mullainathan)
3. Resume generator (identity independent of `_latent_fit`)
4. **Counterfactual pair generator + the hash invariant ← START HERE with the failing test**
5. CI validation checks
6. Manifest + eval-store layout

**The invariant (the airtight core):**
A counterfactual twin must be identical to its base resume in every job-relevant field and differ *only* in a protected proxy (name, school prestige, keyword). Enforced as a content hash:

```python
content_hash(skills, experience, education_fields_except_proxy) == same for both twins
proxy_fields != same for both twins
```

Any score delta between twins is attributable to the proxy alone → clean bias measurement. This is what makes the Amazon demo defensible.

**Write this test first (watch it fail, then build):**

```python
# tests/test_counterfactual.py
def test_twins_share_content_hash_differ_on_proxy():
    base = make_resume(first_name="Emily", last_name="Walsh")
    twin = make_counterfactual(base, axis="gender")  # swap to male name
    assert content_hash(base) == content_hash(twin)   # same job content
    assert base.identity["first_name"] != twin.identity["first_name"]  # different proxy
```

The test will fail with `NameError` (functions don't exist yet). That's correct — it proves the test is real. Build `make_resume`, `make_counterfactual`, and `content_hash` in `src/hrfidelity/data/` to make it pass.

## Key design decisions (don't relitigate these)

- **Screener = subject under test, not the product.** The product is the certification dashboard. Never pitch the screener.
- **Never learn from past hires.** Score against a fixed req-derived rubric only. Learning from historical hires IS Amazon.
- **Candidates only** — not current-employee valuation (wrong audience: hypergrowth shops need external hiring, not succession planning).
- **Identity ⊥ latent_fit** in the base population. Bias must come from the screener/config, not the dataset. Test this.
- **Counterfactual axes:** gender (SSA first-name swap), race-proxy (Bertrand–Mullainathan surname swap), school prestige toggle, keyword inject/remove (the Amazon "women's" probe).
- **Screening config knobs** each annotated with their bias risk; live-wired to the audit so changing a knob updates four-fifths in real time. The "no employment gaps" knob → gray ⚠ warning → turns red + specific when it breaks four-fifths.

## Audit math (Layer 3)

- **Four-fifths:** `selection_rate(group) / selection_rate(top_group)` — flag < 0.80 (EEOC / LL144).
- **Counterfactual drift:** `mean |score(twin_A) − score(twin_B)|` and `% of pairs whose advance/reject verdict flips`.
- **Fidelity κ:** Cohen's κ (AI vs aggregated human votes) vs Fleiss' κ (human–human baseline). Claim = κ_AI-human ≈ κ_human-human, never "AI is correct."

## References (for synthetic data realism)

- **O\*NET** (onetonline.org) — occupation/skills taxonomy. Use for role-appropriate skill lists in the resume generator.
- **SSA baby names** (ssa.gov/oact/babynames/) — first name → inferred sex.
- **US Census surname file** — surname → race/ethnicity probability.
- **Bertrand & Mullainathan (2004)** "Are Emily and Greg More Employable than Lakisha and Jamal?" — the foundational paper. Your counterfactual method is the same as theirs. Grounding in it is a credibility signal.

## What NOT to build yet

- The API / FastAPI server (M4)
- The voting UI (M3/M4)
- The certification dashboard (M4)
- The LLM screener (M2) — needs M1 data to score
- Anything requiring the Anthropic SDK (needs explicit user OK for API spend)
