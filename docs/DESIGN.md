# hr-fidelity — design doc

> Working name. The repo is `hr-fidelity`; the product name is provisional (candidates: **Fidelity**, **Concord for Hiring**, **Fourfifths**). Decide before the README.

**One line:** A trust-and-audit layer that sits on top of an AI resume screener and answers the only two questions that matter before you let it touch a req: *does it agree with your best recruiters, and will it get you sued?*

---

## 1. The problem (who this is for)

Frontier / hypergrowth shops (~80%/yr headcount) have one top-of-funnel emergency: **thousands of resumes per req, recruiters who can't keep up.** So they reach for an AI screener to rank or filter the inbound. Every vendor sells one. The screener is not the hard part.

The hard part is *trust*. A hiring leader at a company growing this fast cannot answer:

- Does this screener actually agree with how my senior recruiters triage — or is it confidently wrong at scale?
- If a rejected candidate sues, or a regulator audits, can I show the screener isn't biased?

Today those questions get hand-waved. This tool answers them with numbers.

## 2. Why now — the Amazon story + the law

**The villain (2014–2017):** Amazon built an AI resume screener trained to match candidates to past winners. Because past winners skewed male, it learned to penalize resumes containing "women's" (as in "women's chess club captain") and downgrade graduates of two all-women's colleges. Amazon scrapped it. It is the canonical cautionary tale, and it failed at *exactly this task* — resume screening.

**The law (now):** NYC **Local Law 144** (in force since 2023) requires a bias audit of any "automated employment decision tool" — i.e. a resume screener — using disparate-impact ratios (the EEOC four-fifths rule). The **EU AI Act** classifies hiring as high-risk and mandates conformity assessment. The **EEOC** has issued AI-hiring discrimination guidance.

So the villain, the regulation, and this product all live on the same act: **screening resumes.** That convergence is the whole pitch — we're not importing an external concern, we're instrumenting the exact thing that already burned a FAANG and is already regulated.

## 3. What this is — and deliberately is NOT

**NOT:** "another AI resume screener." That category is crowded and distrusted, and building it makes *you* look like you built Amazon's tool. We do build a baseline screener — but only as the **subject under test**, never as the headline.

**IS:** the **fidelity + audit layer** that wraps any screener and certifies it per-req. The product surface a buyer cares about is the dashboard that says: *κ = 0.71 agreement with your recruiters · passes four-fifths on all measured groups · here's the evidence.* Judgment — knowing the screener is the dangerous part — is the brand.

## 4. The three layers

```
┌─────────────────────────────────────────────────────────┐
│  LAYER 3 — AUDIT          four-fifths / disparate impact │
│                           counterfactual probe           │
├─────────────────────────────────────────────────────────┤
│  LAYER 2 — FIDELITY       blind A/B vs recruiters (κ)     │  ← concord, reused
├─────────────────────────────────────────────────────────┤
│  LAYER 1 — SCREENER       scores resumes vs a job req     │  ← subject under test
│            (the thing everyone else sells)               │
└─────────────────────────────────────────────────────────┘
```

**Layer 1 — the screener (subject under test).** An LLM scores each synthetic resume against a job req: `advance / borderline / reject`, with a short rationale. Intentionally simple and honest about being the thing we're auditing — not the product. Decided:
- **Off-the-shelf, boring on purpose.** Primary = **LLM-rubric scoring** (where frontier shops are heading; its fidelity is genuinely in question). Secondary = an **embedding-similarity baseline** (sentence-transformers / BM25 cosine) so we can show the audit is model-agnostic — works on more than one screener.
- **Hard rule: never learn from past hires/winners.** A ranker fit to historical outcomes *is* Amazon. The screener scores against a **fixed, req-derived rubric**, not a learned model. This choice is a stated judgment signal.
- **Candidates only** — not current-employee valuation (that's trait-rating / succession planning = the rejected v1 framing for the wrong audience).
- **Rank for job-fit, not soft traits.** Score on concrete req-derived criteria (required skills, experience depth, domain match). Do NOT infer "leadership" etc. from a resume (weakly grounded, a bias vector, drifts back to trait-rating). Exception: a trait scores only if the **req explicitly names it** AND it's **grounded in cited resume evidence** (the groundedness gate) — inferred vibes are rejected.

**Layer 2 — fidelity (concord, reused ~wholesale).** Blind A/B: show a recruiter two resumes for the *same* req, ask "which advances?" Collect verdicts; randomize left/right to control position bias; seed gold pairs as attention checks. Measure **AI–recruiter agreement (Cohen's κ)** against the **recruiter–recruiter baseline (Fleiss' κ)**. The claim is never "the AI is right" — it's "the AI votes like a calibrated recruiter ⇔ κ_AI-human ≈ κ_human-human." A resume pair is structurally identical to concord's code pair, so the harness ports directly.

**Layer 3 — audit.** Two mechanisms:
- **Counterfactual probe:** generate matched resume pairs identical in skills/experience but differing only on a protected proxy (name, gender signal, college, graduation gap). Measure score drift. Zero drift is the target; drift is the Amazon failure, made visible.
- **Disparate impact:** across synthetic protected groups, compute selection-rate ratios and flag any below 0.8 (four-fifths rule, LL144-style report).

## 5. The demo (the recruiter walkthrough = the Amazon redemption arc)

This is the thing you show in an interview, end to end:

1. **Here is a raw AI screener** ranking a batch of synthetic resumes for a real-ish req. Looks great. Fast.
2. **Watch it fail the audit.** Run the counterfactual probe → identical resumes score differently when the name flips. Run disparate impact → one group lands at 0.68, below four-fifths. *This is Amazon, reproduced live.*
3. **Now gate it.** The same screener, wrapped: it ships its ranking for a req **only when** κ-agreement with recruiters clears a threshold **and** it passes four-fifths. The failing config is blocked, with the evidence shown.

The arc — *naive → caught → calibrated* — is the proof of judgment. Build for this walkthrough.

## 6. What's reused from concord vs. what's new

| Reused from concord (~60%) | New for hr-fidelity |
|---|---|
| Blind A/B pairing + position-bias flip | Resume domain model + synthetic generator |
| Gold attention-check pairs | Layer-1 screener prompt + scoring |
| κ computation (Cohen's / Fleiss') | Counterfactual probe (matched-pair generator) |
| FastAPI + SQLite voting backend shape | Four-fifths / disparate-impact report |
| Voting UI state machine + design system | Buyer-facing **certification dashboard** (the new UX centerpiece) |
| "Observed vs derived" eval-store split | Per-req certification artifact (signed, recomputable) |

## 7. Synthetic data — the credibility load-bearing piece

**Full build-ready spec: `docs/data-generator.md` (M1).** Summary: the whole thing is worthless if the resumes look fake. The synthetic set must be defensible:
- **Generated, never real** (privacy + the no-employer-artifacts rule — hard constraint).
- A handful of job reqs across archetypes (ML researcher, backend, infra, recruiter).
- For each req: a spread of resumes at varying real fit, with **skills/experience as the only legitimate signal.**
- **Counterfactual matched pairs:** programmatically hold the substantive content fixed and vary only the protected proxy, so any score drift is unambiguously attributable. This generator is the heart of Layer 3 and must be airtight.
- Document generation method in the repo so a reviewer can see it's honest synthetic, not scraped.

## 8. Screen-by-screen UX (the part recruiters judge)

1. **Landing / explainer** — must be self-explanatory to a recruiter who lands cold and won't read a README (~15s). Amazon story → fidelity thesis → "trust layer, not a screener," with the **synthetic-data disclosure up front** (honesty = the product's point). Full copy in `docs/landing-content.md`. Reuse concord's editorial design system (palette `#111110` / `#1c1c1a` / amber `#d4a853`) for visual continuity across the portfolio.
2. **Req setup** — pick a job req, load its resume batch, see the raw screener's ranking.
3. **Screening config (the interactive heart of the demo)** — the knobs a real ATS exposes, but curated so each one tells a bias story, in two stages: **filter/retrieval (hard gates)** — education filter, min years / "no employment gaps", keyword include/exclude (the literal Amazon mechanism); **rank/score (soft)** — skills-vs-experience-vs-education weighting, screener model + advance/reject threshold. Each knob is annotated with its bias risk and **live-updates the audit panel** — tune one and watch four-fifths flip CERTIFIED→BLOCKED. The config + dashboard form one tune→re-audit→verdict loop. Discipline: keep the knob set small; this demonstrates the audit responding to config, it is NOT a production ATS.
4. **Recruiter calibration (the A/B voting screen)** — concord's voting UI, re-skinned for resume pairs: two candidates, "which advances?", running κ.
5. **Certification dashboard (the centerpiece)** — per req: κ-agreement gauge vs human baseline, four-fifths bar chart per group, counterfactual drift table, a single **CERTIFIED / BLOCKED** verdict with the evidence behind it. This is the screen that gets you hired.
6. **Audit report export** — a clean, LL144-shaped PDF/printable a compliance person could actually hand to a regulator.

### Inline education (microcopy) — the tool teaches as it's used

The recruiter is neither a lawyer nor a statistician, so every knob states its risk *in context* and the calibration step explains *why* a human vote is needed. Default = one inline line; expandable to the fuller "why + legal cite" (progressive disclosure — never wall-of-text).

Per-knob risk lines:
- **Education filter** — ⚠ Degree/school-prestige filters are the most common source of disparate impact; EEOC treats a blanket degree requirement as suspect unless truly job-related.
- **Min years / "no employment gaps"** — ⚠ Experience floors and gap penalties act as age and caregiving (gender) proxies; "no gaps" penalizes parental leave.
- **Keyword include/exclude** — ⚠ The exact mechanism that sank Amazon's tool ("women's"). Keyword rules silently encode bias.
- **Skills weighting** — ✓ (affirming, not a warning) Job-relevant skills are the legitimate signal; weighting them is defensible and usually *improves* fairness — what the audit rewards.
- **Screener model + threshold** — changes who makes the cut; the audit re-runs on whatever you pick (model-agnostic by design).

Why the A/B test (inline at calibration): *"Is the AI right?" has no answer without a human ground truth — there's no "correct" hire on file. Recruiters judge the same candidates blind + head-to-head; we measure whether the AI agrees with them (Cohen's κ) as much as they agree with each other (human baseline). High agreement = votes like a calibrated recruiter; low = confidently doing its own thing, don't trust at scale.*

**Escalation:** the per-knob note starts gray (⚠ risk). The moment a knob breaks four-fifths, that same note turns **red and specific** — "this setting just dropped [group] to 0.71." The abstract warning becomes a live consequence the recruiter *caused*. That escalation is what turns the demo from a dashboard into a lesson. Full copy in `docs/landing-content.md` companion + to be drafted in a `docs/microcopy.md` when built.

## 9. Tech stack (match concord so reuse is real)

- Backend: FastAPI + SQLite, `python -m hrfidelity serve` pattern.
- Frontend: self-contained HTML/JS with the concord design system (no framework needed for the voting + dashboard; consider one only if the dashboard demands it).
- Scoring/judge: Anthropic SDK (Opus for the screener + judge). **Do not incur API spend without explicit OK each time.**
- Eval store: observed (resumes, reqs, votes) vs derived (κ, impact ratios) split, every score recomputable — carry concord's strongest senior-signal forward.

## 10. Scope, milestones, non-goals

**M0 — planning (DONE):** this doc + `landing-content.md` + `data-generator.md`. Story, positioning, architecture, UX flow, microcopy, and the data spec are all captured.
**M1 — data foundation (START HERE):** build per `docs/data-generator.md` — schema, req fixtures, name/demographic loaders, résumé generator with identity⊥fit independence, and the **counterfactual pair generator + hash invariant** (TDD the invariant first). Gate: invariant + independence checks green before anything else.
**M2 — the failing demo:** Layer 1 screener + Layer 3 audit (four-fifths + counterfactual drift), showing the screener *failing* on a rigged config. (Half the demo's power is the failure.)
**M3 — the fidelity layer:** port concord's A/B + κ onto resume pairs; collect a small calibration set.
**M4 — the certification dashboard + landing + config knobs + microcopy:** the UX centerpiece + the explainer (`landing-content.md`) + the live tune→re-audit→verdict loop with inline education. The recruiter-facing payload.

**Non-goals:** real candidate data (ever); "culture fit" scoring (a discrimination vector — naming it as out-of-scope is itself a judgment signal); being a production ATS; trait-rating / succession planning (wrong audience — that was the rejected v1 framing).

---

*Companion: concord (the A/B calibration method this reuses) ships as a methods appendix in `agentic-test-eval`. This repo is the recruiter-facing showcase: full-stack + UX + responsible-AI on the exact task Amazon failed.*
