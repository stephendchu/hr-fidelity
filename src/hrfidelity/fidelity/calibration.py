"""
Fidelity calibration session.

ai_vote            — AI's verdict on an A/B pair via the rubric screener.
simulate_human_vote — synthetic recruiter with configurable agreement rate
                       (demo only; production would collect real votes via the UI).
run_calibration    — compute κ_AI-human and κ_human-human and return a FidelityReport.
"""
from __future__ import annotations

import random

from hrfidelity.data.req_loader import Req
from hrfidelity.fidelity.kappa import cohen_kappa, fleiss_kappa
from hrfidelity.fidelity.protocol import FidelityPair, FidelityReport, KappaResult, Vote
from hrfidelity.screener import rubric_screener
from hrfidelity.screener.protocol import ScreenerConfig

_FAIR = ScreenerConfig()
_CHOICES = ("left", "right", "tie")


def ai_vote(
    pair: FidelityPair,
    req: Req,
    config: ScreenerConfig = _FAIR,
) -> Vote:
    """Score both resumes and vote for the higher-scoring side."""
    left_s = rubric_screener.score(pair.left, req, config)
    right_s = rubric_screener.score(pair.right, req, config)

    if left_s.raw_score > right_s.raw_score:
        choice = "left"
    elif right_s.raw_score > left_s.raw_score:
        choice = "right"
    else:
        choice = "tie"

    return Vote(pair_id=pair.pair_id, judge_id="ai", choice=choice)


def simulate_human_vote(
    pair: FidelityPair,
    req: Req,
    judge_id: str,
    *,
    agreement_rate: float = 0.80,
    rng: random.Random,
) -> Vote:
    """Return a synthetic human vote that agrees with the rubric *agreement_rate* of the time.

    Used for demo / testing only — production collects real recruiter votes via the UI.
    """
    # Determine rubric-grounded "correct" answer
    left_s = rubric_screener.score(pair.left, req, _FAIR)
    right_s = rubric_screener.score(pair.right, req, _FAIR)

    if left_s.raw_score > right_s.raw_score:
        correct = "left"
    elif right_s.raw_score > left_s.raw_score:
        correct = "right"
    else:
        correct = "tie"

    choice = correct if rng.random() < agreement_rate else rng.choice(_CHOICES)
    return Vote(pair_id=pair.pair_id, judge_id=judge_id, choice=choice)


def run_calibration(
    pairs: list[FidelityPair],
    req: Req,
    human_votes_by_judge: dict[str, list[Vote]],
    *,
    config: ScreenerConfig = _FAIR,
    kappa_threshold: float = 0.60,
) -> FidelityReport:
    """Run the calibration session and return a FidelityReport.

    Computes:
      - Cohen's κ between AI and each human judge
      - Fleiss' κ across all human judges (human-human baseline)
      - Gold pair pass rate (AI attention-check accuracy)
    """
    ai_votes = [ai_vote(p, req, config) for p in pairs]
    ai_vote_map = {v.pair_id: v.choice for v in ai_votes}

    # Cohen's κ — AI vs each human judge
    kappa_results: list[KappaResult] = [
        cohen_kappa(ai_votes, votes)
        for votes in human_votes_by_judge.values()
    ]

    mean_kappa = (
        sum(kr.kappa for kr in kappa_results) / len(kappa_results)
        if kappa_results else 0.0
    )

    # Fleiss' κ — human-human baseline
    kappa_hh = (
        fleiss_kappa(list(human_votes_by_judge.values()))
        if len(human_votes_by_judge) >= 2 else 0.0
    )

    # Gold pair accuracy
    gold_pairs = [p for p in pairs if p.is_gold]
    n_gold_correct = sum(
        1 for p in gold_pairs
        if p.gold_winner and ai_vote_map.get(p.pair_id) == p.gold_winner
    )

    return FidelityReport(
        req_id=req.id,
        kappa_ai_human=kappa_results,
        kappa_human_human=kappa_hh,
        mean_kappa_ai_human=mean_kappa,
        n_pairs=len(pairs),
        n_gold=len(gold_pairs),
        gold_pass_rate=n_gold_correct / len(gold_pairs) if gold_pairs else 0.0,
        passes=mean_kappa >= kappa_threshold,
    )
