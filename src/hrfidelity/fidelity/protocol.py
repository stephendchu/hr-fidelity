from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class FidelityPair:
    pair_id: str
    req_id: str
    left: object   # Resume — typed as object to avoid circular at protocol level
    right: object
    is_gold: bool
    gold_winner: Literal["left", "right"] | None  # None for non-gold pairs


@dataclass
class Vote:
    pair_id: str
    judge_id: str
    choice: Literal["left", "right", "tie"]


@dataclass
class KappaResult:
    kappa: float
    judge_a: str
    judge_b: str
    n_pairs: int
    observed_agreement: float
    expected_agreement: float


@dataclass
class FidelityReport:
    req_id: str
    kappa_ai_human: list[KappaResult]
    kappa_human_human: float
    mean_kappa_ai_human: float
    n_pairs: int
    n_gold: int
    gold_pass_rate: float
    passes: bool  # mean_kappa_ai_human >= threshold
