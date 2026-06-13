from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ScreenerVerdict = Literal["advance", "borderline", "reject"]


@dataclass
class Score:
    candidate_id: str
    req_id: str
    verdict: ScreenerVerdict
    rationale: str
    raw_score: float  # 0.0–1.0


@dataclass
class ScreenerConfig:
    # Legitimate scoring knobs
    required_skill_weight: float = 0.6
    nice_to_have_weight: float = 0.2
    experience_weight: float = 0.2
    threshold_advance: float = 0.7
    threshold_borderline: float = 0.4

    # Bias knobs — zero is fair; nonzero recreates the Amazon failure mode.
    # prestige_bonus > 0 favors tier-1 institutions (EEOC suspect for disparate impact).
    # race_proxy_bias / gender_bias apply additive adjustments by identity proxy.
    prestige_bonus: float = 0.0
    race_proxy_bias: dict[str, float] = field(default_factory=dict)
    gender_bias: dict[str, float] = field(default_factory=dict)
