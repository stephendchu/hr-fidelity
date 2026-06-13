"""
Generate A/B fidelity pairs for recruiter calibration.

Each pair shows two resumes from the same req, left/right positions randomized
to control for position bias. Gold pairs (strong vs weak) serve as attention
checks with a known correct answer.
"""
from __future__ import annotations

import hashlib
import random

from hrfidelity.data.req_loader import Req
from hrfidelity.data.schema import Resume
from hrfidelity.fidelity.protocol import FidelityPair


def generate_fidelity_pairs(
    resumes: list[Resume],
    req: Req,
    *,
    n_pairs: int = 20,
    seed: int = 42,
) -> list[FidelityPair]:
    """Return n_pairs A/B pairs from *resumes*, split across fit levels.

    Gold pairs (strong vs weak) are seeded first for attention-check coverage;
    remaining slots fill with strong×medium and medium×weak.
    Position (left/right) is randomized to prevent position bias.
    """
    rng = random.Random(seed)

    strong = [r for r in resumes if r.latent_fit == "strong"]
    medium = [r for r in resumes if r.latent_fit == "medium"]
    weak = [r for r in resumes if r.latent_fit == "weak"]

    rng.shuffle(strong)
    rng.shuffle(medium)
    rng.shuffle(weak)

    # Use ceiling so we always generate enough candidates to fill n_pairs exactly.
    n_each = max(1, (n_pairs + 2) // 3)
    candidates: list[tuple[Resume, Resume, bool]] = []

    # Gold: strong vs weak — unambiguous correct answer
    for s, w in zip(strong[:n_each], weak[:n_each]):
        candidates.append((s, w, True))

    # Strong vs medium
    for s, m in zip(strong[n_each : n_each * 2], medium[:n_each]):
        candidates.append((s, m, False))

    # Medium vs weak
    for m, w in zip(medium[n_each : n_each * 2], weak[n_each : n_each * 2]):
        candidates.append((m, w, False))

    rng.shuffle(candidates)
    candidates = candidates[:n_pairs]

    pairs: list[FidelityPair] = []
    for stronger, weaker, is_gold in candidates:
        # Randomize which resume appears on the left
        if rng.choice([True, False]):
            left, right = stronger, weaker
            gold_winner: str | None = "left" if is_gold else None
        else:
            left, right = weaker, stronger
            gold_winner = "right" if is_gold else None

        pair_id = hashlib.sha256(
            f"{left.candidate_id}:{right.candidate_id}:{req.id}".encode()
        ).hexdigest()[:16]

        pairs.append(FidelityPair(
            pair_id=pair_id,
            req_id=req.id,
            left=left,
            right=right,
            is_gold=is_gold,
            gold_winner=gold_winner,
        ))

    return pairs
