"""
TDD: Layer 2 — fidelity / κ calibration (M3).

Instruments:
  - generate_fidelity_pairs: A/B pairs (same req, different fit) with
      position randomization and gold attention-check pairs.
  - ai_vote: screener-derived vote on a pair (higher score wins).
  - simulate_human_vote: synthetic recruiter with configurable agreement rate.
  - cohen_kappa: agreement between two judges on the same pairs.
  - fleiss_kappa: multi-rater agreement across N human judges.
  - run_calibration: produces a FidelityReport with CERTIFIED / needs-work verdict.

Claim: κ_AI-human ≈ κ_human-human ⟹ AI votes like a calibrated recruiter.
"""
import pathlib
import random
import pytest

from hrfidelity.data.corpus_generator import generate_corpus
from hrfidelity.data.req_loader import load_req
from hrfidelity.fidelity.calibration import ai_vote, run_calibration, simulate_human_vote
from hrfidelity.fidelity.kappa import cohen_kappa, fleiss_kappa
from hrfidelity.fidelity.pairs import generate_fidelity_pairs
from hrfidelity.fidelity.protocol import FidelityPair, FidelityReport, KappaResult, Vote
from hrfidelity.screener.protocol import ScreenerConfig

DATA_REQS = pathlib.Path(__file__).parent.parent / "data" / "reqs"


@pytest.fixture(scope="module")
def req():
    return load_req(DATA_REQS / "backend-eng.json")


@pytest.fixture(scope="module")
def resumes(req):
    rs, _ = generate_corpus([req], n_per_fit=20, seed=42)
    return rs


@pytest.fixture(scope="module")
def pairs(resumes, req):
    return generate_fidelity_pairs(resumes, req, n_pairs=24, seed=42)


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------

class TestGenerateFidelityPairs:
    def test_returns_list_of_fidelity_pairs(self, pairs):
        assert all(isinstance(p, FidelityPair) for p in pairs)

    def test_returns_requested_count(self, resumes, req):
        result = generate_fidelity_pairs(resumes, req, n_pairs=10, seed=0)
        assert len(result) == 10

    def test_pairs_belong_to_same_req(self, pairs, req):
        for p in pairs:
            assert p.req_id == req.id

    def test_left_and_right_are_different_resumes(self, pairs):
        for p in pairs:
            assert p.left.candidate_id != p.right.candidate_id

    def test_gold_pairs_have_winner(self, pairs):
        gold = [p for p in pairs if p.is_gold]
        assert gold, "no gold pairs generated"
        for p in gold:
            assert p.gold_winner in ("left", "right")

    def test_non_gold_pairs_have_no_winner(self, pairs):
        non_gold = [p for p in pairs if not p.is_gold]
        for p in non_gold:
            assert p.gold_winner is None

    def test_position_varies_in_gold_pairs(self, resumes, req):
        # Over many gold pairs, strong resume should appear on both sides.
        result = generate_fidelity_pairs(resumes, req, n_pairs=20, seed=0)
        gold = [p for p in result if p.is_gold]
        lefts = sum(1 for p in gold if p.gold_winner == "left")
        rights = sum(1 for p in gold if p.gold_winner == "right")
        assert lefts > 0 and rights > 0, "position not randomized across gold pairs"

    def test_gold_winner_is_strong_resume(self, pairs):
        for p in pairs:
            if not p.is_gold:
                continue
            winner = p.left if p.gold_winner == "left" else p.right
            assert winner.latent_fit == "strong", (
                f"gold winner has latent_fit={winner.latent_fit!r}, expected 'strong'"
            )

    def test_pair_ids_are_unique(self, pairs):
        ids = [p.pair_id for p in pairs]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# AI vote
# ---------------------------------------------------------------------------

class TestAiVote:
    def test_returns_vote_instance(self, pairs, req):
        v = ai_vote(pairs[0], req)
        assert isinstance(v, Vote)

    def test_vote_pair_id_matches(self, pairs, req):
        p = pairs[0]
        v = ai_vote(p, req)
        assert v.pair_id == p.pair_id

    def test_judge_id_is_ai(self, pairs, req):
        v = ai_vote(pairs[0], req)
        assert v.judge_id == "ai"

    def test_choice_is_valid(self, pairs, req):
        for p in pairs:
            v = ai_vote(p, req)
            assert v.choice in ("left", "right", "tie")

    def test_ai_picks_strong_over_weak_in_gold_pairs(self, pairs, req):
        gold = [p for p in pairs if p.is_gold]
        assert gold
        wrong = [p for p in gold if ai_vote(p, req).choice != p.gold_winner]
        # With fair screener, AI should get every gold pair correct.
        assert not wrong, f"AI got {len(wrong)}/{len(gold)} gold pairs wrong"


# ---------------------------------------------------------------------------
# Simulate human vote
# ---------------------------------------------------------------------------

class TestSimulateHumanVote:
    def test_returns_vote_instance(self, pairs, req):
        rng = random.Random(0)
        v = simulate_human_vote(pairs[0], req, "human-1", rng=rng)
        assert isinstance(v, Vote)

    def test_judge_id_preserved(self, pairs, req):
        rng = random.Random(0)
        v = simulate_human_vote(pairs[0], req, "recruiter-A", rng=rng)
        assert v.judge_id == "recruiter-A"

    def test_high_agreement_rate_mostly_correct_on_gold(self, pairs, req):
        gold = [p for p in pairs if p.is_gold]
        rng = random.Random(0)
        correct = sum(
            1 for p in gold
            if simulate_human_vote(p, req, "h", agreement_rate=0.95, rng=rng).choice == p.gold_winner
        )
        assert correct / len(gold) >= 0.80, "agreement_rate=0.95 should yield ≥80% correct on gold"

    def test_random_agent_near_chance_on_gold(self, pairs, req):
        gold = [p for p in pairs if p.is_gold]
        rng = random.Random(0)
        correct = sum(
            1 for p in gold
            if simulate_human_vote(p, req, "h", agreement_rate=0.0, rng=rng).choice == p.gold_winner
        )
        # Pure random from 3 choices → ~33% chance of matching gold_winner
        assert correct / len(gold) <= 0.70, "zero-agreement agent should rarely match gold"


# ---------------------------------------------------------------------------
# Cohen's κ
# ---------------------------------------------------------------------------

class TestCohenKappa:
    def _make_votes(self, judge, choices):
        return [Vote(f"p{i}", judge, c) for i, c in enumerate(choices)]

    def test_returns_kappa_result(self):
        a = self._make_votes("a", ["left", "right"])
        b = self._make_votes("b", ["left", "right"])
        assert isinstance(cohen_kappa(a, b), KappaResult)

    def test_perfect_agreement_is_one(self):
        choices = ["left", "left", "right", "right"]
        a = self._make_votes("a", choices)
        b = self._make_votes("b", choices)
        result = cohen_kappa(a, b)
        assert abs(result.kappa - 1.0) < 1e-9

    def test_chance_agreement_is_zero(self):
        # A: L L R R, B: L R L R → p_o=0.5, p_e=0.5*0.5+0.5*0.5=0.5 → κ=0
        a = self._make_votes("a", ["left", "left", "right", "right"])
        b = self._make_votes("b", ["left", "right", "left", "right"])
        result = cohen_kappa(a, b)
        assert abs(result.kappa) < 1e-9

    def test_negative_agreement(self):
        # A: L L R R, B: R R L L → p_o=0, p_e=0.5 → κ=-1
        a = self._make_votes("a", ["left", "left", "right", "right"])
        b = self._make_votes("b", ["right", "right", "left", "left"])
        result = cohen_kappa(a, b)
        assert result.kappa < 0

    def test_judge_ids_captured(self):
        a = self._make_votes("ai", ["left"])
        b = self._make_votes("human-1", ["left"])
        result = cohen_kappa(a, b)
        assert result.judge_a == "ai"
        assert result.judge_b == "human-1"

    def test_observed_agreement_correct(self):
        a = self._make_votes("a", ["left", "left", "right", "right"])
        b = self._make_votes("b", ["left", "left", "right", "right"])
        result = cohen_kappa(a, b)
        assert abs(result.observed_agreement - 1.0) < 1e-9

    def test_n_pairs_correct(self):
        a = self._make_votes("a", ["left", "right", "tie"])
        b = self._make_votes("b", ["left", "right", "tie"])
        result = cohen_kappa(a, b)
        assert result.n_pairs == 3


# ---------------------------------------------------------------------------
# Fleiss' κ
# ---------------------------------------------------------------------------

class TestFleissKappa:
    def _make_votes(self, judge, choices):
        return [Vote(f"p{i}", judge, c) for i, c in enumerate(choices)]

    def test_returns_float(self):
        v1 = self._make_votes("1", ["left", "right"])
        v2 = self._make_votes("2", ["left", "right"])
        v3 = self._make_votes("3", ["left", "right"])
        result = fleiss_kappa([v1, v2, v3])
        assert isinstance(result, float)

    def test_perfect_agreement_is_one(self):
        choices = ["left", "right", "left", "right"]
        judges = [self._make_votes(str(j), choices) for j in range(4)]
        result = fleiss_kappa(judges)
        assert abs(result - 1.0) < 1e-9

    def test_result_is_between_neg_one_and_one(self):
        import random as _rnd
        rng = _rnd.Random(42)
        cats = ["left", "right", "tie"]
        votes = [[Vote(f"p{i}", str(j), rng.choice(cats)) for i in range(20)] for j in range(5)]
        result = fleiss_kappa(votes)
        assert -1.0 <= result <= 1.0

    def test_single_rater_returns_zero(self):
        v = self._make_votes("1", ["left", "right"])
        assert fleiss_kappa([v]) == 0.0


# ---------------------------------------------------------------------------
# Full calibration
# ---------------------------------------------------------------------------

class TestRunCalibration:
    @pytest.fixture
    def human_votes(self, pairs, req):
        rng = random.Random(1)
        return {
            "human-1": [simulate_human_vote(p, req, "human-1", agreement_rate=0.85, rng=rng) for p in pairs],
            "human-2": [simulate_human_vote(p, req, "human-2", agreement_rate=0.85, rng=rng) for p in pairs],
            "human-3": [simulate_human_vote(p, req, "human-3", agreement_rate=0.85, rng=rng) for p in pairs],
        }

    def test_returns_fidelity_report(self, pairs, req, human_votes):
        report = run_calibration(pairs, req, human_votes)
        assert isinstance(report, FidelityReport)

    def test_req_id_preserved(self, pairs, req, human_votes):
        report = run_calibration(pairs, req, human_votes)
        assert report.req_id == req.id

    def test_n_pairs_correct(self, pairs, req, human_votes):
        report = run_calibration(pairs, req, human_votes)
        assert report.n_pairs == len(pairs)

    def test_kappa_ai_human_has_one_entry_per_judge(self, pairs, req, human_votes):
        report = run_calibration(pairs, req, human_votes)
        assert len(report.kappa_ai_human) == len(human_votes)

    def test_kappa_results_are_kappa_result_instances(self, pairs, req, human_votes):
        report = run_calibration(pairs, req, human_votes)
        assert all(isinstance(kr, KappaResult) for kr in report.kappa_ai_human)

    def test_high_agreement_screener_passes_threshold(self, pairs, req, human_votes):
        # With fair screener + 0.85 human agreement, κ should exceed 0.4.
        report = run_calibration(pairs, req, human_votes, kappa_threshold=0.4)
        assert report.passes, f"mean κ = {report.mean_kappa_ai_human:.3f}, expected ≥ 0.4"

    def test_gold_pass_rate_between_0_and_1(self, pairs, req, human_votes):
        report = run_calibration(pairs, req, human_votes)
        assert 0.0 <= report.gold_pass_rate <= 1.0

    def test_ai_gets_high_gold_pass_rate(self, pairs, req, human_votes):
        report = run_calibration(pairs, req, human_votes)
        # Fair screener should get all gold pairs correct.
        assert report.gold_pass_rate == 1.0, f"gold_pass_rate={report.gold_pass_rate}"

    def test_mean_kappa_ai_human_is_average(self, pairs, req, human_votes):
        report = run_calibration(pairs, req, human_votes)
        expected = sum(kr.kappa for kr in report.kappa_ai_human) / len(report.kappa_ai_human)
        assert abs(report.mean_kappa_ai_human - expected) < 1e-9

    def test_kappa_human_human_is_fleiss_kappa(self, pairs, req, human_votes):
        from hrfidelity.fidelity.kappa import fleiss_kappa as fk
        report = run_calibration(pairs, req, human_votes)
        expected = fk(list(human_votes.values()))
        assert abs(report.kappa_human_human - expected) < 1e-9
