"""
TDD: counterfactual pair generator — hash invariant.

The invariant (§5 of data-generator.md):
  content_hash(twin) == content_hash(base)   <- job-relevant content is identical
  proxy_field(twin)  != proxy_field(base)    <- one proxy changed

Any score delta between twins is attributable to the proxy alone.
That's what makes the bias measurement defensible.
"""
import pytest

from hrfidelity.data.schema import Education, Experience, Identity, Resume
from hrfidelity.data.counterfactual import content_hash, generate_counterfactual


def _base_resume() -> Resume:
    """Canonical base using Bertrand-Mullainathan names (the paper we cite)."""
    return Resume(
        candidate_id="test-001",
        identity=Identity(
            first_name="James",
            last_name="Walsh",
            inferred_gender="M",
            inferred_race_proxy="white",
            source="bertrand_mullainathan",
        ),
        education=[Education(
            degree="BS",
            field="Computer Science",
            institution="State University",
            prestige_tier=2,
            grad_year=2018,
        )],
        experience=[Experience(
            title="Software Engineer",
            company="Acme Corp",
            start="2018-06",
            end="2022-06",
            bullets=["Built REST APIs", "Led migration to microservices"],
        )],
        skills=["Python", "SQL", "Docker"],
        certifications=["AWS Solutions Architect"],
        latent_fit="strong",
    )


# ---------------------------------------------------------------------------
# content_hash contract
# ---------------------------------------------------------------------------

class TestContentHash:
    def test_same_resume_same_hash(self):
        r = _base_resume()
        assert content_hash(r) == content_hash(r)

    def test_skill_change_changes_hash(self):
        r = _base_resume()
        r2 = _base_resume()
        r2.skills = ["Python", "SQL", "Kubernetes"]
        assert content_hash(r) != content_hash(r2)

    def test_identity_change_does_not_change_hash(self):
        """Identity is the proxy axis — must not bleed into content_hash."""
        r = _base_resume()
        r2 = _base_resume()
        r2.identity.first_name = "Jennifer"
        r2.identity.inferred_gender = "F"
        assert content_hash(r) == content_hash(r2)

    def test_prestige_tier_change_does_not_change_hash(self):
        """prestige_tier is a proxy axis — excluded from content_hash."""
        r = _base_resume()
        r2 = _base_resume()
        r2.education[0].prestige_tier = 1
        assert content_hash(r) == content_hash(r2)


# ---------------------------------------------------------------------------
# The invariant itself — parametrised across all axes
# ---------------------------------------------------------------------------

class TestCounterfactualInvariant:

    @pytest.mark.parametrize("axis", ["gender", "race_proxy", "prestige_tier"])
    def test_twins_share_content_hash(self, axis):
        base = _base_resume()
        twin = generate_counterfactual(base, axis=axis)
        assert content_hash(twin) == content_hash(base), (
            f"axis={axis!r}: twin content_hash must equal base — "
            "job-relevant content must not change between twins"
        )

    # --- gender axis ---

    def test_gender_twin_first_name_differs(self):
        base = _base_resume()
        twin = generate_counterfactual(base, axis="gender")
        assert twin.identity.first_name != base.identity.first_name

    def test_gender_twin_inferred_gender_differs(self):
        base = _base_resume()
        twin = generate_counterfactual(base, axis="gender")
        assert twin.identity.inferred_gender != base.identity.inferred_gender

    def test_gender_twin_last_name_unchanged(self):
        base = _base_resume()
        twin = generate_counterfactual(base, axis="gender")
        assert twin.identity.last_name == base.identity.last_name

    # --- race_proxy axis ---

    def test_race_proxy_twin_last_name_differs(self):
        base = _base_resume()
        twin = generate_counterfactual(base, axis="race_proxy")
        assert twin.identity.last_name != base.identity.last_name

    def test_race_proxy_twin_race_proxy_differs(self):
        base = _base_resume()
        twin = generate_counterfactual(base, axis="race_proxy")
        assert twin.identity.inferred_race_proxy != base.identity.inferred_race_proxy

    def test_race_proxy_twin_first_name_unchanged(self):
        base = _base_resume()
        twin = generate_counterfactual(base, axis="race_proxy")
        assert twin.identity.first_name == base.identity.first_name

    # --- prestige_tier axis ---

    def test_prestige_tier_twin_tier_differs(self):
        base = _base_resume()
        twin = generate_counterfactual(base, axis="prestige_tier")
        assert twin.education[0].prestige_tier != base.education[0].prestige_tier

    def test_prestige_tier_twin_skills_unchanged(self):
        base = _base_resume()
        twin = generate_counterfactual(base, axis="prestige_tier")
        assert twin.skills == base.skills

    # --- housekeeping ---

    def test_twin_gets_unique_candidate_id(self):
        base = _base_resume()
        twin = generate_counterfactual(base, axis="gender")
        assert twin.candidate_id != base.candidate_id

    def test_twin_latent_fit_preserved(self):
        """Ground-truth quality must be identical — only the proxy changed."""
        base = _base_resume()
        twin = generate_counterfactual(base, axis="gender")
        assert twin.latent_fit == base.latent_fit

    def test_base_resume_not_mutated(self):
        """generate_counterfactual must not modify the base in place."""
        base = _base_resume()
        original_name = base.identity.first_name
        generate_counterfactual(base, axis="gender")
        assert base.identity.first_name == original_name
