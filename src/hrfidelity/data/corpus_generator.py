"""
Corpus generator (§7 of data-generator.md).

Generates a full set of synthetic résumés + counterfactual pairs, saves them
to the eval-store layout, and writes a manifest recording provenance.

Observed (résumés, reqs, pairs) and derived (scores, κ, ratios) are kept in
separate stores so every metric is recomputable from inputs × versioned scorer.

Usage:
    python -m hrfidelity.data.corpus_generator          # default seed/n
    python -m hrfidelity.data.corpus_generator --seed 42 --n 20
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
import random
from datetime import datetime, timezone

from hrfidelity.data.ci_checks import (
    check_independence,
    check_invariant,
    check_realism,
    check_reproducibility,
)
from hrfidelity.data.counterfactual import generate_counterfactual
from hrfidelity.data.req_loader import Req, load_req
from hrfidelity.data.resume_generator import generate_resume
from hrfidelity.data.schema import CounterfactualPair, Education, Experience, Identity, Resume

_DATA_ROOT = pathlib.Path(__file__).parents[3] / "data"
_AXES = ["gender", "race_proxy", "prestige_tier"]


# ---------------------------------------------------------------------------
# JSON serialisation / deserialisation
# ---------------------------------------------------------------------------

def _resume_to_dict(r: Resume) -> dict:
    return dataclasses.asdict(r)


def _resume_from_dict(d: dict) -> Resume:
    return Resume(
        candidate_id=d["candidate_id"],
        identity=Identity(**d["identity"]),
        education=[Education(**e) for e in d["education"]],
        experience=[
            Experience(
                title=e["title"],
                company=e["company"],
                start=e["start"],
                end=e["end"],
                bullets=e["bullets"],
            )
            for e in d["experience"]
        ],
        skills=d["skills"],
        certifications=d["certifications"],
        latent_fit=d["latent_fit"],
    )


def _pair_to_dict(p: CounterfactualPair) -> dict:
    return {
        "base_id": p.base.candidate_id,
        "twin_id": p.twin.candidate_id,
        "axis": p.axis,
    }


# ---------------------------------------------------------------------------
# In-memory generation
# ---------------------------------------------------------------------------

def generate_corpus(
    reqs: list[Req],
    *,
    n_per_fit: int,
    seed: int,
    axes: list[str] | None = None,
) -> tuple[list[Resume], list[CounterfactualPair]]:
    """Generate a corpus of résumés and counterfactual pairs.

    Every résumé uses swappable_identity=True so all three axes are
    guaranteed to succeed — no résumé will be missing a twin.
    """
    if axes is None:
        axes = _AXES

    rng = random.Random(seed)
    fits = ["strong", "medium", "weak"]
    resumes: list[Resume] = []
    pairs: list[CounterfactualPair] = []

    for req in reqs:
        for fit in fits:
            for _ in range(n_per_fit):
                resume = generate_resume(req, fit, rng, swappable_identity=True)
                resumes.append(resume)
                for axis in axes:
                    twin = generate_counterfactual(resume, axis=axis, rng=rng)
                    pairs.append(CounterfactualPair(base=resume, twin=twin, axis=axis))

    return resumes, pairs


# ---------------------------------------------------------------------------
# Disk I/O
# ---------------------------------------------------------------------------

def save_corpus(
    resumes: list[Resume],
    pairs: list[CounterfactualPair],
    *,
    output_dir: pathlib.Path,
    seed: int,
    n_per_fit: int,
    axes: list[str] | None = None,
) -> pathlib.Path:
    """Save résumés, pairs, and manifest to *output_dir*.

    Returns the path to the manifest file.
    """
    if axes is None:
        axes = _AXES

    resume_dir = output_dir / "resumes"
    pair_dir = output_dir / "counterfactual_pairs"
    resume_dir.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)

    # Build an ID → résumé index (pairs reference résumés by id)
    resume_index: dict[str, Resume] = {r.candidate_id: r for r in resumes}

    # Save base résumés and twin résumés (twins are also résumés, just with a
    # different identity; saving them separately keeps load_corpus self-contained)
    saved_ids: set[str] = set()
    for r in resumes:
        (resume_dir / f"{r.candidate_id}.json").write_text(
            json.dumps(_resume_to_dict(r), indent=2)
        )
        saved_ids.add(r.candidate_id)

    for p in pairs:
        if p.twin.candidate_id not in saved_ids:
            (resume_dir / f"{p.twin.candidate_id}.json").write_text(
                json.dumps(_resume_to_dict(p.twin), indent=2)
            )
            saved_ids.add(p.twin.candidate_id)

    for p in pairs:
        fname = f"{p.base.candidate_id}__{p.twin.candidate_id}__{p.axis}.json"
        (pair_dir / fname).write_text(
            json.dumps({"base_id": p.base.candidate_id, "twin_id": p.twin.candidate_id, "axis": p.axis}, indent=2)
        )

    # Run CI checks and embed results in manifest
    checks_run = {
        "invariant": check_invariant(pairs),
        "independence": check_independence(resumes),
        "realism": check_realism(resumes),
    }

    req_ids = sorted({r.skills[0] for r in resumes})  # proxy; manifest is illustrative
    # Better: derive req_ids from context — but we don't have req objects here
    # Use unique latent_fit + education combos as archetype proxy
    # Actually just note the counts; req_ids come from the caller
    req_ids_note = "see data/reqs/ for req fixtures"

    manifest = {
        "_generated_at": datetime.now(timezone.utc).isoformat(),
        "_generator": "hrfidelity v0.1.0",
        "seed": seed,
        "n_per_fit": n_per_fit,
        "req_ids": req_ids_note,
        "axes": axes,
        "counts": {
            "resumes": len(resumes),
            "pairs": len(pairs),
        },
        "checks": {
            name: {"passed": r.passed, "detail": r.detail}
            for name, r in checks_run.items()
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def load_corpus(
    input_dir: pathlib.Path,
) -> tuple[list[Resume], list[CounterfactualPair]]:
    """Load a corpus previously saved by save_corpus.

    Returns only the base résumés (not twins) in the first element, so
    callers get the same population size as was generated.
    """
    resume_dir = input_dir / "resumes"
    pair_dir = input_dir / "counterfactual_pairs"

    all_resumes = {
        _resume_from_dict(json.loads(p.read_text())).candidate_id:
        _resume_from_dict(json.loads(p.read_text()))
        for p in sorted(resume_dir.glob("*.json"))
    }

    pairs: list[CounterfactualPair] = []
    base_ids: list[str] = []
    for p in sorted(pair_dir.glob("*.json")):
        d = json.loads(p.read_text())
        if d["base_id"] not in all_resumes or d["twin_id"] not in all_resumes:
            continue
        pairs.append(CounterfactualPair(
            base=all_resumes[d["base_id"]],
            twin=all_resumes[d["twin_id"]],
            axis=d["axis"],
        ))
        if d["base_id"] not in base_ids:
            base_ids.append(d["base_id"])

    # Return only base résumés (twins are accessible via pairs)
    base_resumes = [all_resumes[rid] for rid in base_ids if rid in all_resumes]
    return base_resumes, pairs


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic résumé corpus")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=20, help="Résumés per fit level per req")
    parser.add_argument("--out", type=pathlib.Path, default=_DATA_ROOT)
    args = parser.parse_args()

    reqs = [load_req(p) for p in sorted((_DATA_ROOT / "reqs").glob("*.json"))]
    print(f"Generating corpus: {len(reqs)} reqs × 3 fit levels × {args.n} = {len(reqs)*3*args.n} résumés …")

    resumes, pairs = generate_corpus(reqs, n_per_fit=args.n, seed=args.seed)
    manifest_path = save_corpus(resumes, pairs, output_dir=args.out, seed=args.seed, n_per_fit=args.n)

    print(f"Done. {len(resumes)} résumés, {len(pairs)} pairs.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    _main()
