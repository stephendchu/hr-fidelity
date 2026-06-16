"""
Error-taxonomy observability demo — proves Phoenix captures *distinct* failure
modes of the LLM screener, each with debuggable detail.

Happy-path tracing is easy. The real test of an observability layer is whether
different failures surface with enough context to tell them apart and debug them.
This script triggers one real error per category and lets it propagate through
`score()`, which runs inside our `screen_resume` span — so OpenTelemetry records
the exception and sets span status ERROR on both the screener span and the nested
LLM span.

Every error below is a genuine Anthropic API rejection (real `request_id`,
returned before any generation, so $0). Each category is triggered by patching
one screener knob, then restored.

    ANTHROPIC_API_KEY=... .venv/bin/python scripts/llm_error_demo.py

Produces, in the 'hr-fidelity-errors' Phoenix project, one red trace per category:
    screen_resume (ERROR) -> messages.create (ERROR, exception recorded)
"""
from __future__ import annotations

import sys
from contextlib import contextmanager

from hrfidelity.data.corpus_generator import generate_corpus
from hrfidelity.screener import llm_screener
from hrfidelity.server import app as app_module
from hrfidelity.tracing import setup_tracing

PROJECT = "hr-fidelity-errors"


@contextmanager
def patched(**attrs):
    """Temporarily set module attributes on llm_screener, then restore."""
    saved = {k: getattr(llm_screener, k) for k in attrs}
    try:
        for k, v in attrs.items():
            setattr(llm_screener, k, v)
        yield
    finally:
        for k, v in saved.items():
            setattr(llm_screener, k, v)


def _bad_key_client():
    import anthropic
    return anthropic.Anthropic(api_key="sk-ant-invalid-key-for-demo-000")


# Each category: (label, why it happens in production, context-manager of patches)
CATEGORIES = [
    (
        "model_not_found (404)",
        "wrong/retired model id in config",
        dict(_MODEL="claude-nonexistent-model-v0"),
    ),
    (
        "authentication_error (401)",
        "expired or wrong API key",
        dict(_get_client=_bad_key_client),
    ),
    (
        "invalid_request (400)",
        "malformed request — empty prompt content",
        dict(_build_prompt=lambda resume, req: ""),
    ),
]


def main() -> int:
    traced = setup_tracing(PROJECT)
    print(f"tracing active: {traced}  ->  Phoenix project '{PROJECT}'\n")

    req = app_module._load_reqs()[0]
    resumes, _ = generate_corpus([req], n_per_fit=50, seed=44)
    resume = resumes[0]

    captured = 0
    for label, why, patches in CATEGORIES:
        with patched(**patches):
            try:
                llm_screener.score(resume, req)
            except Exception as exc:  # noqa: BLE001 — expected; the span records it
                captured += 1
                etype = type(exc).__name__
                msg = str(exc).replace("\n", " ")[:90]
                print(f"  [{label}]  ({why})")
                print(f"      -> {etype}: {msg}\n")

    try:
        from opentelemetry import trace as otel_trace
        otel_trace.get_tracer_provider().force_flush()
        print("traces flushed to Phoenix")
    except Exception as e:  # noqa: BLE001
        print(f"trace flush skipped: {e}")

    print(f"\n{captured} distinct ERROR categories sent to project '{PROJECT}'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
