"""
Phoenix (Arize) tracing setup for the LLM screener.

Auto-instruments all Anthropic client calls via openinference.
Supports local Phoenix (default) or Phoenix Cloud via env vars.

Env vars:
  PHOENIX_API_KEY            — Phoenix Cloud API key (uses cloud if set)
  PHOENIX_COLLECTOR_ENDPOINT — override collector URL (default: http://localhost:6006/v1/traces)
  HRFIDELITY_TRACING         — set to "0" or "false" to disable tracing entirely
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def setup_tracing(project_name: str = "hr-fidelity") -> bool:
    """
    Register Phoenix tracer and instrument Anthropic.
    Returns True if tracing was enabled, False if skipped or unavailable.
    """
    if os.environ.get("HRFIDELITY_TRACING", "1").lower() in ("0", "false", "off"):
        logger.info("Tracing disabled via HRFIDELITY_TRACING env var")
        return False

    try:
        from phoenix.otel import register
        from openinference.instrumentation.anthropic import AnthropicInstrumentor
    except ImportError:
        logger.info(
            "Phoenix tracing not available — install with: "
            "pip install arize-phoenix-otel openinference-instrumentation-anthropic"
        )
        return False

    try:
        # batch=True -> BatchSpanProcessor (Phoenix's recommended production
        # default; SimpleSpanProcessor warns). Scripts force_flush() before exit
        # so short-lived runs still export their spans.
        kwargs: dict = {"project_name": project_name, "batch": True}

        api_key = os.environ.get("PHOENIX_API_KEY")
        if api_key:
            # Phoenix Cloud
            kwargs["api_key"] = api_key
        else:
            # Local Phoenix or custom endpoint
            endpoint = os.environ.get(
                "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces"
            )
            kwargs["endpoint"] = endpoint

        tracer_provider = register(**kwargs)
        AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)
        logger.info("Phoenix tracing enabled — project: %s", project_name)
        return True

    except Exception as exc:
        logger.warning("Phoenix tracing setup failed: %s", exc)
        return False
