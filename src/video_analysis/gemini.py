"""Shared Gemini API helpers used by video semantic and composition providers."""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

T = TypeVar("T")

# Fallback order when the requested model is rate limited or failing.
# Only models the free tier actually grants quota to (AI Studio rate-limit
# page, 2026-11-20). The Flash-Lite models come last: weaker, but they carry
# 15 RPM / 500 RPD where the others carry 5 RPM / 20 RPD.
DEFAULT_SEMANTIC_MODELS = (
    "gemini-4-hydrogen",
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-2.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
)
DEFAULT_COMPOSITION_MODELS = (
    "gemini-3.6-flash",
    "gemini-3.8-flash",
    "gemini-4-hydrogen",
    "gemini-2.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
)

# Errors that say "this model, right now" rather than "this request": quota,
# rate limits and server-side failures. Anything else (bad input, a response
# that fails validation, a timeout on a long video) is raised at once, since
# repeating it on another model would spend another paid call for nothing.
_RETRYABLE_MARKERS = (
    "INTERNAL", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "DEADLINE_EXCEEDED",
    "OVERLOADED", "QUOTA", "RATE LIMIT", "TOO MANY REQUESTS",
)
_RETRYABLE_CODES = {429, 500, 502, 503, 504}


def is_retryable_model_error(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, ValueError)):
        return False
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(code, int) and code in _RETRYABLE_CODES:
        return True
    text = str(exc).upper()
    return any(marker in text for marker in _RETRYABLE_MARKERS) or any(
        f" {c}" in text or f"{c} " in text for c in ("429", "503")
    )


def model_chain(primary: str, env_var: str, defaults: Iterable[str]) -> list[str]:
    """The requested model first, then the fallbacks, without repeats.

    The fallbacks come from `env_var` (comma-separated) when set, else
    `defaults`. VIDEO_ANALYSIS_MODEL_FALLBACK=0 turns fallback off.

    At most VIDEO_ANALYSIS_MAX_MODEL_ATTEMPTS models (default 3) are tried:
    when the input itself is what makes Gemini fail, every model fails, and
    each attempt is a paid call.
    """
    if os.environ.get("VIDEO_ANALYSIS_MODEL_FALLBACK", "1").strip() == "0":
        return [primary]
    configured = os.environ.get(env_var)
    rest = [m.strip() for m in configured.split(",")] if configured else list(defaults)
    chain: list[str] = []
    for model in [primary, *rest]:
        if model and model not in chain:
            chain.append(model)
    limit = max(1, int(os.environ.get("VIDEO_ANALYSIS_MAX_MODEL_ATTEMPTS", "3")))
    return chain[:limit]


def run_with_model_fallback(
    models: list[str],
    attempt: Callable[[str], T],
    *,
    log: Callable[[str], None] = lambda message: print(message, file=sys.stderr, flush=True),
) -> tuple[T, str]:
    """Call `attempt(model)` down the chain until one model succeeds.

    Each model is tried once. Moves on only for quota, rate-limit and
    server errors; returns (result, model_used).
    """
    failures: list[str] = []
    for index, model in enumerate(models):
        try:
            result = attempt(model)
        except Exception as exc:  # noqa: BLE001 - classified below
            if not is_retryable_model_error(exc):
                raise
            summary = " ".join(str(exc).split())[:240]
            failures.append(f"{model}: {summary}")
            if index + 1 < len(models):
                log(f"Gemini model {model} unavailable ({summary}); trying {models[index + 1]}")
            continue
        log(f"Gemini model used: {model}")
        return result, model
    raise RuntimeError("Every Gemini model in the fallback chain failed: " + " | ".join(failures))


def create_client() -> Any:
    """Create the shared google-genai client using GEMINI_API_KEY."""
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "Gemini support is optional. Install video-analysis[gemini]."
        ) from exc
    return genai.Client()


def sanitize_response_schema(model_schema: dict[str, Any]) -> dict[str, Any]:
    """Return a conservative schema suitable for Gemini structured output."""
    schema = copy.deepcopy(model_schema)

    def sanitize(node: Any) -> None:
        if isinstance(node, dict):
            node.pop("exclusiveMinimum", None)
            node.pop("maxItems", None)
            for value in node.values():
                sanitize(value)
        elif isinstance(node, list):
            for value in node:
                sanitize(value)

    sanitize(schema)
    return schema


def run_background_interaction(
    client: Any,
    *,
    model: str,
    input_items: list[dict[str, Any]],
    system_instruction: str,
    response_schema: dict[str, Any],
    poll_interval_seconds: float,
    processing_timeout_seconds: float,
    webhook_config: dict[str, Any] | None = None,
) -> Any:
    """Submit a stored background interaction and wait for completion."""
    interaction_kwargs: dict[str, Any] = {
        "model": model,
        "input": input_items,
        "system_instruction": system_instruction,
        "response_format": {
            "type": "text",
            "mime_type": "application/json",
            "schema": response_schema,
        },
        "background": True,
        "store": True,
    }
    if webhook_config:
        interaction_kwargs["webhook_config"] = webhook_config

    interaction = client.interactions.create(**interaction_kwargs)
    interaction_id = getattr(interaction, "id", None)
    if not interaction_id:
        raise RuntimeError(
            "Gemini background interaction did not return an interaction ID."
        )

    deadline = time.monotonic() + processing_timeout_seconds
    while True:
        status = str(getattr(interaction, "status", "")).lower()
        if status == "completed":
            return interaction
        if status in {"failed", "cancelled", "canceled", "incomplete"}:
            raise RuntimeError(
                f"Gemini interaction {interaction_id} ended with status={status}; "
                f"errors={getattr(interaction, 'errors', None)}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Gemini interaction {interaction_id} did not complete within "
                f"{processing_timeout_seconds:.0f} seconds; "
                f"last status={status or 'unknown'}"
            )

        time.sleep(poll_interval_seconds)
        interaction = client.interactions.get(interaction_id)


def persist_raw_response(output_text: str | None, env_var: str, default_name: str) -> Path:
    """Persist the exact Gemini output before local validation."""
    path = Path(os.environ.get(env_var, default_name))
    path.write_text(output_text or "", encoding="utf-8")
    return path


def parse_json_response(
    interaction: Any,
    *,
    artifact_name: str,
    raw_path: Path,
    validator: Callable[[dict[str, Any]], Any],
) -> Any:
    """Parse and validate a completed Gemini JSON response."""
    output_text = getattr(interaction, "output_text", None)
    if not output_text:
        raise RuntimeError(
            f"Gemini returned an empty {artifact_name} response; "
            f"raw response saved to {raw_path}"
        )

    try:
        payload = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Gemini returned invalid JSON for {artifact_name}; "
            f"raw response saved to {raw_path}"
        ) from exc

    try:
        return validator(payload)
    except Exception as exc:
        raise ValueError(
            f"Gemini {artifact_name} response failed local validation. "
            f"Raw response saved to {raw_path}"
        ) from exc
