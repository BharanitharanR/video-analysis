"""Shared Gemini API helpers used by video semantic and composition providers."""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

from . import model_config

T = TypeVar("T")

# Built-in model order, used only when neither an environment variable nor
# the model config file (see model_config) sets one. The order is set by the
# user in the studio's Models window or in the config file; these are just
# the defaults. gemini-3.8-flash is the baseline: Flash-Lite answers were too
# weak for demo narration and gemini-4-hydrogen was very slow on a 173 s demo
# (2026-11-21). Several API keys (see api_keys) get past each model's small
# free quota: a quota error tries the same model on the next key; anything
# else another key cannot fix steps down to the next model.
DEFAULT_SEMANTIC_MODELS = (
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-2.5-flash",
    "gemini-4-hydrogen",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
)
DEFAULT_COMPOSITION_MODELS = (
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-2.5-flash",
    "gemini-4-hydrogen",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
)
STEP_DEFAULTS = {"analysis": DEFAULT_SEMANTIC_MODELS, "composition": DEFAULT_COMPOSITION_MODELS}

# Most Gemini attempts (model x key) per step, whatever the mix. Every
# attempt may be a paid call. VIDEO_ANALYSIS_MAX_MODEL_ATTEMPTS or the
# config file's "max_attempts" overrides.
DEFAULT_MAX_ATTEMPTS = 6

# Keys: GEMINI_API_KEY_1 .. GEMINI_API_KEY_<n>, tried in that order; when
# none is set, GEMINI_API_KEY alone. Keys are never logged; logs say "key 2/5".
MAX_NUMBERED_KEYS = 20


def api_keys() -> list[str]:
    """The Gemini API keys to rotate through, in order, without repeats.

    A key only helps with quota if it belongs to a different Google Cloud
    project: quota is counted per project, not per key.
    """
    keys: list[str] = []
    for n in range(1, MAX_NUMBERED_KEYS + 1):
        value = (os.environ.get(f"GEMINI_API_KEY_{n}") or "").strip()
        if value and value not in keys:
            keys.append(value)
    if not keys:
        single = (os.environ.get("GEMINI_API_KEY") or "").strip()
        if single:
            keys.append(single)
    return keys


def max_attempts() -> int:
    return model_config.max_attempts_setting(DEFAULT_MAX_ATTEMPTS)[0]


class InvalidModelResponse(ValueError):
    """Gemini answered, but the answer is empty, not JSON, or fails our
    validation (for example a timestamp past the end of the video).

    This is a fault of that model's answer, not of the request or the key,
    so the chain moves on to the next model. It subclasses ValueError so callers
    that already catch ValueError keep working.
    """


# Errors that say "this model, right now" rather than "this request": quota,
# rate limits, server-side failures, and an answer that fails validation
# (InvalidModelResponse). Anything else (bad input, a timeout on a long
# video) is raised at once, since repeating it on another model would spend
# another paid call for nothing.
_RETRYABLE_MARKERS = (
    "INTERNAL", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "DEADLINE_EXCEEDED",
    "OVERLOADED", "QUOTA", "RATE LIMIT", "TOO MANY REQUESTS",
)
_RETRYABLE_CODES = {429, 500, 502, 503, 504}


def is_retryable_model_error(exc: BaseException) -> bool:
    if isinstance(exc, InvalidModelResponse):
        return True
    if isinstance(exc, (TimeoutError, ValueError)):
        return False
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(code, int) and code in _RETRYABLE_CODES:
        return True
    text = str(exc).upper()
    return any(marker in text for marker in _RETRYABLE_MARKERS) or any(
        f" {c}" in text or f"{c} " in text for c in ("429", "503")
    )


_QUOTA_MARKERS = ("RESOURCE_EXHAUSTED", "QUOTA", "RATE LIMIT", "TOO MANY REQUESTS")


def is_quota_error(exc: BaseException) -> bool:
    """Quota or rate limit: the one failure another key (project) can fix.

    Server errors, timeouts and invalid answers are not: the next key would
    hit the same model with the same request, so those step down a model.
    """
    if isinstance(exc, (InvalidModelResponse, TimeoutError, ValueError)):
        return False
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code == 429:
        return True
    text = str(exc).upper()
    return any(marker in text for marker in _QUOTA_MARKERS) or " 429" in text or "429 " in text


def model_chain(primary: str | None, env_var: str, defaults: Iterable[str]) -> list[str]:
    """The requested model (if any) first, then the fallbacks, without repeats.

    `primary` is None unless the caller asked for a specific model; the
    command line no longer defaults to one, so the ordered chain decides.

    The fallbacks come from `env_var` (comma-separated) when set, else the
    model config file (see model_config), else `defaults`.
    VIDEO_ANALYSIS_MODEL_FALLBACK=0 turns fallback off.

    The chain is cut at the attempt cap (default 6), the same cap that bounds
    total attempts across models and keys: when the input itself is what
    makes Gemini fail, every model fails, and each attempt may be a paid call.
    """
    step = model_config.ENV_STEP.get(env_var)
    if step:
        rest, _source = model_config.step_models(step, list(defaults))
    else:
        configured = os.environ.get(env_var)
        rest = [m.strip() for m in configured.split(",")] if configured else list(defaults)
    if os.environ.get("VIDEO_ANALYSIS_MODEL_FALLBACK", "1").strip() == "0":
        if primary:
            return [primary]
        return [rest[0] if rest else next(iter(defaults))]
    chain: list[str] = []
    for model in [primary, *rest]:
        if model and model not in chain:
            chain.append(model)
    chain = chain[:max_attempts()]
    if step:
        where = "--gemini-model first, then " + _source if primary else _source
        _stderr(f"Gemini model order for {model_config.STEP_LABEL[step]} ({where}): {', '.join(chain)}")
    return chain


def _stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def run_with_key_and_model_fallback(
    models: list[str],
    attempt: Callable[[str, int], T],
    *,
    key_count: int,
    log: Callable[[str], None] = _stderr,
    limit: int | None = None,
) -> tuple[T, str, int]:
    """Try each model on each key, in that order, until one attempt succeeds.

    For each model, keys 1..key_count are tried in turn, but only a quota or
    rate-limit error moves to the next key. Any other retryable failure
    (server error, DEADLINE_EXCEEDED, an answer that fails validation) steps
    down to the next model, starting again from key 1. Errors that are not
    retryable are raised at once. At most `limit` attempts are made in all
    (default VIDEO_ANALYSIS_MAX_MODEL_ATTEMPTS, 6).

    `attempt(model, key_index)` should call Gemini and validate the answer
    (see call_and_validate). Returns (result, model_used, key_index_used).
    """
    key_count = max(1, key_count)
    limit = max_attempts() if limit is None else limit
    multi = key_count > 1
    label = (lambda k: f" (key {k + 1}/{key_count})") if multi else (lambda k: "")
    failures: list[str] = []
    last_exc: BaseException | None = None
    attempts = 0
    for index, model in enumerate(models):
        key = 0
        while key < key_count:
            if attempts >= limit:
                failures.append(f"stopped at the cap of {limit} attempts")
                raise RuntimeError(
                    "Every Gemini model in the fallback chain failed: " + " | ".join(failures)
                ) from last_exc
            attempts += 1
            try:
                result = attempt(model, key)
            except Exception as exc:  # noqa: BLE001 - classified below
                if not is_retryable_model_error(exc):
                    raise
                last_exc = exc
                summary = " ".join(_describe(exc).split())[:300]
                failures.append(f"{model}{label(key)}: {summary}")
                if is_quota_error(exc) and key + 1 < key_count:
                    log(f"Gemini model {model}{label(key)} out of quota ({summary}); "
                        f"trying key {key + 2}/{key_count}")
                    key += 1
                    continue
                if index + 1 < len(models):
                    if isinstance(exc, InvalidModelResponse):
                        reason = "gave an invalid answer"
                    elif is_quota_error(exc):
                        reason = "is out of quota on every key" if multi else "unavailable"
                    else:
                        reason = "unavailable"
                    log(f"Gemini model {model}{label(key)} {reason} ({summary}); "
                        f"trying {models[index + 1]}" + (" from key 1" if multi else ""))
                break
            else:
                log(f"Gemini model used: {model}{label(key)}")
                return result, model, key
    raise RuntimeError(
        "Every Gemini model in the fallback chain failed: " + " | ".join(failures)
    ) from last_exc


def run_with_model_fallback(
    models: list[str],
    attempt: Callable[[str], T],
    *,
    log: Callable[[str], None] = _stderr,
) -> tuple[T, str]:
    """Single-key form: call `attempt(model)` down the chain, one try each.

    Moves on for quota, rate-limit and server errors, and for an answer that
    fails validation. Returns (result, model_used).
    """
    result, model, _ = run_with_key_and_model_fallback(
        models, lambda model, _key: attempt(model), key_count=1, log=log, limit=len(models)
    )
    return result, model


def _describe(exc: BaseException) -> str:
    """The error text, plus the validation detail an InvalidModelResponse wraps."""
    text = str(exc)
    if isinstance(exc, InvalidModelResponse) and exc.__cause__ is not None:
        cause = " ".join(str(exc.__cause__).split())
        # Pydantic puts the useful line ("Value error, event exceeds video
        # duration") after the field path; keep that part.
        marker = cause.find("Value error")
        text += f" ({cause[marker:marker + 120] if marker >= 0 else cause[:120]})"
    return text


def rejected_response_path(raw_path: Path, model: str) -> Path:
    """Where a rejected answer is kept, so a later model's answer does not
    overwrite it: gemini-response.json -> gemini-response.<model>.rejected.json."""
    return raw_path.with_name(f"{raw_path.stem}.{model}.rejected{raw_path.suffix or '.json'}")


def call_and_validate(
    model: str,
    *,
    call: Callable[[str], Any],
    artifact_name: str,
    raw_env_var: str,
    raw_default_name: str,
    validator: Callable[[dict[str, Any]], T],
) -> T:
    """One model attempt: call Gemini, save the raw answer, validate it.

    Used inside the `attempt` for run_with_key_and_model_fallback, so an
    answer that fails validation moves the chain to the next model. A rejected answer is also
    kept under rejected_response_path for inspection.
    """
    interaction = call(model)
    output_text = getattr(interaction, "output_text", None)
    raw_path = persist_raw_response(output_text, raw_env_var, raw_default_name)
    try:
        return parse_json_response(
            interaction, artifact_name=artifact_name, raw_path=raw_path, validator=validator
        )
    except InvalidModelResponse:
        rejected_response_path(raw_path, model).write_text(output_text or "", encoding="utf-8")
        raise


def create_client(api_key: str | None = None) -> Any:
    """Create a google-genai client for one key.

    The key is passed explicitly, so a GOOGLE_API_KEY in the environment can
    never take precedence over the configured Gemini keys.
    """
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "Gemini support is optional. Install video-analysis[gemini]."
        ) from exc
    return genai.Client(api_key=api_key) if api_key else genai.Client()


def create_clients() -> list[Any]:
    """One client per key from api_keys(), in rotation order."""
    keys = api_keys()
    if not keys:
        raise RuntimeError("Set GEMINI_API_KEY, or GEMINI_API_KEY_1 .. GEMINI_API_KEY_<n>")
    return [create_client(key) for key in keys]


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
    processing_timeout_seconds: float | None,
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

    deadline = None if processing_timeout_seconds is None else time.monotonic() + processing_timeout_seconds
    while True:
        status = str(getattr(interaction, "status", "")).lower()
        if status == "completed":
            return interaction
        if status in {"failed", "cancelled", "canceled", "incomplete"}:
            raise RuntimeError(
                f"Gemini interaction {interaction_id} ended with status={status}; "
                f"errors={getattr(interaction, 'errors', None)}"
            )
        if deadline is not None and time.monotonic() >= deadline:
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
        raise InvalidModelResponse(
            f"Gemini returned an empty {artifact_name} response; "
            f"raw response saved to {raw_path}"
        )

    try:
        payload = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise InvalidModelResponse(
            f"Gemini returned invalid JSON for {artifact_name}; "
            f"raw response saved to {raw_path}"
        ) from exc

    try:
        return validator(payload)
    except Exception as exc:
        raise InvalidModelResponse(
            f"Gemini {artifact_name} response failed local validation. "
            f"Raw response saved to {raw_path}"
        ) from exc
