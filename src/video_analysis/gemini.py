"""Shared Gemini API helpers used by video semantic and composition providers."""
from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path
from typing import Any, Callable


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
