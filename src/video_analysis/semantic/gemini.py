"""Gemini implementation of the semantic-only provider."""
from __future__ import annotations

import base64
import copy
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

from .models import SemanticAnalysis
from .prompt import SYSTEM_INSTRUCTION, build_semantic_prompt

MAX_INLINE_VIDEO_BYTES = 20 * 1024 * 1024
DEFAULT_RAW_RESPONSE_LOG = "gemini-response.raw.json"


def _gemini_response_schema() -> dict[str, Any]:
    """Return a Gemini-compatible copy of the Pydantic JSON schema.

    Pydantic correctly emits JSON Schema constraints such as
    exclusiveMinimum and maxItems. The Gemini structured-output compiler can
    reject some of these constraints depending on the model/API version, so
    application-level Pydantic validation remains authoritative while the
    transport schema is kept conservative.
    """
    schema = copy.deepcopy(SemanticAnalysis.model_json_schema())

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

    # These fields belong to the skill contract, not to Gemini. Asking the
    # model to reproduce fixed metadata wastes output tokens and can still
    # produce a contract-invalid value (for example, "1.0" for schema_version).
    # Remove them from the transport schema and inject the authoritative values
    # after the model response is parsed.
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    for field in ("schema_version", "analysis_type", "duration_seconds"):
        properties.pop(field, None)
        if field in required:
            required.remove(field)

    return schema


class GeminiSemanticProvider:
    """Generate a separate SemanticAnalysis JSON document with Gemini.

    Install the optional dependency with: pip install 'video-analysis[gemini]'
    Set GEMINI_API_KEY in the environment. The API key is never accepted in
    source code or logged by this class.

    The provider uses the Gemini Interactions API because it supports the
    multimodal video input and structured JSON response format used by this
    skill.
    """

    def __init__(self, model: str = "gemini-4-hydrogen") -> None:
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "Gemini support is optional. Install video-analysis[gemini]."
            ) from exc
        self._client = genai.Client()
        self.model = model

    def analyze(
        self,
        video_path: str | Path,
        duration_seconds: float,
        evidence: dict[str, Any] | None = None,
    ) -> SemanticAnalysis:
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(path)

        video_bytes = path.read_bytes()
        if len(video_bytes) > MAX_INLINE_VIDEO_BYTES:
            raise ValueError(
                f"Video is {len(video_bytes) / (1024 * 1024):.1f} MiB, exceeding "
                f"the {MAX_INLINE_VIDEO_BYTES / (1024 * 1024):.0f} MiB inline-video limit. "
                "Use a smaller video or add a Files API URI transport."
            )

        mime_type = mimetypes.guess_type(path.name)[0] or "video/mp4"
        prompt = build_semantic_prompt(duration_seconds, evidence)
        schema = _gemini_response_schema()

        interaction = self._client.interactions.create(
            model=self.model,
            input=[
                {
                    "type": "video",
                    "data": base64.b64encode(video_bytes).decode("utf-8"),
                    "mime_type": mime_type,
                },
                {"type": "text", "text": prompt},
            ],
            system_instruction=SYSTEM_INSTRUCTION,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": schema,
            },
            store=False,
        )

        output_text = getattr(interaction, "output_text", None)

        # Persist the exact model response before any parsing or Pydantic
        # validation. If validation fails, the paid Gemini response is still
        # available locally for debugging instead of requiring another request.
        raw_response_path = Path(
            os.environ.get("VIDEO_ANALYSIS_GEMINI_RAW_RESPONSE", DEFAULT_RAW_RESPONSE_LOG)
        )
        raw_response_path.write_text(output_text or "", encoding="utf-8")

        if not output_text:
            raise RuntimeError(
                f"Gemini returned an empty semantic response; raw response logged to "
                f"{raw_response_path}"
            )

        payload = json.loads(output_text)

        # Contract-owned fields are authoritative locally; Gemini does not
        # need to spend output tokens generating them.
        payload["schema_version"] = "0.1"
        payload["analysis_type"] = "semantic"
        payload["duration_seconds"] = duration_seconds

        try:
            return SemanticAnalysis.model_validate(payload)
        except Exception as exc:
            raise ValueError(
                f"Gemini semantic response failed local validation. "
                f"Raw response saved to {raw_response_path}"
            ) from exc
