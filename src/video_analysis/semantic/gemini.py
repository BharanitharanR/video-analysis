"""Gemini implementation of the semantic-only provider."""
from __future__ import annotations

import base64
import copy
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

from .models import SemanticAnalysis
from .prompt import SYSTEM_INSTRUCTION, build_semantic_prompt

MAX_INLINE_VIDEO_BYTES = 20 * 1024 * 1024
DEFAULT_FILE_POLL_INTERVAL_SECONDS = 2.0
DEFAULT_FILE_PROCESSING_TIMEOUT_SECONDS = 300.0
DEFAULT_INTERACTION_POLL_INTERVAL_SECONDS = 5.0
DEFAULT_INTERACTION_PROCESSING_TIMEOUT_SECONDS = 1800.0
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
    # produce a contract-invalid value.
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    for field in ("schema_version", "analysis_type", "duration_seconds"):
        properties.pop(field, None)
        if field in required:
            required.remove(field)

    return schema


def _file_state_name(file: Any) -> str:
    """Return a normalized Gemini Files API state name."""
    state = getattr(file, "state", None)
    name = getattr(state, "name", state)
    return str(name).upper()


class GeminiSemanticProvider:
    """Generate a separate SemanticAnalysis JSON document with Gemini.

    Install the optional dependency with: pip install 'video-analysis[gemini]'
    Set GEMINI_API_KEY in the environment. The API key is never accepted in
    source code or logged by this class.

    Small videos are sent inline. Larger videos are uploaded through the
    Gemini Files API and referenced by URI, so the provider can handle normal
    phone-camera files without requiring the caller to transcode them first.
    """

    def __init__(
        self,
        model: str = "gemini-4-hydrogen",
        *,
        file_poll_interval_seconds: float = DEFAULT_FILE_POLL_INTERVAL_SECONDS,
        file_processing_timeout_seconds: float = DEFAULT_FILE_PROCESSING_TIMEOUT_SECONDS,
        interaction_poll_interval_seconds: float = DEFAULT_INTERACTION_POLL_INTERVAL_SECONDS,
        interaction_processing_timeout_seconds: float = DEFAULT_INTERACTION_PROCESSING_TIMEOUT_SECONDS,
        webhook_url: str | None = None,
    ) -> None:
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "Gemini support is optional. Install video-analysis[gemini]."
            ) from exc
        self._client = genai.Client()
        self.model = model
        self.file_poll_interval_seconds = file_poll_interval_seconds
        self.file_processing_timeout_seconds = file_processing_timeout_seconds
        self.interaction_poll_interval_seconds = interaction_poll_interval_seconds
        self.interaction_processing_timeout_seconds = interaction_processing_timeout_seconds
        self.webhook_url = webhook_url or os.environ.get("VIDEO_ANALYSIS_GEMINI_WEBHOOK_URL")

    def _upload_video(self, path: Path) -> Any:
        """Upload a large video and wait until Gemini has processed it."""
        uploaded_file = self._client.files.upload(file=str(path))
        deadline = time.monotonic() + self.file_processing_timeout_seconds

        while True:
            state = _file_state_name(uploaded_file)
            if state == "ACTIVE":
                return uploaded_file
            if state in {"FAILED", "ERROR"}:
                raise RuntimeError(
                    f"Gemini Files API failed to process video {path}: "
                    f"state={state}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Gemini Files API did not make video active within "
                    f"{self.file_processing_timeout_seconds:.0f} seconds; "
                    f"last state={state}"
                )

            time.sleep(self.file_poll_interval_seconds)
            file_name = getattr(uploaded_file, "name", None)
            if not file_name:
                raise RuntimeError(
                    "Gemini Files API upload response did not include a file name."
                )
            uploaded_file = self._client.files.get(name=file_name)

    def analyze(
        self,
        video_path: str | Path,
        duration_seconds: float,
        evidence: dict[str, Any] | None = None,
    ) -> SemanticAnalysis:
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(path)

        file_size = path.stat().st_size
        mime_type = mimetypes.guess_type(path.name)[0] or "video/mp4"

        if file_size <= MAX_INLINE_VIDEO_BYTES:
            video_bytes = path.read_bytes()
            video_input = {
                "type": "video",
                "data": base64.b64encode(video_bytes).decode("utf-8"),
                "mime_type": mime_type,
            }
        else:
            uploaded_file = self._upload_video(path)
            file_uri = getattr(uploaded_file, "uri", None)
            if not file_uri:
                raise RuntimeError(
                    "Gemini Files API upload did not return a file URI."
                )
            video_input = {
                "type": "video",
                "uri": file_uri,
                "mime_type": getattr(uploaded_file, "mime_type", None) or mime_type,
            }

        prompt = build_semantic_prompt(duration_seconds, evidence)
        schema = _gemini_response_schema()

        interaction_kwargs: dict[str, Any] = {
            "model": self.model,
            "input": [
                video_input,
                {"type": "text", "text": prompt},
            ],
            "system_instruction": SYSTEM_INSTRUCTION,
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": schema,
            },
            # Background interactions avoid holding the initial HTTP request
            # open for long video inference. This is the important distinction
            # from the synchronous request that can fail with server-side 504s.
            "background": True,
            "store": True,
        }

        if self.webhook_url:
            interaction_kwargs["webhook_config"] = {
                "uris": [self.webhook_url],
                "user_metadata": {
                    "video_path": str(path),
                    "analysis_type": "semantic",
                },
            }

        interaction = self._client.interactions.create(**interaction_kwargs)
        interaction_id = getattr(interaction, "id", None)
        if not interaction_id:
            raise RuntimeError("Gemini background interaction did not return an interaction ID.")

        deadline = time.monotonic() + self.interaction_processing_timeout_seconds
        while True:
            status = str(getattr(interaction, "status", "")).lower()
            if status == "completed":
                break
            if status in {"failed", "cancelled", "canceled", "incomplete"}:
                errors = getattr(interaction, "errors", None)
                raise RuntimeError(
                    f"Gemini interaction {interaction_id} ended with status={status}; "
                    f"errors={errors}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Gemini interaction {interaction_id} did not complete within "
                    f"{self.interaction_processing_timeout_seconds:.0f} seconds; "
                    f"last status={status or 'unknown'}"
                )

            time.sleep(self.interaction_poll_interval_seconds)
            interaction = self._client.interactions.get(interaction_id)

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
        payload["schema_version"] = "0.2"
        payload["analysis_type"] = "semantic"
        payload["duration_seconds"] = duration_seconds

        try:
            return SemanticAnalysis.model_validate(payload)
        except Exception as exc:
            raise ValueError(
                f"Gemini semantic response failed local validation. "
                f"Raw response saved to {raw_response_path}"
            ) from exc
