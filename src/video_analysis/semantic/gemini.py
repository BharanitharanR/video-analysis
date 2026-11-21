"""Gemini implementation of the semantic-only provider."""
from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from typing import Any

from ..gemini import (
    DEFAULT_SEMANTIC_MODELS,
    call_and_validate,
    create_client,
    model_chain,
    run_background_interaction,
    run_with_model_fallback,
    sanitize_response_schema,
)
from ..gemini_input import copy_enabled, make_gemini_copy
from .models import SemanticAnalysis
from .prompt import SYSTEM_INSTRUCTION, build_semantic_prompt

MAX_INLINE_VIDEO_BYTES = 20 * 1024 * 1024
DEFAULT_FILE_POLL_INTERVAL_SECONDS = 2.0
DEFAULT_FILE_PROCESSING_TIMEOUT_SECONDS = 300.0
DEFAULT_INTERACTION_POLL_INTERVAL_SECONDS = 5.0
DEFAULT_INTERACTION_PROCESSING_TIMEOUT_SECONDS = 1800.0
DEFAULT_RAW_RESPONSE_LOG = "gemini-response.raw.json"


def _gemini_response_schema() -> dict[str, Any]:
    """Return a conservative Gemini schema for SemanticAnalysis."""
    schema = sanitize_response_schema(SemanticAnalysis.model_json_schema())

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

    Small videos are sent inline. Larger videos are uploaded through the Gemini
    Files API and referenced by URI, so the provider can handle normal
    phone-camera files without requiring the caller to transcode them first.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        file_poll_interval_seconds: float = DEFAULT_FILE_POLL_INTERVAL_SECONDS,
        file_processing_timeout_seconds: float = DEFAULT_FILE_PROCESSING_TIMEOUT_SECONDS,
        interaction_poll_interval_seconds: float = DEFAULT_INTERACTION_POLL_INTERVAL_SECONDS,
        interaction_processing_timeout_seconds: float = DEFAULT_INTERACTION_PROCESSING_TIMEOUT_SECONDS,
        webhook_url: str | None = None,
    ) -> None:
        self._client = create_client()
        self.model = model
        self.file_poll_interval_seconds = file_poll_interval_seconds
        self.file_processing_timeout_seconds = file_processing_timeout_seconds
        self.interaction_poll_interval_seconds = interaction_poll_interval_seconds
        self.interaction_processing_timeout_seconds = interaction_processing_timeout_seconds
        self.webhook_url = webhook_url or os.environ.get("VIDEO_ANALYSIS_GEMINI_WEBHOOK_URL")
        # Send Gemini a deterministic 720p copy, never the original file
        # (see ..gemini_input). VIDEO_ANALYSIS_GEMINI_COPY=0 turns it off.
        self.make_copy = copy_enabled()
        self.gemini_input: dict[str, Any] | None = None

    def _upload_video(self, path: Path) -> Any:
        """Upload a large video and wait until Gemini has processed it."""
        uploaded_file = self._client.files.upload(file=str(path))
        import time

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

        # What Gemini receives: the deterministic copy when enabled (the
        # original stays untouched), otherwise the original.
        upload_path = path
        self.gemini_input = None
        if getattr(self, "make_copy", False):
            self.gemini_input = make_gemini_copy(path)
            upload_path = path.with_name(self.gemini_input["path"])

        file_size = upload_path.stat().st_size
        mime_type = mimetypes.guess_type(upload_path.name)[0] or "video/mp4"

        if file_size <= MAX_INLINE_VIDEO_BYTES:
            video_bytes = upload_path.read_bytes()
            video_input = {
                "type": "video",
                "data": base64.b64encode(video_bytes).decode("utf-8"),
                "mime_type": mime_type,
            }
        else:
            uploaded_file = self._upload_video(upload_path)
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

        webhook_config = None
        if self.webhook_url:
            webhook_config = {
                "uris": [self.webhook_url],
                "user_metadata": {
                    "video_path": str(path),
                    "analysis_type": "semantic",
                },
            }

        def call(model: str) -> Any:
            return run_background_interaction(
                self._client,
                model=model,
                input_items=[
                    video_input,
                    {"type": "text", "text": prompt},
                ],
                system_instruction=SYSTEM_INSTRUCTION,
                response_schema=schema,
                poll_interval_seconds=self.interaction_poll_interval_seconds,
                processing_timeout_seconds=self.interaction_processing_timeout_seconds,
                webhook_config=webhook_config,
            )

        def validate(payload: dict[str, Any]) -> SemanticAnalysis:
            # Contract-owned fields are authoritative locally; Gemini does not
            # need to spend output tokens generating them.
            payload["schema_version"] = "0.3"
            payload["analysis_type"] = "semantic"
            payload["duration_seconds"] = duration_seconds
            return SemanticAnalysis.model_validate(payload)

        # The upload above happens once; every model in the chain reads the
        # same uploaded file or inline bytes. Each attempt is validated, so an
        # answer that fails validation moves up to the next model.
        semantic, self.model_used = run_with_model_fallback(
            model_chain(self.model, "VIDEO_ANALYSIS_SEMANTIC_MODELS", DEFAULT_SEMANTIC_MODELS),
            lambda model: call_and_validate(
                model,
                call=call,
                artifact_name="semantic",
                raw_env_var="VIDEO_ANALYSIS_GEMINI_RAW_RESPONSE",
                raw_default_name=DEFAULT_RAW_RESPONSE_LOG,
                validator=validate,
            ),
        )
        return semantic
