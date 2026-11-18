"""Gemini provider for composition planning from analysis.json."""
from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path
from typing import Any

from .models import CompositionPlan
from .prompt import SYSTEM_INSTRUCTION, build_composition_prompt

DEFAULT_MODEL = "gemini-3.6-flash"
DEFAULT_POLL_INTERVAL_SECONDS = 2.0
DEFAULT_TIMEOUT_SECONDS = 300.0


def _schema() -> dict[str, Any]:
    schema = copy.deepcopy(CompositionPlan.model_json_schema())

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


class GeminiCompositionProvider:
    """Generate a CompositionPlan from a completed analysis.json artifact."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        processing_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "Gemini support is optional. Install video-analysis[gemini]."
            ) from exc
        self._client = genai.Client()
        self.model = model
        self.poll_interval_seconds = poll_interval_seconds
        self.processing_timeout_seconds = processing_timeout_seconds

    def compose(self, analysis: dict[str, Any] | str | Path) -> CompositionPlan:
        if isinstance(analysis, (str, Path)):
            payload = json.loads(Path(analysis).read_text(encoding="utf-8"))
        else:
            payload = analysis

        prompt = build_composition_prompt(payload)
        interaction = self._client.interactions.create(
            model=self.model,
            input=[{"type": "text", "text": prompt}],
            system_instruction=SYSTEM_INSTRUCTION,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": _schema(),
            },
            background=True,
            store=True,
        )

        interaction_id = getattr(interaction, "id", None)
        if not interaction_id:
            raise RuntimeError("Gemini composition interaction did not return an interaction ID.")

        deadline = time.monotonic() + self.processing_timeout_seconds
        while True:
            status = str(getattr(interaction, "status", "")).lower()
            if status == "completed":
                break
            if status in {"failed", "cancelled", "canceled", "incomplete"}:
                raise RuntimeError(
                    f"Gemini composition interaction {interaction_id} ended with status={status}; "
                    f"errors={getattr(interaction, 'errors', None)}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Gemini composition interaction {interaction_id} did not complete within "
                    f"{self.processing_timeout_seconds:.0f} seconds"
                )
            time.sleep(self.poll_interval_seconds)
            interaction = self._client.interactions.get(interaction_id)

        output_text = getattr(interaction, "output_text", None)
        raw_path = Path(os.environ.get("VIDEO_ANALYSIS_COMPOSITION_RAW_RESPONSE", "composition-response.raw.json"))
        raw_path.write_text(output_text or "", encoding="utf-8")
        if not output_text:
            raise RuntimeError(f"Gemini returned an empty composition response; raw response saved to {raw_path}")

        try:
            return CompositionPlan.model_validate(json.loads(output_text))
        except Exception as exc:
            raise ValueError(
                f"Gemini composition response failed local validation. Raw response saved to {raw_path}"
            ) from exc
