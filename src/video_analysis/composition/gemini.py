"""Gemini provider for composition planning from analysis.json."""
from __future__ import annotations

import json
from pathlib import Path

from ..gemini import (
    create_client,
    parse_json_response,
    persist_raw_response,
    run_background_interaction,
    sanitize_response_schema,
)
from .models import CompositionPlan
from .prompt import SYSTEM_INSTRUCTION, build_composition_prompt

DEFAULT_MODEL = "gemini-3.6-flash"
DEFAULT_POLL_INTERVAL_SECONDS = 2.0
DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_RAW_RESPONSE_LOG = "composition-response.raw.json"


def _schema() -> dict[str, object]:
    return sanitize_response_schema(CompositionPlan.model_json_schema())


class GeminiCompositionProvider:
    """Generate a CompositionPlan from a completed analysis.json artifact."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        processing_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._client = create_client()
        self.model = model
        self.poll_interval_seconds = poll_interval_seconds
        self.processing_timeout_seconds = processing_timeout_seconds

    def compose(self, analysis: dict | str | Path) -> CompositionPlan:
        if isinstance(analysis, (str, Path)):
            payload = json.loads(Path(analysis).read_text(encoding="utf-8"))
        else:
            payload = analysis

        prompt = build_composition_prompt(payload)
        interaction = run_background_interaction(
            self._client,
            model=self.model,
            input_items=[{"type": "text", "text": prompt}],
            system_instruction=SYSTEM_INSTRUCTION,
            response_schema=_schema(),
            poll_interval_seconds=self.poll_interval_seconds,
            processing_timeout_seconds=self.processing_timeout_seconds,
        )

        raw_path = persist_raw_response(
            getattr(interaction, "output_text", None),
            "VIDEO_ANALYSIS_COMPOSITION_RAW_RESPONSE",
            DEFAULT_RAW_RESPONSE_LOG,
        )

        return parse_json_response(
            interaction,
            artifact_name="composition",
            raw_path=raw_path,
            validator=CompositionPlan.model_validate,
        )
