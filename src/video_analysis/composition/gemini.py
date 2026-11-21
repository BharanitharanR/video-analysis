"""Gemini provider for composition planning from analysis.json."""
from __future__ import annotations

import json
from pathlib import Path

from ..gemini import (
    DEFAULT_COMPOSITION_MODELS,
    create_client,
    model_chain,
    parse_json_response,
    persist_raw_response,
    run_background_interaction,
    run_with_model_fallback,
    sanitize_response_schema,
)
from .models import CompositionPlan
from .voice import apply_narration_bed, load_voice_timeline, summarize_voice_timeline
from .prompt import SYSTEM_INSTRUCTION, build_composition_prompt

DEFAULT_MODEL = None  # the ordered chain in ..gemini decides
DEFAULT_POLL_INTERVAL_SECONDS = 2.0
DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_RAW_RESPONSE_LOG = "composition-response.raw.json"


def _schema() -> dict[str, object]:
    return sanitize_response_schema(CompositionPlan.model_json_schema())


class GeminiCompositionProvider:
    """Generate a CompositionPlan from a completed analysis.json artifact."""

    def __init__(
        self,
        model: str | None = DEFAULT_MODEL,
        *,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        processing_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._client = create_client()
        self.model = model
        self.poll_interval_seconds = poll_interval_seconds
        self.processing_timeout_seconds = processing_timeout_seconds

    def compose(self, analysis: dict | str | Path, voice_timeline: dict | str | Path | None = None) -> CompositionPlan:
        if isinstance(analysis, (str, Path)):
            payload = json.loads(Path(analysis).read_text(encoding="utf-8"))
        else:
            payload = analysis

        duration = payload.get("video", {}).get("dimensions", {}).get("technical", {}).get("duration_seconds")
        if duration is None:
            raise ValueError(
                "analysis.json is missing video.dimensions.technical.duration_seconds"
            )

        timeline = load_voice_timeline(voice_timeline) if voice_timeline is not None else None
        voice_summary = summarize_voice_timeline(timeline, float(duration)) if timeline else None
        prompt = build_composition_prompt(payload, voice_summary)
        interaction, self.model_used = run_with_model_fallback(
            model_chain(self.model, "VIDEO_ANALYSIS_COMPOSITION_MODELS", DEFAULT_COMPOSITION_MODELS),
            lambda model: run_background_interaction(
                self._client,
                model=model,
                input_items=[{"type": "text", "text": prompt}],
                system_instruction=SYSTEM_INSTRUCTION,
                response_schema=_schema(),
                poll_interval_seconds=self.poll_interval_seconds,
                processing_timeout_seconds=self.processing_timeout_seconds,
            ),
        )

        raw_path = persist_raw_response(
            getattr(interaction, "output_text", None),
            "VIDEO_ANALYSIS_COMPOSITION_RAW_RESPONSE",
            DEFAULT_RAW_RESPONSE_LOG,
        )

        def validate(result: dict) -> CompositionPlan:
            # Video duration is authoritative from deterministic analysis.
            result["duration_seconds"] = float(duration)
            return CompositionPlan.model_validate(result)

        plan = parse_json_response(
            interaction,
            artifact_name="composition",
            raw_path=raw_path,
            validator=validate,
        )
        if timeline is not None:
            apply_narration_bed(plan, timeline)
        return plan
