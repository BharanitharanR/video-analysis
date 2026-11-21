"""Gemini provider for composition planning from analysis.json."""
from __future__ import annotations

import json
from pathlib import Path

from ..gemini import (
    DEFAULT_COMPOSITION_MODELS,
    _stderr,
    call_and_validate,
    create_clients,
    model_chain,
    run_background_interaction,
    run_with_key_and_model_fallback,
    sanitize_response_schema,
)
from .. import model_config
from .models import CompositionPlan
from .voice import apply_narration_bed, load_voice_timeline, summarize_voice_timeline
from .prompt import SYSTEM_INSTRUCTION, build_composition_prompt

DEFAULT_MODEL = None  # the ordered chain in ..gemini decides
DEFAULT_POLL_INTERVAL_SECONDS = 2.0
DEFAULT_TIMEOUT_SECONDS = None  # no limit: wait until Gemini answers
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
        processing_timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        # One client per key (GEMINI_API_KEY_1..n, else GEMINI_API_KEY).
        self._clients = create_clients()
        self._client = self._clients[0]
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
        clients = list(getattr(self, "_clients", None) or [self._client])

        thinking, thinking_source = model_config.thinking_setting("composition")
        _stderr(f"Gemini thinking level for Music plan: {thinking or 'Gemini default'} ({thinking_source})")

        def call(model: str, key: int):
            return run_background_interaction(
                clients[key],
                model=model,
                input_items=[{"type": "text", "text": prompt}],
                system_instruction=SYSTEM_INSTRUCTION,
                response_schema=_schema(),
                poll_interval_seconds=self.poll_interval_seconds,
                processing_timeout_seconds=self.processing_timeout_seconds,
                thinking_level=thinking,
            )

        def validate(result: dict) -> CompositionPlan:
            # Video duration is authoritative from deterministic analysis.
            result["duration_seconds"] = float(duration)
            return CompositionPlan.model_validate(result)

        # Each model is tried on key 1, 2, ... (next key only on a quota
        # error); each attempt is validated, so an invalid answer moves on to
        # the next model.
        plan, self.model_used, self.key_used = run_with_key_and_model_fallback(
            model_chain(self.model, "VIDEO_ANALYSIS_COMPOSITION_MODELS", DEFAULT_COMPOSITION_MODELS),
            lambda model, key: call_and_validate(
                model,
                call=lambda m: call(m, key),
                artifact_name="composition",
                raw_env_var="VIDEO_ANALYSIS_COMPOSITION_RAW_RESPONSE",
                raw_default_name=DEFAULT_RAW_RESPONSE_LOG,
                validator=validate,
            ),
            key_count=len(clients),
        )
        if timeline is not None:
            apply_narration_bed(plan, timeline)
        return plan
