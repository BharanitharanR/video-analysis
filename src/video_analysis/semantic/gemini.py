"""Gemini implementation of the semantic-only provider."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import SemanticAnalysis
from .prompt import SYSTEM_INSTRUCTION, build_semantic_prompt


class GeminiSemanticProvider:
    """Generate a separate SemanticAnalysis JSON document with Gemini.

    Install the optional dependency with: pip install 'video-analysis[gemini]'
    Set GEMINI_API_KEY in the environment. The API key is never accepted in
    source code or logged by this class.
    """

    def __init__(self, model: str = "gemini-2.5-flash") -> None:
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

        uploaded = self._client.files.upload(file=str(path))
        try:
            schema = SemanticAnalysis.model_json_schema()
            prompt = build_semantic_prompt(duration_seconds, evidence)
            response = self._client.models.generate_content(
                model=self.model,
                contents=[uploaded, prompt],
                config={
                    "system_instruction": SYSTEM_INSTRUCTION,
                    "response_mime_type": "application/json",
                    "response_schema": schema,
                    "temperature": 0.1,
                },
            )
            text = getattr(response, "text", None)
            if not text:
                raise RuntimeError("Gemini returned an empty semantic response")
            payload = json.loads(text)
            return SemanticAnalysis.model_validate(payload)
        finally:
            # Uploaded file cleanup is best-effort; preserve the primary error.
            try:
                self._client.files.delete(name=uploaded.name)
            except Exception:
                pass
