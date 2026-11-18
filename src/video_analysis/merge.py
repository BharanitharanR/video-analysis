"""Merge deterministic and semantic analysis artifacts.

The deterministic artifact remains authoritative for measured dimensions.
The semantic artifact is validated independently and added under
video.dimensions.semantic.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .semantic.merge import merge_semantic_analysis
from .semantic.models import SemanticAnalysis


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {source}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"JSON artifact must contain an object: {source}")
    return data


def merge_analysis_artifacts(
    deterministic_path: str | Path,
    semantic_path: str | Path,
) -> dict[str, Any]:
    """Load two artifacts and return the merged analysis document."""
    deterministic = load_json(deterministic_path)
    semantic = load_json(semantic_path)

    if not isinstance(deterministic.get("video"), dict):
        raise ValueError("Deterministic artifact is missing a video object.")
    if not isinstance(deterministic["video"].get("dimensions"), dict):
        raise ValueError("Deterministic artifact is missing video.dimensions.")

    semantic_model = SemanticAnalysis.model_validate(semantic)
    return merge_semantic_analysis(deterministic, semantic_model)


def write_merged_analysis(
    deterministic_path: str | Path,
    semantic_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Merge artifacts and write analysis.json."""
    result = merge_analysis_artifacts(deterministic_path, semantic_path)
    output = Path(output_path)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result
