"""Explicitly merge semantic dimensions into the deterministic report."""
from __future__ import annotations

from typing import Any
from .models import SemanticAnalysis


def merge_semantic_analysis(
    deterministic_report: dict[str, Any],
    semantic: SemanticAnalysis | dict[str, Any],
) -> dict[str, Any]:
    """Return a new report with semantic-only dimensions added.

    This function never overwrites deterministic dimensions. Semantic fields are
    nested under dimensions['semantic'] as a single independent namespace.
    """
    semantic_model = (
        semantic if isinstance(semantic, SemanticAnalysis)
        else SemanticAnalysis.model_validate(semantic)
    )
    semantic_json = semantic_model.model_dump(mode="json")
    result = {
        **deterministic_report,
        "video": {
            **deterministic_report["video"],
            "dimensions": {
                **deterministic_report["video"].get("dimensions", {}),
                "semantic": semantic_json,
                # Same object, so downstream skills can read the type from
                # either location without it drifting.
                "video_type": semantic_json["video_type"],
            },
        },
    }
    return result
