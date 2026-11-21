"""The Gemini model order, read from a config file instead of code.

One JSON file, shared by the Video Score Studio "Models" window, the command
line and the MCP wrapper. The order of each list is the order models are
tried in:

    {
      "analysis":    ["gemini-3.8-flash", "gemini-3.6-flash", ...],
      "composition": ["gemini-3.8-flash", "gemini-3.6-flash", ...],
      "disabled":    {"analysis": ["gemini-4-hydrogen"], "composition": []},
      "max_attempts": 6
    }

"disabled" only remembers models switched off in the studio window; they are
never tried. Every field is optional; a missing list falls back to the
built-in default for that step.

Where it lives: VIDEO_ANALYSIS_MODELS_CONFIG, else
~/.config/video-score/gemini-models.json. It is read fresh at the start of
every Gemini step, so an edit applies to the next run without a restart.

Which setting wins, per step:
  1. --gemini-model (moves that model to the front)
  2. VIDEO_ANALYSIS_SEMANTIC_MODELS / VIDEO_ANALYSIS_COMPOSITION_MODELS
  3. this config file
  4. the built-in defaults in ..gemini
The attempt cap: VIDEO_ANALYSIS_MAX_MODEL_ATTEMPTS, else "max_attempts" here,
else 6.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

CONFIG_ENV = "VIDEO_ANALYSIS_MODELS_CONFIG"
DEFAULT_CONFIG_PATH = Path("~/.config/video-score/gemini-models.json")

STEPS = ("analysis", "composition")
STEP_ENV = {
    "analysis": "VIDEO_ANALYSIS_SEMANTIC_MODELS",
    "composition": "VIDEO_ANALYSIS_COMPOSITION_MODELS",
}
ENV_STEP = {env: step for step, env in STEP_ENV.items()}
STEP_LABEL = {"analysis": "Video analysis", "composition": "Music plan"}

MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
MAX_ATTEMPTS_LIMIT = 50


class ModelConfigError(ValueError):
    """The config file exists but cannot be used. Raised before any Gemini call."""


def config_path() -> Path:
    return Path(os.environ.get(CONFIG_ENV) or DEFAULT_CONFIG_PATH).expanduser()


def _clean_list(value: Any, where: str) -> list[str]:
    if not isinstance(value, list):
        raise ModelConfigError(f"{where} must be a list of model names")
    models: list[str] = []
    for item in value:
        if not isinstance(item, str) or not MODEL_NAME.match(item.strip()):
            raise ModelConfigError(f"{where} has an invalid model name: {item!r}")
        name = item.strip()
        if name not in models:
            models.append(name)
    return models


def validate(data: Any) -> dict[str, Any]:
    """Return a cleaned copy of a config document, or raise ModelConfigError."""
    if not isinstance(data, dict):
        raise ModelConfigError("the model config must be a JSON object")
    clean: dict[str, Any] = {}
    disabled_in = data.get("disabled") or {}
    if not isinstance(disabled_in, dict):
        raise ModelConfigError('"disabled" must be an object of lists')
    disabled: dict[str, list[str]] = {}
    for step in STEPS:
        off = _clean_list(disabled_in.get(step, []), f'"disabled.{step}"')
        if step in data:
            models = _clean_list(data[step], f'"{step}"')
            if not [m for m in models if m not in off]:
                raise ModelConfigError(f'"{step}" needs at least one model switched on')
            clean[step] = models
        if off:
            disabled[step] = off
    if disabled:
        clean["disabled"] = disabled
    if "max_attempts" in data:
        cap = data["max_attempts"]
        if isinstance(cap, bool) or not isinstance(cap, int) or not 1 <= cap <= MAX_ATTEMPTS_LIMIT:
            raise ModelConfigError(f'"max_attempts" must be a whole number from 1 to {MAX_ATTEMPTS_LIMIT}')
        clean["max_attempts"] = cap
    return clean


def load(path: Path | None = None) -> dict[str, Any] | None:
    """The cleaned config, or None when there is no file."""
    path = path or config_path()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelConfigError(f"cannot read the model config {path}: {exc}") from exc
    try:
        return validate(data)
    except ModelConfigError as exc:
        raise ModelConfigError(f"{path}: {exc}") from exc


def save(data: dict[str, Any], path: Path | None = None) -> Path:
    """Validate and write the config atomically. Returns the path written."""
    clean = validate(data)
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".gemini-models.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(clean, handle, indent=2)
            handle.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def step_models(step: str, defaults: list[str] | tuple[str, ...]) -> tuple[list[str], str]:
    """(ordered models to try, where the order came from) for one step.

    Environment variable, then config file, then defaults. Disabled models
    are left out.
    """
    env = os.environ.get(STEP_ENV[step])
    if env and env.strip():
        models = [m.strip() for m in env.split(",") if m.strip()]
        return list(dict.fromkeys(models)), f"environment variable {STEP_ENV[step]}"
    config = load()
    if config and step in config:
        off = set(config.get("disabled", {}).get(step, []))
        return [m for m in config[step] if m not in off], f"config file {config_path()}"
    return list(defaults), "built-in defaults"


def max_attempts_setting(default: int) -> tuple[int, str]:
    env = os.environ.get("VIDEO_ANALYSIS_MAX_MODEL_ATTEMPTS")
    if env and env.strip():
        return max(1, int(env)), "environment variable VIDEO_ANALYSIS_MAX_MODEL_ATTEMPTS"
    config = load()
    if config and "max_attempts" in config:
        return config["max_attempts"], f"config file {config_path()}"
    return default, "built-in default"
