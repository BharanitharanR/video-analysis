from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyzer import VideoAnalyzer
from .composition.compiler import compile_song_spec
from .composition.gemini import GeminiCompositionProvider
from .merge import write_merged_analysis
from .pipeline import VideoAnalysisPipeline


def main():
    parser = argparse.ArgumentParser(prog="video-analysis")
    sub = parser.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze")
    a.add_argument("video")
    a.add_argument("-o", "--output", default=None)
    a.add_argument("--no-scenes", action="store_true")
    a.add_argument("--interval", type=float, default=1.0)
    a.add_argument("--semantic", choices=["none", "gemini"], default="none",
                   help="Enable semantic AI analysis")
    a.add_argument("--gemini-model", default=None,
                   help="Force this Gemini model first (default: the order in the model config; see `video-analysis models`)")

    m = sub.add_parser(
        "merge",
        help="Merge deterministic.json and semantic.json into analysis.json",
    )
    m.add_argument("deterministic")
    m.add_argument("semantic")
    m.add_argument("-o", "--output", default="analysis.json")

    c = sub.add_parser(
        "compose",
        help="Convert analysis.json into a SongSpec 1.0 for text-to-compose",
    )
    c.add_argument("analysis", help="Completed analysis.json")
    c.add_argument("-o", "--output", default="song-spec.json")
    c.add_argument("--gemini-model", default=None,
                   help="Force this Gemini model first (default: the order in the model config; see `video-analysis models`)")

    c.add_argument("--voice-timeline", default=None,
                   help="voiceover-timeline.json; plans a sparse bed under narration")

    mo = sub.add_parser(
        "models",
        help="Show the Gemini model order each step will use, and where it comes from",
    )
    mo.add_argument("--json", action="store_true", help="Machine-readable output (used by the studio)")
    mo.add_argument("--write-defaults", action="store_true",
                    help="Create the config file with the built-in order, if it does not exist")
    mo.add_argument("--save", metavar="JSON", default=None,
                    help="Validate and save a new config: a JSON document, or - to read it from stdin")

    args = parser.parse_args()

    if args.command == "models":
        raise SystemExit(_models_command(args))

    if args.command == "analyze":
        deterministic = VideoAnalyzer(
            scene_detection=not args.no_scenes,
            sample_interval=args.interval,
        )
        provider = None
        if args.semantic == "gemini":
            from .semantic.gemini import GeminiSemanticProvider
            provider = GeminiSemanticProvider(model=args.gemini_model)
        result = VideoAnalysisPipeline(
            deterministic_analyzer=deterministic,
            semantic_provider=provider,
        ).analyze(args.video)
        text = json.dumps(result, indent=2)
        if args.output:
            Path(args.output).write_text(text + "\n", encoding="utf-8")
        else:
            print(text)

    elif args.command == "merge":
        write_merged_analysis(args.deterministic, args.semantic, args.output)
        print(f"Merged analysis written to {args.output}")

    elif args.command == "compose":
        provider = GeminiCompositionProvider(model=args.gemini_model)
        plan = provider.compose(args.analysis, voice_timeline=args.voice_timeline)
        song_spec = compile_song_spec(plan)
        Path(args.output).write_text(
            json.dumps(song_spec, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"SongSpec 1.0 written to {args.output}")


def _models_command(args) -> int:
    import sys

    from . import model_config
    from .gemini import DEFAULT_MAX_ATTEMPTS, STEP_DEFAULTS, api_keys

    try:
        if args.save is not None:
            text = sys.stdin.read() if args.save == "-" else args.save
            try:
                document = json.loads(text)
            except json.JSONDecodeError as exc:
                raise model_config.ModelConfigError(f"--save needs a JSON document: {exc}") from exc
            print(f"Saved {model_config.save(document)}", file=sys.stderr)
        if args.write_defaults:
            path = model_config.config_path()
            if path.exists():
                print(f"{path} already exists; left unchanged", file=sys.stderr)
            else:
                model_config.save({step: list(STEP_DEFAULTS[step]) for step in model_config.STEPS}
                                  | {"max_attempts": DEFAULT_MAX_ATTEMPTS})
                print(f"Wrote {path}", file=sys.stderr)
        config = model_config.load() or {}
        cap, cap_source = model_config.max_attempts_setting(DEFAULT_MAX_ATTEMPTS)
        steps = {}
        for step in model_config.STEPS:
            order, source = model_config.step_models(step, list(STEP_DEFAULTS[step]))
            steps[step] = {
                "label": model_config.STEP_LABEL[step],
                "order": order,
                "source": source,
                "configured": config.get(step, list(STEP_DEFAULTS[step])),
                "disabled": config.get("disabled", {}).get(step, []),
                "defaults": list(STEP_DEFAULTS[step]),
            }
    except model_config.ModelConfigError as exc:
        print(f"Model config error: {exc}", file=sys.stderr)
        return 2
    report = {
        "config_path": str(model_config.config_path()),
        "config_exists": model_config.config_path().is_file(),
        "max_attempts": cap,
        "max_attempts_source": cap_source,
        "key_count": len(api_keys()),
        "steps": steps,
    }
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print(f"Config file: {report['config_path']}" + ("" if report["config_exists"] else " (not created yet)"))
    print(f"Attempt cap: {cap} ({cap_source}); API keys found: {report['key_count']}")
    for step, info in steps.items():
        print(f"\n{info['label']} ({info['source']}):")
        for n, model in enumerate(info["order"], start=1):
            print(f"  {n}. {model}")
        if info["disabled"]:
            print(f"  switched off: {', '.join(info['disabled'])}")
    return 0


if __name__ == "__main__":
    main()
