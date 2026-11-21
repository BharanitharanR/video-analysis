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
    a.add_argument("--gemini-model", default="gemini-4-hydrogen",
                   help="Gemini model used when --semantic=gemini")

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
    c.add_argument("--gemini-model", default="gemini-3.6-flash",
                   help="Gemini model used for composition planning")

    c.add_argument("--voice-timeline", default=None,
                   help="voiceover-timeline.json; plans a sparse bed under narration")

    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
