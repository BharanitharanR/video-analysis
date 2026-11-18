from __future__ import annotations
import argparse, json
from pathlib import Path
from .analyzer import VideoAnalyzer
from .pipeline import VideoAnalysisPipeline
from .merge import write_merged_analysis


def main():
    parser=argparse.ArgumentParser(prog="video-analysis")
    sub=parser.add_subparsers(dest="command",required=True)

    a=sub.add_parser("analyze")
    a.add_argument("video")
    a.add_argument("-o","--output",default=None)
    a.add_argument("--no-scenes",action="store_true")
    a.add_argument("--interval",type=float,default=1.0)
    a.add_argument("--semantic",choices=["none","gemini"],default="none",help="Enable semantic AI analysis")
    a.add_argument("--gemini-model",default="gemini-4-hydrogen",help="Gemini model used when --semantic=gemini")

    m=sub.add_parser("merge",help="Merge deterministic.json and semantic.json into analysis.json")
    m.add_argument("deterministic",help="Path to deterministic analysis JSON")
    m.add_argument("semantic",help="Path to semantic analysis JSON")
    m.add_argument("-o","--output",default="analysis.json",help="Output merged analysis JSON")

    args=parser.parse_args()

    if args.command=="analyze":
        deterministic=VideoAnalyzer(scene_detection=not args.no_scenes,sample_interval=args.interval)
        provider=None
        if args.semantic == "gemini":
            from .semantic.gemini import GeminiSemanticProvider
            provider=GeminiSemanticProvider(model=args.gemini_model)
        result=VideoAnalysisPipeline(
            deterministic_analyzer=deterministic,
            semantic_provider=provider,
        ).analyze(args.video)
        text=json.dumps(result,indent=2)
        if args.output: Path(args.output).write_text(text+"\n",encoding="utf-8")
        else: print(text)

    elif args.command=="merge":
        write_merged_analysis(args.deterministic,args.semantic,args.output)
        print(f"Merged analysis written to {args.output}")


if __name__=="__main__": main()
