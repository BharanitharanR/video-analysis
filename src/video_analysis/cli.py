from __future__ import annotations
import argparse, json
from pathlib import Path
from .analyzer import VideoAnalyzer

def main():
    parser=argparse.ArgumentParser(prog="video-analysis")
    sub=parser.add_subparsers(dest="command",required=True)
    a=sub.add_parser("analyze")
    a.add_argument("video")
    a.add_argument("-o","--output",default=None)
    a.add_argument("--no-scenes",action="store_true")
    a.add_argument("--interval",type=float,default=1.0)
    args=parser.parse_args()
    if args.command=="analyze":
        result=VideoAnalyzer(scene_detection=not args.no_scenes,sample_interval=args.interval).analyze(args.video)
        text=json.dumps(result,indent=2)
        if args.output: Path(args.output).write_text(text+"\n",encoding="utf-8")
        else: print(text)

if __name__=="__main__": main()
