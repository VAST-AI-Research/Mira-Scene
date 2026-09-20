#!/usr/bin/env python3
"""Automatic or interactive Mira-Scene segmentation.

Automatic: ``python infer_scripts/0_segmentation.py --input image.jpg --output out --config config.yml``
Review UI: ``python infer_scripts/0_segmentation.py --web --input images --output out --config config.yml``
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path: sys.path.insert(0,str(HERE.parent))

from infer_scripts.core.cases import discover_images, prepare_cases
from infer_scripts.core.config import get, load_config, path_value
from infer_scripts.core.manifest import signature, update_stage


def build_engine(config):
    from infer_scripts.segmentation.engine import SegmentationEngine, Stage0Settings
    from infer_scripts.segmentation.backends.sam3 import Sam3Backend
    from infer_scripts.segmentation.backends.vlm import VLMClient
    settings=Stage0Settings(
        object_profile=str(get(config,"segmentation.object_profile","major_v6")),
        room_prompt=str(get(config,"segmentation.room_prompt","list_objects_major_v5.txt")),
        tabletop_prompt=str(get(config,"segmentation.tabletop_prompt","list_objects_tabletop_v1.txt")),
        sam3_confidence=float(get(config,"segmentation.sam3_confidence",.5)),
        recycle=bool(get(config,"segmentation.recycle",True)),
        recycle_verifier_mode=str(get(config,"segmentation.recycle_verifier_mode","identity_upgrade")),
        missing_object_critic_rounds=int(get(config,"segmentation.missing_object_critic_rounds",0)),
        missing_object_critic_prompt=str(get(config,"segmentation.missing_object_critic_prompt","missing_objects_major_v2.txt")),
        missing_object_critic_max_overlap=float(get(config,"segmentation.missing_object_critic_max_overlap",.2)),
        save_debug=bool(get(config,"segmentation.save_debug",False)),
    )
    return SegmentationEngine(
        Sam3Backend(path_value(config,"external.sam3.repo",required=True),
                    path_value(config,"external.sam3.checkpoint",required=True),
                    str(get(config,"segmentation.device","cuda")),
                    float(get(config,"segmentation.sam3_confidence",.5))),
        VLMClient(str(get(config,"segmentation.vlm_model","gemini-2.5-pro")),
                  str(get(config,"api.base_url","https://lumina.tripo3d.com/v1")),
                  float(get(config,"api.timeout",300))), settings)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,help="image or flat image directory")
    p.add_argument("--output",type=Path,required=True,help="case output root")
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--case",action="append",help="existing case id; repeatable")
    p.add_argument("--web",action="store_true")
    p.add_argument("--host",default="127.0.0.1"); p.add_argument("--port",type=int,default=8890)
    p.add_argument("--force",action="store_true"); p.add_argument("--no-scene-graph",action="store_true")
    return p.parse_args()


def main():
    args=parse_args(); config=load_config(args.config); root=args.output.expanduser().resolve(); root.mkdir(parents=True,exist_ok=True)
    if args.web:
        # Web mode prepares input images up front, but deliberately does not
        # invoke the segmentation engine.  Segmentation remains an explicit
        # action in the review UI.
        if args.input:
            prepare_cases(discover_images(args.input), root)
        from infer_scripts.segmentation.interactive.webapp import run_web
        run_web(root,config,args.host,args.port); return
    cases=[]
    if args.input: cases.extend(prepare_cases(discover_images(args.input),root))
    if args.case: cases.extend(root/name for name in dict.fromkeys(args.case))
    cases=list(dict.fromkeys(cases))
    if not cases: raise ValueError("provide --input or --case (or use --web)")
    engine=build_engine(config); failures=[]
    for case in cases:
        try:
            if not args.force and (case/"review/annotation.json").is_file() and (case/"scene_graph.json").is_file():
                print(f"{case.name}: existing segmentation loaded; use --force to replace"); continue
            engine.run(case,generate_graph=not args.no_scene_graph)
            sig=signature(case,"segmentation",config,HERE/"segmentation")
            update_stage(case,"segmentation","complete",sig,args=sys.argv)
            print(f"{case.name}: segmentation complete")
        except Exception as exc:
            failures.append((case.name,exc)); print(f"{case.name}: ERROR {type(exc).__name__}: {exc}")
    if failures: raise SystemExit(1)


if __name__=="__main__":
    from core.stage_logging import run_logged
    run_logged(main,"00_segmentation.log",primary_root_flags=("--output",))
