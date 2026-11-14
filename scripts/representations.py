#!/usr/bin/env python3
"""Representation-learning campaign using the shared gated optimization workflow."""
import argparse,json,logging
from pathlib import Path
import campaign as c


def load(phase='discovery'):
    c.load(phase,config_path=Path(__file__).resolve().parents[1]/'configs/representations.yaml',
           additional_sources=['scripts/representations.py','scripts/denoise_gpu.py','stealth/representation_models.py'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['init','cpu','select','confirm','package','report'])
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    load('confirmation' if args.stage in ('confirm','package') else 'discovery')
    if args.stage=='cpu':c.cpu()
    elif args.stage=='select':print(json.dumps(c.selection(),indent=2))
    elif args.stage=='confirm':print(json.dumps(c.confirm(),indent=2))
    elif args.stage=='package':print(c.package())
    elif args.stage=='report':c.r.report()


if __name__=='__main__':main()
