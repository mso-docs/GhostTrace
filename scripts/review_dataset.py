#!/usr/bin/env python3
"""Prepare/apply an explicit human review, producing a new dataset version."""
import argparse
import json
from pathlib import Path
from ghosttrace.config import EngineError
from ghosttrace.review import prepare, apply, render_html

parser=argparse.ArgumentParser(description=__doc__)
sub=parser.add_subparsers(dest='command',required=True)
p=sub.add_parser('prepare');p.add_argument('--source',default='datasets/synthetic-v1');p.add_argument('--output',required=True);p.add_argument('--html',help='Optional self-contained offline HTML review page')
p=sub.add_parser('apply');p.add_argument('--source',default='datasets/synthetic-v1');p.add_argument('--decisions',required=True);p.add_argument('--output',required=True);p.add_argument('--reviewer',required=True);p.add_argument('--attest-human',action='store_true')
a=parser.parse_args()
try:
    if a.command=='prepare':
        out=Path(a.output)
        if out.exists():raise EngineError('CONFLICT','Review packet output already exists')
        packet=prepare(a.source)
        out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(packet,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        if a.html:
            html=Path(a.html)
            if html.exists():raise EngineError('CONFLICT','HTML review output already exists')
            html.parent.mkdir(parents=True,exist_ok=True);html.write_text(render_html(packet),encoding='utf-8')
        print(f'Prepared review packet: {out}. A human must review all entries; no item was approved.')
    else:print(json.dumps(apply(a.source,json.loads(Path(a.decisions).read_text()),a.output,a.reviewer,a.attest_human),indent=2))
except EngineError as e:parser.exit(1,json.dumps({'error':{'code':e.code,'message':e.message}})+'\n')
