#!/usr/bin/env python3
"""Explicit model download: resolves an immutable commit and inventories safetensors."""
import argparse
import json
import sys
from pathlib import Path

parser=argparse.ArgumentParser(description='Explicitly download a Qwen model (~1–3 GB) to the Hugging Face local cache. Never called by replay or inference.')
parser.add_argument('--repo',choices=['Qwen/Qwen2.5-0.5B-Instruct','Qwen/Qwen2.5-1.5B-Instruct'],default='Qwen/Qwen2.5-0.5B-Instruct')
parser.add_argument('--revision',default='main')
parser.add_argument('--output',default='artifacts/model-download.json')
args=parser.parse_args()
from huggingface_hub import HfApi,snapshot_download
from ghosttrace.config import file_digest,now
info=HfApi().model_info(args.repo,revision=args.revision)
print(f'Downloading {args.repo} at immutable commit {info.sha}. Weights may require several GB; progress follows.',file=sys.stderr)
path=Path(snapshot_download(args.repo,revision=info.sha,allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja','LICENSE','LICENSE.*'],ignore_patterns=['*.bin','*.pt','*.pth']))
weights=list(path.glob('*.safetensors'))
if not weights:raise SystemExit('No Safetensors weights found; model unavailable.')
record={'schema_version':'1.0','created_at':now(),'repo_id':args.repo,'revision':info.sha,'tokenizer_revision':info.sha,'snapshot_path':str(path),'files':{p.name:file_digest(p) for p in path.iterdir() if p.is_file()},'total_bytes':sum(p.stat().st_size for p in path.iterdir() if p.is_file())}
output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
