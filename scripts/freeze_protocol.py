#!/usr/bin/env python3
"""Explicit human preregistration freeze. Never runs or unseals a holdout."""
import argparse
import json
from pathlib import Path
import yaml
from ghosttrace.config import load_config,read_dataset,digest,file_digest,now,EngineError
from ghosttrace.storage.artifacts import atomic_json,atomic_bytes

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/balanced.yaml');p.add_argument('--model-profile',choices=['qwen25-05b-cpu','qwen25-15b-local-validated']);p.add_argument('--dataset',required=True);p.add_argument('--protocol',default='preregistrations/pilot-v1.md');p.add_argument('--output-protocol',required=True);p.add_argument('--output-config',required=True);p.add_argument('--reviewer',required=True);p.add_argument('--attest-human',action='store_true')
a=p.parse_args()
try:
    if not a.attest_human or not a.reviewer.strip():raise EngineError('INVALID_CONFIG','Explicit human review and preregistration attestation required')
    dataset=Path(a.dataset);review=json.loads((dataset/'human-review.json').read_text());scenarios=read_dataset(dataset/'scenarios.jsonl');templates=json.loads((dataset/'templates.json').read_text())
    if review.get('reviewed_dataset_hash')!=digest(scenarios) or review.get('reviewed_templates_hash')!=digest(templates):raise EngineError('CONFLICT','Reviewed dataset or templates changed; repeat human review')
    config=load_config(a.config)
    if a.model_profile:config['model_profile']=a.model_profile
    config['dataset']=str(dataset);config['preregistration']=a.output_protocol
    outputs=[Path(a.output_protocol),Path(a.output_protocol).with_suffix('.json'),Path(a.output_config)]
    if any(out.exists() for out in outputs):raise EngineError('CONFLICT','Freeze outputs already exist; choose a new version')
    timestamp=now();body=Path(a.protocol).read_text()
    frozen=f'# Frozen preregistration receipt\n\nFrozen at {timestamp} by {a.reviewer.strip()}. The named human attests that this protocol was reviewed and fixed before confirmatory results were accessed.\n\nDataset SHA-256: `{digest(scenarios)}`\n\nHuman review SHA-256: `{digest(review)}`\n\n---\n\n'+body
    atomic_bytes(outputs[0],frozen.encode());receipt={'schema_version':'1.0','created_at':timestamp,'reviewer':a.reviewer.strip(),'attestation':'Protocol fixed before confirmatory results were accessed.','config_hash':digest(config),'dataset_hash':digest(scenarios),'template_hash':digest(templates),'human_review_hash':digest(review),'protocol_sha256':file_digest(outputs[0])}
    atomic_json(outputs[1],receipt);atomic_bytes(outputs[2],yaml.safe_dump(config,sort_keys=False).encode())
    print(json.dumps({'frozen':True,'config':str(outputs[2]),'protocol':str(outputs[0]),'receipt':str(outputs[1]),'next_step':'Validate the reviewed dataset against the pinned tokenizer and freeze an experiment after hardware preflight.'},indent=2))
except (EngineError,OSError,ValueError) as e:p.exit(1,str(e)+'\n')
