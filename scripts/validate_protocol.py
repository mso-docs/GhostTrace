#!/usr/bin/env python3
"""Validate protocol/token alignment using the pinned local tokenizer; no inference."""
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer
from ghosttrace.config import read_dataset,canonical,digest,now,EngineError
from ghosttrace.engine.render import messages,align_continuation

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',default='datasets/synthetic-v1');p.add_argument('--repo',default='Qwen/Qwen2.5-0.5B-Instruct');p.add_argument('--revision',required=True);p.add_argument('--output',required=True)
a=p.parse_args()
if len(a.revision)!=40 or any(c not in '0123456789abcdef' for c in a.revision):p.error('Revision must be an immutable 40-character hex commit')
source=Path(a.dataset);rows=read_dataset(source/'scenarios.jsonl');templates=json.loads((source/'templates.json').read_text());results=[]
tokenizer=AutoTokenizer.from_pretrained(a.repo,revision=a.revision,local_files_only=True,trust_remote_code=False)
if (source/'factual-controls.jsonl').exists():rows+=read_dataset(source/'factual-controls.jsonl')
for scenario in rows:
    expected=None;positions=[]
    for template in ('A','B'):
        for condition in ('neutral','remorseful','unremorseful'):
            rendered=tokenizer.apply_chat_template(messages(scenario,condition,template,templates),tokenize=False,add_generation_prompt=True)
            aligned=align_continuation(tokenizer,rendered,scenario['common_continuation'],expected)
            if not 16<=len(aligned['continuation_ids'])<=32:raise EngineError('INVALID_CONFIG','Continuation must be 16–32 tokens',{'scenario_id':scenario['id']})
            expected=aligned['continuation_ids'];positions.append({'template':template,'condition':condition,'prefix_token_count':aligned['prefix_token_count'],'total_tokens':len(aligned['token_ids'])})
    results.append({'scenario_id':scenario['id'],'continuation_token_count':len(expected),'continuation_ids':expected,'alignment_verified':True,'positions':positions,'absolute_position_confound':len({v['prefix_token_count'] for v in positions})>1})
report={'schema_version':'1.0','created_at':now(),'repo_id':a.repo,'tokenizer_revision':a.revision,'dataset_hash':digest(rows),'templates_hash':digest(templates),'passed':True,'scenarios_checked':len(rows),'human_review_completed':all(s.get('review_status')=='reviewed' for s in rows),'results':results}
output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_bytes(canonical(report));print(json.dumps({'passed':True,'scenarios_checked':len(rows),'output':str(output),'human_review_completed':report['human_review_completed']}))
