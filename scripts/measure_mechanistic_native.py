#!/usr/bin/env python3
"""Opt-in, offline native Qwen mech-interp validation; never downloads weights."""
import argparse
import json
from pathlib import Path
import numpy as np
from ghosttrace.config import canonical,now,digest
from ghosttrace.models.mechanistic import MechanisticAdapter
from ghosttrace.storage.db import Store

p=argparse.ArgumentParser();p.add_argument('--output',default='artifacts/mechanistic/native-validation');p.add_argument('--profile',default='qwen25-05b-cpu');a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=True);adapter=MechanisticAdapter()
try:
    profile=Store().get_record('profiles',a.profile);adapter.load(profile);adapter.seed(73)
    prompts={}
    for condition,color in [('donor','blue'),('recipient','red')]:
        messages=[{'role':'system','content':'You are a precise copying assistant. Answer with only the code color word, no punctuation.'},{'role':'user','content':f'The code color is {color}. Repeat the code color:'}]
        rendered=adapter.render(messages);ids=adapter.tokenize(rendered);prompts[condition]={'messages':messages,'rendered':rendered,'token_ids':ids}
    positive=adapter.tokenize('blue');negative=adapter.tokenize('red')
    if len(positive)!=1 or len(negative)!=1:raise SystemExit('Pinned single-token targets required; do not silently replace the preregistered target.')
    target={'kind':'next_token_logit_diff','positive_token_id':positive[0],'negative_token_id':negative[0]}
    donor=prompts['donor']['token_ids'];recipient=prompts['recipient']['token_ids']
    if len(donor)!=len(recipient):raise SystemExit('Copying prompts were not position-matched; no automatic word selection permitted.')
    positions=list(range(len(donor)-3,len(donor)));layers=[0,len(adapter.blocks)//2-1,len(adapter.blocks)-1]
    result={'schema_version':'2.0','created_at':now(),'measurement_status':'measured','purpose':'Prespecified copying diagnostic; one example, not a population-level mechanistic claim.','prompts':prompts,'target':target,'captures':{},'interventions':{},'profile':adapter.describe()}
    for condition,prompt in prompts.items():
        capture=adapter.inspect(prompt['token_ids'],layers=layers,positions=positions,target=target,attention=True)
        arrays=capture.pop('artifact_arrays');np.savez_compressed(out/f'{condition}.npz',**arrays);capture['arrays_file']=f'{condition}.npz';result['captures'][condition]=capture
    for name,site,layer,head,channel,operation in [('residual_patch','resid_post',layers[1],None,None,'patch'),('last_residual_positive_control','resid_post',layers[-1],None,None,'patch'),('head_zero','head_pre',layers[1],0,None,'zero_ablate'),('mlp_mean','mlp_post_gate',layers[1],None,0,'mean_ablate')]:
        r=adapter.intervene(donor,recipient,site=site,layer=layer,positions=[positions[-1]],operation=operation,target=target,head=head,channel=channel);r.pop('artifact_arrays');result['interventions'][name]=r
    attribution=adapter.attribute(donor,recipient,site='resid_post',layer=layers[1],positions=[positions[-1]],target=target)
    np.savez_compressed(out/'attribution.npz',**attribution.pop('artifact_arrays'));result['attribution']=attribution
    path=adapter.path_patch(donor,recipient,source_site='resid_post',source_layer=layers[1],destination_site='resid_pre',destination_layer=layers[-1],positions=[positions[-1]],target=target);path.pop('artifact_arrays');result['path']=path
    (out/'results.json').write_bytes(canonical(result))
    print(json.dumps({'output':str(out),'token_count':len(donor),'positions':positions,'target':target,'capture_parity':{c:r['metrics']['logit_parity_max_abs_error'] for c,r in result['captures'].items()},'interventions':{c:r['metrics'] for c,r in result['interventions'].items()},'attribution':attribution['metrics'],'path':path['metrics']},indent=2))
finally:adapter.unload()
