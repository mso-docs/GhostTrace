#!/usr/bin/env python3
"""Execute the measured v2 copying study through the same queued services as API."""
import argparse
import json
from pathlib import Path
from ghosttrace.storage.db import Store
from ghosttrace.research.service import freeze_study,queue_capture,queue_intervention,get_bundle
from ghosttrace.research.worker import execute_research
from ghosttrace.config import canonical

p=argparse.ArgumentParser();p.add_argument('--config',default='configs/research-copy.json');p.add_argument('--output',default='artifacts/mechanistic/service-validation');a=p.parse_args()
store=Store();study=freeze_study(store,json.loads(Path(a.config).read_text()));out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
def execute(queued):
    job=store.claim()
    if not job or job['id']!=queued['id']:raise SystemExit('Another job is ahead of this diagnostic. Run the explicit worker; do not discard queued jobs.')
    execute_research(store,job);finished=store.job(job['id'])
    print(json.dumps({'job':job['id'],'state':finished['state'],'error_code':finished.get('error_code'),'error_message':finished.get('error_message')}),flush=True)
    if finished['state']!='completed':raise SystemExit(1)
execute(queue_capture(store,study['id'],idempotency_key='measured-v2-copy-components'))
common={'donor_condition_id':'donor','recipient_condition_id':'recipient','site':'resid_post','layer':11,'positions':[-1],'alignment':{'kind':'exact_tokens'}}
for name,spec in [('middle_patch',{}),('final_residual_wiring_control',{'layer':23}),('head_zero',{'site':'head_pre','head':0,'operation':'zero_ablate'}),('mlp_channel_mean',{'site':'mlp_post_gate','channel':0,'operation':'mean_ablate'}),('gradient_estimate',{'kind':'attribution'}),('source_receiver_path',{'kind':'path','destination_site':'resid_pre','destination_layer':23})]:
    execute(queue_intervention(store,study['id'],{**common,**spec},idempotency_key='measured-v2-'+name))
bundle=get_bundle(store,study['id']);(out/'bundle.json').write_bytes(canonical(bundle));(out/'study-id.txt').write_text(study['id']+'\n');print(json.dumps({'study_id':study['id'],'output':str(out),'captures':len(bundle['captures']),'interventions':len(bundle['interventions'])}),flush=True)
