#!/usr/bin/env python3
"""Attach explicitly nonlearned coordinate controls and a separately fitted lens."""
import argparse,json,zipfile
from pathlib import Path
import numpy as np
from ghosttrace.config import canonical,file_digest
from ghosttrace.storage.db import Store
from ghosttrace.research.storage import ResearchRepository
from ghosttrace.research.service import get_bundle
from ghosttrace.research.features import import_dictionary,encode_features
from ghosttrace.research.extensions import intervene_features,apply_lens_job,attribute_features
p=argparse.ArgumentParser();p.add_argument('--directory',default='artifacts/mechanistic/service-validation-final');a=p.parse_args()
out=Path(a.directory);store=Store();repo=ResearchRepository(store);sid=(out/'study-id.txt').read_text().strip();bundle=get_bundle(store,sid);profile=bundle['study']['model_profile'];width=profile['hidden_size']
weights=out/'coordinate-weights.npz';np.savez_compressed(weights,W_enc=np.eye(width,dtype=np.float32),b_enc=np.zeros(width,dtype=np.float32),W_dec=np.eye(width,dtype=np.float32),b_dec=np.zeros(width,dtype=np.float32))
meta={'name':'Identity coordinate diagnostic — not a trained SAE','kind':'sae','model_repo_id':profile['repo_id'],'model_revision':profile['revision'],'dictionary_revision':'coordinate-control-v1','license':'CC0-1.0','input_site':'resid_post','output_site':'resid_post','layer':11,'input_width':width,'output_width':width,'feature_count':width,'preprocessing':'none','encoder_activation':'identity','normalization_state':'raw_component','fidelity_evidence':'Exact identity coordinate basis for software and reconstruction controls only. No learned features or semantic interpretations claimed.'}
package=out/'coordinate-dictionary.zip'
with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:z.writestr('dictionary.json',canonical({'metadata':meta,'weights_sha256':file_digest(weights)}));z.write(weights,'weights.npz')
dictionary=import_dictionary(store,package)
for capture in bundle['captures']:
    feature=encode_features(store,sid,capture['id'],dictionary['id']);print(json.dumps({'feature':feature['id'],'condition':capture['condition_id']}),flush=True)
recipient=next(c for c in bundle['captures'] if c['condition_id']=='recipient')
payload={'study_id':sid,'capture_id':recipient['id'],'dictionary_id':dictionary['id'],'feature_indices':[0],'operation':'ablate','value':0}
edit=intervene_features(store,payload);print(json.dumps({'feature_intervention':edit['id'],'metrics':edit['metrics']}),flush=True)
attribution=attribute_features(store,{'study_id':sid,'capture_id':recipient['id'],'dictionary_id':dictionary['id'],'top_k':64});print(json.dumps({'feature_attribution':attribution['id']}),flush=True)
lens=json.loads(Path('artifacts/mechanistic/lens-diagnostic/result.json').read_text())
for capture in bundle['captures']:
    result=apply_lens_job(store,{'study_id':sid,'capture_id':capture['id'],'lens_id':lens['id'],'decode':True,'layers':[11]});print(json.dumps({'lens_result':result['id']}),flush=True)
bundle=get_bundle(store,sid);(out/'bundle.json').write_bytes(canonical(bundle));print(json.dumps({'study_id':sid,'dictionary_id':dictionary['id']}),flush=True)
