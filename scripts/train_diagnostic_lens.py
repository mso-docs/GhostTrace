#!/usr/bin/env python3
"""Actual small-corpus predictive-KL lens diagnostic; never a generalization claim."""
import json
from pathlib import Path
import numpy as np
from ghosttrace.config import canonical,digest,now
from ghosttrace.storage.db import Store
from ghosttrace.models.mechanistic import MechanisticAdapter
from ghosttrace.research.lenses import fit_lens_from_arrays

store=Store();adapter=MechanisticAdapter();source=Path('datasets/lens-diagnostic-v1/corpus.jsonl');rows=[json.loads(x) for x in source.read_text().splitlines() if x.strip()];output=Path('artifacts/mechanistic/lens-diagnostic');output.mkdir(parents=True,exist_ok=True)
try:
 profile=store.get_record('profiles','qwen25-05b-cpu');adapter.load(profile);adapter.seed(73);inputs=[];teachers=[]
 for i,row in enumerate(rows):
  ids=adapter.tokenize(adapter.render(row['messages']));vectors=adapter.capture(ids,'residual_post_block_pre_final_norm',[len(ids)-1]);inputs.append(vectors[11,0]);teachers.append(vectors[-1,0]);print(json.dumps({'captured':i+1,'total':len(rows),'split':row['split']}),flush=True)
 x=np.stack(inputs).astype(np.float32);y=np.stack(teachers).astype(np.float32);mask=np.array([r['split']=='development' for r in rows],dtype=np.bool_);groups=[r['group_id'] for r in rows]
 metadata={'schema_version':'2.0-lens','name':'Small generic-corpus predictive-KL diagnostic (not a general lens)','model_repo_id':profile['repo_id'],'model_revision':profile['revision'],'lens_revision':'diagnostic-v1','license':'CC0-1.0','hidden_size':adapter.model.config.hidden_size,'layers':[11],'input_site':'resid_post','normalization_state':'raw_component','transform':'residual_affine_delta','objective':'predictive_kl','training_dataset_sha256':digest(rows),'training_split':'development','holdout_used_for_fit':False,'training_sample_count':int(mask.sum()),'development_group_ids':sorted(set(g for g,m in zip(groups,mask) if m)),'holdout_group_ids':sorted(set(g for g,m in zip(groups,mask) if not m)),'excludes_persona_fixture':True,'training_scope':'small_corpus_diagnostic','training_description':'32 AI-assisted generic prompts from separate topic groups;24 development and8 heldout. Software diagnostic only, not representative language validation. No persona scenarios used.'}
 record=fit_lens_from_arrays(store,metadata,{11:x},y,mask,groups,decoder=adapter.residual_logits_torch,optimization={'steps':8,'batch_size':4,'learning_rate':.002,'seed':73,'max_eval_rows':32})
 np.savez_compressed(output/'calibration-vectors.npz',inputs=x,teacher_final=y,development_mask=mask.astype(np.float32));(output/'result.json').write_bytes(canonical(record));print(json.dumps({'lens_id':record['id'],'metrics':record['metrics'],'output':str(output)},indent=2))
finally:adapter.unload()
