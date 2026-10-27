"""Explicit research artifact application and training; no implicit model fetches."""
from ghosttrace.config import EngineError,digest
from ghosttrace.research.storage import ResearchRepository
from ghosttrace.research.service import _load,resolve_target

def apply_lens_job(store,payload):
    from ghosttrace.research.lenses import apply_lens
    repo=ResearchRepository(store);study=repo.get(payload['study_id'],'study');adapter=None
    try:
        decoder=None
        if payload.get('decode',True):
            adapter=_load(store,study);target=resolve_target(adapter,study['config']['target'])
            decoder=lambda vectors,layer:adapter.decode_residual(vectors,target=target,topk=min(20,study['config']['capture']['top_k']))
        return apply_lens(store,study['id'],payload['capture_id'],payload['lens_id'],decoder=decoder,layers=payload.get('layers'))
    finally:
        if adapter is not None:adapter.unload()

def train_lens_package(store,path):
    """Safe supplied calibration matrices; fitting never selects development by outcome."""
    from ghosttrace.research.features import _read_package
    from ghosttrace.research.lenses import fit_lens_from_arrays,LensMetadata
    temp,document,arrays,archive_hash=_read_package(path,'training.json');adapter=None
    try:
        import numpy as np
        metadata=LensMetadata.model_validate(document.get('metadata',{})).model_dump(mode='json')
        profile_id=document.get('model_profile_id');profile=store.get_record('profiles',profile_id)
        if not profile.get('preflight_passed') or metadata['model_repo_id']!=profile['repo_id'] or metadata['model_revision']!=profile['revision'] or metadata['hidden_size']!=profile['hidden_size']:raise EngineError('MODEL_NOT_READY','Calibration requires an exact compatible validated local model profile')
        if len(metadata['layers'])*metadata['hidden_size']**2*4>100_000_000:raise EngineError('DISK_QUOTA','Selected affine lens weights would exceed the 100 MB per-tensor bound; train fewer layers')
        expected={f'inputs.L{layer}' for layer in metadata['layers']}|{'targets'}
        if set(arrays)!=expected:raise EngineError('INVALID_BUNDLE','Calibration arrays must contain targets and inputs.L<block> for every declared layer')
        groups=document.get('group_ids');mask=document.get('development_mask')
        if not isinstance(groups,list) or not isinstance(mask,list) or len(groups)!=len(mask) or any(type(v)is not bool for v in mask):raise EngineError('INVALID_CONFIG','Supply explicit boolean development_mask and same-length group_ids')
        decoder=None;optimization=document.get('optimization') or {}
        if metadata['objective']=='predictive_kl':
            study={'model_profile':profile,'model_profile_hash':digest(profile),'config':{'model_profile_id':profile_id,'order_seed':optimization.get('seed',73)}}
            adapter=_load(store,study);decoder=lambda vectors:adapter.residual_logits_torch(vectors,normalized=False)
        record=fit_lens_from_arrays(store,metadata,{layer:arrays[f'inputs.L{layer}'] for layer in metadata['layers']},arrays['targets'],np.asarray(mask,dtype=np.bool_),groups,ridge=document.get('ridge',1.0),decoder=decoder,optimization=optimization)
        return record
    finally:
        temp.cleanup()
        if adapter is not None:adapter.unload()

def intervene_features(store,payload):
    import numpy as np
    from ghosttrace.research.features import validate_feature_pair,encode_dictionary,modify_features,decode_features
    from ghosttrace.research.arrays import read_arrays
    from ghosttrace.research.service import _tokens
    repo=ResearchRepository(store);study=repo.get(payload['study_id'],'study');pair=validate_feature_pair(store,study['id'],payload['capture_id'],payload['dictionary_id']);capture=repo.get(payload['capture_id'],'capture');dictionary=repo.get(payload['dictionary_id'],'dictionary')
    indices=payload['feature_indices']
    if any(index>=dictionary['metadata']['feature_count'] for index in indices):raise EngineError('INVALID_CONFIG','Feature index exceeds compatible dictionary')
    capture_path,capture_sha=repo.artifact_location(capture);dictionary_path,dictionary_sha=repo.artifact_location(dictionary)
    captured=read_arrays(capture_path,capture_sha);weights=read_arrays(dictionary_path,dictionary_sha)
    source=captured[pair['input_component']['array_key']];features,reconstruction=encode_dictionary(source,dictionary['metadata'],weights)
    edit=modify_features(features,dictionary['metadata'],weights,indices,'zero' if payload.get('operation','ablate')=='ablate' else payload['operation'],payload.get('value',0.0))
    decoded=edit['reconstruction']
    adapter=_load(store,study)
    try:
        condition=next(c for c in study['config']['conditions'] if c['id']==capture['condition_id']);text,ids=_tokens(adapter,condition,study['config']['max_input_tokens'])
        if ids!=capture['token_ids']:raise EngineError('TOKEN_ALIGNMENT_FAILED','Current pinned rendering differs from the source capture')
        target=resolve_target(adapter,study['config']['target']);output=pair['output_component'];positions=output['token_indices']
        result=adapter.feature_intervene(ids,site=dictionary['metadata']['output_site'],layer=dictionary['metadata']['layer'],positions=positions,reconstruction=np.asarray(reconstruction,dtype=np.float32),modified_reconstruction=np.asarray(decoded,dtype=np.float32),target=target,feature_indices=indices,dictionary_id=dictionary['id'],seed=study['config']['order_seed'])
        from ghosttrace.research.service import _artifact
        import uuid
        record_id=str(uuid.uuid4());artifact=_artifact(store,study,record_id,result.pop('artifact_arrays',{}))
        return repo.put('feature_intervention',{'id':record_id,'study_id':study['id'],'capture_id':capture['id'],'dictionary_id':dictionary['id'],'artifact':artifact,'spec':{**payload,'kind':'feature','site':dictionary['metadata']['output_site'],'layer':dictionary['metadata']['layer'],'positions':positions},'raw_delta':result.get('metrics',{}).get('delta'),'status':'completed','metrics':result.get('metrics',{}),'metadata':result.get('metadata',{}),'graph':result.get('graph',{}),'evidence_type':'intervention','provenance':{'capture_record_hash':capture['record_hash'],'dictionary_record_hash':dictionary['record_hash'],'weights_sha256':dictionary_sha,'backend':adapter.describe()},'limitations':['The edit preserves the recorded reconstruction error; reconstruction-only and self controls are reported separately.','An effect of a dictionary coordinate is not proof of a semantic or monosemantic feature.']},study['id'])
    finally:adapter.unload()


def attribute_features(store,payload):
    """First-order decoded-coordinate zero-ablation estimates with explicit residuals."""
    import numpy as np
    import uuid
    from ghosttrace.research.features import validate_feature_pair,encode_dictionary,reconstruction_metrics
    from ghosttrace.research.arrays import read_arrays
    from ghosttrace.research.service import _tokens,_artifact
    repo=ResearchRepository(store);study=repo.get(payload['study_id'],'study');pair=validate_feature_pair(store,study['id'],payload['capture_id'],payload['dictionary_id']);capture=repo.get(payload['capture_id'],'capture');dictionary=repo.get(payload['dictionary_id'],'dictionary')
    path,sha=repo.artifact_location(capture);values=read_arrays(path,sha);weights_path,weights_sha=repo.artifact_location(dictionary);weights=read_arrays(weights_path,weights_sha)
    x=values[pair['input_component']['array_key']];y=values[pair['output_component']['array_key']];features,reconstruction=encode_dictionary(x,dictionary['metadata'],weights)
    error=(y-reconstruction).astype(np.float32);adapter=_load(store,study)
    try:
        condition=next(c for c in study['config']['conditions'] if c['id']==capture['condition_id']);text,ids=_tokens(adapter,condition,study['config']['max_input_tokens'])
        if ids!=capture['token_ids']:raise EngineError('TOKEN_ALIGNMENT_FAILED','Attribution rendering differs from its source capture')
        target=resolve_target(adapter,study['config']['target']);positions=pair['output_component']['token_indices']
        native=adapter.attribute(ids,ids,site=dictionary['metadata']['output_site'],layer=dictionary['metadata']['layer'],positions=positions,target=target)
        gradient=native.get('artifact_arrays',{}).get('gradient')
        if gradient is None or gradient.shape!=y.shape:raise EngineError('UNSUPPORTED_CAPTURE','A compatible original-model component gradient is required')
        gradients_by_feature=gradient.astype(np.float64) @ weights['W_dec'].astype(np.float64).T
        token_estimates=-(features.astype(np.float64)*gradients_by_feature);estimates=token_estimates.sum(axis=0)
        selected=np.argsort(-np.abs(estimates),kind='stable')[:payload.get('top_k',64)].tolist();record_id=str(uuid.uuid4())
        artifact=_artifact(store,study,record_id,{'component_gradient':gradient.astype(np.float32),'feature_zero_ablation_estimates':estimates.astype(np.float32),'token_feature_estimates':token_estimates.astype(np.float32),'reconstruction_error':error})
        target_node=f'feature-attribution:{record_id}:target';nodes=[{'id':target_node,'kind':'target','label':'Specified target metric','evidence_kind':'attribution','measured':False}];edges=[];rows=[]
        tests=[]
        for intervention in repo.list('feature_intervention',study['id']):
            if intervention.get('capture_id')==capture['id'] and intervention.get('dictionary_id')==dictionary['id'] and intervention.get('status')=='completed' and intervention.get('spec',{}).get('operation')=='ablate':
                indices=intervention['spec']['feature_indices'];estimate=float(estimates[indices].sum());actual=intervention['metrics']['delta'];tests.append({'intervention_id':intervention['id'],'feature_indices':indices,'estimated_delta':estimate,'exact_delta':actual,'linearization_error':estimate-actual,'controls':intervention['metrics'].get('controls',{})})
        for index in selected:
            identifier=f'dictionary:{dictionary["id"]}:capture:{capture["id"]}:feature:{index}';value=float(estimates[index]);matched=[test['intervention_id'] for test in tests if test['feature_indices']==[index]]
            row={'feature_index':index,'zero_ablation_estimated_delta':value,'max_absolute_activation':float(np.abs(features[:,index]).max()),'exact_intervention_ids':matched};rows.append(row)
            nodes.append({'id':identifier,'kind':'feature','feature_index':index,'layer':dictionary['metadata']['layer'],'site':dictionary['metadata']['output_site'],'label':f'F{index} · literal dictionary coordinate','label_status':'uninterpreted_coordinate','evidence_kind':'attribution','measured':False,'exact_intervention_ids':matched})
            edges.append({'id':identifier+'->'+target_node,'source':identifier,'target':target_node,'kind':'attribution','evidence_type':'attribution','value':value,'measured':False,'label':'First-order zero-coordinate edit estimate'})
        error_contribution=float(-(error.astype(np.float64)*gradient.astype(np.float64)).sum());bias_contribution=float(-(weights['b_dec'].astype(np.float64)[None,:]*gradient.astype(np.float64)).sum())
        for suffix,label,value,kind in [('error','Unexplained reconstruction residual',error_contribution,'error'),('bias','Decoder bias contribution',bias_contribution,'component')]:
            identifier=f'feature-attribution:{record_id}:{suffix}';nodes.append({'id':identifier,'kind':kind,'label':label,'evidence_kind':'attribution','measured':False});edges.append({'id':identifier+'->'+target_node,'source':identifier,'target':target_node,'kind':'attribution','evidence_type':'attribution','value':value,'measured':False,'label':'First-order contribution; retained separately'})
        total=float(np.abs(estimates).sum());visible=float(np.abs(estimates[selected]).sum())
        coverage={'displayed_features':len(selected),'total_features':len(estimates),'hidden_node_count':len(estimates)-len(selected),'visible_absolute_estimate_fraction':visible/total if total>1e-12 else None,'coverage_scope':'Fraction of absolute first-order coordinate estimates, not causal completeness or dictionary faithfulness.'}
        graph={'schema_version':'2.0','nodes':nodes,'edges':edges,'hidden_node_count':coverage['hidden_node_count'],'hidden_edge_count':coverage['hidden_node_count'],'coverage':coverage}
        return repo.put('feature_attribution',{'id':record_id,'study_id':study['id'],'capture_id':capture['id'],'dictionary_id':dictionary['id'],'status':'completed','method':'first_order_feature_zero_ablation','target':target,'baseline':native['metrics']['baseline'],'features':rows,'gradient_scope':{'site':dictionary['metadata']['output_site'],'layer':dictionary['metadata']['layer'],'token_indices':positions},'reconstruction':reconstruction_metrics(y,reconstruction,features),'error_estimated_contribution':error_contribution,'decoder_bias_estimated_contribution':bias_contribution,'exact_intervention_comparisons':tests,'graph':graph,'artifact':artifact,'coverage':coverage,'provenance':{'capture_record_hash':capture['record_hash'],'dictionary_record_hash':dictionary['record_hash'],'weights_sha256':weights_sha,'backend':adapter.describe()},'limitations':['Every feature edge is a local first-order approximation, never an automatically discovered causal path.','Ablation estimates preserve the original reconstruction error; decoder bias and unexplained residual contributions remain separate.','Ranking by absolute estimate is display selection; omitted features are counted and their full arrays retained.','Exact original-model coordinate interventions, where present, are reported separately; untested features remain untested.']},study['id'])
    finally:adapter.unload()
