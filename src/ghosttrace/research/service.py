from __future__ import annotations
import json
import random
import time
import uuid
from copy import deepcopy
from pathlib import Path
from ghosttrace.config import EngineError,canonical,digest,now
from ghosttrace.provenance import code_provenance
from ghosttrace.engine.runner import require_compatible_backend,measured_peak_memory
from ghosttrace.research.schemas import StudyCreate,InterventionRequest,HypothesisCreate
from ghosttrace.research.storage import ResearchRepository
from ghosttrace.research.arrays import write_arrays,read_arrays,browser_tile

LIMITATIONS=['Architectural components and coordinates have literal identities; they are not discovered concepts.','Logit lens readouts are exploratory output projections, not model thoughts.','Node intervention effects apply to the specified target, input, site and mask; they do not establish an untested computational path.','Feature labels and hypothesis conclusions are annotations requiring evidence; reconstruction and unexplained residuals remain visible.']

def freeze_study(store,config):
    request=StudyCreate.model_validate(config)
    profile=store.get_record('profiles',request.model_profile_id)
    if not profile.get('preflight_passed'):raise EngineError('MODEL_NOT_READY','Complete a real local model preflight before freezing a research study')
    layers=request.capture.layers or sorted(set([profile['layers']//2-1,profile['layers']-1]))
    if any(layer>=profile['layers'] for layer in layers):raise EngineError('UNSUPPORTED_CAPTURE','Capture block outside validated model')
    maximum_positions=len(request.capture.positions);width=profile['hidden_size']
    heads=profile.get('query_heads',profile.get('num_attention_heads',max(1,width//32)))
    intermediate=profile.get('intermediate_size',width*8);head_dim=profile.get('head_dim',max(1,width//heads))
    per_condition=len(layers)*maximum_positions*4*(7*width+heads*head_dim+heads*width+intermediate+(heads*request.max_input_tokens if request.capture.attention else 0))+maximum_positions*width*4
    if per_condition>64_000_000:raise EngineError('DISK_QUOTA','Estimated native capture exceeds its 64 MB per-forward budget; select fewer layers or token positions',{'estimate':True,'conservative_dimensions':not all(k in profile for k in ('query_heads','intermediate_size','head_dim')),'estimated_bytes':per_condition})
    estimate=len(request.conditions)*per_condition
    if estimate>request.capture.max_artifact_bytes:raise EngineError('DISK_QUOTA','Estimated component payload exceeds the explicit research quota',{'estimated_uncompressed_bytes':estimate,'estimate':True})
    config=request.model_dump(mode='json')
    base={'schema_version':'2.0','config':config,'model_profile':deepcopy(profile),'model_profile_hash':digest(profile),'resolved_capture':{**config['capture'],'layers':layers,'browser_tile_values':max(128,min(4096,1_000_000//max(1,len(request.conditions)*len(layers)*len(request.capture.sites))))},'protocol':'general-mechanistic-study-v2','classification':'exploratory','estimated_uncompressed_bytes':estimate}
    study_hash=digest(base);repo=ResearchRepository(store)
    for prior in repo.list('study'):
        if prior['study_hash']==study_hash:return prior
    study=repo.put('study',{**base,'study_hash':study_hash,'code_provenance':code_provenance(),'measurement_status':'not_run','limitations':LIMITATIONS})
    from ghosttrace.storage.artifacts import atomic_json
    atomic_json(store.root/'research'/study_hash/'study.json',study)
    return study

def merge_graph(captures,interventions,feature_interventions=(),circuits=(),feature_attributions=()):
    nodes={};edges={};dimensions={}
    for record in [*captures,*interventions,*feature_interventions,*circuits,*feature_attributions]:
        graph=record.get('graph',{});external=record.get('source')=='external_import'
        if external:
            prefix=f"external:{record['id']}:";mapping={node['id']:prefix+node['id'] for node in graph.get('nodes',[])}
            graph={**graph,'nodes':[{**node,'id':mapping[node['id']],'external_source_id':node['id'],'parent':mapping.get(node.get('parent'),node.get('parent'))} for node in graph.get('nodes',[])],'edges':[{**edge,'id':prefix+(edge.get('id') or digest(edge)),'source':mapping[edge['source']],'target':mapping[edge['target']]} for edge in graph.get('edges',[])]}
        if not external:
            for key,value in graph.get('dimensions',{}).items():
                if key in dimensions and dimensions[key]!=value:raise EngineError('CONFLICT','Architectural graph dimensions differ within one study')
                dimensions[key]=value
        for node in graph.get('nodes',[]):
            previous=nodes.get(node['id'],{})
            observations=list(previous.get('observation_refs',[]));effects=list(previous.get('intervention_refs',[]))
            if 'condition_id' in record and record.get('status')=='completed':
                for component in record.get('components',[]):
                    if component.get('site')==node.get('site') and component.get('layer')==node.get('layer'):
                        observation={'capture_id':record['id'],'condition_id':record['condition_id'],'array_key':component['array_key'],'token_indices':component.get('token_indices',record.get('selected_positions',[])),'head':node.get('head'),'channel':node.get('channel')}
                        if observation not in observations:observations.append(observation)
            elif record.get('status')=='completed' and 'spec' in record:
                spec=record['spec'];matches_source=spec.get('site')==node.get('site') and spec.get('layer')==node.get('layer')
                matches_destination=spec.get('kind')=='path' and spec.get('destination_site')==node.get('site') and spec.get('destination_layer')==node.get('layer')
                selected_head=spec.get('head') if matches_source else spec.get('destination_head')
                selected_channel=spec.get('channel') if matches_source else spec.get('destination_channel')
                if (matches_source or matches_destination) and (selected_head is None or selected_head==node.get('head')) and (selected_channel is None or selected_channel==node.get('channel')):
                    effect={'intervention_id':record['id'],'delta':record.get('raw_delta'),'estimated_delta':record.get('estimated_delta'),'method':spec['kind'],'operation':spec['operation'],'evidence_type':record.get('evidence_type'),'role':'source' if matches_source else 'receiver'}
                    if effect not in effects:effects.append(effect)
            merged={**previous,**node,'observation_refs':observations,'intervention_refs':effects,'measured':bool(previous.get('measured') or (node.get('measured') and not external) or observations or effects)}
            merged.pop('intervention_effect',None) # No single effect can stand in for multiple tested conditions.
            analyses=list(previous.get('analysis_refs',[]))
            if record.get('method')=='first_order_feature_zero_ablation':analyses.append({'record_id':record['id'],'evidence_type':'attribution'})
            if external:analyses.append({'record_id':record['id'],'evidence_type':'external_reported'})
            merged['analysis_refs']=analyses
            merged['source_record_ids']=sorted({ref['capture_id'] for ref in observations}|{ref['intervention_id'] for ref in effects}|{ref['record_id'] for ref in analyses})
            merged['evidence_types']=sorted(set((['external_reported'] if external else ['structural'] if node.get('kind')!='selected_component' else [])+(['activation'] if observations else [])+[ref.get('evidence_type') or 'intervention' for ref in effects]))
            if observations:merged['evidence_kind']='observed_activation'
            nodes[node['id']]=merged
        for edge in graph.get('edges',[]):
            evidence=edge.get('evidence_type',edge.get('kind','structural'))
            evidence={'estimated_attribution':'attribution'}.get(evidence,evidence)
            key=(edge.get('id') or digest(edge))+(f":{record['id']}" if evidence!='structural' else '')
            edges[key]={**edge,'id':key,'evidence_type':evidence,'evidence_record_id':record['id'] if evidence!='structural' else None,'evidence_origin':'external_reported' if external else 'local_record'}
    return {'nodes':list(nodes.values()),'edges':list(edges.values()),'dimensions':dimensions,'hidden_node_count':sum(r.get('graph',{}).get('hidden_node_count',0) for r in [*circuits,*feature_attributions]),'hidden_edge_count':sum(r.get('graph',{}).get('hidden_edge_count',0) for r in [*circuits,*feature_attributions])}

def get_bundle(store,study_id):
    repo=ResearchRepository(store);study=repo.get(study_id,'study');captures=repo.list('capture',study_id);interventions=repo.list('intervention',study_id);feature_interventions=repo.list('feature_intervention',study_id);circuits=repo.list('circuit',study_id);feature_attributions=repo.list('feature_attribution',study_id)
    return {'schema_version':'2.0','study':study,'graph':merge_graph(captures,interventions,feature_interventions,circuits,feature_attributions),'captures':captures,'interventions':interventions,'feature_interventions':feature_interventions,'feature_attributions':feature_attributions,'lenses':[lens for lens in repo.list('lens') if lens.get('metadata',{}).get('model_repo_id')==study['model_profile'].get('repo_id') and lens.get('metadata',{}).get('model_revision')==study['model_profile'].get('revision')],'lens_results':repo.list('lens_result',study_id),'circuits':circuits,'hypotheses':repo.list('hypothesis',study_id),'dictionaries':[d for d in repo.list('dictionary') if d.get('metadata',{}).get('model_repo_id')==study['model_profile'].get('repo_id') and d.get('metadata',{}).get('model_revision')==study['model_profile'].get('revision')],'features':repo.list('feature',study_id),'feature_annotations':repo.list('annotation',study_id),'provenance':{'study_hash':study['study_hash'],'model_profile_hash':study['model_profile_hash'],'recorded':True,'model_loaded_for_viewing':False},'limitations':LIMITATIONS}

def require_local_profile(store,study):
    try:trusted=store.get_record('profiles',study['config']['model_profile_id'])
    except EngineError as error:
        if error.code=='NOT_FOUND':raise EngineError('MODEL_NOT_READY','Historical research is read-only until a matching local preflight exists; freeze a new local study after doctor') from error
        raise
    if not trusted.get('preflight_passed') or digest(trusted)!=study['model_profile_hash']:raise EngineError('MODEL_NOT_READY','The local preflight changed; freeze a new local study')
    return trusted

def queue_capture(store,study_id,condition_ids=None,idempotency_key=None):
    repo=ResearchRepository(store);study=repo.get(study_id,'study');known=[c['id'] for c in study['config']['conditions']]
    require_local_profile(store,study)
    chosen=condition_ids or known
    if len(set(chosen))!=len(chosen) or set(chosen)-set(known):raise EngineError('INVALID_CONFIG','Unknown or duplicate research condition')
    return store.queue(None,{'study_id':study_id,'condition_ids':chosen},len(chosen),'research_capture',f'{study_id}:{idempotency_key}' if idempotency_key else None)

def queue_intervention(store,study_id,spec,idempotency_key=None):
    request=InterventionRequest.model_validate(spec);study=ResearchRepository(store).get(study_id,'study');known={c['id'] for c in study['config']['conditions']}
    require_local_profile(store,study)
    if request.donor_condition_id not in known or request.recipient_condition_id not in known or request.same_condition_condition_id is not None and request.same_condition_condition_id not in known:raise EngineError('INVALID_CONFIG','Intervention conditions must belong to the frozen study')
    if request.same_condition_condition_id in (request.donor_condition_id,request.recipient_condition_id):raise EngineError('INVALID_CONFIG','An independent same-condition control must reference a distinct condition')
    patch_sites={'resid_pre','attn_in','attn_out','resid_mid','mlp_in','mlp_out','resid_post','final_norm','head_pre','mlp_post_gate'}
    if request.site not in patch_sites or request.kind=='path' and request.destination_site not in patch_sites:raise EngineError('UNSUPPORTED_CAPTURE','Selected site is read-only; patch a residual/component output, head_pre or MLP gated channel instead')
    if request.head is not None and request.site!='head_pre' or request.destination_head is not None and request.destination_site!='head_pre':raise EngineError('UNSUPPORTED_CAPTURE','Head selection applies only to head_pre before output projection')
    if request.layer>=study['model_profile']['layers'] or request.destination_layer is not None and request.destination_layer>=study['model_profile']['layers']:raise EngineError('UNSUPPORTED_CAPTURE','Intervention block outside validated model')
    return store.queue(None,{'study_id':study_id,'spec':request.model_dump(mode='json')},1,'research_intervention',f'{study_id}:{idempotency_key}' if idempotency_key else None)

def resolve_target(adapter,request):
    value={k:v for k,v in request.items() if v is not None};kind=value['kind']
    if 'positive' in value:
        positive=adapter.tokenize(value['positive']);negative=adapter.tokenize(value['negative'])
        if not positive or not negative:raise EngineError('INVALID_CONFIG','Targets must contain actual tokenizer tokens')
        if kind=='next_token_logit_diff':
            if len(positive)!=1 or len(negative)!=1:raise EngineError('TOKEN_ALIGNMENT_FAILED','Next-token targets must each encode to exactly one token; retain leading spaces when appropriate',{'positive_tokens':positive,'negative_tokens':negative})
            return {'kind':kind,'positive_token_id':positive[0],'negative_token_id':negative[0]}
        if len(positive)>64 or len(negative)>64:raise EngineError('CONTEXT_LIMIT','Candidate continuation exceeds 64 tokens')
        return {'kind':kind,'positive_ids':positive,'negative_ids':negative,'normalization':value.get('normalization','sum')}
    if kind=='next_token_logit_diff':return {k:value[k] for k in ('kind','positive_token_id','negative_token_id')}
    return {k:value[k] for k in ('kind','positive_ids','negative_ids','normalization')}

def _tokens(adapter,condition,max_tokens):
    rendered=adapter.render(condition['messages']);ids=adapter.tokenize(rendered)
    if not ids or len(ids)>max_tokens:raise EngineError('CONTEXT_LIMIT','Research condition exceeds its explicit context limit',{'condition_id':condition['id'],'tokens':len(ids),'maximum':max_tokens})
    return rendered,ids

def _positions(requested,ids):
    positions=[position if position>=0 else len(ids)+position for position in requested]
    if any(p<0 or p>=len(ids) for p in positions) or len(set(positions))!=len(positions):raise EngineError('CONTEXT_LIMIT','Selected token position is absent or duplicated after resolving relative indices')
    return positions

def _load(store,study,adapter=None):
    # An imported profile is historical provenance, never a trusted local preflight.
    require_local_profile(store,study)
    if adapter is None:
        from ghosttrace.models.mechanistic import MechanisticAdapter
        adapter=MechanisticAdapter()
    validated=deepcopy(study['model_profile'])
    try:
        adapter.load(deepcopy(validated));adapter.seed(study['config']['order_seed'])
        require_compatible_backend(validated,adapter.describe())
        return adapter
    except BaseException:
        adapter.unload();raise

def _artifact(store,study,record_id,arrays):
    if not arrays:return None
    root=store.root/'research'/study['study_hash'];path=root/'records'/record_id/'arrays.npz'
    artifact=write_arrays(path,arrays,study['resolved_capture']['max_artifact_bytes'],root)
    artifact['relative_path']=str(path.relative_to(store.root));return artifact

def _token_labels(adapter,ids):
    if hasattr(adapter.tokenizer,'convert_ids_to_tokens'):return adapter.tokenizer.convert_ids_to_tokens(ids)
    return [str(token) for token in ids]

def run_captures(store,job,adapter=None):
    repo=ResearchRepository(store);study=repo.get(job['payload']['study_id'],'study');adapter=_load(store,study,adapter)
    ids_out=[];completed=failed=0
    try:
        target=resolve_target(adapter,study['config']['target']);conditions={c['id']:c for c in study['config']['conditions']}
        for condition_id in job['payload']['condition_ids']:
            if store.job(job['id'])['cancel_requested']:
                store.update_job(job['id'],state='cancelled',stage='cancelled_between_research_forwards',completed_units=completed,failed_units=failed);return None
            unit_id=digest({'study_hash':study['study_hash'],'condition_id':condition_id,'capture':study['resolved_capture'],'target':target,'implementation_sha256':adapter.describe().get('code_provenance',{}).get('source_sha256')})
            previous=[r for r in repo.list('capture',study['id']) if r['unit_id']==unit_id and r['status']=='completed']
            if previous:
                artifact=previous[0].get('artifact')
                if artifact:
                    path,checksum=repo.artifact_location(previous[0]);read_arrays(path,checksum)
                ids_out.append(previous[0]['id']);completed+=1;continue
            record_id=str(uuid.uuid4());started=time.monotonic();base={'id':record_id,'study_id':study['id'],'unit_id':unit_id,'condition_id':condition_id,'job_id':job['id'],'target':target,'provenance':deepcopy(adapter.describe())}
            try:
                rendered,token_ids=_tokens(adapter,conditions[condition_id],study['config']['max_input_tokens']);positions=_positions(study['resolved_capture']['positions'],token_ids)
                result=adapter.inspect(token_ids,layers=study['resolved_capture']['layers'],positions=positions,target=target,attention=study['resolved_capture']['attention'],top_k=study['resolved_capture']['top_k'])
                arrays=result.pop('artifact_arrays',{});components=result.get('components',[]);wanted=set(study['resolved_capture']['sites'])
                components=[c for c in components if c.get('site') in wanted]
                keys={c['array_key'] for c in components};arrays={k:v for k,v in arrays.items() if k in keys}
                artifact=_artifact(store,study,record_id,arrays)
                tiles=[]
                for component in components:
                    key=component['array_key']
                    if key not in arrays:continue
                    tile={**component,**browser_tile(arrays[key],component,max_values=study['resolved_capture'].get('browser_tile_values',4096)),'dtype':'float32','unit_id':unit_id,'artifact_hash':artifact['sha256'],'token_indices':component.get('token_indices',positions),'normalization_state':component.get('normalization_state','literal_component_output')}
                    tiles.append(tile)
                capture={**base,'status':'completed','rendered_text':rendered,'input_hash':digest(rendered),'token_ids':token_ids,'tokens':_token_labels(adapter,token_ids),'selected_positions':positions,'requested_sites':study['resolved_capture']['sites'],'unavailable_sites':[{'site':site,'reason':'not_returned_by_adapter'} for site in wanted-{c['site'] for c in components}],'components':components,'sites':tiles,'tiles':tiles,'artifact':artifact,'metrics':result.get('metrics',{}),'target_readout':result.get('metrics',{}),'logit_lens':result.get('metrics',{}).get('logit_lens',[]),'checks':{k:v for k,v in result.get('metrics',{}).items() if k not in ('logit_lens','target_score')},'metadata':result.get('metadata',{}),'graph':result.get('graph',{}),'seconds':time.monotonic()-started,'peak_memory':measured_peak_memory(study['model_profile'].get('device','cpu'))}
                repo.put('capture',capture,study['id']);ids_out.append(record_id);completed+=1
            except EngineError as error:
                repo.put('capture',{**base,'status':'failed','error':{'code':error.code,'message':error.message,'details':error.details},'seconds':time.monotonic()-started},study['id']);failed+=1
            except Exception as error:
                code='OUT_OF_MEMORY' if 'out of memory' in str(error).lower() else 'RESEARCH_CAPTURE_FAILED'
                repo.put('capture',{**base,'status':'failed','error':{'code':code,'message':str(error)},'seconds':time.monotonic()-started},study['id']);failed+=1
            store.update_job(job['id'],completed_units=completed,failed_units=failed,stage='component_capture')
        if failed:raise EngineError('RESEARCH_UNIT_FAILURES',f'{failed} research capture units failed; successful immutable units remain available')
        return {'resource_type':'research_capture','study_id':study['id'],'capture_ids':ids_out,'resource_id':ids_out[-1] if ids_out else study['id']}
    finally:adapter.unload()

def run_intervention(store,job,adapter=None):
    repo=ResearchRepository(store);study=repo.get(job['payload']['study_id'],'study');spec=job['payload']['spec'];adapter=_load(store,study,adapter);started=time.monotonic();record_id=str(uuid.uuid4())
    try:
        conditions={c['id']:c for c in study['config']['conditions']};target=resolve_target(adapter,study['config']['target'])
        donor_text,donor=_tokens(adapter,conditions[spec['donor_condition_id']],study['config']['max_input_tokens']);recipient_text,recipient=_tokens(adapter,conditions[spec['recipient_condition_id']],study['config']['max_input_tokens'])
        donor_positions=_positions(spec['positions'],donor);recipient_positions=_positions(spec['positions'],recipient)
        if len(donor)!=len(recipient) or donor_positions!=recipient_positions:raise EngineError('TOKEN_ALIGNMENT_FAILED','Interventions require equal sequence lengths and the same explicitly matched absolute positions')
        if spec['alignment']['kind']=='exact_tokens' and any(donor[p]!=recipient[p] for p in donor_positions):raise EngineError('TOKEN_ALIGNMENT_FAILED','Selected donor/recipient token IDs differ; declare and explain a genuine semantic correspondence or choose matched tokens')
        common={'positions':donor_positions,'target':target,'head':spec.get('head'),'channel':spec.get('channel')}
        if spec['kind']=='path':result=adapter.path_patch(donor,recipient,source_site=spec['site'],source_layer=spec['layer'],destination_site=spec['destination_site'],destination_layer=spec['destination_layer'],destination_head=spec.get('destination_head'),destination_channel=spec.get('destination_channel'),**common)
        elif spec['kind']=='attribution':result=adapter.attribute(donor,recipient,site=spec['site'],layer=spec['layer'],**common)
        else:
            independent={}
            if spec.get('same_condition_condition_id'):
                same_condition=conditions[spec['same_condition_condition_id']]
                recipient_group=conditions[spec['recipient_condition_id']].get('group_id');same_group=same_condition.get('group_id')
                if recipient_group and same_group and recipient_group!=same_group:raise EngineError('INVALID_CONFIG','Declared same-condition control belongs to a different experimental group')
                same_text,same_ids=_tokens(adapter,same_condition,study['config']['max_input_tokens'])
                if len(same_ids)!=len(recipient) or _positions(spec['positions'],same_ids)!=recipient_positions:raise EngineError('TOKEN_ALIGNMENT_FAILED','Independent same-condition control must match absolute positions and sequence length')
                independent['same_condition_token_ids']=same_ids
            result=adapter.intervene(donor,recipient,site=spec['site'],layer=spec['layer'],operation=spec['operation'],**common,**independent)
        arrays=result.pop('artifact_arrays',{});artifact=_artifact(store,study,record_id,arrays);metrics=result.get('metrics',{})
        record={'id':record_id,'study_id':study['id'],'job_id':job['id'],'unit_id':digest({'study_hash':study['study_hash'],'spec':spec,'target':target}),'status':'completed','spec':spec,'target':target,'donor_token_ids':donor,'recipient_token_ids':recipient,'selected_positions':donor_positions,'input_hashes':{'donor':digest(donor_text),'recipient':digest(recipient_text)},'metrics':metrics,'baseline_metric':metrics.get('baseline'),'intervened_metric':metrics.get('patched'),'raw_delta':metrics.get('delta'),'estimated_delta':metrics.get('estimated_delta'),'exact_reference_delta':metrics.get('exact_patch_delta'),'linearization_error':metrics.get('linearization_error'),'recovery':metrics.get('recovery'),'recovery_missing_reason':metrics.get('recovery_missing_reason'),'controls':metrics.get('controls',{}),'checks':metrics.get('checks',{}),'graph':result.get('graph',{}),'metadata':result.get('metadata',{}),'artifact':artifact,'provenance':deepcopy(adapter.describe()),'seconds':time.monotonic()-started,'peak_memory':measured_peak_memory(study['model_profile'].get('device','cpu')),'baseline_scope':'donor_selected_positions' if spec['operation']=='mean_ablate' else 'explicit_donor_and_recipient','evidence_type':{'exact':'intervention','attribution':'attribution','path':'tested_path'}[spec['kind']],'limitations':['Normalized recovery is unavailable for small baseline gaps; raw deltas are never clamped.','Mean ablation uses the mean of the declared donor positions, not an undeclared global corpus.']}
        repo.put('intervention',record,study['id'])
        return {'resource_type':'research_intervention','study_id':study['id'],'intervention_id':record_id,'resource_id':record_id}
    except Exception as error:
        details={'code':error.code,'message':error.message,'details':error.details} if isinstance(error,EngineError) else {'code':'OUT_OF_MEMORY' if 'out of memory' in str(error).lower() else 'RESEARCH_INTERVENTION_FAILED','message':str(error)}
        repo.put('intervention',{'id':record_id,'study_id':study['id'],'job_id':job['id'],'spec':spec,'status':'failed','error':details,'seconds':time.monotonic()-started,'provenance':deepcopy(adapter.describe())},study['id'])
        raise
    finally:adapter.unload()

def save_hypothesis(store,study_id,spec):
    repo=ResearchRepository(store);study=repo.get(study_id,'study');request=HypothesisCreate.model_validate(spec)
    for kind,ids in [('capture',request.source_capture_ids),('intervention',request.source_intervention_ids)]:
        for record_id in ids:
            if repo.get(record_id,kind)['study_id']!=study_id:raise EngineError('CONFLICT','Hypothesis evidence belongs to another study')
    return repo.put('hypothesis',{'study_id':study_id,**request.model_dump(mode='json'),'evidence_type':'human_interpretation','registered_before_results':not repo.list('capture',study_id) and not repo.list('intervention',study_id),'interpretation':'A recorded research interpretation; not automatic scientific certification.'},study_id)
