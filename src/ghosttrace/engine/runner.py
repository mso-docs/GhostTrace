from __future__ import annotations
import json
import threading
import time
import traceback
import uuid
from pathlib import Path
from copy import deepcopy
from ghosttrace.config import EngineError,SITE,canonical,digest,now
from ghosttrace.engine.render import messages,align_continuation,token_accounting
from ghosttrace.storage.artifacts import atomic_json,write_activation,read_activation
from ghosttrace.models.qwen import QwenAdapter


def preflight(store,profile,adapter=None):
    """Real-model checks only. Download must already have been explicitly requested."""
    import numpy as np
    adapter=adapter or QwenAdapter()
    started=time.monotonic()
    try:
        metadata=adapter.load(profile)
        outputs=[]
        for prompt in ['Describe a quiet room in one sentence.','What is two plus two?','Describe a notebook in one sentence.']:
            ids=adapter.tokenize(adapter.render([{'role':'user','content':prompt}]))
            outputs.append(adapter.generate(ids,{'max_new_tokens':16}))
        ids=adapter.tokenize(adapter.render([{'role':'user','content':'Describe what happens next.'}]))
        adapter.seed(73)
        a=adapter.capture(ids,SITE,[len(ids)-1]);adapter.seed(73);b=adapter.capture(ids,SITE,[len(ids)-1])
        max_error=float(np.max(np.abs(a-b)))
        tolerance=1e-5 if profile.get('compute_dtype','float32')=='float32' else 1e-3
        timing=adapter.generate(ids,{'max_new_tokens':128})
        memory=measured_peak_memory(profile.get('device','cpu'))
        peak_ram=memory['ram_bytes'];peak_vram=memory['vram_bytes']
        metadata={**metadata,'schema_version':'1.0','created_at':now(),'preflight_passed':max_error<=tolerance,'repeat_max_abs_error':max_error,'repeat_tolerance':tolerance,'peak_ram_bytes':peak_ram,'peak_vram_bytes':peak_vram,'peak_memory':memory,'generation_tokens_per_second':timing['generated_token_count']/max(timing['seconds'],1e-8),'generation_timing_tokens':timing['generated_token_count'],'generation_timing_max_tokens':128,'load_and_test_seconds':time.monotonic()-started,'smoke_generations':outputs}
        adapter.unload();adapter.load(profile);adapter.unload()
        # Include the full unload/reload lifecycle in elapsed and memory evidence.
        memory=measured_peak_memory(profile.get('device','cpu'))
        metadata.update(peak_ram_bytes=memory['ram_bytes'],peak_vram_bytes=memory['vram_bytes'],peak_memory=memory,load_and_test_seconds=time.monotonic()-started)
        metadata['reload_passed']=True
        store.put_record('profiles',metadata)
        if not metadata['preflight_passed']:raise EngineError('INVALID_CONFIG','Deterministic-repeat check failed; canonical pilot blocked',metadata)
        return metadata
    finally:adapter.unload()



def measured_peak_memory(device='cpu'):
    result={'ram_bytes':None,'vram_bytes':None,'ram_scope':'Process lifetime high-water RSS; includes model load and prior units, not an incremental per-unit peak.','vram_scope':'CUDA allocator high-water mark since its last reset; not an incremental per-unit peak.','ram_unavailable_reason':None,'vram_unavailable_reason':'CUDA measurement is unavailable on this backend.'}
    try:
        import resource,sys
        result['ram_bytes']=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024)
    except (ImportError,OSError,ValueError) as error:result['ram_unavailable_reason']=str(error)
    if device.startswith('cuda'):
        try:
            import torch
            if torch.cuda.is_available():
                result['vram_bytes']=int(torch.cuda.max_memory_allocated(device));result['vram_unavailable_reason']=None
        except (ImportError,RuntimeError,ValueError) as error:result['vram_unavailable_reason']=str(error)
    return result



NUMERICAL_PROVENANCE_KEYS=(
    'repo_id','revision','tokenizer_revision','config_hash','weight_file_hashes',
    'tokenizer_file_hashes','chat_template','chat_template_hash','layers','hidden_size',
    'device','compute_dtype','storage_dtype','quantization','attention_backend',
    'threads','interop_threads','torch_version','transformers_version','numpy_version',
    'package_versions','python_version','os','use_sliding_window','deterministic_algorithms',
)

def require_compatible_backend(validated,actual,canonical_holdout=False):
    """Canonical execution must use the numerical environment tested by doctor."""
    if not isinstance(actual,dict):raise EngineError('MODEL_NOT_READY','Loaded adapter did not describe its numerical provenance')
    mismatches=[key for key in NUMERICAL_PROVENANCE_KEYS if (key in validated or key in actual) and validated.get(key)!=actual.get(key)]
    validated_source=validated.get('code_provenance',{}).get('source_sha256')
    actual_source=actual.get('code_provenance',{}).get('source_sha256')
    if canonical_holdout and validated_source and actual_source and validated_source!=actual_source:mismatches.append('code_provenance.source_sha256')
    if mismatches:
        raise EngineError('CONFLICT','Loaded backend differs from validated preflight ('+', '.join(mismatches)+'); run doctor again and freeze a new experiment.',{'mismatched_fields':mismatches,'requires_new_preflight':True})


def run_job(store,job,adapter=None):
    adapter=adapter or QwenAdapter()
    stop=threading.Event()
    def heartbeat():
        while not stop.wait(5):
            store.update_job(job['id'],heartbeat_at=now())
            store.worker_heartbeat('busy')
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    try:
        if job['kind']=='preflight':
            result=preflight(store,job['payload']['profile'],adapter)
            store.update_job(job['id'],state='completed',stage='preflight_complete',completed_units=1)
            return result
        experiment=store.experiment(job['experiment_id'])
        config=experiment['config']
        profile=store.get_record('profiles',config['model_profile'])
        if not profile.get('preflight_passed'):raise EngineError('MODEL_NOT_READY','Complete a successful hardware preflight first')
        if digest(profile)!=experiment['model_profile_hash']:raise EngineError('CONFLICT','Validated model profile changed; freeze a new experiment')
        if experiment['preregistration_hash'] is None:raise EngineError('INVALID_CONFIG','Missing preregistration')
        if any(s['split']=='holdout' for s in experiment['scenarios']):
            if any(s.get('review_status')!='reviewed' for s in experiment['scenarios']+experiment.get('control_scenarios',[])) or any(t.get('review_status')!='reviewed' for t in experiment.get('prompt_templates',[])):
                raise EngineError('INVALID_CONFIG','Human scenario, factual-control, and template review is required before a canonical pilot containing holdout scenarios')
            review=experiment.get('human_review')
            receipt=experiment.get('preregistration_receipt')
            if not review or review.get('reviewed_dataset_hash')!=experiment['dataset_hash'] or review.get('reviewed_templates_hash')!=digest(experiment.get('prompt_templates',[])) or review.get('reviewed_controls_hash')!=digest(experiment.get('control_scenarios',[])) or experiment.get('human_review_hash')!=digest(review) or not review.get('reviewer'):
                raise EngineError('INVALID_CONFIG','Canonical holdout requires a hash-bound human review receipt; use scripts/review_dataset.py')
            if not receipt or receipt.get('config_hash')!=digest(config) or receipt.get('dataset_hash')!=experiment['dataset_hash'] or receipt.get('template_hash')!=digest(experiment.get('prompt_templates',[])) or experiment.get('preregistration_receipt_hash')!=digest(receipt) or receipt.get('human_review_hash')!=digest(review) or receipt.get('protocol_sha256')!=experiment['preregistration_hash']:
                raise EngineError('INVALID_CONFIG','Freeze a reviewed preregistration before canonical holdout; use scripts/freeze_protocol.py')

        validated_profile=deepcopy(profile)
        adapter.load(profile);adapter.seed(config.get('order_seed',73))
        require_compatible_backend(validated_profile,adapter.describe(),canonical_holdout=any(s['split']=='holdout' for s in experiment['scenarios']))
        root=store.root/experiment['config_hash']
        atomic_json(root/'environment.json',adapter.describe())
        scenarios={s['id']:s for s in experiment['scenarios']+experiment.get('control_scenarios',[])}
        completed=failed=0
        expected={}
        store.update_job(job['id'],stage='executing_units')
        for unit in experiment['ordered_units']:
            if store.job(job['id'])['cancel_requested']:
                store.update_job(job['id'],state='cancelled',stage='cancelled_between_units',completed_units=completed,failed_units=failed);return
            prior=None
            with store.connect() as db:prior=db.execute('SELECT * FROM units WHERE id=?',(unit['id'],)).fetchone()
            if prior and prior['status']=='completed':
                if prior['artifact_path']:read_activation(store.root/prior['artifact_path'],prior['artifact_sha256'])
                completed+=1
                with store.connect() as db:db.execute('INSERT OR IGNORE INTO job_units VALUES(?,?)',(job['id'],unit['id']))
                store.update_job(job['id'],completed_units=completed,failed_units=failed);continue
            scenario=scenarios[unit['scenario_id']]
            record={**unit,'schema_version':'1.0','job_id':job['id'],'experiment_id':experiment['id'],'created_at':now(),'status':'running','backend_fingerprint':digest(adapter.describe()),'model_profile':adapter.describe(),'scenario':scenario,'split':scenario['split'],'family_id':scenario['family_id']}
            activation_path=checksum=None
            start=time.monotonic()
            try:
                rendered_messages=messages(scenario,unit['condition'],unit['template_id'],experiment.get('prompt_templates'))
                rendered=adapter.render(rendered_messages)
                record.update(rendered_messages=rendered_messages,rendered_text=rendered,input_hash=digest(rendered),token_ids=adapter.tokenize(rendered))
                limit=config.get('limits',{}).get('max_input_tokens',512)
                if len(record['token_ids'])>limit:raise EngineError('CONTEXT_LIMIT','Prompt exceeds configured input limit')
                record['visible_tokens']=adapter.tokenizer.convert_ids_to_tokens(record['token_ids'])
                record.update(token_accounting(adapter.tokenizer,rendered,rendered_messages))
                if unit['mode']=='greedy_generation':
                    generated=adapter.generate(record['token_ids'],config.get('generation',{}),config.get('order_seed',73))
                    record.update(generated)
                    record['generation_wall_seconds']=generated.get('seconds')
                    record['generation_tokens_per_second']=generated['generated_token_count']/max(generated['seconds'],1e-8) if generated.get('seconds') is not None else None
                    special_ids=getattr(adapter.tokenizer,'all_special_ids',None)
                    record['generated_special_token_mask']=[int(token in set(special_ids)) for token in generated['generated_ids']] if special_ids is not None else None
                    record['generated_role_span']={'role':'assistant','token_start':len(record['token_ids']),'token_end':len(record['token_ids'])+len(generated['generated_ids']),'token_ids_field':'generated_ids','alignment':'Generated texts have no assumed token correspondence across conditions.'}
                elif unit['mode']=='fixed_continuation':
                    pair_key=(scenario['id'],unit['template_id'])
                    aligned=align_continuation(adapter.tokenizer,rendered,scenario['common_continuation'],expected.get(pair_key),limit)
                    expected.setdefault(pair_key,aligned['continuation_ids'])
                    # A full rendered-prefix tokenization validates the named user-content anchor.
                    user_start=rendered.rfind(scenario['text']);user_end=user_start+len(scenario['text'])
                    if user_start<0:raise EngineError('TOKEN_ALIGNMENT_FAILED','Scenario span absent from rendering')
                    offsets=adapter.tokenizer(rendered+scenario['common_continuation'],add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
                    user_indices=[i for i,(a,b) in enumerate(offsets) if a<user_end and b>user_start]
                    if not user_indices:raise EngineError('TOKEN_ALIGNMENT_FAILED','Scenario content token map unavailable')
                    scenario_anchor=user_indices[-1]
                    record['segment_spans']={'scenario':{'character_start':user_start,'character_end':user_end,'token_indices':user_indices},'continuation':{'token_start':aligned['continuation_offset'],'token_end':len(aligned['token_ids'])}}
                    positions=[scenario_anchor]+aligned['token_indices']
                    labels=['last_scenario_token']+aligned['semantic_position_labels']
                    capture_started=time.monotonic()
                    activations=adapter.capture(aligned['token_ids'],SITE,positions)
                    record['capture_wall_seconds']=time.monotonic()-capture_started
                    record['capture_input_tokens_per_second']=len(aligned['token_ids'])/max(record['capture_wall_seconds'],1e-8)
                    record['capture_timing_scope']='One teacher-forced forward including selected residual slices transferred to CPU.'
                    artifact_file=root/'runs'/unit['id']/f'attempt-{job["id"]}'/'activations.npz'
                    checksum=write_activation(artifact_file,activations,config.get('limits',{}).get('max_artifact_bytes',2_000_000_000),root)
                    activation_path=str(artifact_file.relative_to(store.root))
                    scoring_started=time.monotonic()
                    nll=adapter.score_continuation(aligned['token_ids'][:aligned['continuation_offset']],aligned['continuation_ids'][:16])
                    record['continuation_scoring_wall_seconds']=time.monotonic()-scoring_started
                    record.update(aligned)
                    record.update(token_accounting(adapter.tokenizer,rendered,rendered_messages,scenario['common_continuation']))
                    record['token_indices']=positions
                    record['absolute_positions']=positions
                    record['visible_tokens']=adapter.tokenizer.convert_ids_to_tokens(record['token_ids'])
                    record['semantic_position_labels']=labels
                    record['continuation_nll']=nll
                    record['activation_artifact']={'id':digest({'unit':unit['id'],'sha256':checksum}),'relative_path':activation_path,'sha256':checksum,'site':SITE,'axis_names':['layer','selected_token','hidden'],'shape':list(activations.shape),'dtype':'float32','layer_indices':list(range(activations.shape[0])),'token_indices':positions,'semantic_position_labels':labels,'normalization_state':'before_final_model_norm','pool_mask':list(range(2,18))}
                else:raise EngineError('UNSUPPORTED_CAPTURE','Replay-generated capture is exploratory and not enabled in the canonical runner')
                record.update(status='completed',seconds=time.monotonic()-start)
                completed+=1
            except EngineError as e:
                record.update(status='failed',error={'code':e.code,'message':e.message,'details':e.details},seconds=time.monotonic()-start);failed+=1
            except Exception as e:
                code='OUT_OF_MEMORY' if 'out of memory' in str(e).lower() else 'UNIT_FAILED'
                record.update(status='failed',error={'code':code,'message':str(e)},seconds=time.monotonic()-start);failed+=1
            record['peak_memory']=measured_peak_memory(profile.get('device','cpu'))
            atomic_json(root/'runs'/unit['id']/f'attempt-{job["id"]}'/'record.json',record)
            with store.connect() as db:
                db.execute('INSERT INTO units VALUES(?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET record=excluded.record,status=excluded.status,artifact_path=excluded.artifact_path,artifact_sha256=excluded.artifact_sha256,updated_at=excluded.updated_at',(unit['id'],experiment['id'],canonical(record).decode(),record['status'],activation_path,checksum,now()))
                db.execute('INSERT OR IGNORE INTO job_units VALUES(?,?)',(job['id'],unit['id']))
                store.event(db,job['id'],'unit_completed' if record['status']=='completed' else 'unit_failed',{'unit_id':unit['id'],'status':record['status'],'error':record.get('error')})
            store.update_job(job['id'],completed_units=completed,failed_units=failed)
        store.update_job(job['id'],state='completed' if not failed else 'failed',stage='complete' if not failed else 'partial_results',completed_units=completed,failed_units=failed,error_code='UNIT_FAILURES' if failed else None)
    except EngineError as e:store.update_job(job['id'],state='failed',stage='error',error_code=e.code,error_message=e.message)
    except Exception as e:store.update_job(job['id'],state='failed',stage='error',error_code='WORKER_ERROR',error_message=str(e))
    finally:
        stop.set();thread.join(timeout=1);adapter.unload()
