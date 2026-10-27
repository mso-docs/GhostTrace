"""Untrusted evidence bundles: validate, never execute, never fetch a model."""
from __future__ import annotations
import hashlib
import json
import math
import re
import shutil
import stat
import tempfile
import uuid
import zipfile
from pathlib import Path,PurePosixPath
from ghosttrace.config import EngineError,canonical,digest,file_digest,now,validate_config,validate_dataset,validate_templates,SITE
from ghosttrace.storage.artifacts import atomic_json,atomic_bytes,read_activation

MAX_ARCHIVE_BYTES=2_000_000_000
MAX_UNCOMPRESSED_BYTES=2_000_000_000
MAX_ENTRIES=10_000
MAX_JSON_BYTES=64_000_000
HEX=re.compile(r'^[a-f0-9]{64}$')
TOP_LEVEL={'README.md','model-profile.json','manifest.json','analysis.json','report.md','environment.json','dataset.jsonl','prompts.json','preregistration.md','factual-controls.jsonl','ratings.json','metrics.csv','jobs.jsonl','events.jsonl','inventory.json','human-review.json','preregistration-receipt.json'}
ATTEMPT_PATH=re.compile(r'^attempts/([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})/([a-f0-9]{64})/record\.json$')
RUN_PATH=re.compile(r'^runs/([a-f0-9]{64})/(record\.json|activations\.npz)$')

def _json(content,label):
    if len(content)>MAX_JSON_BYTES:raise EngineError('INVALID_BUNDLE',f'{label} exceeds JSON limit')
    try:
        result=json.loads(content,parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite value')))
        canonical(result)
        return result
    except (ValueError,TypeError,UnicodeError,RecursionError) as e:raise EngineError('INVALID_BUNDLE',f'Invalid finite UTF-8 JSON in {label}') from e

def _uuid(value,label):
    try:
        parsed=str(uuid.UUID(value))
        if parsed!=value:raise ValueError('UUID must be lowercase canonical form')
        return parsed
    except (ValueError,TypeError,AttributeError):raise EngineError('INVALID_BUNDLE',f'{label} must be UUID')

def _manifest(manifest):
    _uuid(manifest.get('id'),'Experiment id')
    if manifest.get('schema_version')!='1.0' or not HEX.fullmatch(manifest.get('config_hash','')):raise EngineError('INVALID_BUNDLE','Invalid manifest version/hash')
    validate_config(manifest.get('config',{}))
    validate_dataset(manifest.get('scenarios',[]))
    controls=manifest.get('control_scenarios',[])
    if controls:
        validate_dataset(controls)
        validate_dataset(manifest['scenarios']+controls)
    if digest(manifest['scenarios'])!=manifest.get('dataset_hash') or (digest(controls) if controls else None)!=manifest.get('factual_dataset_hash'):
        raise EngineError('ARTIFACT_CORRUPT','Dataset hash does not match frozen manifest')
    if digest(manifest.get('model_profile'))!=manifest.get('model_profile_hash'):raise EngineError('ARTIFACT_CORRUPT','Model profile hash mismatch')
    templates=manifest.get('prompt_templates')
    if not isinstance(templates,list) or not templates or any(not isinstance(t,dict) or not isinstance(t.get('family'),str) for t in templates):
        raise EngineError('INVALID_BUNDLE','Frozen prompt templates are required for direct family-hash verification')
    for family in manifest['config']['templates']:
        selected=[t for t in templates if t['family']==family]
        if not selected or digest(selected)!=manifest.get('template_hashes',{}).get(family):raise EngineError('ARTIFACT_CORRUPT','Selected prompt-template family hash mismatch')
    if 'prompt_templates_hash' in manifest and digest(templates)!=manifest['prompt_templates_hash']:
        raise EngineError('ARTIFACT_CORRUPT','Full prompt-template inventory hash mismatch')
    validate_templates(templates,manifest['config']['templates'])
    base={key:manifest[key] for key in ('schema_version','config','dataset_hash','factual_dataset_hash','model_profile_hash','template_hashes','preregistration_hash')}
    if 'prompt_templates_hash' in manifest:base['prompt_templates_hash']=manifest['prompt_templates_hash']
    for hash_key,record_key in [('human_review_hash','human_review'),('preregistration_receipt_hash','preregistration_receipt')]:
        if hash_key in manifest:
            base[hash_key]=manifest[hash_key]
            if digest(manifest.get(record_key))!=manifest[hash_key]:raise EngineError('ARTIFACT_CORRUPT',f'{record_key} receipt hash mismatch')
    if digest(base)!=manifest['config_hash']:raise EngineError('ARTIFACT_CORRUPT','Manifest content hash mismatch')
    units=manifest.get('ordered_units',[])
    if not isinstance(units,list) or len(units)>2000 or not units:raise EngineError('INVALID_BUNDLE','Invalid unit inventory')
    ids=set();scenarios={s['id'] for s in manifest['scenarios']+controls}
    for unit in units:
        if unit.get('id') in ids or not HEX.fullmatch(unit.get('id','')) or unit.get('scenario_id') not in scenarios:raise EngineError('INVALID_BUNDLE','Invalid/duplicate run unit identity')
        ids.add(unit['id'])
        expected=digest({'experiment_hash':manifest['config_hash'],**{k:v for k,v in unit.items() if k!='id'}})
        if expected!=unit['id']:raise EngineError('ARTIFACT_CORRUPT','Unit identity hash mismatch')
    if manifest.get('ordered_unit_ids')!=[u['id'] for u in units]:raise EngineError('INVALID_BUNDLE','Ordered unit IDs mismatch')
    profile=manifest['model_profile']
    if not isinstance(profile.get('layers'),int) or not 1<=profile['layers']<=128 or not isinstance(profile.get('hidden_size'),int) or not 1<=profile['hidden_size']<=65536:raise EngineError('INVALID_BUNDLE','Unsupported model shape')
    return {u['id']:u for u in units}

def verify_bundle(path):
    """Stream checksums before parsing arrays, then return validated JSON metadata."""
    path=Path(path)
    if not path.is_file() or path.stat().st_size>MAX_ARCHIVE_BYTES:raise EngineError('PAYLOAD_TOO_LARGE','Evidence archive missing or exceeds 2 GB')
    try:
        with zipfile.ZipFile(path) as archive:
            infos=archive.infolist();names=[i.filename for i in infos]
            if len(infos)>MAX_ENTRIES or len(names)!=len(set(names)) or sum(i.file_size for i in infos)>MAX_UNCOMPRESSED_BYTES:raise EngineError('INVALID_BUNDLE','Duplicate entries or archive exceeds count/expanded-size limits')
            for info in infos:
                name=info.filename;p=PurePosixPath(name)
                if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name or str(p)!=name or stat.S_ISLNK(info.external_attr>>16) or info.is_dir() or info.flag_bits&1:
                    raise EngineError('INVALID_BUNDLE','Unsafe, encrypted, directory, or symbolic-link ZIP entry')
                if name not in TOP_LEVEL and not RUN_PATH.fullmatch(name) and not ATTEMPT_PATH.fullmatch(name):raise EngineError('INVALID_BUNDLE',f'Unsupported bundle entry {name}')
                cap=100_000_000 if name.endswith('.npz') else MAX_JSON_BYTES
                if info.file_size>cap:raise EngineError('INVALID_BUNDLE',f'Oversized bundle entry {name}')
            if not {'manifest.json','analysis.json','inventory.json'}<=set(names):raise EngineError('INVALID_BUNDLE','Missing manifest, analysis, or inventory')
            inventory=_json(archive.read('inventory.json'),'inventory.json');checksums=inventory.get('file_inventory_with_checksums')
            if not isinstance(checksums,dict) or set(checksums)!=set(names)-{'inventory.json'}:raise EngineError('INVALID_BUNDLE','Inventory must cover every archive entry exactly once')
            for name,expected in checksums.items():
                if not isinstance(expected,str) or not HEX.fullmatch(expected):raise EngineError('INVALID_BUNDLE','Invalid inventory checksum')
                h=hashlib.sha256()
                with archive.open(name) as source:
                    for chunk in iter(lambda:source.read(1024*1024),b''):h.update(chunk)
                if h.hexdigest()!=expected:raise EngineError('ARTIFACT_CORRUPT',f'Checksum mismatch: {name}')
            manifest=_json(archive.read('manifest.json'),'manifest.json');units=_manifest(manifest)
            if 'model-profile.json' in names:
                model_bytes=archive.read('model-profile.json')
                if hashlib.sha256(model_bytes).hexdigest()!=manifest['model_profile_hash'] or _json(model_bytes,'model-profile.json')!=manifest['model_profile']:raise EngineError('ARTIFACT_CORRUPT','Canonical model-profile artifact differs from frozen model profile')
            for filename,field in [('dataset.jsonl','scenarios'),('factual-controls.jsonl','control_scenarios')]:
                if filename in names:
                    rows=[_json(line,filename) for line in archive.read(filename).splitlines() if line.strip()]
                    if rows!=manifest.get(field,[]):raise EngineError('ARTIFACT_CORRUPT','Dataset artifact differs from frozen scenario inventory')
            if 'prompts.json' in names and _json(archive.read('prompts.json'),'prompts.json')!=manifest['prompt_templates']:raise EngineError('ARTIFACT_CORRUPT','Prompt artifact differs from frozen template inventory')
            if 'preregistration.md' in names and hashlib.sha256(archive.read('preregistration.md')).hexdigest()!=manifest.get('preregistration_hash'):raise EngineError('ARTIFACT_CORRUPT','Preregistration artifact differs from frozen protocol')
            for filename,field in [('human-review.json','human_review'),('preregistration-receipt.json','preregistration_receipt')]:
                if filename in names and _json(archive.read(filename),filename)!=manifest.get(field):raise EngineError('ARTIFACT_CORRUPT','Review receipt artifact differs from frozen manifest')
            if inventory.get('manifest_hash')!=manifest['config_hash']:raise EngineError('ARTIFACT_CORRUPT','Inventory manifest hash mismatch')
            analysis=_json(archive.read('analysis.json'),'analysis.json');_uuid(analysis.get('id'),'Analysis id')
            if analysis.get('experiment_id')!=manifest['id'] or analysis.get('schema_version')!='1.0':raise EngineError('INVALID_BUNDLE','Analysis belongs to another experiment or schema')
            settings=analysis.get('settings',{})
            if settings.get('metric_version')!='normalized-l2-v1' or settings.get('split') not in ('development','holdout','all') or digest(settings)!=analysis.get('settings_hash'):raise EngineError('INVALID_BUNDLE','Unsupported or inconsistent analysis settings')
            records=[]
            for name in names:
                match=RUN_PATH.fullmatch(name)
                if not match or match[2]!='record.json':continue
                record=_json(archive.read(name),name);unit=units.get(match[1])
                if not unit or record.get('id')!=match[1] or record.get('experiment_id')!=manifest['id']:raise EngineError('INVALID_BUNDLE','Run record not in frozen experiment')
                if any(record.get(k)!=v for k,v in unit.items()):raise EngineError('ARTIFACT_CORRUPT','Run treatment differs from frozen unit')
                if record.get('status') not in ('completed','failed'):raise EngineError('INVALID_BUNDLE','Only finalized run records may be imported')
                scenario_lookup={s['id']:s for s in manifest['scenarios']+manifest.get('control_scenarios',[])}
                if 'scenario' in record and record['scenario']!=scenario_lookup[record['scenario_id']]:raise EngineError('ARTIFACT_CORRUPT','Run scenario differs from frozen scenario')
                if 'rendered_text' in record and digest(record['rendered_text'])!=record.get('input_hash'):raise EngineError('ARTIFACT_CORRUPT','Rendered input hash mismatch')
                if 'model_profile' in record and digest(record['model_profile'])!=record.get('backend_fingerprint'):raise EngineError('ARTIFACT_CORRUPT','Run backend fingerprint mismatch')
                if 'continuation_offset' in record and type(record['continuation_offset']) is not int:raise EngineError('INVALID_BUNDLE','Invalid continuation offset type')
                if record.get('mode')=='fixed_continuation' and 'continuation_offset' in record and record.get('continuation_ids')!=record.get('token_ids',[])[record['continuation_offset']:]:raise EngineError('ARTIFACT_CORRUPT','Fixed continuation IDs differ from rendered token suffix')
                artifact=record.get('activation_artifact')
                raw_name=f"runs/{record['id']}/activations.npz"
                if artifact:
                    shape=artifact.get('shape',[]);mask=artifact.get('pool_mask',[])
                    if artifact.get('normalization_state')!='before_final_model_norm' or artifact.get('site')!=SITE or artifact.get('dtype')!='float32' or artifact.get('axis_names')!=['layer','selected_token','hidden'] or len(shape)!=3 or shape[0]!=manifest['model_profile']['layers'] or shape[2]!=manifest['model_profile']['hidden_size'] or not all(isinstance(x,int) and x>0 for x in shape) or math.prod(shape)>25_000_000:
                        raise EngineError('INVALID_BUNDLE','Invalid activation metadata')
                    if shape[1]!=18 or mask!=list(range(2,18)) or artifact.get('layer_indices')!=list(range(shape[0])) or len(artifact.get('token_indices',[]))!=shape[1] or len(mask)!=16 or len(set(mask))!=16 or any(not isinstance(x,int) or not 0<=x<shape[1] for x in mask):raise EngineError('INVALID_BUNDLE','Invalid primary pooling mask')
                    if not isinstance(record.get('continuation_ids'),list) or len(record['continuation_ids'])<16:raise EngineError('INVALID_BUNDLE','Missing continuation token alignment')
                    if artifact.get('semantic_position_labels')!=['last_scenario_token','last_prompt_token']+[f'continuation_{i}' for i in range(16)]:raise EngineError('INVALID_BUNDLE','Invalid semantic capture labels')
                    offset=record.get('continuation_offset')
                    if type(offset) is not int or offset<1 or artifact['token_indices'][1:]!=[offset-1]+list(range(offset,offset+16)):
                        raise EngineError('INVALID_BUNDLE','Primary pool is not the first 16 aligned continuation tokens')
                    if any(type(v) is not int or v<0 or v>=len(record.get('token_ids',[])) for v in artifact['token_indices']):raise EngineError('INVALID_BUNDLE','Selected token index outside rendered input')
                    if raw_name in names and checksums[raw_name]!=artifact.get('sha256'):raise EngineError('ARTIFACT_CORRUPT','Activation metadata checksum mismatch')
                elif raw_name in names:raise EngineError('INVALID_BUNDLE','Raw array has no metadata')
                if 'generated_token_count' in record and (type(record['generated_token_count']) is not int or not 0<=record['generated_token_count']<=96):raise EngineError('INVALID_BUNDLE','Invalid generated token count')
                if 'truncated' in record and type(record['truncated']) is not bool:raise EngineError('INVALID_BUNDLE','Invalid truncation flag')
                for metric in ('seconds','continuation_nll'):
                    if metric in record and record[metric] is not None and (type(record[metric]) not in (int,float) or not math.isfinite(record[metric]) or record[metric]<0):raise EngineError('INVALID_BUNDLE','Invalid timing or continuation NLL')
                for field,maximum in [('token_ids',8192),('continuation_ids',512),('generated_ids',96)]:
                    values=record.get(field,[])
                    if not isinstance(values,list) or len(values)>maximum or any(not isinstance(v,int) or v<0 or v>10_000_000 for v in values):raise EngineError('INVALID_BUNDLE',f'Invalid bounded {field}')
                records.append(record)
            history_jobs=[];history_events=[]
            for filename,limit,target in [('jobs.jsonl',1000,history_jobs),('events.jsonl',50000,history_events)]:
                if filename in names:
                    lines=archive.read(filename).splitlines()
                    if len(lines)>limit:raise EngineError('INVALID_BUNDLE','Execution history exceeds count limit')
                    target.extend(_json(line,filename) for line in lines if line.strip())
            history_job_ids=set()
            for job in history_jobs:
                _uuid(job.get('id'),'Historical job id')
                if job['id'] in history_job_ids or job.get('experiment_id')!=manifest['id'] or not isinstance(job.get('created_at'),str) or job.get('state') not in ('draft','queued','running','completed','failed','cancelled','interrupted'):raise EngineError('INVALID_BUNDLE','Historical job identity conflicts with this experiment')
                history_job_ids.add(job['id'])
            event_ids=set()
            for event in history_events:
                key=(event.get('job_id'),event.get('seq'))
                if key in event_ids or event.get('job_id') not in history_job_ids or type(event.get('seq')) is not int or event['seq']<1 or not isinstance(event.get('created_at'),str) or not isinstance(event.get('type'),str) or not isinstance(event.get('payload'),dict):raise EngineError('INVALID_BUNDLE','Historical event identity is invalid')
                event_ids.add(key)
            attempt_count=0
            for name in names:
                match=ATTEMPT_PATH.fullmatch(name)
                if not match:continue
                attempt_count+=1;record=_json(archive.read(name),name)
                if match[1] not in history_job_ids or match[2] not in units or record.get('id')!=match[2] or record.get('job_id')!=match[1] or record.get('experiment_id')!=manifest['id'] or record.get('status') not in ('completed','failed'):
                    raise EngineError('INVALID_BUNDLE','Historical attempt does not match its frozen unit and job')
                if any(record.get(k)!=v for k,v in units[match[2]].items()):raise EngineError('INVALID_BUNDLE','Historical attempt treatment differs from the frozen unit')
            if attempt_count>5000:raise EngineError('INVALID_BUNDLE','Too many historical attempts')
            raw_ids={RUN_PATH.fullmatch(n)[1] for n in names if n.endswith('/activations.npz')}
            record_ids={r['id'] for r in records}
            if not raw_ids<=record_ids:raise EngineError('INVALID_BUNDLE','Orphan activation arrays')
            if not inventory.get('includes_raw_activations') and raw_ids:raise EngineError('INVALID_BUNDLE','Undeclared raw arrays')
            ratings=_json(archive.read('ratings.json'),'ratings.json') if 'ratings.json' in names else []
            if not isinstance(ratings,list) or len(ratings)>5000:raise EngineError('INVALID_BUNDLE','Ratings must be a bounded list')
            rating_ids=set()
            for rating in ratings:
                _uuid(rating.get('id'),'Rating id')
                scores=rating.get('dimension_scores',{})
                if rating['id'] in rating_ids or rating.get('run_unit_id') not in record_ids or rating.get('rubric_version')!='expressed-behavior-v1' or set(scores)!={'regret','responsibility','repair','justification_deflection'} or any(type(v) is not int or v not in (0,1,2) for v in scores.values()) or not isinstance(rating.get('rater_id'),str) or not 1<=len(rating['rater_id'])<=100:
                    raise EngineError('INVALID_BUNDLE','Invalid rating record')
                rating_ids.add(rating['id'])
            return {'manifest':manifest,'analysis':analysis,'inventory':inventory,'ratings':ratings,'history_jobs':history_jobs,'history_events':history_events,'records':records,'raw_unit_ids':sorted(raw_ids),'archive_sha256':file_digest(path),'verified_entries':len(infos)}
    except EngineError:raise
    except (OSError,ValueError,KeyError,TypeError,zipfile.BadZipFile,RuntimeError) as e:raise EngineError('INVALID_BUNDLE',f'Invalid evidence archive: {e}') from e

def _compare(a,b,path='',absolute=1e-6,relative=1e-5):
    if path.endswith('/caveat'):return [] # Editorial caveats are not numeric replay endpoints.
    if isinstance(a,bool) or isinstance(b,bool):return [] if a==b else [path]
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):
        return [] if math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,abs_tol=absolute,rel_tol=relative) else [path]
    if isinstance(a,dict) and isinstance(b,dict):
        if set(a)!=set(b):return [path+':keys']
        return [issue for k in a for issue in _compare(a[k],b[k],path+'/'+str(k),absolute,relative)]
    if isinstance(a,list) and isinstance(b,list):
        if len(a)!=len(b):return [path+':length']
        return [issue for i,(x,y) in enumerate(zip(a,b)) for issue in _compare(x,y,path+'/'+str(i),absolute,relative)]
    return [] if a==b else [path]

def replay_experiment(store,experiment_id,analysis_id=None,verify_metrics=True):
    """Replay uses only stored arrays; importing this module never imports Transformers."""
    from ghosttrace.analysis.report import analyze
    exp=store.experiment(experiment_id)
    if analysis_id is None:
        with store.connect() as db:row=db.execute('SELECT record FROM analyses WHERE experiment_id=? ORDER BY rowid DESC LIMIT 1',(experiment_id,)).fetchone()
        if not row:raise EngineError('NOT_FOUND','No recorded analysis to replay')
        original=json.loads(row['record'])
    else:original=store.get_record('analyses',analysis_id)
    if original['experiment_id']!=experiment_id:raise EngineError('CONFLICT','Analysis belongs to a different experiment')
    rows=store.unit_rows(experiment_id)
    needed=[r for r in rows if json.loads(r['record']).get('activation_artifact') and r['status']=='completed']
    available=bool(needed) and all(r['artifact_path'] for r in needed)
    if not available:
        return {'experiment_id':experiment_id,'analysis_id':original['id'],'recomputed':False,'raw_activations_available':False,'verification':{'status':'summary_only','reason':'Raw arrays omitted or no completed captures; saved metrics are viewable but cannot be independently recomputed.','absolute_tolerance':1e-6,'relative_tolerance':1e-5}}
    for row in needed:read_activation(store.root/row['artifact_path'],row['artifact_sha256'])
    recomputed=analyze(store,experiment_id,original['settings']['split'])
    differences=[]
    if verify_metrics:
        for key in ('aggregates','token_maps','included_units','exclusions_with_reasons','expected_unit_count','observed_unit_count','settings'):
            old_value=original.get(key);new_value=recomputed.get(key)
            # Schema 1.0 gained optional secondary metrics without changing the L2 endpoint.
            if key=='aggregates' and isinstance(old_value,dict):
                new_value={contrast:{k:v for k,v in value.items() if k!='metrics' or 'metrics' in old_value.get(contrast,{})} for contrast,value in new_value.items()}
            if key=='token_maps' and isinstance(old_value,list):
                old_contrasts={row['contrast'] for row in old_value}
                new_value=[row for row in new_value if row['contrast'] not in ('neutral-wording','factual-wording') or row['contrast'] in old_contrasts]
            differences.extend(_compare(old_value,new_value,key))
        if 'behavioral' in original:differences.extend(_compare(original['behavioral'],recomputed['behavioral'],'behavioral'))
        if differences:raise EngineError('REPLAY_MISMATCH','Recomputed metrics differ from recorded analysis',{'paths':differences[:20],'absolute_tolerance':1e-6,'relative_tolerance':1e-5,'analysis_id':recomputed['id']})
    return {'experiment_id':experiment_id,'analysis_id':recomputed['id'],'source_analysis_id':original['id'],'recomputed':True,'raw_activations_available':True,'verification':{'status':'matched' if verify_metrics else 'recomputed_uncompared','absolute_tolerance':1e-6,'relative_tolerance':1e-5,'model_loaded':False}}

def import_bundle(store,path,verify_metrics=True):
    """Stage and validate all content before a transactional metadata registration."""
    verified=verify_bundle(path);exp=verified['manifest'];analysis=verified['analysis'];archive_hash=verified['archive_sha256']
    with store.connect() as db:
        existing=db.execute('SELECT record FROM imports WHERE json_extract(record,\'$.archive_sha256\')=?',(archive_hash,)).fetchone()
        if existing:
            existing_value=json.loads(existing['record'])
            if not verify_metrics or existing_value['verification']['status'] in ('matched','summary_only'):return existing_value
        old=db.execute('SELECT record FROM experiments WHERE id=? OR config_hash=?',(exp['id'],exp['config_hash'])).fetchone()
        if old and json.loads(old['record'])!=exp:raise EngineError('CONFLICT','Experiment ID or hash conflicts with existing local evidence')
        for record in verified['records']:
            old_unit=db.execute('SELECT record FROM units WHERE id=?',(record['id'],)).fetchone()
            if old_unit and json.loads(old_unit['record'])!=record:raise EngineError('CONFLICT','Run unit conflicts with existing immutable local evidence')
        for rating in verified['ratings']:
            old_rating=db.execute('SELECT record FROM ratings WHERE id=?',(rating['id'],)).fetchone()
            if old_rating and json.loads(old_rating['record'])!=rating:raise EngineError('CONFLICT','Rating ID conflicts with local evidence')
        old_analysis=db.execute('SELECT record FROM analyses WHERE id=?',(analysis['id'],)).fetchone()
        if old_analysis and json.loads(old_analysis['record'])!=analysis:raise EngineError('CONFLICT','Analysis ID conflicts with local evidence')
    imports_root=store.root/'imports';imports_root.mkdir(exist_ok=True)
    with zipfile.ZipFile(path) as archive:expanded_bytes=sum(item.file_size for item in archive.infolist())
    if shutil.disk_usage(imports_root).free<expanded_bytes+64_000_000:raise EngineError('DISK_QUOTA','Insufficient disk space to stage and verify this evidence bundle')
    staging=Path(tempfile.mkdtemp(prefix='.staging-',dir=imports_root));destination=imports_root/archive_hash
    try:
        with zipfile.ZipFile(path) as archive:
            for name in verified['inventory']['file_inventory_with_checksums']:
                target=staging/name;target.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(name) as source,target.open('wb') as output:shutil.copyfileobj(source,output,1024*1024)
            atomic_json(staging/'inventory.json',verified['inventory'])
        for record in verified['records']:
            raw=staging/'runs'/record['id']/'activations.npz'
            if raw.exists():
                array=read_activation(raw,record['activation_artifact']['sha256'])
                if list(array.shape)!=record['activation_artifact']['shape']:raise EngineError('ARTIFACT_CORRUPT','Array shape differs from declared capture metadata')
        # Check replay in a temporary isolated Store before exposing anything to the application.
        from ghosttrace.storage.db import Store
        validation_store=Store(staging/'validation')
        _register(validation_store,exp,analysis,verified['records'],staging,allow_existing=False,ratings=verified['ratings'])
        replay=replay_experiment(validation_store,exp['id'],analysis['id'],verify_metrics) if verified['raw_unit_ids'] else {'analysis_id':analysis['id'],'recomputed':False,'raw_activations_available':False,'verification':{'status':'summary_only','reason':'Raw activation arrays were omitted.'}}
        shutil.rmtree(staging/'validation')
        if destination.exists():shutil.rmtree(staging)
        else:staging.rename(destination)
        _register(store,exp,analysis,verified['records'],destination,allow_existing=True,ratings=verified['ratings'])
        result={'id':str(uuid.uuid4()),'schema_version':'1.0','created_at':now(),'archive_sha256':archive_hash,'experiment_id':exp['id'],'analysis_id':analysis['id'],'raw_activations_available':bool(verified['raw_unit_ids']) and replay['raw_activations_available'],'recomputed':replay['recomputed'],'verification':{**replay['verification'],'inventory_sha256_verified':True,'verified_entries':verified['verified_entries'],'authenticity':'Checksums verify integrity, not authorship or scientific validity.','prompt_binding':'full_inventory' if 'prompt_templates_hash' in exp else 'legacy_selected_families_only','legacy_unbound_control_hash':'prompt_templates_hash' not in exp,'model_profile_binding':'canonical_artifact' if 'model-profile.json' in verified['inventory']['file_inventory_with_checksums'] else 'legacy_inline_python_canonical','legacy_metadata_note':'Older records may lack role, mask, timing or memory fields; absent metadata remains unavailable.'},'recorded_experiment':True,'model_profile':exp['model_profile'],'relative_path':str(destination.relative_to(store.root)),'execution_history':{'job_count':len(verified.get('history_jobs',[])),'event_count':len(verified.get('history_events',[])),'inert':True}}
        store.put_record('imports',result)
        return result
    finally:
        if staging.exists():shutil.rmtree(staging)

def _register(store,exp,analysis,records,root,allow_existing,ratings=None):
    """Portable raw records remain byte-equivalent; SQLite holds relocated local paths."""
    with store.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        db.execute('INSERT OR IGNORE INTO experiments VALUES(?,?,?,?)',(exp['id'],exp['config_hash'],canonical(exp).decode(),exp['created_at']))
        db.execute('INSERT OR IGNORE INTO analyses VALUES(?,?,?)',(analysis['id'],exp['id'],canonical(analysis).decode()))
        for record in records:
            raw=root/'runs'/record['id']/'activations.npz'
            relative=str(raw.relative_to(store.root)) if raw.exists() and raw.is_relative_to(store.root) else None
            # Validation store references staged arrays outside its root by a verified private copy.
            if raw.exists() and relative is None:
                target=store.root/'runs'/record['id']/'activations.npz';target.parent.mkdir(parents=True,exist_ok=True)
                try:
                    import os
                    os.link(raw,target)
                except OSError:shutil.copyfile(raw,target)
                relative=str(target.relative_to(store.root))
            checksum=record.get('activation_artifact',{}).get('sha256') if relative else None
            db.execute('INSERT INTO units VALUES(?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET artifact_path=COALESCE(units.artifact_path,excluded.artifact_path),artifact_sha256=COALESCE(units.artifact_sha256,excluded.artifact_sha256)',(record['id'],exp['id'],canonical(record).decode(),record['status'],relative,checksum,record.get('created_at',now())))

        for rating in ratings or []:
            db.execute('INSERT OR IGNORE INTO ratings VALUES(?,?,?)',(rating['id'],rating['run_unit_id'],canonical(rating).decode()))
