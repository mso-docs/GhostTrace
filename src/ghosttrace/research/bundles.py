"""Portable v2 evidence. No imported job, module, model or code is executed."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import stat
import tempfile
import uuid
import zipfile
from ghosttrace.config import EngineError,canonical,digest,file_digest,now
from ghosttrace.research.schemas import StudyCreate
from ghosttrace.research.storage import ResearchRepository
from ghosttrace.research.service import get_bundle,merge_graph
from ghosttrace.research.arrays import read_arrays,browser_tile

MAX_BUNDLE=500_000_000
MAX_JSON=64_000_000
KINDS={'study','capture','intervention','hypothesis','dictionary','feature','annotation','lens','lens_fit','lens_result','feature_intervention','feature_attribution','circuit'}
BASE_KEYS=('schema_version','config','model_profile','model_profile_hash','resolved_capture','protocol','classification','estimated_uncompressed_bytes')

def _json(content):
    if len(content)>MAX_JSON:raise EngineError('INVALID_BUNDLE','Research JSON exceeds 64 MB')
    try:
        value=json.loads(content,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')));canonical(value);return value
    except (ValueError,UnicodeError,RecursionError,TypeError) as error:raise EngineError('INVALID_BUNDLE','Invalid finite UTF-8 research JSON') from error

def _uuid(value):
    try:
        if str(uuid.UUID(value))!=value:raise ValueError('Noncanonical UUID')
        return value
    except (ValueError,TypeError,AttributeError):raise EngineError('INVALID_BUNDLE','Research IDs must be canonical UUIDs')

def _inventory_records(repo,study_id):
    bundle=get_bundle(repo.store,study_id);records=[{'kind':'study','study_id':None,'record':bundle['study']}]
    ids={bundle['study']['id']}
    for kind,values in [('capture',bundle['captures']),('intervention',bundle['interventions']),('hypothesis',bundle['hypotheses']),('dictionary',bundle['dictionaries']),('feature',bundle['features']),('annotation',bundle['feature_annotations']),('lens',bundle.get('lenses',[])),('lens_result',bundle.get('lens_results',[])),('feature_intervention',bundle.get('feature_interventions',[])),('circuit',bundle.get('circuits',[])),('feature_attribution',bundle.get('feature_attributions',[]))]:
        for record in values:
            if record['id'] not in ids:records.append({'kind':kind,'study_id':record.get('study_id'),'record':record});ids.add(record['id'])
    for kind in ('lens','lens_fit','lens_result','feature_intervention','feature_attribution','circuit'):
        for record in repo.list(kind,study_id):
            if record['id'] not in ids:records.append({'kind':kind,'study_id':study_id,'record':record});ids.add(record['id'])
    return bundle,records

def export_research(store,study_id,include_arrays=True):
    repo=ResearchRepository(store);bundle,records=_inventory_records(repo,study_id);study=bundle['study'];export_id=str(uuid.uuid4());root=store.root/'research'/study['study_hash']/'exports'/export_id;root.mkdir(parents=True,exist_ok=True)
    definition={key:study[key] for key in BASE_KEYS}
    files={'bundle.json':canonical(bundle),'records.json':canonical(records),'study-definition.json':canonical(definition),'model-profile.json':canonical(study['model_profile']),'README.md':b'GhostTrace v2 recorded mechanistic research. No model weights, executable modules, or active jobs. Inspect interpretation and controls before drawing conclusions. Importing verifies integrity and saved numerical tiles; it never reruns a model.\n'}
    raw={};omitted=[]
    for entry in records:
        record=entry['record']
        if not record.get('artifact'):continue
        if not include_arrays:omitted.append(record['id']);continue
        try:path,checksum=repo.artifact_location(record)
        except EngineError as error:
            if error.code=='UNSUPPORTED_CAPTURE':omitted.append(record['id']);continue
            raise
        read_arrays(path,checksum);raw[f"arrays/{record['id']}.npz"]=(path,checksum)
    if any(len(content)>MAX_JSON for content in files.values()):raise EngineError('DISK_QUOTA','Research viewer JSON exceeds 64 MB; use smaller selected capture windows')
    size=sum(len(v) for v in files.values())+sum(path.stat().st_size for path,sha in raw.values())
    if size>MAX_BUNDLE:raise EngineError('DISK_QUOTA','Research evidence exceeds 500 MB export budget')
    hashes={name:hashlib.sha256(content).hexdigest() for name,content in files.items()};hashes.update({name:sha for name,(path,sha) in raw.items()})
    inventory={'schema_version':'2.0','kind':'ghosttrace-research-bundle','study_id':study_id,'study_hash':study['study_hash'],'model_profile_hash':study['model_profile_hash'],'created_at':now(),'files':hashes,'includes_arrays':include_arrays,'omitted_array_record_ids':omitted,'verification_scope':'SHA-256, immutable record/study/profile bindings, bounded raw arrays and saved capture tiles. Model outputs and intervention causality are not independently re-executed during import.'}
    files['inventory.json']=canonical(inventory)
    fd,temp=tempfile.mkstemp(prefix='.pending-',suffix='.zip',dir=root);os.close(fd)
    try:
        with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,allowZip64=True) as archive:
            for name,content in files.items():archive.writestr(name,content)
            for name,(path,sha) in raw.items():archive.write(path,name)
        os.replace(temp,root/'evidence.zip')
    finally:
        if os.path.exists(temp):os.unlink(temp)
    return repo.put('export',{'id':export_id,'study_id':study_id,'relative_path':str((root/'evidence.zip').relative_to(store.root)),'inventory':inventory,'status':'completed'},study_id)

def verify_research(path):
    path=Path(path)
    if not path.is_file() or path.stat().st_size>MAX_BUNDLE:raise EngineError('PAYLOAD_TOO_LARGE','Research archive missing or exceeds 500 MB')
    try:
        with zipfile.ZipFile(path) as archive:
            infos=archive.infolist();names=[item.filename for item in infos]
            if len(infos)>4000 or len(set(names))!=len(names) or sum(item.file_size for item in infos)>MAX_BUNDLE:raise EngineError('INVALID_BUNDLE','Research archive exceeds count/expanded-size limits or has duplicates')
            allowed={'bundle.json','records.json','study-definition.json','model-profile.json','README.md','inventory.json'}
            for item in infos:
                p=PurePosixPath(item.filename)
                if p.is_absolute() or str(p)!=item.filename or '..' in p.parts or '\\' in item.filename or ':' in item.filename or item.is_dir() or stat.S_ISLNK(item.external_attr>>16) or item.flag_bits&1:raise EngineError('INVALID_BUNDLE','Unsafe research archive path/type')
                if item.filename not in allowed:
                    if len(p.parts)!=2 or p.parts[0]!='arrays' or p.suffix!='.npz':raise EngineError('INVALID_BUNDLE','Unknown research archive entry')
                    _uuid(p.stem)
                if item.file_size>(200_000_000 if item.filename.endswith('.npz') else MAX_JSON):raise EngineError('INVALID_BUNDLE','Oversized research archive member')
            if not {'bundle.json','records.json','study-definition.json','model-profile.json','inventory.json'}<=set(names):raise EngineError('INVALID_BUNDLE','Missing research evidence bindings')
            inventory=_json(archive.read('inventory.json'))
            if inventory.get('kind')!='ghosttrace-research-bundle' or inventory.get('schema_version')!='2.0' or set(inventory.get('files',{}))!=set(names)-{'inventory.json'}:raise EngineError('INVALID_BUNDLE','Research inventory must cover every member exactly')
            for name,expected in inventory['files'].items():
                sha=hashlib.sha256()
                with archive.open(name) as source:
                    for chunk in iter(lambda:source.read(1024*1024),b''):sha.update(chunk)
                if sha.hexdigest()!=expected:raise EngineError('ARTIFACT_CORRUPT','Research member checksum mismatch',{'file':name})
            bundle=_json(archive.read('bundle.json'));records=_json(archive.read('records.json'));definition=_json(archive.read('study-definition.json'));study=bundle.get('study',{})
            StudyCreate.model_validate(study.get('config',{}));_uuid(study.get('id'))
            if inventory['study_id']!=study['id'] or inventory['study_hash']!=study.get('study_hash') or digest(definition)!=study['study_hash'] or definition!={key:study[key] for key in BASE_KEYS}:raise EngineError('ARTIFACT_CORRUPT','Research study definition hash mismatch')
            if hashlib.sha256(archive.read('model-profile.json')).hexdigest()!=study['model_profile_hash'] or _json(archive.read('model-profile.json'))!=study['model_profile']:raise EngineError('ARTIFACT_CORRUPT','Research model-profile binding mismatch')
            if not isinstance(records,list) or len(records)>3000:raise EngineError('INVALID_BUNDLE','Research record count exceeds limit')
            ids={};conditions={c['id'] for c in study['config']['conditions']}
            for entry in records:
                kind=entry.get('kind');record=entry.get('record',{});record_id=_uuid(record.get('id'))
                if kind not in KINDS or record_id in ids or record.get('schema_version')!='2.0':raise EngineError('INVALID_BUNDLE','Unknown/duplicate research record')
                if digest({k:v for k,v in record.items() if k!='record_hash'})!=record.get('record_hash'):raise EngineError('ARTIFACT_CORRUPT','Research immutable record hash mismatch')
                if kind=='study' and record!=study or kind!='study' and entry.get('study_id') not in (None,study['id']):raise EngineError('INVALID_BUNDLE','Research record belongs to another study')
                if kind=='capture' and record.get('condition_id') not in conditions:raise EngineError('INVALID_BUNDLE','Capture condition absent from study')
                ids[record_id]=entry
                raw_name=f'arrays/{record_id}.npz'
                if raw_name in names and inventory['files'][raw_name]!=record.get('artifact',{}).get('sha256'):raise EngineError('ARTIFACT_CORRUPT','Research raw-array hash differs from record')
            if sum(entry['kind']=='study' for entry in records)!=1:raise EngineError('INVALID_BUNDLE','Research bundle requires exactly one frozen study')
            for key,kind in [('captures','capture'),('interventions','intervention'),('hypotheses','hypothesis'),('dictionaries','dictionary'),('features','feature'),('feature_annotations','annotation'),('lenses','lens'),('lens_results','lens_result'),('feature_interventions','feature_intervention'),('circuits','circuit'),('feature_attributions','feature_attribution')]:
                for record in bundle.get(key,[]):
                    if record.get('id') not in ids or ids[record['id']]['kind']!=kind or ids[record['id']]['record']!=record:raise EngineError('ARTIFACT_CORRUPT','Research viewer differs from immutable record inventory')
            reconstructed_graph=merge_graph(bundle.get('captures',[]),bundle.get('interventions',[]),bundle.get('feature_interventions',[]),bundle.get('circuits',[]),bundle.get('feature_attributions',[]))
            if reconstructed_graph!=bundle.get('graph'):raise EngineError('ARTIFACT_CORRUPT','Research graph view differs from immutable evidence records')
            for name in names:
                if name.startswith('arrays/') and PurePosixPath(name).stem not in ids:raise EngineError('INVALID_BUNDLE','Orphan research array file')
            return {'bundle':bundle,'records':records,'inventory':inventory,'archive_sha256':file_digest(path),'entries':len(infos)}
    except EngineError:raise
    except (OSError,ValueError,TypeError,KeyError,zipfile.BadZipFile) as error:raise EngineError('INVALID_BUNDLE','Invalid research evidence archive',{'reason':type(error).__name__}) from error

def _verify_tiles(record,arrays):
    components={c['array_key']:c for c in record.get('components',[])}
    for tile in record.get('tiles',[]):
        key=tile.get('array_key')
        if key not in arrays or key not in components:raise EngineError('ARTIFACT_CORRUPT','Saved research tile has no source array')
        actual=browser_tile(arrays[key],components[key])
        # The saved tile may have a smaller explicit budget than the default.
        import numpy as np
        from ghosttrace.research.arrays import tile_matrix
        matrix,row_axis,trailing,shape=tile_matrix(arrays[key],components[key]);values=np.asarray(tile.get('values'))
        if values.ndim!=2 or values.shape[0]>matrix.shape[0] or values.shape[1]>matrix.shape[1] or not np.allclose(values,matrix[:values.shape[0],:values.shape[1]],atol=1e-6,rtol=1e-5):raise EngineError('REPLAY_MISMATCH','Saved research tile differs from the recorded raw tensor')
        if tile.get('shape')!=list(arrays[key].shape) or tile.get('row_axis') not in (None,row_axis):raise EngineError('ARTIFACT_CORRUPT','Research tile axis metadata mismatch')

def import_research(store,path):
    checked=verify_research(path);repo=ResearchRepository(store);archive_hash=checked['archive_sha256'];study=checked['bundle']['study']
    for prior in repo.list('import'):
        if prior.get('archive_sha256')==archive_hash:return prior
    for entry in checked['records']:
        try:existing=repo.get(entry['record']['id'])
        except EngineError as error:
            if error.code!='NOT_FOUND':raise
        else:
            if existing!=entry['record']:raise EngineError('CONFLICT','Imported research ID conflicts with immutable local evidence')
    root=store.root/'research'/'imports';root.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.staging-',dir=root));destination=root/archive_hash;raw_count=0;verified_tiles=0
    try:
        with zipfile.ZipFile(path) as archive:
            for name in checked['inventory']['files']:
                target=stage/name;target.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(name) as source,target.open('wb') as output:shutil.copyfileobj(source,output,1024*1024)
        for entry in checked['records']:
            record=entry['record'];raw=stage/'arrays'/f"{record['id']}.npz"
            if raw.exists():
                arrays=read_arrays(raw,record['artifact']['sha256']);raw_count+=1
                declared=record['artifact'].get('arrays',{})
                if set(arrays)!=set(declared) or any(list(array.shape)!=declared[key]['shape'] for key,array in arrays.items()):raise EngineError('ARTIFACT_CORRUPT','Research array shapes differ from recorded metadata')
                if entry['kind']=='capture':_verify_tiles(record,arrays);verified_tiles+=len(record.get('tiles',[]))
        if destination.exists():shutil.rmtree(stage)
        else:stage.rename(destination)
        # One metadata transaction: imported profiles never enter trusted v1 profiles.
        with store.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            for entry in checked['records']:
                record=entry['record'];db.execute('INSERT OR IGNORE INTO research_records VALUES(?,?,?,?,?)',(record['id'],entry['kind'],entry.get('study_id'),canonical(record).decode(),record['created_at']))
                raw=destination/'arrays'/f"{record['id']}.npz"
                if raw.exists():db.execute('INSERT OR IGNORE INTO research_artifacts VALUES(?,?,?)',(record['id'],str(raw.relative_to(store.root)),record['artifact']['sha256']))
        return repo.put('import',{'study_id':study['id'],'archive_sha256':archive_hash,'relative_path':str(destination.relative_to(store.root)),'verification':{'inventory_verified':True,'immutable_bindings_verified':True,'raw_artifacts_verified':raw_count,'capture_tiles_recomputed':verified_tiles,'absolute_tolerance':1e-6,'relative_tolerance':1e-5,'model_loaded':False,'model_output_or_causal_reexecution':False,'authenticity':'Integrity and numerical tile agreement do not establish authorship or scientific validity.'},'recorded':True},study['id'])
    finally:
        if stage.exists():shutil.rmtree(stage)
