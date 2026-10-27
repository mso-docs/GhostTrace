"""Immutable, content-addressed experiment contracts. No model import at load time."""
from __future__ import annotations
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = '1.0'
CONDITIONS = ('neutral', 'remorseful', 'unremorseful')
SITE = 'residual_post_block_pre_final_norm'

class EngineError(Exception):
    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code, self.message, self.details = code, message, details or {}

def now():
    return datetime.now(timezone.utc).isoformat()

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')

def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def file_digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def _keys(obj, allowed, name):
    if not isinstance(obj, dict):
        raise EngineError('INVALID_CONFIG', f'{name} must be an object')
    extra = set(obj) - set(allowed)
    if extra:
        raise EngineError('INVALID_CONFIG', f'Unknown {name} keys: {sorted(extra)}')

def validate_config(config):
    _keys(config, ['schema_version','name','model_profile','dataset','split_policy','conditions','templates','modes','capture','generation','limits','analysis','order_seed','preregistration'], 'config')
    try: c = json.loads(canonical(config))
    except (ValueError, TypeError, RecursionError): raise EngineError('INVALID_CONFIG', 'Configuration must contain finite JSON values')
    def integer(value, lo, hi, label):
        if type(value) is not int or not lo <= value <= hi:
            raise EngineError('INVALID_CONFIG', f'{label} must be an integer in {lo}..{hi}')
    if c.get('schema_version') != SCHEMA_VERSION:
        raise EngineError('INVALID_CONFIG', 'schema_version must be 1.0')
    for key in ('name','dataset','model_profile'):
        if not isinstance(c.get(key), str) or not c[key].strip() or len(c[key]) > 500:
            raise EngineError('INVALID_CONFIG', f'{key} must be a nonempty bounded string')
    if c.get('split_policy','family-heldout-v1') != 'family-heldout-v1':
        raise EngineError('INVALID_CONFIG', 'Only the family-heldout-v1 split policy is supported')
    if c.get('conditions') != list(CONDITIONS):
        raise EngineError('INVALID_CONFIG', 'All three ordered conditions are required')
    templates = c.get('templates')
    if not isinstance(templates,list) or templates not in (['A'],['A','B']):
        raise EngineError('INVALID_CONFIG', 'templates must be [A] or [A, B]; A is required for mandatory controls')
    modes = c.get('modes')
    if not isinstance(modes,list) or not modes or any(not isinstance(m,str) for m in modes) or len(set(modes)) != len(modes):
        raise EngineError('INVALID_CONFIG', 'modes must be a nonempty list without duplicates')
    if set(modes) - {'fixed_continuation','greedy_generation'}:
        raise EngineError('UNSUPPORTED_CAPTURE', 'Only fixed_continuation and greedy_generation are implemented; replay_generated is unavailable')
    capture = c.setdefault('capture', {})
    _keys(capture, ['site','layers','anchors','continuation_tokens','storage_dtype','use_cache'], 'capture')
    if capture.get('site', SITE) != SITE:
        raise EngineError('UNSUPPORTED_CAPTURE', 'Only pre-final-norm block outputs are supported')
    if capture.get('layers','all') != 'all' or capture.get('anchors',['last_scenario_token','last_prompt_token']) != ['last_scenario_token','last_prompt_token']:
        raise EngineError('UNSUPPORTED_CAPTURE', 'Canonical capture requires all blocks and both named anchors')
    if type(capture.get('continuation_tokens',16)) is not int or capture.get('continuation_tokens',16)!=16 or capture.get('storage_dtype','float32')!='float32' or capture.get('use_cache',False) is not False:
        raise EngineError('INVALID_CONFIG', 'Canonical capture requires 16 tokens, float32 storage and use_cache=false')
    generation = c.setdefault('generation', {})
    _keys(generation, ['do_sample','max_new_tokens','stop'], 'generation')
    if generation.get('do_sample',False) is not False or generation.get('stop','model_eos')!='model_eos':
        raise EngineError('INVALID_CONFIG', 'Canonical generation requires do_sample=false and model_eos stop')
    integer(generation.get('max_new_tokens',96),1,96,'max_new_tokens')
    limits = c.setdefault('limits', {})
    _keys(limits, ['max_input_tokens','max_total_tokens','batch_size','max_artifact_bytes'], 'limits')
    integer(limits.get('batch_size',1),1,1,'batch_size')
    integer(limits.get('max_input_tokens',512),32,4096,'max_input_tokens')
    integer(limits.get('max_total_tokens',640),33,8192,'max_total_tokens')
    if limits.get('max_total_tokens',640) < limits.get('max_input_tokens',512) + generation.get('max_new_tokens',96):
        raise EngineError('INVALID_CONFIG', 'max_total_tokens cannot fit input and generation limits')
    integer(limits.get('max_artifact_bytes',2_000_000_000),1,20_000_000_000,'max_artifact_bytes')
    analysis = c.setdefault('analysis', {})
    _keys(analysis, ['primary_layer','primary_metric','bootstrap_draws','bootstrap_seed'], 'analysis')
    if analysis.get('primary_layer','middle_block')!='middle_block' or analysis.get('primary_metric','normalized_l2')!='normalized_l2':
        raise EngineError('INVALID_CONFIG', 'Canonical analysis requires middle_block and normalized_l2')
    integer(analysis.get('bootstrap_draws',2000),2000,2000,'bootstrap_draws')
    integer(analysis.get('bootstrap_seed',20261027),0,2**32-1,'bootstrap_seed')
    integer(c.get('order_seed',73),0,2**32-1,'order_seed')
    if 'preregistration' in c and (not isinstance(c['preregistration'],str) or not c['preregistration']):
        raise EngineError('INVALID_CONFIG','preregistration must be a nonempty path')
    return c

def load_config(path):
    import yaml
    p=Path(path)
    try:
        if p.stat().st_size>5_000_000:raise EngineError('INVALID_CONFIG','Config exceeds 5 MB')
        config=yaml.safe_load(p.read_text(encoding='utf-8'))
    except (OSError,UnicodeError,yaml.YAMLError,RecursionError) as error:
        raise EngineError('INVALID_CONFIG','Could not read a valid UTF-8 YAML configuration',{'path':str(p)[:500],'reason':type(error).__name__}) from error
    return validate_config(config)

def read_json_file(path,expected_type=None,max_bytes=5_000_000):
    p=Path(path)
    try:
        if p.stat().st_size>max_bytes:raise EngineError('INVALID_CONFIG','JSON file exceeds size limit',{'path':str(p)[:500]})
        value=json.loads(p.read_text(encoding='utf-8'))
        canonical(value)
    except (OSError,UnicodeError,ValueError,RecursionError) as error:
        raise EngineError('INVALID_CONFIG','Could not read finite UTF-8 JSON',{'path':str(p)[:500],'reason':type(error).__name__}) from error
    if expected_type and not isinstance(value,expected_type):raise EngineError('INVALID_CONFIG','JSON file has the wrong top-level type',{'path':str(p)[:500]})
    return value

def checked_file_digest(path):
    try:return file_digest(path)
    except OSError as error:raise EngineError('INVALID_CONFIG','Could not read provenance file',{'path':str(path)[:500],'reason':type(error).__name__}) from error

def validate_dataset(rows):
    if not isinstance(rows,list) or not 1 <= len(rows) <= 200:
        raise EngineError('INVALID_CONFIG', 'Dataset must have 1..200 scenarios')
    ids, family_splits = set(), {}
    for row in rows:
        _keys(row, ['schema_version','id','family_id','split','text','common_continuation','language','provenance','review_status','sha256','created_at','analysis_role'], 'scenario')
        for field in ('id','family_id','text','common_continuation'):
            if not isinstance(row.get(field),str) or not row[field].strip():
                raise EngineError('INVALID_CONFIG', f'Scenario {field} must be a nonempty string')
        if len(row['id'])>128 or len(row['family_id'])>128:
            raise EngineError('INVALID_CONFIG','Scenario identifiers exceed 128 characters')
        if row.get('id') in ids or not row.get('id'):
            raise EngineError('INVALID_CONFIG','Duplicate or empty scenario id')
        ids.add(row['id'])
        if row.get('split') not in ('development','holdout'):
            raise EngineError('INVALID_CONFIG','Invalid scenario split')
        family = row.get('family_id')
        if not family or family in family_splits and family_splits[family] != row['split']:
            raise EngineError('INVALID_CONFIG','Scenario family overlaps development and holdout')
        family_splits[family] = row['split']
        if not 1 <= len(row['text']) <= 2000 or not 1 <= len(row['common_continuation']) <= 2000:
            raise EngineError('INVALID_CONFIG','Scenario text or continuation missing/too long')
        if row.get('schema_version',SCHEMA_VERSION)!=SCHEMA_VERSION:
            raise EngineError('INVALID_CONFIG','Unsupported scenario schema_version')
        if row.get('sha256') and row['sha256']!=digest({k:v for k,v in row.items() if k!='sha256'}):
            raise EngineError('INVALID_CONFIG','Scenario content checksum mismatch',{'scenario_id':row['id']})
    return {'schema_version': SCHEMA_VERSION, 'count':len(rows), 'dataset_hash':digest(rows), 'families':family_splits, 'warnings':['Human review must be recorded before a canonical pilot.'] if any(r.get('review_status') != 'reviewed' for r in rows) else []}

def read_dataset(path):
    p = Path(path)
    try:
        if p.stat().st_size > 5_000_000:raise EngineError('INVALID_CONFIG','Dataset exceeds 5 MB')
        rows = [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]
        canonical(rows)
    except (OSError,UnicodeError,ValueError,RecursionError) as error:
        raise EngineError('INVALID_CONFIG','Could not read valid finite UTF-8 JSONL scenarios',{'path':str(p)[:500],'reason':type(error).__name__}) from error
    validate_dataset(rows)
    return rows


def validate_templates(rows,families=('A','B')):
    if not isinstance(rows,list) or not 1<=len(rows)<=7:raise EngineError('INVALID_CONFIG','Expected a bounded prompt-template list')
    known={f'{family}-{condition}' for family in ('A','B') for condition in CONDITIONS}|{'label-only-neutral'}
    required={f'{family}-{condition}' for family in families for condition in CONDITIONS}|{'label-only-neutral'}
    ids=set()
    for row in rows:
        _keys(row,['schema_version','created_at','id','family','condition','system_text','render_version','review_status','sha256'],'prompt template')
        identity=row.get('id')
        if not isinstance(identity,str) or identity not in known or identity in ids:raise EngineError('INVALID_CONFIG','Unknown or duplicate prompt-template id')
        ids.add(identity)
        expected_family='control' if identity=='label-only-neutral' else identity.split('-',1)[0]
        expected_condition='neutral' if identity=='label-only-neutral' else identity.split('-',1)[1]
        if row.get('family')!=expected_family or row.get('condition')!=expected_condition:raise EngineError('INVALID_CONFIG','Prompt-template identity does not match family and condition')
        if not isinstance(row.get('system_text'),str) or not 1<=len(row['system_text'])<=4000:raise EngineError('INVALID_CONFIG','Prompt-template system_text must be a bounded nonempty string')
        if row.get('schema_version',SCHEMA_VERSION)!=SCHEMA_VERSION:raise EngineError('INVALID_CONFIG','Unsupported prompt-template schema version')
        if 'sha256' in row and row['sha256']!=digest({k:v for k,v in row.items() if k!='sha256'}):raise EngineError('INVALID_CONFIG','Prompt-template content checksum mismatch',{'template_id':identity})
    if not required<=ids:raise EngineError('INVALID_CONFIG','Required persona or label-only prompt templates are missing',{'missing_ids':sorted(required-ids)})
    return rows
