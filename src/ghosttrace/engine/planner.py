from __future__ import annotations
import json
import random
import uuid
from pathlib import Path
from ghosttrace.config import canonical,digest,now,validate_config,read_dataset,EngineError,file_digest,read_json_file,checked_file_digest,validate_templates
from ghosttrace.engine.render import TEMPLATES
from ghosttrace.models.profiles import get_profile
from ghosttrace.storage.artifacts import atomic_json,atomic_bytes

def plan(store,config):
    config=validate_config(config)
    dataset_path=Path(config['dataset'])
    if dataset_path.is_dir():dataset_path=dataset_path/'scenarios.jsonl'
    if not dataset_path.exists():dataset_path=Path('datasets')/config['dataset']/'scenarios.jsonl'
    scenarios=read_dataset(dataset_path)
    factual_path=dataset_path.parent/'factual-controls.jsonl'
    factual=read_dataset(factual_path) if factual_path.exists() and len(scenarios)==30 and config['templates']==['A','B'] else []
    if factual:
        # Validate the joint namespace and family split before composing scenario maps.
        from ghosttrace.config import validate_dataset
        validate_dataset(scenarios+factual)
    template_path=dataset_path.parent/'templates.json'
    if not template_path.exists(): template_path=Path('datasets/synthetic-v1/templates.json')
    frozen_templates=read_json_file(template_path,list)
    validate_templates(frozen_templates,config['templates'])
    profile=get_profile(config['model_profile'])
    try:validated=store.get_record('profiles',profile['id'])
    except EngineError:validated=None
    if validated:profile=validated
    prereg_path=Path(config.get('preregistration','preregistrations/pilot-v1.md'))
    prereg_hash=checked_file_digest(prereg_path) if prereg_path.exists() else None
    experiment_base={'schema_version':'1.0','config':config,'dataset_hash':digest(scenarios),'factual_dataset_hash':digest(factual) if factual else None,'model_profile_hash':digest(profile),'prompt_templates_hash':digest(frozen_templates),'template_hashes':{t:digest([r for r in frozen_templates if r['family']==t] or TEMPLATES[t]) for t in config['templates']},'preregistration_hash':prereg_hash}
    review_path=dataset_path.parent/'human-review.json'
    review_receipt=read_json_file(review_path,dict) if review_path.exists() else None
    prereg_receipt_path=prereg_path.with_suffix('.json')
    prereg_receipt=read_json_file(prereg_receipt_path,dict) if prereg_receipt_path.exists() else None
    if review_receipt:experiment_base['human_review_hash']=digest(review_receipt)
    if prereg_receipt:experiment_base['preregistration_receipt_hash']=digest(prereg_receipt)
    config_hash=digest(experiment_base)
    with store.connect() as db:
        old=db.execute('SELECT record FROM experiments WHERE config_hash=?',(config_hash,)).fetchone()
        if old:return json.loads(old['record'])
    rng=random.Random(config.get('order_seed',73))
    units=[]
    for scenario in scenarios:
        block=[]
        for template in config['templates']:
            for condition in config['conditions']:
                for mode in config['modes']:
                    unit={'scenario_id':scenario['id'],'condition':condition,'template_id':template,'mode':mode,'seed':None,'control':None}
                    unit['id']=digest({'experiment_hash':config_hash,**unit});block.append(unit)
        # Separate mandatory floor/lexical controls; smoke runs also get these.
        for condition,control in [('neutral','neutral_repeat'),('label_only','label_only')]:
            unit={'scenario_id':scenario['id'],'condition':condition,'template_id':'A','mode':'fixed_continuation','seed':None,'control':control}
            unit['id']=digest({'experiment_hash':config_hash,**unit});block.append(unit)
        rng.shuffle(block);units.extend(block)
    for scenario in factual:
        for template in config['templates']:
            unit={'scenario_id':scenario['id'],'condition':'neutral','template_id':template,'mode':'fixed_continuation','seed':None,'control':'factual_task'}
            unit['id']=digest({'experiment_hash':config_hash,**unit});units.append(unit)
    estimated=sum(1 for u in units if u['mode']=='fixed_continuation')*profile['layers']*18*profile['hidden_size']*4
    warnings=[]
    if not validated or not validated.get('preflight_passed'):warnings.append('Hardware preflight required before queueing; no live measurements are available.')
    if prereg_hash is None:warnings.append('Preregistration is missing; canonical holdout execution is blocked.')
    if any(s.get('review_status')!='reviewed' for s in scenarios):warnings.append('Dataset requires independent human review before canonical pilot.')
    if estimated>config.get('limits',{}).get('max_artifact_bytes',2_000_000_000):raise EngineError('DISK_QUOTA','Estimated activation payload exceeds artifact quota')
    record={**experiment_base,'id':str(uuid.uuid4()),'config_hash':config_hash,'created_at':now(),'model_profile':profile,'scenarios':scenarios,'control_scenarios':factual,'prompt_templates':frozen_templates,'ordered_units':units,'ordered_unit_ids':[u['id'] for u in units],'estimated_units':len(units),'estimated_bytes':estimated,'warnings':warnings,'measurement_status':'not_run','human_review':review_receipt,'preregistration_receipt':prereg_receipt}
    root=store.root/config_hash
    atomic_json(root/'manifest.json',record)
    if review_receipt:atomic_json(root/'human-review.json',review_receipt)
    if prereg_receipt:atomic_json(root/'preregistration-receipt.json',prereg_receipt)
    atomic_bytes(root/'dataset.jsonl',b'\n'.join(canonical(s) for s in scenarios)+b'\n')
    atomic_json(root/'prompts.json',frozen_templates or TEMPLATES)
    if factual:atomic_bytes(root/'factual-controls.jsonl',b'\n'.join(canonical(s) for s in factual)+b'\n')
    if prereg_path.exists():atomic_bytes(root/'preregistration.md',prereg_path.read_bytes())
    with store.connect() as db:db.execute('INSERT INTO experiments VALUES(?,?,?,?)',(record['id'],config_hash,canonical(record).decode(),now()))
    return record
