from __future__ import annotations
import json
import statistics
import subprocess
import uuid
import zipfile
from collections import defaultdict
from pathlib import Path
from ghosttrace.config import EngineError,canonical,digest,file_digest,now
from ghosttrace.analysis.bootstrap import bootstrap_scenarios
from ghosttrace.analysis.metrics import distances
from ghosttrace.storage.artifacts import atomic_json,atomic_bytes,read_activation

CONTRASTS={'remorseful-neutral':('remorseful','neutral'),'unremorseful-neutral':('unremorseful','neutral'),'remorseful-unremorseful':('remorseful','unremorseful'),'neutral-repeat':('neutral_repeat','neutral'),'neutral-wording':('neutral_B','neutral_A'),'label-only-neutral':('label_only','neutral'),'factual-wording':('factual_A','factual_B')}

def analyze(store,experiment_id,split='development'):
    import numpy as np
    if split not in ('development','holdout','all'):raise EngineError('INVALID_CONFIG','Invalid analysis split')
    exp=store.experiment(experiment_id)
    # Explicit API/CLI selection is required to unseal holdout, and is permanently recorded.
    if split in ('holdout','all') and any(s['split']=='holdout' for s in exp['scenarios']):
        review=exp.get('human_review');receipt=exp.get('preregistration_receipt')
        if not review or not receipt or not exp.get('preregistration_hash') or not review.get('reviewer'):
            raise EngineError('CONFLICT','Freeze hash-bound human review and preregistration receipts before holdout analysis')
        if review.get('reviewed_dataset_hash')!=exp['dataset_hash'] or review.get('reviewed_templates_hash')!=digest(exp.get('prompt_templates',[])) or review.get('reviewed_controls_hash')!=digest(exp.get('control_scenarios',[])) or exp.get('human_review_hash')!=digest(review):
            raise EngineError('CONFLICT','Human review does not match this frozen dataset, templates, and controls')
        if receipt.get('config_hash')!=digest(exp['config']) or receipt.get('dataset_hash')!=exp['dataset_hash'] or receipt.get('template_hash')!=digest(exp.get('prompt_templates',[])) or receipt.get('human_review_hash')!=digest(review) or receipt.get('protocol_sha256')!=exp['preregistration_hash'] or exp.get('preregistration_receipt_hash')!=digest(receipt):
            raise EngineError('CONFLICT','Preregistration receipt does not match the frozen protocol')
        if any(s.get('review_status')!='reviewed' for s in exp['scenarios']+exp.get('control_scenarios',[])) or any(t.get('review_status')!='reviewed' for t in exp.get('prompt_templates',[])):
            raise EngineError('CONFLICT','Canonical holdout requires every source item to have human review')
    records=[json.loads(r['record']) for r in store.unit_rows(experiment_id)]
    included=[];exclusions=[];groups=defaultdict(dict)
    scenario_by_id={s['id']:s for s in exp['scenarios']+exp.get('control_scenarios',[])}
    expected_units=[u for u in exp['ordered_units'] if split=='all' or scenario_by_id[u['scenario_id']]['split']==split]
    observed_ids={r['id'] for r in records}
    for unit in expected_units:
        if unit['id'] not in observed_ids:exclusions.append({'unit_id':unit['id'],'reason':'not_run','mode':unit['mode']})
    for record in records:
        if split!='all' and record.get('split')!=split:continue
        if record['status']!='completed':
            exclusions.append({'unit_id':record['id'],'reason':record.get('error',{}).get('code','not_completed')});continue
        if record['mode']!='fixed_continuation':continue
        artifact=record.get('activation_artifact')
        if not artifact:
            exclusions.append({'unit_id':record['id'],'reason':'unavailable_capture'});continue
        try:
            path,checksum=store.activation_location(record)
            array=read_activation(path,checksum)
        except EngineError as e:
            if e.code=='UNSUPPORTED_CAPTURE':
                exclusions.append({'unit_id':record['id'],'reason':'raw_activations_omitted'});continue
            raise
        key=(record['scenario_id'],record['template_id'])
        label=record.get('control') or record['condition']
        groups[key][label]=(record,array)
    layers=exp['model_profile']['layers'];primary=layers//2-1
    all_curves=defaultdict(lambda:defaultdict(list));secondary_curves=defaultdict(lambda:defaultdict(lambda:defaultdict(list)));token_maps=[]
    for (scenario,template),group in groups.items():
        for contrast,(a,b) in CONTRASTS.items():
            if contrast in ('neutral-wording','factual-wording'):continue
            if a not in group or b not in group:continue
            ra,aa=group[a];rb,ab=group[b]
            if ra['continuation_ids']!=rb['continuation_ids']:
                exclusions.append({'unit_ids':[ra['id'],rb['id']],'reason':'TOKEN_ALIGNMENT_FAILED'});continue
            mask_a=ra['activation_artifact']['pool_mask'];mask_b=rb['activation_artifact']['pool_mask']
            va=aa[:,mask_a,:].mean(axis=1,dtype=np.float32);vb=ab[:,mask_b,:].mean(axis=1,dtype=np.float32)
            metric_rows=[distances(x.tolist(),y.tolist()) for x,y in zip(va,vb)]
            curve=[m['normalized_l2'] for m in metric_rows]
            for metric in ('cosine_distance','norm_ratio'):secondary_curves[contrast][metric][scenario].append([m[metric] for m in metric_rows])
            all_curves[contrast][scenario].append(curve)
            included.extend([ra['id'],rb['id']])
            token_maps.append({'scenario_id':scenario,'template':template,'contrast':contrast,'unit_ids':[ra['id'],rb['id']],'pool_mask':mask_a,'absolute_position_difference':[x-y for x,y in zip(ra['absolute_positions'],rb['absolute_positions'])],'values':[[distances(aa[l,x,:].tolist(),ab[l,y,:].tolist())['normalized_l2'] for x,y in zip(mask_a,mask_b)] for l in range(layers)]})
        if template=='A':
            other=groups.get((scenario,'B'),{}).get('neutral')
            if 'factual_task' in group:
                other_factual=groups.get((scenario,'B'),{}).get('factual_task')
                if other_factual:
                    ra,aa=group['factual_task'];rb,ab=other_factual
                    if ra['continuation_ids']==rb['continuation_ids']:
                        va=aa[:,ra['activation_artifact']['pool_mask'],:].mean(axis=1);vb=ab[:,rb['activation_artifact']['pool_mask'],:].mean(axis=1)
                        wording_metrics=[distances(x.tolist(),y.tolist()) for x,y in zip(va,vb)]
                        all_curves['factual-wording'][scenario].append([m['normalized_l2'] for m in wording_metrics])
                        for metric in ('cosine_distance','norm_ratio'):secondary_curves['factual-wording'][metric][scenario].append([m[metric] for m in wording_metrics])
                        included.extend([ra['id'],rb['id']])
                        ma=ra['activation_artifact']['pool_mask'];mb=rb['activation_artifact']['pool_mask']
                        token_maps.append({'scenario_id':scenario,'template':'A','contrast':'factual-wording','unit_ids':[ra['id'],rb['id']],'pool_mask':ma,'absolute_position_difference':[x-y for x,y in zip(ra['absolute_positions'],rb['absolute_positions'])],'values':[[distances(aa[l,x,:].tolist(),ab[l,y,:].tolist())['normalized_l2'] for x,y in zip(ma,mb)] for l in range(layers)]})
            if 'neutral' in group and other:
                ra,aa=group['neutral'];rb,ab=other
                if ra['continuation_ids']==rb['continuation_ids']:
                    va=aa[:,ra['activation_artifact']['pool_mask'],:].mean(axis=1);vb=ab[:,rb['activation_artifact']['pool_mask'],:].mean(axis=1)
                    wording_metrics=[distances(x.tolist(),y.tolist()) for x,y in zip(va,vb)]
                    all_curves['neutral-wording'][scenario].append([m['normalized_l2'] for m in wording_metrics])
                    for metric in ('cosine_distance','norm_ratio'):secondary_curves['neutral-wording'][metric][scenario].append([m[metric] for m in wording_metrics])
                    included.extend([ra['id'],rb['id']])
                    ma=ra['activation_artifact']['pool_mask'];mb=rb['activation_artifact']['pool_mask']
                    token_maps.append({'scenario_id':scenario,'template':'A','contrast':'neutral-wording','unit_ids':[ra['id'],rb['id']],'pool_mask':ma,'absolute_position_difference':[x-y for x,y in zip(ra['absolute_positions'],rb['absolute_positions'])],'values':[[distances(aa[l,x,:].tolist(),ab[l,y,:].tolist())['normalized_l2'] for x,y in zip(ma,mb)] for l in range(layers)]})
                else:exclusions.append({'unit_ids':[ra['id'],rb['id']],'reason':'TOKEN_ALIGNMENT_FAILED'})
    aggregates={}
    seed=exp['config'].get('analysis',{}).get('bootstrap_seed',20261027)
    for contrast in CONTRASTS:
        scenarios=all_curves[contrast]
        scenario_means={s:[statistics.fmean(c[l] for c in curves) for l in range(layers)] for s,curves in scenarios.items()}
        curves=[{'layer':l,**bootstrap_scenarios([v[l] for v in scenario_means.values()],seed=seed)} for l in range(layers)]
        family_values=defaultdict(list)
        family_by_id={s['id']:s['family_id'] for s in exp['scenarios']+exp.get('control_scenarios',[])}
        for sid,v in scenario_means.items():family_values[family_by_id[sid]].append(v[primary])
        aggregates[contrast]={'primary':curves[primary],'layers':curves,'scenario_values':scenario_means,'family_sensitivity':{k:{'mean':statistics.fmean(v),'n':len(v)} for k,v in family_values.items()}}
        aggregates[contrast]['metrics']={}
        for metric in ('cosine_distance','norm_ratio'):
            metric_means={}
            for scenario,template_curves in secondary_curves[contrast][metric].items():
                metric_means[scenario]=[statistics.fmean(valid) if (valid:=[c[l] for c in template_curves if c[l] is not None]) else None for l in range(layers)]
            metric_layers=[{'layer':l,**bootstrap_scenarios([v[l] for v in metric_means.values() if v[l] is not None],seed=seed),'missing_scenarios':sum(v[l] is None for v in metric_means.values())} for l in range(layers)]
            aggregates[contrast]['metrics'][metric]={'primary':metric_layers[primary],'layers':metric_layers,'scenario_values':metric_means,'interpretation':'Descriptive distance; unavailable for zero-norm vectors.' if metric=='cosine_distance' else 'Vector norm ratio a/b, not a distance or emotional score.'}
    try:commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception:commit='uncommitted'
    from ghosttrace.provenance import code_provenance
    source=code_provenance()
    if source.get('workspace_dirty'):commit='uncommitted:'+commit
    settings={'split':split,'metric_version':'normalized-l2-v1','bootstrap_draws':2000,'bootstrap_seed':seed,'primary_layer':primary}
    from ghosttrace.analysis.ratings import rating_summary
    from ghosttrace.analysis.behavior import descriptive_outputs
    selected_records=[r for r in records if split=='all' or r.get('split')==split]
    reliability={'matched_pair_count':len(token_maps),'alignment_excluded_pair_count':sum(x['reason']=='TOKEN_ALIGNMENT_FAILED' for x in exclusions),'matched_pair_token_identity_rate':len(token_maps)/(len(token_maps)+sum(x['reason']=='TOKEN_ALIGNMENT_FAILED' for x in exclusions)) if token_maps or any(x['reason']=='TOKEN_ALIGNMENT_FAILED' for x in exclusions) else None,'interpretation':'Identity rate counts attempted valid/invalid contrast pairs, not independent scenarios. Review absolute-position differences separately.'}
    performance={'completed_unit_count':sum(r['status']=='completed' for r in selected_records),'failed_unit_count':sum(r['status']=='failed' for r in selected_records),'missing_planned_unit_count':sum(x['reason']=='not_run' for x in exclusions),'measured_unit_seconds_total':sum(r.get('seconds',0) for r in selected_records if type(r.get('seconds')) in (int,float)),'artifact_bytes':sum((store.root/r['artifact_path']).stat().st_size for r in store.unit_rows(experiment_id) if r['artifact_path'] and (store.root/r['artifact_path']).is_file()),'device':exp['model_profile'].get('device'),'compute_dtype':exp['model_profile'].get('compute_dtype'),'preflight':{k:exp['model_profile'].get(k) for k in ('repeat_max_abs_error','repeat_tolerance','peak_ram_bytes','peak_vram_bytes','generation_tokens_per_second','load_and_test_seconds')},'interpretation':'Elapsed unit timings are measured. Missing preflight fields remain null; summed unit time excludes queue/load overhead.'}
    analysis_label='Development diagnostic — exploratory' if split=='development' else ('Frozen holdout pilot' if split=='holdout' else 'Combined-split exploratory view')
    result={'analysis_label':analysis_label,'uncertainty_label':'Descriptive scenario bootstrap interval','reliability':reliability,'performance':performance,'behavioral':descriptive_outputs(selected_records,expected_units,seed),'ratings':rating_summary(store,experiment_id),'id':str(uuid.uuid4()),'schema_version':'1.0','created_at':now(),'experiment_id':experiment_id,'code_commit':commit,'code_provenance':source,'settings':settings,'settings_hash':digest(settings),'expected_unit_count':len(expected_units),'observed_unit_count':sum(r['id'] in {u['id'] for u in expected_units} for r in records),'included_units':sorted(set(included)),'exclusions_with_reasons':exclusions,'aggregates':aggregates,'token_maps':token_maps,'measurement_status':'measured' if included else 'not_run','limitations':['Persona instructions describe experimental conditions, not feelings.','Activation distance is descriptive, not causal evidence.','Absolute token positions may differ across prompts.','Human ratings and family sensitivity are required to contextualize this pilot.'],'projection_artifact':None}
    root=store.root/exp['config_hash']/'analyses'/(result['settings_hash']+'-'+result['id'])
    atomic_json(root/'metrics.json',result);atomic_json(root/'exclusions.json',exclusions)
    store.put_record('analyses',result,experiment_id)
    return result

def report_markdown(exp,analysis):
    lines=['# GhostTrace evidence report','',f"Experiment: {exp['id']}",f"Manifest SHA-256: {exp['config_hash']}",f"Analysis: {analysis['id']}",f"Status: {analysis['measurement_status']}",f"Split: {analysis['settings']['split']}",analysis.get('analysis_label','Descriptive pilot analysis'),'','## Interpretation','Persona conditions change instructions. These measurements do not establish remorse, consciousness, guilt, or a moral center. Distances are descriptive, not causal evidence.','','## Primary endpoint',f"Zero-based block {analysis['settings']['primary_layer']}; normalized L2; first 16 verified continuation tokens. Scenario-level paired means, 2,000 bootstrap draws.",'','| Contrast | Mean | 95% interval | Scenarios |','|---|---:|---|---:|']
    for key,value in analysis['aggregates'].items():
        p=value['primary'];fmt=lambda n:'Unavailable' if n is None else f'{n:.6f}'
        lines.append(f"| {key} | {fmt(p['mean'])} | {fmt(p['lower'])}–{fmt(p['upper'])} | {p['n']} |")
    lines+=['','## Secondary representation metrics','Cosine distance is unavailable when either vector norm is near zero. Norm ratio a/b measures magnitude and is not a distance.','','| Contrast | Cosine distance at primary block | Norm ratio at primary block |','|---|---:|---:|']
    for contrast,value in analysis['aggregates'].items():
        metrics=value.get('metrics',{})
        cosine=metrics.get('cosine_distance',{}).get('primary',{}).get('mean');ratio=metrics.get('norm_ratio',{}).get('primary',{}).get('mean')
        lines.append(f"| {contrast} | {cosine if cosine is not None else 'Unavailable'} | {ratio if ratio is not None else 'Unavailable'} |")
    behavior=analysis.get('behavioral')
    if behavior:
        lines+=['','## Descriptive output behavior',behavior['interpretation'],'','| Condition | Completed / planned outputs | Mean generated tokens | Mean words | Truncation fraction | Continuation NLL |','|---|---:|---:|---:|---:|---:|']
        for condition,data in behavior['conditions'].items():
            metric=data['metrics'];fmt=lambda value:'Unavailable' if value is None else f'{value:.4f}'
            lines.append(f"| {condition} | {data['completed_units']} / {data['expected_units']} | {fmt(metric['generated_token_count']['mean'])} | {fmt(metric['word_count']['mean'])} | {fmt(metric['truncated']['mean'])} | {fmt(behavior['continuation_nll']['conditions'][condition]['mean'])} |")
        lines+=['',behavior['flags_limitation']]
    ratings=analysis.get('ratings')
    if ratings:
        lines+=['','## Blinded expressed-behavior ratings',f"Raters: {ratings['rater_count']}; unique ratings: {ratings['unique_rating_count']}."]
        for comparison in ratings['pairwise_agreement']:
            lines.append(f"Paired responses: {comparison['paired_response_count']}; raters: {', '.join(comparison['raters'])}.")
            for dimension,agreement in comparison['dimensions'].items():
                lines.append(f"- {dimension}: agreement={agreement['agreement']}, quadratic weighted Cohen kappa={agreement['weighted_kappa']}, n={agreement['n']} ({agreement.get('reason') or 'defined'}).")
        lines += ['- '+limitation for limitation in ratings['limitations']]
    lines+=['','## Missingness',f"Excluded records/pairs: {len(analysis['exclusions_with_reasons'])}",'','## Limitations']+['- '+x for x in analysis['limitations']]
    return '\n'.join(lines)+'\n'

def export_bundle(store,analysis_id,include_activations=False):
    """Stream raw arrays into the archive; keep only small JSON payloads in memory."""
    import hashlib
    import os
    import tempfile
    analysis=store.get_record('analyses',analysis_id);exp=store.experiment(analysis['experiment_id'])
    export_id=str(uuid.uuid4());root=store.root/exp['config_hash'];output=root/'exports'/export_id
    output.mkdir(parents=True,exist_ok=True)
    report=report_markdown(exp,analysis);atomic_bytes(output/'report.md',report.encode())
    files={'README.md':b'GhostTrace recorded evidence bundle. No model weights or executable code. Checksums verify integrity, not scientific validity. Import offline with ghosttrace import evidence.zip; raw arrays are required for metric recomputation.\n','manifest.json':canonical(exp),'model-profile.json':canonical(exp['model_profile']),'analysis.json':canonical(analysis),'report.md':report.encode()}
    with store.connect() as db:
        imported=db.execute("SELECT record FROM imports WHERE json_extract(record,'$.experiment_id')=? ORDER BY rowid DESC LIMIT 1",(exp['id'],)).fetchone()
        ratings=[json.loads(r['record']) for r in db.execute('SELECT ratings.record FROM ratings JOIN units ON units.id=ratings.run_unit_id WHERE units.experiment_id=?',(exp['id'],))]
    sources=[root]
    if imported:sources.append(store.root/json.loads(imported['record'])['relative_path'])
    for name in ['environment.json','dataset.jsonl','prompts.json','preregistration.md','factual-controls.jsonl','human-review.json','preregistration-receipt.json']:
        for source in sources:
            if (source/name).exists():files[name]=(source/name).read_bytes();break
    from ghosttrace.storage.history import portable_history
    history_files,history=portable_history(store,exp['id'],root,sources)
    files.update(history_files)
    files['ratings.json']=canonical(ratings)
    import csv,io
    table=io.StringIO();writer=csv.writer(table);writer.writerow(['contrast','metric','zero_based_block','mean','lower_95','upper_95','n_scenarios'])
    for contrast,aggregate in analysis['aggregates'].items():
        for metric,values in [('normalized_l2',aggregate),*aggregate.get('metrics',{}).items()]:
            for layer in values['layers']:writer.writerow([contrast,metric,layer['layer'],layer['mean'],layer['lower'],layer['upper'],layer['n']])
    files['metrics.csv']=table.getvalue().encode('utf-8')
    raw_files={}
    for row in store.unit_rows(exp['id']):
        record=json.loads(row['record']);files[f"runs/{record['id']}/record.json"]=canonical(record)
        if include_activations and row['artifact_path']:
            read_activation(store.root/row['artifact_path'],row['artifact_sha256'])
            raw_files[f"runs/{record['id']}/activations.npz"]=(store.root/row['artifact_path'],row['artifact_sha256'])
    inventory={name:hashlib.sha256(content).hexdigest() for name,content in files.items()}
    inventory.update({name:checksum for name,(path,checksum) in raw_files.items()})
    metadata={'id':export_id,'schema_version':'1.0','created_at':now(),'analysis_id':analysis_id,'manifest_hash':exp['config_hash'],'file_inventory_with_checksums':inventory,'includes_raw_activations':include_activations,'raw_array_count':len(raw_files),'execution_history':history,'redaction_notes':'No model weights. Synthetic prompts, output text, and supplied rater identifiers are included.','status':'completed','relative_path':str((output/'evidence.zip').relative_to(store.root))}
    files['inventory.json']=canonical(metadata)
    fd,temp_path=tempfile.mkstemp(prefix='.pending-',suffix='.zip',dir=output);os.close(fd)
    try:
        with zipfile.ZipFile(temp_path,'w',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as z:
            for name,content in files.items():z.writestr(name,content)
            for name,(path,checksum) in raw_files.items():z.write(path,name)
        os.replace(temp_path,output/'evidence.zip')
    finally:
        if os.path.exists(temp_path):os.unlink(temp_path)
    atomic_json(output/'inventory.json',metadata);store.put_record('exports',metadata)
    return metadata
