#!/usr/bin/env python3
"""Package a genuinely measured development smoke for model-free offline replay."""
import argparse
import json
import shutil
from pathlib import Path
from ghosttrace.storage.db import Store
from ghosttrace.config import EngineError,canonical,file_digest,now
from ghosttrace.analysis.report import analyze,export_bundle

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',required=True);p.add_argument('--data-dir',default='artifacts');p.add_argument('--output',default='web/public/demo');p.add_argument('--raw-output',default='artifacts/measured-smoke-raw.zip')
a=p.parse_args()
try:
    store=Store(a.data_dir);exp=store.experiment(a.experiment)
    if any(s['split']!='development' for s in exp['scenarios']):raise EngineError('INVALID_CONFIG','Bundled demo must contain development scenarios only; holdout export is an explicit separate action')
    units=store.unit_rows(exp['id'])
    if len(units)!=exp['estimated_units'] or any(r['status']!='completed' for r in units):raise EngineError('CONFLICT','Demo requires all planned units completed; retain partial runs separately')
    profile=exp['model_profile']
    if not profile.get('preflight_passed') or not profile.get('weight_file_hashes'):raise EngineError('MODEL_NOT_READY','Demo requires real immutable model provenance and successful preflight')
    analysis=analyze(store,exp['id'],'development')
    if analysis['measurement_status']!='measured' or not analysis['included_units']:raise EngineError('CONFLICT','No measured aligned artifacts available')
    summary=export_bundle(store,analysis['id'],False);raw=export_bundle(store,analysis['id'],True)
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True);source=store.root/summary['relative_path']
    if source.stat().st_size>5_000_000:raise EngineError('DISK_QUOTA','Bundled summary exceeds5MB budget')
    shutil.copyfile(source,output/'evidence.zip');raw_path=Path(a.raw_output);raw_path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(store.root/raw['relative_path'],raw_path)
    descriptor={'schema_version':'1.0','created_at':now(),'measurement_status':'measured','interpretation':'Recorded development tooling diagnostic, not a confirmatory pilot. No emotional-state or causal inference.','experiment_id':exp['id'],'analysis_id':analysis['id'],'model':profile['repo_id'],'revision':profile['revision'],'device':profile['device'],'compute_dtype':profile['compute_dtype'],'scenario_count':len(exp['scenarios']),'unit_count':len(units),'templates':exp['config']['templates'],'bundle_sha256':file_digest(output/'evidence.zip'),'includes_raw_activations':False,'raw_recomputation_bundle':str(raw_path),'human_review_status':'pending','recorded_at':exp['created_at']}
    (output/'descriptor.json').write_bytes(canonical(descriptor));(output/'README.md').write_text('# Recorded GhostTrace smoke\n\nThis bundle contains actual local measurements, not simulated fixture values. It covers only development scenarios and is a tooling diagnostic, not a confirmatory pilot. Human review remains pending. Model weights are not bundled. Raw activations are omitted from the browser bundle and included in the separate local raw-recomputation bundle.\n\n'+json.dumps(descriptor,indent=2)+'\n')
    print(json.dumps(descriptor,indent=2))
except EngineError as e:p.exit(1,json.dumps({'error':{'code':e.code,'message':e.message}})+'\n')
