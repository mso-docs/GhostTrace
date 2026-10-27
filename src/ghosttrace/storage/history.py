"""Bounded, inert snapshots of execution history for portable evidence."""
import json
import uuid
from pathlib import Path
from ghosttrace.config import EngineError,canonical,now

MAX_HISTORY_JSON_BYTES=64_000_000

def _jsonl(values,limit,label):
    if len(values)>limit:raise EngineError('DISK_QUOTA',f'{label} exceeds portable history count limit; no history was silently truncated')
    payload=b'\n'.join(canonical(v) for v in values)+b'\n'
    if len(payload)>MAX_HISTORY_JSON_BYTES:raise EngineError('DISK_QUOTA',f'{label} exceeds portable history byte limit')
    return payload

def portable_history(store,experiment_id,root,sources):
    jobs={};events={};attempts={}
    # Imported histories remain inert files; they never become active local jobs.
    for source in reversed(sources):
        for filename,target,key in [('jobs.jsonl',jobs,lambda x:x['id']),('events.jsonl',events,lambda x:(x['job_id'],x['seq']))]:
            path=source/filename
            if path.is_file():
                if path.stat().st_size>MAX_HISTORY_JSON_BYTES:raise EngineError('DISK_QUOTA','Saved history exceeds byte limit')
                for line in path.read_text().splitlines():
                    if line.strip():
                        value=json.loads(line);target[key(value)]=value
        if (source/'attempts').exists():
            for path in (source/'attempts').glob('*/*/record.json'):
                if not path.resolve().is_relative_to(source.resolve()):raise EngineError('ARTIFACT_CORRUPT','Unsafe history source path')
                attempts[str(path.relative_to(source))]=path
    with store.connect() as db:
        rows=db.execute('SELECT * FROM jobs WHERE experiment_id=? ORDER BY created_at LIMIT 1001',(experiment_id,)).fetchall()
        for row in rows:
            value=dict(row);value['payload']=json.loads(value['payload']);value['result']=json.loads(value['result']) if value.get('result') else None;jobs[value['id']]=value
        rows=db.execute('SELECT events.* FROM events JOIN jobs ON jobs.id=events.job_id WHERE jobs.experiment_id=? ORDER BY events.seq LIMIT 50001',(experiment_id,)).fetchall()
        for row in rows:
            value=dict(row);value['payload']=json.loads(value['payload']);events[(value['job_id'],value['seq'])]=value
    files={'jobs.jsonl':_jsonl(sorted(jobs.values(),key=lambda j:(j['created_at'],j['id'])),1000,'Job history'),'events.jsonl':_jsonl(sorted(events.values(),key=lambda e:(e['created_at'],e['job_id'],e['seq'])),50000,'Event history')}
    for path in (root/'runs').glob('*/attempt-*/record.json'):
        if not path.resolve().is_relative_to(root.resolve()):raise EngineError('ARTIFACT_CORRUPT','Unsafe attempt source path')
        unit_id=path.parent.parent.name;job_id=path.parent.name.removeprefix('attempt-')
        try:
            if str(uuid.UUID(job_id))!=job_id:raise ValueError('Noncanonical job ID')
        except ValueError as error:raise EngineError('ARTIFACT_CORRUPT','Invalid attempt job identity') from error
        attempts[f'attempts/{job_id}/{unit_id}/record.json']=path
    if len(attempts)>5000:raise EngineError('DISK_QUOTA','Attempt history exceeds 5000 records; no failures were silently omitted')
    for name,path in attempts.items():
        if path.stat().st_size>5_000_000:raise EngineError('DISK_QUOTA','Attempt record exceeds 5 MB')
        files[name]=path.read_bytes()
    return files,{'snapshot_at':now(),'job_count':len(jobs),'event_count':len(events),'attempt_record_count':len(attempts),'includes_attempt_activations':False,'attempt_activation_note':'Only latest canonical activation arrays are optionally exported. Earlier-attempt raw arrays are omitted; their output/error records remain included.','import_behavior':'Execution history is read-only provenance; imported jobs and events never enter the active queue.'}
