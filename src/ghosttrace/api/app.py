"""Localhost-only API; the worker is an explicitly separate process."""
from __future__ import annotations
import base64
import json
import os
import time
import secrets
import uuid
from datetime import datetime,timezone
from pathlib import Path
from typing import Literal,Annotated
from urllib.parse import urlsplit
from fastapi import FastAPI,Request,Header,Query
from fastapi.responses import JSONResponse,FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel,ConfigDict,Field
from ghosttrace.config import EngineError,now,canonical,digest,validate_dataset
from ghosttrace.storage.db import Store
from ghosttrace.engine.planner import plan
from ghosttrace.api.schemas import ExperimentConfigSchema
from ghosttrace.models.profiles import PROFILES,get_profile
from ghosttrace.analysis.report import analyze,export_bundle
from ghosttrace.storage.artifacts import read_activation
from ghosttrace.storage.replay import MAX_ARCHIVE_BYTES
from starlette.concurrency import run_in_threadpool
import os
import tempfile

class StrictModel(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
class PreflightRequest(StrictModel):
    profile_id:str
    revision:str=Field(pattern=r'^[a-f0-9]{40}$')
    device:Literal['cpu','cuda','mps']='cpu'
    compute_dtype:Literal['float32','float16','bfloat16']='float32'
class AnalysisRequest(StrictModel):
    split:Literal['development','holdout','all']='development'
class ExportRequest(StrictModel):
    analysis_id:uuid.UUID
    include_activations:bool=False
class ReplayRequest(StrictModel):
    analysis_id:uuid.UUID|None=None
    verify_metrics:bool=True
class RatingRequest(StrictModel):
    run_unit_id:str=Field(pattern=r'^[a-f0-9]{64}$')
    rubric_version:Literal['expressed-behavior-v1']='expressed-behavior-v1'
    blinded_display_id:str=Field(min_length=1,max_length=100)
    rater_id:str=Field(min_length=1,max_length=100)
    dimension_scores:dict[str,Annotated[int,Field(strict=True,ge=0,le=2)]]
    refusal:bool=False
    malformed:bool=False


def create_app(root=None):
    store=Store(root);mutation_token=secrets.token_urlsafe(32);app=FastAPI(title='GhostTrace local research API',version='0.1.0')
    app.state.store=store
    @app.middleware('http')
    async def safety(request,call_next):
        request.state.request_id=str(uuid.uuid4())
        host_header=request.headers.get('host','')
        try:
            parsed_host=urlsplit('http://'+host_header)
            host=parsed_host.hostname
            valid_host=host in ('127.0.0.1','localhost','testserver','::1') and not parsed_host.username and not parsed_host.password and parsed_host.path=='' and not parsed_host.query and not parsed_host.fragment
            parsed_host.port # Validate a bounded numeric port.
        except ValueError:valid_host=False
        if not valid_host:
            return JSONResponse({'error':{'code':'FORBIDDEN','message':'Localhost only'},'request_id':request.state.request_id},403)
        origin=request.headers.get('origin')
        if origin and origin!=f'{request.url.scheme}://{host_header}' and origin not in ('http://127.0.0.1:8000','http://localhost:8000','http://localhost:5173','http://127.0.0.1:5173'):
            return JSONResponse({'error':{'code':'FORBIDDEN','message':'Origin not allowed'},'request_id':request.state.request_id},403)
        if request.method in ('POST','PUT','PATCH','DELETE') and request.headers.get('x-ghosttrace-token') != mutation_token:
            return JSONResponse({'error':{'code':'FORBIDDEN','message':'Fetch /api/v1/health and provide X-GhostTrace-Token for mutations'},'request_id':request.state.request_id},403)
        try:content_length=int(request.headers.get('content-length','0'))
        except ValueError:return JSONResponse({'error':{'code':'INVALID_CONFIG','message':'Invalid Content-Length'},'request_id':request.state.request_id},400)
        body_limit=MAX_ARCHIVE_BYTES if request.url.path=='/api/v1/imports' else (500_000_000 if request.url.path in ('/api/v2/research/imports','/api/v2/research/dictionaries/imports','/api/v2/research/lenses/imports','/api/v2/research/lenses/train') else 5_000_000)
        if content_length>body_limit or content_length<0:
            return JSONResponse({'error':{'code':'PAYLOAD_TOO_LARGE','message':f'Maximum request body is {body_limit} bytes'},'request_id':request.state.request_id},413)
        response=await call_next(request);response.headers['X-Request-ID']=request.state.request_id
        return response
    from ghosttrace.api.limits import BodyLimitMiddleware
    app.add_middleware(BodyLimitMiddleware)
    @app.exception_handler(EngineError)
    async def engine_error(request,e):
        status={'NOT_FOUND':404,'CONFLICT':409,'MODEL_NOT_READY':409,'WORKER_UNAVAILABLE':503,'DISK_QUOTA':413,'ARTIFACT_CORRUPT':409,'REPLAY_MISMATCH':409,'PAYLOAD_TOO_LARGE':413}.get(e.code,422)
        return JSONResponse({'error':{'code':e.code,'message':e.message,'details':e.details},'request_id':request.state.request_id},status)
    @app.exception_handler(RequestValidationError)
    async def validation_error(request,e):
        return JSONResponse({'error':{'code':'INVALID_CONFIG','message':'Request validation failed','details':{'errors':str(e)}},'request_id':request.state.request_id},422)
    def wrapped(request,**value):return {'schema_version':'1.0','request_id':request.state.request_id,**value}
    def cursor_offset(cursor):
        try:return int(base64.urlsafe_b64decode(cursor).decode()) if cursor else 0
        except Exception:raise EngineError('INVALID_CONFIG','Invalid cursor')
    def listing(request,items,limit,cursor):
        offset=cursor_offset(cursor)
        if offset<0:raise EngineError('INVALID_CONFIG','Invalid cursor')
        page=items[offset:offset+limit]
        return wrapped(request,items=page,next_cursor=base64.urlsafe_b64encode(str(offset+limit).encode()).decode() if offset+limit<len(items) else None)
    @app.get('/api/v1/health')
    def health(request:Request):
        store.interrupt_stale()
        with store.connect() as db:
            row=db.execute('SELECT * FROM worker_state WHERE id=1').fetchone()
        active=bool(row and (datetime.now(timezone.utc)-datetime.fromisoformat(row['heartbeat_at'])).total_seconds()<15)
        return wrapped(request,version='0.1.0',api='healthy',worker=row['state'] if active else 'unavailable',offline=True,replay=True,model_loaded=bool(active and row['state']=='busy'),mutation_token=mutation_token)
    @app.get('/api/v1/models')
    def models(request:Request,limit:int=Query(100,ge=1,le=100),cursor:str|None=None):
        values=[]
        for p in PROFILES:
            try:v=store.get_record('profiles',p['id'])
            except EngineError:v={**p,'preflight_passed':False,'download_state':'not_verified','revision':None,'capabilities':['generation','residual_capture','continuation_scoring']}
            values.append(v)
        return listing(request,values,limit,cursor)
    @app.post('/api/v1/preflights',status_code=202)
    def preflights(body:PreflightRequest,request:Request):
        profile={**get_profile(body.profile_id),'revision':body.revision,'tokenizer_revision':body.revision,'device':body.device,'compute_dtype':body.compute_dtype}
        return wrapped(request,**store.queue(None,{'profile':profile},1,'preflight'))
    @app.post('/api/v1/datasets/validate')
    async def datasets(request:Request):
        body=await request.body()
        if len(body)>5_000_000:raise EngineError('INVALID_CONFIG','Dataset exceeds 5 MB')
        try:
            rows=json.loads(body) if body.lstrip().startswith(b'[') else [json.loads(x) for x in body.decode().splitlines() if x.strip()]
        except (ValueError,UnicodeError,RecursionError):raise EngineError('INVALID_CONFIG','Expected UTF-8 JSON array or JSONL')
        return wrapped(request,**validate_dataset(rows))
    @app.post('/api/v1/experiments',status_code=201)
    def experiments(body:ExperimentConfigSchema,request:Request):return wrapped(request,**plan(store,body.model_dump(exclude_unset=True)))
    @app.get('/api/v1/experiments')
    def experiment_list(request:Request,limit:int=Query(100,ge=1,le=100),cursor:str|None=None):
        with store.connect() as db:values=[json.loads(r['record']) for r in db.execute('SELECT record FROM experiments ORDER BY created_at DESC')]
        return listing(request,values,limit,cursor)
    @app.get('/api/v1/experiments/{experiment_id}')
    def experiment(experiment_id:uuid.UUID,request:Request):return wrapped(request,**store.experiment(str(experiment_id)))
    @app.post('/api/v1/experiments/{experiment_id}/jobs',status_code=202)
    def jobs(experiment_id:uuid.UUID,request:Request,body:dict|None=None,idempotency_key:str=Header(...)):
        exp=store.experiment(str(experiment_id))
        try:profile=store.get_record('profiles',exp['config']['model_profile'])
        except EngineError:raise EngineError('MODEL_NOT_READY','Complete local model preflight before queueing')
        if not profile.get('preflight_passed'):raise EngineError('MODEL_NOT_READY','Complete local model preflight')
        return wrapped(request,**store.queue(str(experiment_id),body or {},exp['estimated_units'],idempotency_key=idempotency_key))
    @app.get('/api/v1/jobs/{job_id}')
    def job(job_id:uuid.UUID,request:Request):store.interrupt_stale();return wrapped(request,**store.job(str(job_id)))
    @app.post('/api/v1/jobs/{job_id}/cancel')
    def cancel(job_id:uuid.UUID,request:Request):
        j=store.job(str(job_id))
        if j['state']=='queued':store.update_job(j['id'],state='cancelled',stage='cancelled_before_start',cancel_requested=1)
        elif j['state']=='running':store.update_job(j['id'],cancel_requested=1)
        return wrapped(request,**store.job(j['id']))
    @app.post('/api/v1/jobs/{job_id}/resume',status_code=202)
    def resume(job_id:uuid.UUID,request:Request):
        j=store.job(str(job_id))
        if not j['recoverable']:raise EngineError('CONFLICT','Only failed/interrupted/cancelled jobs can resume')
        return wrapped(request,**store.queue(j['experiment_id'],j['payload'],j['total_units'],j['kind'],idempotency_key=f'resume:{j["id"]}'))
    @app.get('/api/v1/jobs/{job_id}/events')
    def events(job_id:uuid.UUID,request:Request,after:int=Query(0,ge=0),limit:int=Query(100,ge=1,le=100)):
        store.job(str(job_id))
        with store.connect() as db:rows=db.execute('SELECT * FROM events WHERE job_id=? AND seq>? ORDER BY seq LIMIT ?',(str(job_id),after,limit)).fetchall()
        items=[{**dict(r),'payload':json.loads(r['payload'])} for r in rows]
        return wrapped(request,items=items,after=items[-1]['seq'] if items else after)
    @app.get('/api/v1/experiments/{experiment_id}/runs')
    def runs(experiment_id:uuid.UUID,request:Request,scenario:str|None=None,condition:str|None=None,template:str|None=None,mode:str|None=None,status:str|None=None,limit:int=Query(100,ge=1,le=100),cursor:str|None=None):
        store.experiment(str(experiment_id));items=[json.loads(r['record']) for r in store.unit_rows(str(experiment_id))]
        for key,value in [('scenario_id',scenario),('condition',condition),('template_id',template),('mode',mode),('status',status)]:
            if value:items=[r for r in items if r.get(key)==value]
        return listing(request,items,limit,cursor)
    @app.get('/api/v1/runs/{run_id}')
    def run(run_id:str,request:Request):return wrapped(request,**store.get_record('units',run_id))
    @app.get('/api/v1/runs/{run_id}/activations')
    def activations(run_id:str,request:Request,layer_start:int=Query(0,ge=0),layer_end:int|None=Query(None,ge=1),token_start:int=Query(0,ge=0),token_end:int|None=Query(None,ge=1),raw:bool=False):
        r=store.get_record('units',run_id);artifact=r.get('activation_artifact')
        if not artifact:raise EngineError('UNSUPPORTED_CAPTURE','This unit has no activation capture')
        path,checksum=store.activation_location(r)
        array=read_activation(path,checksum)
        sliced=array[layer_start:layer_end,token_start:token_end,:]
        if raw and sliced.nbytes>1_000_000:raise EngineError('INVALID_CONFIG','Slice exceeds response size; narrow layer/token range to <=1 MB raw float payload')
        result=sliced.tolist() if raw else {'norms':((sliced*sliced).sum(axis=2)**.5).tolist()}
        response=wrapped(request,artifact=artifact,layer_start=layer_start,token_start=token_start,values=result,raw=raw)
        if len(canonical(response))>5_000_000:raise EngineError('INVALID_CONFIG','Response exceeds 5 MB; request a smaller slice')
        return response
    @app.post('/api/v1/experiments/{experiment_id}/analyses',status_code=202)
    def analyses(experiment_id:uuid.UUID,body:AnalysisRequest,request:Request,idempotency_key:str|None=Header(None)):
        store.experiment(str(experiment_id))
        return wrapped(request,**store.queue(str(experiment_id),body.model_dump(),1,'analysis',idempotency_key))
    @app.get('/api/v1/analyses/{analysis_id}')
    def analysis(analysis_id:uuid.UUID,request:Request):return wrapped(request,**store.get_record('analyses',str(analysis_id)))
    @app.post('/api/v1/ratings',status_code=201)
    def ratings(body:RatingRequest,request:Request):
        if set(body.dimension_scores)!={'regret','responsibility','repair','justification_deflection'} or any(v not in (0,1,2) for v in body.dimension_scores.values()):raise EngineError('INVALID_CONFIG','Four rubric dimensions scored 0..2 are required')
        rated_unit=store.get_record('units',body.run_unit_id)
        if rated_unit['status']!='completed' or rated_unit['mode']!='greedy_generation' or rated_unit.get('control'):
            raise EngineError('INVALID_CONFIG','Only completed free-generation responses can be rated')
        expected_display=digest({'experiment':rated_unit['experiment_id'],'rater':body.rater_id,'unit':body.run_unit_id,'version':'blind-v1'})[:20]
        if body.blinded_display_id!=expected_display:raise EngineError('INVALID_CONFIG','Use the blinded display ID issued by rating-tasks for this rater and response')
        with store.connect() as db:
            existing=db.execute("SELECT record FROM ratings WHERE run_unit_id=? AND json_extract(record,'$.rater_id')=? AND json_extract(record,'$.rubric_version')=? ORDER BY rowid LIMIT 1",(body.run_unit_id,body.rater_id,body.rubric_version)).fetchone()
        if existing:
            original=json.loads(existing['record'])
            if all(original.get(k)==v for k,v in body.model_dump().items()):return wrapped(request,**original)
            raise EngineError('CONFLICT','An independent rating is immutable; retain disagreements and record adjudication separately.')
        value={**body.model_dump(),'id':str(uuid.uuid4()),'schema_version':'1.0','created_at':now(),'adjudication':None}
        store.put_record('ratings',value);return wrapped(request,**value)
    @app.post('/api/v1/exports',status_code=202)
    def exports(body:ExportRequest,request:Request,idempotency_key:str|None=Header(None)):
        analysis=store.get_record('analyses',str(body.analysis_id))
        return wrapped(request,**store.queue(analysis['experiment_id'],{'analysis_id':str(body.analysis_id),'include_activations':body.include_activations},1,'export',idempotency_key))
    @app.get('/api/v1/exports/{export_id}')
    def export_metadata(export_id:uuid.UUID,request:Request):return wrapped(request,**store.get_record('exports',str(export_id)))
    @app.post('/api/v1/imports',status_code=202)
    async def import_evidence(request:Request,verify_metrics:bool=True):
        if request.headers.get('content-type','').split(';')[0] not in ('application/zip','application/octet-stream'):
            raise EngineError('INVALID_CONFIG','Upload raw application/zip bytes')
        uploads=store.root/'uploads';uploads.mkdir(exist_ok=True)
        fd,filename=tempfile.mkstemp(suffix='.zip',prefix='evidence-',dir=uploads)
        size=0
        try:
            with os.fdopen(fd,'wb') as output:
                async for chunk in request.stream():
                    size+=len(chunk)
                    if size>MAX_ARCHIVE_BYTES:raise EngineError('PAYLOAD_TOO_LARGE','Evidence upload exceeds 2 GB')
                    output.write(chunk)
            if size==0:raise EngineError('INVALID_BUNDLE','Empty evidence upload')
            return wrapped(request,**store.queue(None,{'relative_path':str(Path(filename).relative_to(store.root)),'verify_metrics':verify_metrics},1,'import'))
        except Exception:
            Path(filename).unlink(missing_ok=True);raise
    @app.get('/api/v1/imports/{import_id}')
    def imported(import_id:uuid.UUID,request:Request):return wrapped(request,**store.get_record('imports',str(import_id)))
    @app.post('/api/v1/experiments/{experiment_id}/replay',status_code=202)
    def replay(experiment_id:uuid.UUID,body:ReplayRequest,request:Request,idempotency_key:str|None=Header(None)):
        store.experiment(str(experiment_id))
        payload={'analysis_id':str(body.analysis_id) if body.analysis_id else None,'verify_metrics':body.verify_metrics}
        return wrapped(request,**store.queue(str(experiment_id),payload,1,'replay',idempotency_key))
    @app.get('/api/v1/experiments/{experiment_id}/ratings')
    def ratings_overview(experiment_id:uuid.UUID,request:Request,limit:int=Query(100,ge=1,le=100),cursor:str|None=None):
        store.experiment(str(experiment_id))
        with store.connect() as db:
            values=[json.loads(r['record']) for r in db.execute('SELECT ratings.record FROM ratings JOIN units ON units.id=ratings.run_unit_id WHERE units.experiment_id=? ORDER BY ratings.id',(str(experiment_id),))]
        return listing(request,values,limit,cursor)
    @app.get('/api/v1/experiments/{experiment_id}/rating-summary')
    def ratings_summary(experiment_id:uuid.UUID,request:Request):
        from ghosttrace.analysis.ratings import rating_summary
        return wrapped(request,**rating_summary(store,str(experiment_id)))
    @app.get('/api/v1/experiments/{experiment_id}/rating-tasks')
    def rating_tasks(experiment_id:uuid.UUID,request:Request,rater_id:str=Query(...,min_length=1,max_length=100),limit:int=Query(36,ge=1,le=100)):
        from ghosttrace.analysis.ratings import blinded_tasks
        return wrapped(request,**blinded_tasks(store,str(experiment_id),rater_id,limit))
    @app.get('/api/v1/exports/{export_id}/download')
    def download(export_id:uuid.UUID):
        e=store.get_record('exports',str(export_id));path=store.root/e['relative_path']
        if not path.exists():raise EngineError('NOT_FOUND','Export file unavailable')
        return FileResponse(path,media_type='application/zip',filename=f'ghosttrace-{export_id}.zip')
    from ghosttrace.research.routes import router_for
    app.include_router(router_for(store))
    frontend=Path('dist')
    if frontend.is_dir():app.mount('/',StaticFiles(directory=frontend,html=True),name='workbench')
    return app

app=create_app()
