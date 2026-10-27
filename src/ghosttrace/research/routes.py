from __future__ import annotations
from pathlib import Path
import os
import tempfile
import uuid
from fastapi import APIRouter,Request,Header,Query
from fastapi.responses import FileResponse
from ghosttrace.config import EngineError
from ghosttrace.research.schemas import StudyCreate,CaptureRequest,InterventionRequest,HypothesisCreate,ResearchExportRequest,FeatureRequest,FeatureAnnotation,LensApplyRequest,CircuitImportRequest,FeatureInterventionRequest,FeatureAttributionRequest
from ghosttrace.research.storage import ResearchRepository
from ghosttrace.research.service import freeze_study,get_bundle,queue_capture,queue_intervention,save_hypothesis
from ghosttrace.research.arrays import read_arrays,tile_matrix


def router_for(store):
    router=APIRouter(prefix='/api/v2/research',tags=['Research v2'])
    def wrap(request,**value):return {**value,'schema_version':'2.0','request_id':request.state.request_id}
    def page(request,values,limit,offset):return wrap(request,items=values[offset:offset+limit],next_offset=offset+limit if len(values)>offset+limit else None,total=len(values))
    @router.get('/capabilities')
    def capabilities(request:Request):
        return wrap(request,execution='local_worker_only',v1_protocol_unchanged=True,sites=['resid_pre','attn_in','attn_out','resid_mid','mlp_in','mlp_out','resid_post','final_norm','head_pre','head_contribution','mlp_post_gate','attention_pattern'],patch_sites=['resid_pre','attn_in','attn_out','resid_mid','mlp_in','mlp_out','resid_post','final_norm','head_pre','mlp_post_gate'],operations=['patch','zero_ablate','mean_ablate','norm_matched_random'],evidence_types=['structural','activation','attribution','intervention','tested_path','human_interpretation'],feature_dictionaries={'formats':['2.0-dictionary NPZ/JSON ZIP'],'model_compatibility':'Exact model revision, site, layer, dimensions and preprocessing required'},limits={'conditions':12,'positions':32,'layers':32,'input_tokens':512,'research_bundle_bytes':500_000_000,'array_artifact_bytes':200_000_000},limitations=['Capabilities require validated local weights; no endpoint downloads models.','A supported architectural site does not establish a discovered circuit.'])
    @router.post('/studies',status_code=201)
    def studies(body:StudyCreate,request:Request):return wrap(request,**freeze_study(store,body.model_dump(mode='json')))
    @router.get('/studies')
    def list_studies(request:Request,limit:int=Query(50,ge=1,le=100),offset:int=Query(0,ge=0,le=10000)):return page(request,ResearchRepository(store).list('study'),limit,offset)
    @router.get('/studies/{study_id}')
    def study(study_id:uuid.UUID,request:Request):return wrap(request,**ResearchRepository(store).get(str(study_id),'study'))
    @router.get('/studies/{study_id}/bundle')
    def bundle(study_id:uuid.UUID,request:Request):return wrap(request,**get_bundle(store,str(study_id)))
    @router.post('/studies/{study_id}/captures',status_code=202)
    def capture(study_id:uuid.UUID,body:CaptureRequest,request:Request,idempotency_key:str|None=Header(None)):return wrap(request,**queue_capture(store,str(study_id),body.condition_ids,idempotency_key))
    @router.get('/captures/{capture_id}')
    def get_capture(capture_id:uuid.UUID,request:Request):return wrap(request,**ResearchRepository(store).get(str(capture_id),'capture'))
    @router.get('/captures/{capture_id}/tiles')
    def tiles(capture_id:uuid.UUID,request:Request,array:str=Query(...,min_length=1,max_length=120),row_start:int=Query(0,ge=0),row_count:int=Query(16,ge=1,le=32),channel_start:int=Query(0,ge=0),channel_count:int=Query(64,ge=1,le=128)):
        record=ResearchRepository(store).get(str(capture_id),'capture');artifact=record.get('artifact')
        if not artifact:raise EngineError('UNSUPPORTED_CAPTURE','Raw research arrays are unavailable in this record or summary bundle')
        path,checksum=ResearchRepository(store).artifact_location(record)
        arrays=read_arrays(path,checksum)
        if array not in arrays:raise EngineError('NOT_FOUND','Unknown array key')
        value=arrays[array];component=next((c for c in record.get('components',[]) if c['array_key']==array),{})
        matrix,row_axis,trailing_axes,trailing_shape=tile_matrix(value,component)
        if row_start>=matrix.shape[0] or channel_start>=matrix.shape[1]:raise EngineError('INVALID_CONFIG','Requested tile starts outside the actual tensor')
        selected=matrix[row_start:row_start+row_count,channel_start:channel_start+channel_count]
        return wrap(request,array_key=array,shape=list(value.shape),matrix_shape=list(matrix.shape),row_indices=list(range(row_start,row_start+selected.shape[0])),channel_indices=list(range(channel_start,channel_start+selected.shape[1])),values=selected.tolist(),row_axis=row_axis,flattened_trailing_axes=trailing_axes,trailing_axis_shape=trailing_shape,dtype='float32',artifact_hash=artifact['sha256'],omitted_value_count=int(value.size-selected.size),interpretation='Literal selected coordinates; trailing axes are flattened in their recorded order.')
    @router.post('/studies/{study_id}/interventions',status_code=202)
    def intervention(study_id:uuid.UUID,body:InterventionRequest,request:Request,idempotency_key:str|None=Header(None)):return wrap(request,**queue_intervention(store,str(study_id),body.model_dump(mode='json'),idempotency_key))
    @router.get('/interventions/{intervention_id}')
    def get_intervention(intervention_id:uuid.UUID,request:Request):return wrap(request,**ResearchRepository(store).get(str(intervention_id),'intervention'))
    @router.post('/studies/{study_id}/hypotheses',status_code=201)
    def hypothesis(study_id:uuid.UUID,body:HypothesisCreate,request:Request):return wrap(request,**save_hypothesis(store,str(study_id),body.model_dump(mode='json')))
    @router.post('/studies/{study_id}/exports',status_code=202)
    def export(study_id:uuid.UUID,body:ResearchExportRequest,request:Request,idempotency_key:str|None=Header(None)):
        ResearchRepository(store).get(str(study_id),'study')
        return wrap(request,**store.queue(None,{'study_id':str(study_id),'include_arrays':body.include_arrays},1,'research_export',f'{study_id}:{idempotency_key}' if idempotency_key else None))
    @router.get('/exports/{export_id}')
    def export_metadata(export_id:uuid.UUID,request:Request):return wrap(request,**ResearchRepository(store).get(str(export_id),'export'))
    @router.get('/exports/{export_id}/download')
    def export_download(export_id:uuid.UUID):
        record=ResearchRepository(store).get(str(export_id),'export');path=(store.root/record['relative_path']).resolve()
        if not path.is_relative_to(store.root) or not path.is_file():raise EngineError('NOT_FOUND','Research export unavailable')
        return FileResponse(path,media_type='application/zip',filename=f'ghosttrace-research-{export_id}.zip')
    async def upload(request,kind):
        if request.headers.get('content-type','').split(';')[0] not in ('application/zip','application/octet-stream'):raise EngineError('INVALID_CONFIG','Upload raw application/zip bytes')
        root=store.root/'research'/'uploads';root.mkdir(parents=True,exist_ok=True);fd,name=tempfile.mkstemp(prefix='upload-',suffix='.zip',dir=root);size=0
        try:
            with os.fdopen(fd,'wb') as output:
                async for chunk in request.stream():
                    size+=len(chunk)
                    if size>500_000_000:raise EngineError('PAYLOAD_TOO_LARGE','Research archive exceeds 500 MB')
                    output.write(chunk)
            if size==0:raise EngineError('INVALID_BUNDLE','Empty research archive')
            return store.queue(None,{'relative_path':str(Path(name).relative_to(store.root))},1,kind)
        except BaseException:Path(name).unlink(missing_ok=True);raise
    @router.post('/imports',status_code=202)
    async def imported(request:Request):return wrap(request,**await upload(request,'research_import'))
    @router.post('/dictionaries/imports',status_code=202)
    async def dictionary_import(request:Request):return wrap(request,**await upload(request,'research_dictionary_import'))
    @router.post('/lenses/imports',status_code=202)
    async def lens_import(request:Request):return wrap(request,**await upload(request,'research_lens_import'))
    @router.post('/lenses/train',status_code=202)
    async def lens_train(request:Request):return wrap(request,**await upload(request,'research_lens_train'))
    @router.get('/lenses')
    def lenses(request:Request,limit:int=Query(50,ge=1,le=100),offset:int=Query(0,ge=0,le=10000)):return page(request,ResearchRepository(store).list('lens'),limit,offset)
    @router.get('/lenses/{lens_id}')
    def lens(lens_id:uuid.UUID,request:Request):return wrap(request,**ResearchRepository(store).get(str(lens_id),'lens'))
    @router.post('/studies/{study_id}/lenses/apply',status_code=202)
    def lens_apply(study_id:uuid.UUID,body:LensApplyRequest,request:Request,idempotency_key:str|None=Header(None)):
        repo=ResearchRepository(store);repo.get(str(study_id),'study');capture=repo.get(body.capture_id,'capture');repo.get(body.lens_id,'lens')
        if capture['study_id']!=str(study_id):raise EngineError('CONFLICT','Lens capture belongs to another study')
        return wrap(request,**store.queue(None,{'study_id':str(study_id),**body.model_dump(mode='json')},1,'research_lens_apply',f'{study_id}:{idempotency_key}' if idempotency_key else None))
    @router.post('/studies/{study_id}/circuits',status_code=201)
    def circuit_import(study_id:uuid.UUID,body:CircuitImportRequest,request:Request):
        from ghosttrace.research.circuits import import_circuit
        return wrap(request,**import_circuit(store,str(study_id),body.document,body.source_metadata))
    @router.get('/studies/{study_id}/circuits')
    def circuits(study_id:uuid.UUID,request:Request):
        repo=ResearchRepository(store);repo.get(str(study_id),'study');return wrap(request,items=repo.list('circuit',str(study_id)))
    @router.post('/studies/{study_id}/feature-interventions',status_code=202)
    def feature_intervention(study_id:uuid.UUID,body:FeatureInterventionRequest,request:Request,idempotency_key:str|None=Header(None)):
        from ghosttrace.research.features import validate_feature_pair
        validate_feature_pair(store,str(study_id),body.capture_id,body.dictionary_id)
        return wrap(request,**store.queue(None,{'study_id':str(study_id),**body.model_dump(mode='json')},1,'research_feature_intervention',f'{study_id}:{idempotency_key}' if idempotency_key else None))
    @router.post('/studies/{study_id}/feature-attributions',status_code=202)
    def feature_attribution(study_id:uuid.UUID,body:FeatureAttributionRequest,request:Request,idempotency_key:str|None=Header(None)):
        from ghosttrace.research.features import validate_feature_pair
        validate_feature_pair(store,str(study_id),body.capture_id,body.dictionary_id)
        return wrap(request,**store.queue(None,{'study_id':str(study_id),**body.model_dump(mode='json')},1,'research_feature_attribution',f'{study_id}:{idempotency_key}' if idempotency_key else None))
    @router.get('/dictionaries')
    def dictionaries(request:Request,limit:int=Query(50,ge=1,le=100),offset:int=Query(0,ge=0,le=10000)):return page(request,ResearchRepository(store).list('dictionary'),limit,offset)
    @router.get('/dictionaries/{dictionary_id}')
    def dictionary(dictionary_id:uuid.UUID,request:Request):return wrap(request,**ResearchRepository(store).get(str(dictionary_id),'dictionary'))
    @router.post('/studies/{study_id}/features',status_code=202)
    def features(study_id:uuid.UUID,body:FeatureRequest,request:Request,idempotency_key:str|None=Header(None)):
        from ghosttrace.research.features import validate_feature_pair
        validate_feature_pair(store,str(study_id),body.capture_id,body.dictionary_id)
        return wrap(request,**store.queue(None,{'study_id':str(study_id),**body.model_dump(mode='json')},1,'research_features',f'{study_id}:{idempotency_key}' if idempotency_key else None))
    @router.post('/studies/{study_id}/dictionaries/{dictionary_id}/annotations',status_code=201)
    def feature_annotation(study_id:uuid.UUID,dictionary_id:uuid.UUID,body:FeatureAnnotation,request:Request):
        repo=ResearchRepository(store);repo.get(str(study_id),'study');dictionary=repo.get(str(dictionary_id),'dictionary')
        if body.feature_index>=dictionary['metadata']['feature_count']:raise EngineError('INVALID_CONFIG','Feature index outside dictionary')
        for record_id in body.evidence_capture_ids:
            if repo.get(record_id,'capture')['study_id']!=str(study_id):raise EngineError('CONFLICT','Feature annotation evidence belongs to another study')
        return wrap(request,**repo.put('annotation',{'study_id':str(study_id),'dictionary_id':str(dictionary_id),**body.model_dump(mode='json'),'evidence_type':'human_interpretation','limitation':'A feature label is an annotation, not ground truth.'},str(study_id)))
    return router
