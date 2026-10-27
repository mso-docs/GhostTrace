"""Versioned affine lenses, never a transcript of hidden thoughts.

Canonical tuned-lens training minimizes a predictive KL objective. This module
also supports explicitly labeled ridge hidden-state alignment, useful for supplied
calibration corpora but not mislabeled as canonical predictive-KL training.
"""
from __future__ import annotations
from pathlib import Path
from typing import Literal
import math
from pydantic import Field,model_validator
from ghosttrace.config import EngineError,canonical,digest
from ghosttrace.research.schemas import Contract
from ghosttrace.research.features import _read_package,_component
from ghosttrace.research.arrays import read_arrays,write_arrays,browser_tile
from ghosttrace.research.storage import ResearchRepository

class LensMetadata(Contract):
    schema_version:Literal['2.0-lens']='2.0-lens'
    name:str=Field(min_length=1,max_length=200)
    model_repo_id:str=Field(min_length=1,max_length=200)
    model_revision:str=Field(pattern=r'^[a-f0-9]{40}$')
    lens_revision:str=Field(min_length=1,max_length=200)
    license:str=Field(min_length=1,max_length=200)
    hidden_size:int=Field(ge=1,le=65536)
    layers:list[int]=Field(min_length=1,max_length=32)
    input_site:Literal['resid_pre','resid_mid','resid_post']='resid_post'
    normalization_state:Literal['raw_component']='raw_component'
    transform:Literal['affine','residual_affine_delta']='affine'
    objective:Literal['predictive_kl','ridge_hidden_alignment','external_unspecified']
    training_dataset_sha256:str=Field(pattern=r'^[a-f0-9]{64}$')
    training_split:Literal['development']='development'
    holdout_used_for_fit:Literal[False]=False
    training_sample_count:int=Field(ge=1,le=100_000_000)
    development_group_ids:list[str]=Field(min_length=1,max_length=10000)
    holdout_group_ids:list[str]=Field(default_factory=list,max_length=10000)
    excludes_persona_fixture:Literal[True]=True
    training_scope:Literal['general_corpus','small_corpus_diagnostic']='general_corpus'
    training_description:str=Field(min_length=20,max_length=8000)
    @model_validator(mode='after')
    def splits(self):
        if any(type(v)is not int or not 0<=v<128 for v in self.layers) or len(set(self.layers))!=len(self.layers):raise ValueError('Lens layers must be distinct zero-based blocks')
        for group in self.development_group_ids+self.holdout_group_ids:
            if not group or len(group)>200:raise ValueError('Group IDs must be bounded nonempty strings')
        if len(set(self.development_group_ids))!=len(self.development_group_ids) or len(set(self.holdout_group_ids))!=len(self.holdout_group_ids):raise ValueError('Group inventories cannot contain duplicates')
        if set(self.development_group_ids)&set(self.holdout_group_ids):raise ValueError('Lens development/holdout groups overlap')
        return self


def validate_lens_arrays(metadata,arrays):
    import numpy as np
    try:meta=LensMetadata.model_validate(metadata).model_dump(mode='json')
    except ValueError as error:raise EngineError('INVALID_CONFIG','Invalid affine lens metadata',{'validation':str(error)}) from error
    n=len(meta['layers']);width=meta['hidden_size']
    if set(arrays)!={'W','b'} or arrays['W'].shape!=(n,width,width) or arrays['b'].shape!=(n,width):raise EngineError('ARTIFACT_CORRUPT','Lens requires W[layer,input,output] and b[layer,output] matching metadata')
    if any(v.dtype!=np.float32 or not np.isfinite(v).all() for v in arrays.values()):raise EngineError('ARTIFACT_CORRUPT','Lens weights must be finite float32')
    return meta


def import_lens(store,archive_path):
    temp,document,arrays,package_hash=_read_package(archive_path,'lens.json')
    try:
        raw=document.get('metadata',{k:v for k,v in document.items() if k!='weights_sha256'});meta=validate_lens_arrays(raw,arrays);repo=ResearchRepository(store)
        for prior in repo.list('lens'):
            if prior.get('source_archive_sha256')==package_hash:return prior
        path=store.root/'research'/'lenses'/package_hash/'weights.npz';artifact=write_arrays(path,arrays);artifact['relative_path']=str(path.relative_to(store.root))
        return repo.put('lens',{'metadata':meta,'name':meta['name'],'source_archive_sha256':package_hash,'artifact':artifact,'training_provenance_status':'externally_declared_not_independently_verified','method':'affine_translator','objective':meta['objective'],'canonical_predictive_tuned_lens':meta['objective']=='predictive_kl','compatibility_status':'requires_capture_validation','limitations':['An affine lens is an output-readout instrument, not model thoughts.','Imported training provenance is declared by the artifact provider; integrity is not authorship.','Ridge hidden alignment and unspecified external objectives must not be described as canonical predictive-KL tuned-lens training.']})
    finally:temp.cleanup()


def transform_hidden(vectors,metadata,arrays,layer):
    import numpy as np
    meta=validate_lens_arrays(metadata,arrays)
    if layer not in meta['layers']:raise EngineError('UNSUPPORTED_CAPTURE','Lens has no translator for this block')
    value=np.asarray(vectors)
    if value.ndim!=2 or value.shape[1]!=meta['hidden_size'] or value.dtype!=np.float32 or not np.isfinite(value).all():raise EngineError('INVALID_CONFIG','Lens input must be finite float32 token × hidden vectors')
    index=meta['layers'].index(layer);result=value @ arrays['W'][index]+arrays['b'][index]
    if meta['transform']=='residual_affine_delta':result=result+value
    if not np.isfinite(result).all():raise EngineError('INVALID_CONFIG','Lens transform overflowed')
    return result.astype(np.float32)


def fit_affine_translator(inputs,targets,development_mask,group_ids,ridge=1.0):
    """Fit on development rows only; validation rows never enter normal equations."""
    import numpy as np
    x=np.asarray(inputs);y=np.asarray(targets);mask=np.asarray(development_mask)
    if x.dtype!=np.float32 or y.dtype!=np.float32 or x.ndim!=2 or x.shape!=y.shape or x.shape[0]>100000 or x.shape[1]>4096 or not np.isfinite(x).all() or not np.isfinite(y).all():raise EngineError('INVALID_CONFIG','Affine calibration requires matching bounded finite float32 matrices')
    if mask.dtype!=np.bool_ or mask.shape!=(len(x),) or len(group_ids)!=len(x) or not mask.any() or mask.all():raise EngineError('INVALID_CONFIG','Explicit development and held-out rows are both required')
    if not all(isinstance(g,str) and 0<len(g)<=200 for g in group_ids):raise EngineError('INVALID_CONFIG','Calibration group IDs must be bounded strings')
    dev={g for g,m in zip(group_ids,mask) if m};held={g for g,m in zip(group_ids,mask) if not m}
    if dev&held:raise EngineError('CONFLICT','Calibration groups overlap development and holdout')
    if type(ridge) not in (int,float) or not math.isfinite(ridge) or ridge<=0:raise EngineError('INVALID_CONFIG','Ridge regularization must be finite and positive')
    xd=x[mask].astype(np.float64);yd=y[mask].astype(np.float64);xm=xd.mean(axis=0);ym=yd.mean(axis=0);xc=xd-xm;yc=yd-ym;width=x.shape[1]
    try:weight=np.eye(width)+np.linalg.solve(xc.T@xc+float(ridge)*np.eye(width),xc.T@(yc-xc))
    except np.linalg.LinAlgError as error:raise EngineError('INVALID_CONFIG','Affine calibration could not solve its regularized system') from error
    bias=ym-xm@weight;weight=weight.astype(np.float32);bias=bias.astype(np.float32)
    if not np.isfinite(weight).all() or not np.isfinite(bias).all():raise EngineError('INVALID_CONFIG','Affine calibration weights are not finite')
    pred=x.astype(np.float64)@weight.astype(np.float64)+bias
    mse=lambda values:float(np.mean(values*values))
    return {'W':weight,'b':bias,'metrics':{'development_mse':mse(pred[mask]-y[mask]),'holdout_mse':mse(pred[~mask]-y[~mask]),'development_identity_mse':mse(x[mask]-y[mask]),'holdout_identity_mse':mse(x[~mask]-y[~mask]),'development_rows':int(mask.sum()),'holdout_rows':int((~mask).sum()),'ridge':float(ridge),'objective':'ridge_hidden_alignment','heldout_used_for_fit':False,'canonical_predictive_kl':False},'development_group_ids':sorted(dev),'holdout_group_ids':sorted(held)}


def fit_lens_from_arrays(store,metadata,inputs_by_layer,targets,development_mask,group_ids,ridge=1.0,decoder=None,optimization=None):
    """User-supplied diverse-corpus workflow, intentionally excluding the30 fixtures."""
    import numpy as np
    try:meta=LensMetadata.model_validate(metadata).model_dump(mode='json')
    except ValueError as error:raise EngineError('INVALID_CONFIG','Invalid lens training metadata',{'validation':str(error)}) from error
    mask=np.asarray(development_mask)
    if mask.dtype!=np.bool_ or mask.shape!=(len(targets),) or len(group_ids)!=len(targets) or not mask.any() or mask.all() or any(not isinstance(g,str) or not 1<=len(g)<=200 for g in group_ids):raise EngineError('INVALID_CONFIG','Calibration split and groups must match supplied rows')
    minimum_rows,minimum_groups=(32,16) if meta['training_scope']=='small_corpus_diagnostic' else (1024,100)
    if len(targets)<minimum_rows or len({g for g,m in zip(group_ids,mask) if m})<minimum_groups:raise EngineError('INVALID_CONFIG','Supplied lens corpus is below its declared training scope minimum; never use the30 persona fixtures as a calibration corpus')
    if meta['objective']=='predictive_kl' and (decoder is None or meta['transform']!='residual_affine_delta'):raise EngineError('UNSUPPORTED_CAPTURE','Predictive-KL training needs a differentiable compatible model decoder and residual_affine_delta transform')
    if meta['objective'] not in ('ridge_hidden_alignment','predictive_kl'):raise EngineError('UNSUPPORTED_CAPTURE','Local lens fitting requires an explicitly implemented objective')
    if meta['objective']=='ridge_hidden_alignment' and meta['transform']!='affine':raise EngineError('INVALID_CONFIG','Ridge hidden alignment stores an affine transform')
    if set(inputs_by_layer)!=set(meta['layers']):raise EngineError('INVALID_CONFIG','Every declared lens layer requires supplied calibration vectors')
    actual_dev=sorted({g for g,m in zip(group_ids,mask) if m});actual_held=sorted({g for g,m in zip(group_ids,mask) if not m})
    if set(actual_dev)&set(actual_held) or sorted(meta['development_group_ids'])!=actual_dev or sorted(meta['holdout_group_ids'])!=actual_held or meta['training_sample_count']!=int(mask.sum()):raise EngineError('CONFLICT','Lens training provenance does not match actual supplied split/groups')
    if any(np.asarray(value).ndim!=2 or np.asarray(value).shape!=(len(targets),meta['hidden_size']) for value in [targets,*inputs_by_layer.values()]):raise EngineError('INVALID_CONFIG','Calibration vectors must match the declared hidden width and row count')
    fitted=[fit_affine_kl(inputs_by_layer[layer],targets,mask,group_ids,decoder,**(optimization or {})) if meta['objective']=='predictive_kl' else fit_affine_translator(inputs_by_layer[layer],targets,mask,group_ids,ridge) for layer in meta['layers']]
    actual_dev=fitted[0]['development_group_ids'];actual_held=fitted[0]['holdout_group_ids']
    if sorted(meta['development_group_ids'])!=actual_dev or sorted(meta['holdout_group_ids'])!=actual_held or meta['training_sample_count']!=int(mask.sum()):raise EngineError('CONFLICT','Lens training provenance does not match actual supplied split/groups')
    arrays={'W':np.stack([f['W'] for f in fitted]),'b':np.stack([f['b'] for f in fitted])};validate_lens_arrays(meta,arrays)
    training_hash=digest({'inputs':{str(k):__import__('hashlib').sha256(v.tobytes()).hexdigest() for k,v in inputs_by_layer.items()},'targets':__import__('hashlib').sha256(np.asarray(targets).tobytes()).hexdigest(),'development_mask':mask.tolist(),'group_ids':list(group_ids),'ridge':float(ridge),'objective':meta['objective'],'optimization':optimization or {}})
    path=store.root/'research'/'lenses'/training_hash/'weights.npz';artifact=write_arrays(path,arrays);artifact['relative_path']=str(path.relative_to(store.root))
    return ResearchRepository(store).put('lens',{'metadata':meta,'name':meta['name'],'method':'affine_translator','objective':meta['objective'],'canonical_predictive_tuned_lens':meta['objective']=='predictive_kl','training_provenance_status':'locally_fitted_on_supplied_development_vectors','training_hash':training_hash,'metrics':{str(layer):f['metrics'] for layer,f in zip(meta['layers'],fitted)},'artifact':artifact,'limitations':['Predictive-KL and ridge hidden-alignment objectives are distinct and recorded explicitly; neither readout is a transcript of thoughts.','A sufficiently diverse supplied corpus and held-out groups are required; statistical quality is not guaranteed by the minimum sample count.','Output predictions require the compatible model final normalization and unembedding.']})


def apply_lens(store,study_id,capture_id,lens_id,decoder=None,layers=None):
    import numpy as np
    repo=ResearchRepository(store);study=repo.get(study_id,'study');capture=repo.get(capture_id,'capture');lens=repo.get(lens_id,'lens');meta=lens['metadata'];profile=study['model_profile']
    if capture.get('study_id')!=study_id or capture.get('status')!='completed':raise EngineError('CONFLICT','Lens requires a completed capture from this study')
    if meta['model_repo_id']!=profile['repo_id'] or meta['model_revision']!=profile['revision'] or meta['hidden_size']!=profile['hidden_size'] or any(layer>=profile['layers'] for layer in meta['layers']):raise EngineError('CONFLICT','Lens base model/revision/width differs from the frozen study')
    capture_path,capture_hash=repo.artifact_location(capture);lens_path,lens_hash=repo.artifact_location(lens);vectors=read_arrays(capture_path,capture_hash);weights=read_arrays(lens_path,lens_hash)
    translated={};readouts=[];tiles=[]
    available={c.get('layer') for c in capture.get('components',[]) if c.get('site')==meta['input_site']}
    if layers is not None and (not isinstance(layers,list) or not layers or len(set(layers))!=len(layers) or any(type(layer)is not int or layer not in meta['layers'] for layer in layers)):raise EngineError('UNSUPPORTED_CAPTURE','Requested lens layers must be distinct layers present in the artifact')
    if layers is not None and not set(layers)<=available:raise EngineError('UNSUPPORTED_CAPTURE','Requested lens layer/site was not captured')
    chosen=[layer for layer in (meta['layers'] if layers is None else layers) if layer in available]
    if not chosen:raise EngineError('UNSUPPORTED_CAPTURE','Capture has no compatible layer/site translator inputs')
    for layer in chosen:
        component=_component(capture,meta['input_site'],layer)
        if component.get('normalization_state')!=meta['normalization_state']:raise EngineError('CONFLICT','Lens normalization state is incompatible')
        result=transform_hidden(vectors[component['array_key']],meta,weights,layer);key=f'L{layer}.translated';translated[key]=result
        tiles.append({'layer':layer,'site':meta['input_site'],**browser_tile(result,{'axis_names':['token','hidden']})})
        if decoder is not None:readouts.append({'layer':layer,'readout':decoder(result,layer)})
    identity=digest({'capture':capture['record_hash'],'lens':lens['record_hash'],'layers':chosen,'decoded':decoder is not None,'method':'affine-readout-v1'})
    path=store.root/'research'/'lens-readouts'/identity/'arrays.npz';artifact=write_arrays(path,translated);artifact['relative_path']=str(path.relative_to(store.root))
    return repo.put('lens_result',{'study_id':study_id,'capture_id':capture_id,'lens_id':lens_id,'method':'affine_translator','objective':meta['objective'],'canonical_predictive_tuned_lens':lens.get('canonical_predictive_tuned_lens',False),'tiles':tiles,'applied_layers':chosen,'unavailable_lens_layers':[layer for layer in meta['layers'] if layer not in chosen],'readouts':readouts,'prediction_status':'measured_model_readout' if decoder is not None else 'not_available','prediction_unavailable_reason':None if decoder is not None else 'Compatible model decoder was not loaded; translated vectors are not token predictions.','artifact':artifact,'training_split':'development','heldout_used_for_fit':False,'provenance':{'lens_record_hash':lens['record_hash'],'capture_record_hash':capture['record_hash'],'base_revision':profile['revision'],'weights_sha256':lens_hash},'limitations':lens['limitations']},study_id)


def fit_affine_kl(inputs,teacher_final_vectors,development_mask,group_ids,decoder,*,steps=64,batch_size=16,learning_rate=.01,seed=73,max_eval_rows=256):
    """Train a residual affine translator against detached final-output distributions.

    ``decoder`` is the compatible frozen model's final normalization/unembedding.
    autograd.grad requests ONLY translator gradients; model parameter .grad buffers
    are never populated, zeroed or otherwise mutated by this function.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    x=np.asarray(inputs);y=np.asarray(teacher_final_vectors);mask=np.asarray(development_mask)
    if x.dtype!=np.float32 or y.dtype!=np.float32 or x.ndim!=2 or x.shape!=y.shape or x.shape[0]>100000 or x.shape[1]>4096 or not np.isfinite(x).all() or not np.isfinite(y).all():raise EngineError('INVALID_CONFIG','Predictive lens calibration requires bounded finite float32 input/final-vector matrices')
    if mask.dtype!=np.bool_ or mask.shape!=(len(x),) or len(group_ids)!=len(x) or not mask.any() or mask.all():raise EngineError('INVALID_CONFIG','Separate development/holdout rows are required')
    if not all(isinstance(g,str) and 0<len(g)<=200 for g in group_ids):raise EngineError('INVALID_CONFIG','Calibration group IDs are invalid')
    dev={g for g,m in zip(group_ids,mask) if m};held={g for g,m in zip(group_ids,mask) if not m}
    if dev&held:raise EngineError('CONFLICT','Predictive lens calibration groups overlap development and holdout')
    if type(steps)is not int or not 1<=steps<=1000 or type(batch_size)is not int or not 1<=batch_size<=64 or type(seed)is not int or not 0<=seed<2**32 or type(max_eval_rows)is not int or not 1<=max_eval_rows<=1024 or type(learning_rate) not in (int,float) or not math.isfinite(learning_rate) or not 0<learning_rate<=1:raise EngineError('INVALID_CONFIG','Lens optimization settings exceed explicit bounds')
    width=x.shape[1];weight=torch.zeros((width,width),dtype=torch.float32,requires_grad=True);bias=torch.zeros(width,dtype=torch.float32,requires_grad=True)
    tx=torch.from_numpy(x.copy());ty=torch.from_numpy(y.copy());development=np.flatnonzero(mask);holdout=np.flatnonzero(~mask);rng=np.random.default_rng(seed)
    def divergence(indices,trained):
        source=tx[indices];translated=source+source@weight+bias if trained else source
        with torch.no_grad():teacher=decoder(ty[indices]).detach().float().log_softmax(-1)
        logits=decoder(translated).float()
        if logits.ndim!=2 or logits.shape!=teacher.shape or logits.shape[0]!=len(indices) or logits.shape[1]>1_000_000 or not torch.isfinite(logits).all() or not torch.isfinite(teacher).all():raise EngineError('INVALID_CONFIG','Compatible decoder must return bounded finite batch × vocabulary logits')
        return F.kl_div(logits.log_softmax(-1),teacher,log_target=True,reduction='batchmean')
    losses=[]
    try:
        for _ in range(steps):
            indices=rng.choice(development,size=min(batch_size,len(development)),replace=False).tolist();loss=divergence(indices,True)
            gradients=torch.autograd.grad(loss,(weight,bias),allow_unused=False)
            if not torch.isfinite(loss) or any(not torch.isfinite(g).all() for g in gradients):raise EngineError('INVALID_CONFIG','Predictive lens optimization produced nonfinite gradients')
            with torch.no_grad():
                for parameter,gradient in zip((weight,bias),gradients):parameter.add_(gradient,alpha=-float(learning_rate))
            losses.append(float(loss.detach()))
        def evaluate(indices,trained):
            selected=indices[:max_eval_rows];total=0.
            with torch.no_grad():
                for start in range(0,len(selected),batch_size):
                    part=selected[start:start+batch_size].tolist();total+=float(divergence(part,trained))*len(part)
            return total/len(selected)
        metrics={'objective':'predictive_kl','canonical_predictive_kl':True,'heldout_used_for_fit':False,'development_kl':evaluate(development,True),'holdout_kl':evaluate(holdout,True),'development_identity_kl':evaluate(development,False),'holdout_identity_kl':evaluate(holdout,False),'development_rows':len(development),'holdout_rows':len(holdout),'development_evaluation_rows':min(len(development),max_eval_rows),'holdout_evaluation_rows':min(len(holdout),max_eval_rows),'evaluation_selection':'First bounded rows in each declared split, independent of observed results.','steps':steps,'batch_size':batch_size,'learning_rate':float(learning_rate),'seed':seed,'optimizer':'SGD without momentum','training_device':'cpu translator; decoder retains its explicit model backend','training_losses':losses,'model_parameters_updated':False}
        return {'W':weight.detach().cpu().numpy().copy(),'b':bias.detach().cpu().numpy().copy(),'transform':'residual_affine_delta','metrics':metrics,'development_group_ids':sorted(dev),'holdout_group_ids':sorted(held)}
    except EngineError:raise
    except RuntimeError as error:raise EngineError('UNSUPPORTED_CAPTURE','Predictive lens decoder must support differentiable normalization/unembedding without inference-mode tensors',{'reason':str(error)}) from error
