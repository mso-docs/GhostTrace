"""Safe compatible linear dictionaries; feature labels never establish concepts.

Supported format is deliberately explicit: float32 linear encoder/decoder with
ReLU or identity activation. Pickled checkpoints and custom model code are never
loaded. A transcoder maps a specified captured input site to its output site.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import shutil
import stat
import tempfile
import zipfile
from ghosttrace.config import EngineError,canonical,digest,file_digest,now
from ghosttrace.research.arrays import read_arrays,write_arrays,MAX_ARRAY_BYTES,MAX_TOTAL_BYTES
from ghosttrace.research.schemas import DictionaryMetadata
from ghosttrace.research.storage import ResearchRepository

HEX=re.compile(r'^[a-f0-9]{64}$')
MAX_PACKAGE_BYTES=210_000_000


def _read_package(path,metadata_name):
    """Return parsed JSON and a safely staged NPZ; caller owns temporary directory."""
    path=Path(path)
    if not path.is_file() or path.stat().st_size>MAX_PACKAGE_BYTES:raise EngineError('PAYLOAD_TOO_LARGE','Dictionary/lens package exceeds 210 MB')
    temp=tempfile.TemporaryDirectory(prefix='ghosttrace-research-artifact-')
    try:
        with zipfile.ZipFile(path) as archive:
            entries=archive.infolist();names=[e.filename for e in entries]
            if len(entries)!=2 or set(names)!={metadata_name,'weights.npz'} or len(set(names))!=2:raise EngineError('INVALID_BUNDLE',f'Expected exactly {metadata_name} and weights.npz')
            if sum(e.file_size for e in entries)>MAX_PACKAGE_BYTES or any(e.flag_bits&1 or stat.S_ISLNK(e.external_attr>>16) or e.is_dir() for e in entries):raise EngineError('INVALID_BUNDLE','Package is oversized, encrypted, or contains links')
            if archive.getinfo(metadata_name).file_size>1_000_000:raise EngineError('INVALID_BUNDLE','Artifact metadata exceeds 1 MB')
            try:
                document=json.loads(archive.read(metadata_name),parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)));canonical(document)
            except (ValueError,TypeError,UnicodeError,RecursionError) as error:raise EngineError('INVALID_BUNDLE','Artifact metadata must be finite UTF-8 JSON') from error
            if not isinstance(document,dict):raise EngineError('INVALID_BUNDLE','Artifact metadata must be an object')
            weights=Path(temp.name)/'weights.npz'
            with archive.open('weights.npz') as source,weights.open('wb') as target:shutil.copyfileobj(source,target,1_048_576)
            checksum=document.get('weights_sha256')
            if not isinstance(checksum,str) or not HEX.fullmatch(checksum) or file_digest(weights)!=checksum:raise EngineError('ARTIFACT_CORRUPT','Declared weights_sha256 does not match weights.npz')
            arrays=read_arrays(weights,checksum)
        return temp,document,arrays,file_digest(path)
    except EngineError:temp.cleanup();raise
    except (OSError,ValueError,TypeError,KeyError,RuntimeError,zipfile.BadZipFile) as error:
        temp.cleanup();raise EngineError('INVALID_BUNDLE','Unreadable bounded research artifact package',{'reason':type(error).__name__}) from error


def validate_dictionary_arrays(metadata,arrays):
    import numpy as np
    if not isinstance(metadata,dict):raise EngineError('INVALID_CONFIG','Dictionary metadata must be an object')
    try:meta=DictionaryMetadata.model_validate(metadata).model_dump(mode='json')
    except ValueError as error:raise EngineError('INVALID_CONFIG','Unsupported dictionary metadata or architecture',{'validation':str(error)}) from error
    expected={'W_enc':(meta['input_width'],meta['feature_count']),'b_enc':(meta['feature_count'],),'W_dec':(meta['feature_count'],meta['output_width']),'b_dec':(meta['output_width'],)}
    if meta['encoder_activation']=='jumprelu':expected['threshold']=(meta['feature_count'],)
    if set(arrays)!=set(expected):raise EngineError('UNSUPPORTED_CAPTURE','Dictionary tensor inventory differs from the explicit encoder/decoder architecture')
    if meta['encoder_activation']=='topk':
        if type(meta.get('top_k'))is not int or not 1<=meta['top_k']<=meta['feature_count']:raise EngineError('INVALID_CONFIG','TopK encoder requires1<=top_k<=feature_count')
    elif meta.get('top_k') is not None:raise EngineError('INVALID_CONFIG','top_k must be absent for non-TopK encoders')
    for key,shape in expected.items():
        value=np.asarray(arrays[key])
        if value.shape!=shape or value.dtype!=np.float32 or not np.isfinite(value).all():raise EngineError('ARTIFACT_CORRUPT',f'{key} must be finite float32 with declared shape',{ 'expected_shape':list(shape)})
    if meta['encoder_activation']=='jumprelu' and (arrays['threshold']<0).any():raise EngineError('INVALID_CONFIG','JumpReLU requires nonnegative threshold values, not unchecked log-thresholds')
    if meta['kind']=='sae' and (meta['input_width']!=meta['output_width'] or meta['input_site']!=meta['output_site']):raise EngineError('INVALID_CONFIG','An SAE reconstructs the same site/width; use kind=transcoder for cross-site mappings')
    if meta['preprocessing']=='center_decoder_bias' and meta['input_width']!=meta['output_width']:raise EngineError('UNSUPPORTED_CAPTURE','Decoder-bias input centering requires equal input/output widths')
    return meta


def import_dictionary(store,archive_path):
    temp,document,arrays,package_hash=_read_package(archive_path,'dictionary.json')
    try:
        raw=document.get('metadata',{k:v for k,v in document.items() if k!='weights_sha256'})
        meta=validate_dictionary_arrays(raw,arrays);repo=ResearchRepository(store)
        for prior in repo.list('dictionary'):
            if prior.get('source_archive_sha256')==package_hash:return prior
        import numpy as np
        coordinate_basis=meta['encoder_activation']=='identity' and meta['input_width']==meta['output_width']==meta['feature_count'] and np.array_equal(arrays['W_enc'],np.eye(meta['input_width'],dtype=np.float32)) and np.array_equal(arrays['W_dec'],arrays['W_enc']) and not np.any(arrays['b_enc']) and not np.any(arrays['b_dec'])
        destination=store.root/'research'/'dictionaries'/package_hash/'weights.npz'
        artifact=write_arrays(destination,arrays)
        artifact['relative_path']=str(destination.relative_to(store.root))
        return repo.put('dictionary',{'metadata':meta,'name':meta['name'],'kind':meta['kind'],'source_archive_sha256':package_hash,'metadata_hash':digest(meta),'artifact':artifact,'weights_sha256':artifact['sha256'],'imported_weights_sha256':document['weights_sha256'],'dictionary_classification':'identity_coordinate_diagnostic' if coordinate_basis else 'compatible_linear_dictionary','learned_features':False if coordinate_basis else None,'compatibility_status':'requires_capture_validation','architecture':{'encoder':'linear_'+meta['encoder_activation'],'decoder':'linear','orientation':'x @ W_enc + b_enc; features @ W_dec + b_dec','preprocessing':meta['preprocessing'],'jump_relu_convention':'z=pre*(pre>threshold); threshold is already exponentiated/nonnegative' if meta['encoder_activation']=='jumprelu' else None,'topk_convention':'Keep k largest positive ReLU candidates; stable feature-index tie break' if meta['encoder_activation']=='topk' else None},'provenance':{'source_archive_sha256':package_hash,'model_repo_id':meta['model_repo_id'],'model_revision':meta['model_revision'],'dictionary_revision':meta['dictionary_revision'],'license':meta['license'],'authorship':'Checksums verify file integrity, not authorship or dictionary quality.'},'limitations':['Feature coordinates are not automatically monosemantic or causal.','Published fidelity claims are external metadata; current-capture reconstruction is measured separately.','Behavioral fidelity requires an intervention in the original model and is not inferred from reconstruction error.']})
    finally:temp.cleanup()


def _component(capture,site,layer):
    matches=[c for c in capture.get('components',[]) if c.get('site')==site and c.get('layer')==layer]
    if len(matches)!=1:raise EngineError('UNSUPPORTED_CAPTURE','Dictionary requires exactly one compatible captured component',{'site':site,'layer':layer,'matches':len(matches)})
    return matches[0]


def validate_feature_pair(store,study_id,capture_id,dictionary_id):
    repo=ResearchRepository(store);study=repo.get(study_id,'study');capture=repo.get(capture_id,'capture');dictionary=repo.get(dictionary_id,'dictionary');meta=dictionary['metadata'];profile=study['model_profile']
    if capture.get('study_id')!=study_id or capture.get('status')!='completed':raise EngineError('CONFLICT','Features require a completed capture from this frozen study')
    if meta['model_repo_id']!=profile.get('repo_id') or meta['model_revision']!=profile.get('revision'):raise EngineError('CONFLICT','Dictionary base model/revision differs from the frozen study')
    if meta['layer']>=profile['layers']:raise EngineError('UNSUPPORTED_CAPTURE','Dictionary layer is outside the resolved model')
    first=_component(capture,meta['input_site'],meta['layer']);second=_component(capture,meta['output_site'],meta['layer'])
    input_state=meta.get('input_normalization_state') or meta['normalization_state'];output_state=meta.get('output_normalization_state') or meta['normalization_state']
    for component,width,expected_state in [(first,meta['input_width'],input_state),(second,meta['output_width'],output_state)]:
        shape=component.get('shape') or capture.get('artifact',{}).get('arrays',{}).get(component.get('array_key'),{}).get('shape',[])
        axes=component.get('axis_names',[])
        if len(shape)!=2 or shape[-1]!=width or axes not in (['token','hidden'],['selected_token','hidden']):raise EngineError('UNSUPPORTED_CAPTURE','Dictionary requires token × hidden captured vectors of the declared width')
        if component.get('normalization_state')!=expected_state:raise EngineError('CONFLICT','Dictionary normalization state differs from capture metadata',{'dictionary':expected_state,'capture':component.get('normalization_state')})
    if first.get('token_indices',capture.get('selected_positions'))!=second.get('token_indices',capture.get('selected_positions')):raise EngineError('TOKEN_ALIGNMENT_FAILED','Transcoder input/output token positions must correspond exactly')
    return {'passed':True,'study_id':study_id,'capture_id':capture_id,'dictionary_id':dictionary_id,'input_component':first,'output_component':second,'model_repo_id':profile['repo_id'],'model_revision':profile['revision'],'layer':meta['layer'],'normalization_state':output_state,'input_normalization_state':input_state,'output_normalization_state':output_state}


def encode_dictionary(inputs,metadata,arrays):
    import numpy as np
    meta=validate_dictionary_arrays(metadata,arrays);x=np.asarray(inputs)
    if x.dtype!=np.float32 or x.ndim!=2 or x.shape[1]!=meta['input_width'] or x.shape[0]>4096 or not np.isfinite(x).all():raise EngineError('INVALID_CONFIG','Dictionary input must be bounded finite float32 token × hidden vectors')
    if x.shape[0]*meta['feature_count']*4>MAX_ARRAY_BYTES or x.shape[0]*meta['output_width']*4>MAX_ARRAY_BYTES:raise EngineError('DISK_QUOTA','Projected feature/reconstruction tensor would exceed100 MB before allocation')
    centered=x-arrays['b_dec'] if meta['preprocessing']=='center_decoder_bias' else x
    features=centered @ arrays['W_enc']+arrays['b_enc']
    if meta['encoder_activation']=='relu':features=np.maximum(features,0)
    elif meta['encoder_activation']=='jumprelu':features=features*(features>arrays['threshold'])
    elif meta['encoder_activation']=='topk':
        positive=np.maximum(features,0);indices=np.argsort(-positive,axis=1,kind='stable')[:,:meta['top_k']];features=np.zeros_like(positive);np.put_along_axis(features,indices,np.take_along_axis(positive,indices,axis=1),axis=1)
    decoded=decode_features(features,meta,arrays)
    if not np.isfinite(features).all() or not np.isfinite(decoded).all():raise EngineError('INVALID_CONFIG','Dictionary projection overflowed; incompatible numeric scale')
    return features.astype(np.float32),decoded.astype(np.float32)



def decode_features(features,metadata,arrays):
    import numpy as np
    meta=validate_dictionary_arrays(metadata,arrays);value=np.asarray(features)
    if value.dtype!=np.float32 or value.ndim!=2 or value.shape[1]!=meta['feature_count'] or not np.isfinite(value).all():raise EngineError('INVALID_CONFIG','Decoded features must be finite float32 token × feature vectors')
    if value.shape[0]*meta['output_width']*4>MAX_ARRAY_BYTES:raise EngineError('DISK_QUOTA','Decoded tensor exceeds100 MB before allocation')
    result=value @ arrays['W_dec']+arrays['b_dec']
    if not np.isfinite(result).all():raise EngineError('INVALID_CONFIG','Decoded dictionary values overflowed')
    return result.astype(np.float32)


def modify_features(features,metadata,arrays,feature_indices,operation='zero',value=0.0):
    import numpy as np
    meta=validate_dictionary_arrays(metadata,arrays);z=np.asarray(features)
    if z.dtype!=np.float32 or z.ndim!=2 or z.shape[1]!=meta['feature_count'] or not np.isfinite(z).all():raise EngineError('INVALID_CONFIG','Feature edits require finite encoded coordinates')
    if not isinstance(feature_indices,list) or not 1<=len(feature_indices)<=64 or len(set(feature_indices))!=len(feature_indices) or any(type(i)is not int or not 0<=i<meta['feature_count'] for i in feature_indices):raise EngineError('INVALID_CONFIG','Feature edit indices must be1–64 distinct available coordinates')
    if operation not in ('zero','set','scale') or type(value)not in (int,float) or not np.isfinite(value) or abs(value)>1000000:raise EngineError('INVALID_CONFIG','Feature edit operation/value is outside explicit bounds')
    modified=z.copy()
    if operation=='zero':modified[:,feature_indices]=0
    elif operation=='set':modified[:,feature_indices]=np.float32(value)
    else:modified[:,feature_indices]*=np.float32(value)
    original=decode_features(z,meta,arrays);changed=decode_features(modified,meta,arrays)
    return {'features':modified,'reconstruction':changed,'decoded_delta':changed-original,'error_policy':'Preserve the original unexplained residual: patch y_original + decoded_delta; reconstruction_only is a separate fidelity intervention.'}

def reconstruction_metrics(targets,reconstruction,features):
    import numpy as np
    y=np.asarray(targets,dtype=np.float64);pred=np.asarray(reconstruction,dtype=np.float64)
    if y.shape!=pred.shape or y.ndim!=2 or not np.isfinite(y).all() or not np.isfinite(pred).all():raise EngineError('INVALID_CONFIG','Reconstruction targets must match finite decoded vectors')
    error=y-pred;norm=float(np.linalg.norm(y));centered=y-y.mean(axis=0,keepdims=True);variance=float((centered*centered).sum());squared=float((error*error).sum())
    return {'mse':float(np.mean(error*error)),'relative_l2':float(np.linalg.norm(error)/norm) if norm>1e-12 else None,'relative_l2_unavailable_reason':None if norm>1e-12 else 'zero_target_norm','explained_variance':1-squared/variance if variance>1e-12 else None,'explained_variance_unavailable_reason':None if variance>1e-12 else 'zero_centered_target_variance','l0_mean':float(np.count_nonzero(features,axis=1).mean()),'error_norm_by_token':np.linalg.norm(error,axis=1).tolist(),'target_norm_by_token':np.linalg.norm(y,axis=1).tolist(),'error_node':{'kind':'reconstruction_error','label':'Unexplained reconstruction residual','norm_by_token':np.linalg.norm(error,axis=1).tolist(),'interpretation':'This residual remains part of the original model computation; it is not silently removed.'}}


def feature_delta(features,arrays,feature_index,new_activation=0.0):
    """A decoded delta preserves the original reconstruction-error residual."""
    import numpy as np
    z=np.asarray(features,dtype=np.float32)
    if z.ndim!=2 or type(feature_index)is not int or not 0<=feature_index<z.shape[1] or not np.isfinite(new_activation):raise EngineError('INVALID_CONFIG','Feature intervention index/value is invalid')
    return ((np.float32(new_activation)-z[:,feature_index])[:,None]*arrays['W_dec'][feature_index][None,:]).astype(np.float32)


def encode_features(store,study_id,capture_id,dictionary_id):
    import numpy as np
    pair=validate_feature_pair(store,study_id,capture_id,dictionary_id);repo=ResearchRepository(store);capture=repo.get(capture_id,'capture');dictionary=repo.get(dictionary_id,'dictionary')
    capture_path,capture_hash=repo.artifact_location(capture);dictionary_path,dictionary_hash=repo.artifact_location(dictionary)
    payload=read_arrays(capture_path,capture_hash);weights=read_arrays(dictionary_path,dictionary_hash)
    x=payload[pair['input_component']['array_key']];y=payload[pair['output_component']['array_key']];features,reconstruction=encode_dictionary(x,dictionary['metadata'],weights)
    identity=digest({'capture_hash':capture['record_hash'],'dictionary_hash':dictionary['record_hash'],'method':'linear-dictionary-projection-v1'})
    for old in repo.list('feature',study_id):
        if old.get('analysis_hash')==identity:return old
    path=store.root/'research'/'features'/identity/'arrays.npz';artifact=write_arrays(path,{'features':features,'reconstruction':reconstruction,'error':(y-reconstruction).astype(np.float32)})
    artifact['relative_path']=str(path.relative_to(store.root))
    positions=pair['input_component'].get('token_indices',capture.get('selected_positions',list(range(len(x)))))
    ranking=np.argsort(-np.max(np.abs(features),axis=0),kind='stable');shown=ranking[:min(128,features.shape[1])]
    rows=[{'feature_index':int(index),'activations':features[:,index].tolist(),'max_activation':float(features[:,index].max()),'mean_activation':float(features[:,index].mean()),'nonzero_token_count':int(np.count_nonzero(features[:,index])),'top_token_indices':[positions[i] for i in np.argsort(-np.abs(features[:,index]),kind='stable')[:min(8,len(x))]],'annotation':None,'test_status':'literal_coordinate_not_learned_feature' if dictionary.get('learned_features') is False else 'unvalidated_feature_coordinate'} for index in shown]
    metrics=reconstruction_metrics(y,reconstruction,features)
    return repo.put('feature',{'study_id':study_id,'capture_id':capture_id,'dictionary_id':dictionary_id,'analysis_hash':identity,'method':'linear-dictionary-projection-v1','measurement_status':'measured_projection','dictionary_classification':dictionary.get('dictionary_classification','compatible_linear_dictionary'),'learned_features':dictionary.get('learned_features'),'compatibility':pair,'features':rows,'selection':'Top 128 coordinates by maximum absolute activation; full feature arrays retained. This is display selection, not a semantic discovery.','omitted_feature_count':int(features.shape[1]-len(rows)),'token_indices':positions,'reconstruction':metrics,'behavioral_fidelity':{'status':'not_measured','value':None,'reason':'Requires replacing this component in the original model and comparing the specified target; reconstruction alone is insufficient.'},'error_node':metrics['error_node'],'artifact':artifact,'provenance':{'capture_record_hash':capture['record_hash'],'dictionary_record_hash':dictionary['record_hash'],'dictionary_weights_sha256':dictionary['artifact']['sha256'],'base_revision':pair['model_revision'],'site':dictionary['metadata']['output_site'],'layer':pair['layer'],'normalization_state':pair['normalization_state']},'limitations':dictionary['limitations']},study_id)
