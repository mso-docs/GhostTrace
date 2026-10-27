"""Bounded numerical artifacts. NPY headers are checked before allocation."""
from __future__ import annotations
import math
import os
from pathlib import Path
import re
import tempfile
import zipfile
from ghosttrace.config import EngineError,file_digest

NAME=re.compile(r'^[A-Za-z0-9_.-]{1,120}$')
MAX_ARRAY_BYTES=100_000_000
MAX_TOTAL_BYTES=200_000_000

def write_arrays(path,arrays,quota=MAX_TOTAL_BYTES,study_root=None):
    import numpy as np
    if not isinstance(arrays,dict) or len(arrays)>400:raise EngineError('INVALID_CONFIG','Research array inventory is not bounded')
    clean={};total=0
    for key,array in arrays.items():
        if not NAME.fullmatch(key):raise EngineError('INVALID_CONFIG','Unsafe research array name')
        value=np.asarray(array)
        if value.dtype!=np.float32 or not 1<=value.ndim<=4 or any(n<1 for n in value.shape) or value.nbytes>MAX_ARRAY_BYTES or not np.isfinite(value).all():raise EngineError('INVALID_CONFIG','Research arrays must be bounded finite float32 tensors with 1–4 axes')
        total+=value.nbytes;clean[key]=value
    if total>MAX_TOTAL_BYTES:raise EngineError('DISK_QUOTA','Research arrays exceed 200 MB per artifact')
    if study_root:
        used=sum(p.stat().st_size for p in Path(study_root).rglob('*') if p.is_file())
        if used+total>quota:raise EngineError('DISK_QUOTA','Research study artifact quota exceeded',{'used_bytes':used,'new_uncompressed_bytes':total,'quota_bytes':quota})
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fd,temp=tempfile.mkstemp(dir=path.parent,prefix='.pending-')
    try:
        with os.fdopen(fd,'wb') as output:
            np.savez_compressed(output,**clean);output.flush();os.fsync(output.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp):os.unlink(temp)
    return {'relative_path':str(path),'sha256':file_digest(path),'uncompressed_bytes':total,'arrays':{key:{'shape':list(value.shape),'dtype':'float32','bytes':value.nbytes} for key,value in clean.items()}}

def read_arrays(path,sha256=None):
    import numpy as np
    path=Path(path)
    if not path.is_file() or sha256 and file_digest(path)!=sha256:raise EngineError('ARTIFACT_CORRUPT','Research artifact missing or checksum mismatch')
    try:
        with zipfile.ZipFile(path) as archive:
            infos=archive.infolist();names=[i.filename for i in infos]
            if len(infos)>400 or len(set(names))!=len(names) or sum(i.file_size for i in infos)>MAX_TOTAL_BYTES+400*8192:raise EngineError('ARTIFACT_CORRUPT','Research artifact exceeds file or expanded-size limits')
            for info in infos:
                if not info.filename.endswith('.npy') or not NAME.fullmatch(info.filename[:-4]) or info.file_size>MAX_ARRAY_BYTES+8192:raise EngineError('ARTIFACT_CORRUPT','Invalid research tensor entry')
                with archive.open(info) as member:
                    version=np.lib.format.read_magic(member)
                    if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(member,max_header_size=4096)
                    elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(member,max_header_size=4096)
                    else:raise EngineError('ARTIFACT_CORRUPT','Unsupported research tensor version')
                    count=math.prod(shape)
                    if not 1<=len(shape)<=4 or any(type(n)is not int or n<1 for n in shape) or dtype!=np.dtype('float32') or count*4>MAX_ARRAY_BYTES or member.tell()+count*4!=info.file_size:raise EngineError('ARTIFACT_CORRUPT','Invalid research tensor header/payload')
        with np.load(path,allow_pickle=False,max_header_size=4096) as payload:
            values={key:payload[key].copy() for key in payload.files}
        if any(not np.isfinite(v).all() for v in values.values()):raise EngineError('ARTIFACT_CORRUPT','Research artifact contains nonfinite values')
        return values
    except EngineError:raise
    except (OSError,ValueError,EOFError,KeyError,zipfile.BadZipFile) as error:raise EngineError('ARTIFACT_CORRUPT','Unreadable research artifact',{'reason':type(error).__name__}) from error

def tile_matrix(array,component):
    import numpy as np
    value=np.asarray(array);axes=component.get('axis_names',component.get('axes',[]))
    if len(axes)!=value.ndim:axes=[f'axis_{i}' for i in range(value.ndim)]
    row_axis=axes.index('token') if 'token' in axes else (axes.index('query_token') if 'query_token' in axes else 0)
    moved=np.moveaxis(value,row_axis,0)
    matrix=moved.reshape(moved.shape[0],-1)
    trailing_axes=[axis for i,axis in enumerate(axes) if i!=row_axis]
    return matrix,axes[row_axis],trailing_axes,list(moved.shape[1:])

def browser_tile(array,component,max_values=4096):
    """Deterministic literal-coordinate tiles with every flattened axis disclosed."""
    import numpy as np
    value=np.asarray(array);matrix,row_axis,trailing_axes,trailing_shape=tile_matrix(value,component)
    rows=min(matrix.shape[0],32);columns=min(matrix.shape[1],max(1,max_values//max(1,rows)),128)
    tile=matrix[:rows,:columns]
    coordinates=[list(np.unravel_index(i,tuple(trailing_shape))) if trailing_shape else [] for i in range(columns)]
    labels=[' / '.join(f'{axis}:{int(index)}' for axis,index in zip(trailing_axes,indices)) or 'value' for indices in coordinates]
    return {'values':tile.tolist(),'tile_shape':list(tile.shape),'shape':list(value.shape),'axes':component.get('axis_names',component.get('axes',[])),'row_axis':row_axis,'channel_indices':list(range(columns)),'column_indices':[[int(x) for x in indices] for indices in coordinates],'column_labels':labels,'selected_row_indices':list(range(rows)),'flattened_trailing_axes':trailing_axes,'trailing_axis_shape':trailing_shape,'omitted_value_count':int(value.size-tile.size),'selection':'First selected token/query rows and first literal coordinates. Flattened axes and original coordinates are explicit; omitted values are unavailable, not zero.','summary':{'minimum':float(value.min()),'maximum':float(value.max()),'mean':float(value.mean(dtype=np.float64)),'std':float(value.std(dtype=np.float64)),'l2_norm':float(np.linalg.norm(value.astype(np.float64)))}}
