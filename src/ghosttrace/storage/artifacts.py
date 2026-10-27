import json
import os
import tempfile
from pathlib import Path
from ghosttrace.config import EngineError, canonical, file_digest

def atomic_bytes(path,contents):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.pending-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            f.write(contents); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    return file_digest(path)

def atomic_json(path,value):return atomic_bytes(path,canonical(value))

def write_activation(path,array,quota,root):
    import numpy as np
    if array.ndim!=3 or array.dtype!=np.float32 or not np.isfinite(array).all():raise EngineError('INVALID_CONFIG','Activation arrays must be finite float32 [layer,token,hidden]')
    used=sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file())
    if used+array.nbytes>quota:raise EngineError('DISK_QUOTA','Artifact quota exceeded',{'used':used,'next_unit':array.nbytes,'quota':quota})
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.pending-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            np.savez_compressed(f,activations=array);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    return file_digest(path)

def read_activation(path,checksum=None):
    import numpy as np
    import zipfile
    import math
    path=Path(path)
    if not path.exists() or checksum and file_digest(path)!=checksum:raise EngineError('ARTIFACT_CORRUPT','Activation checksum mismatch or file missing')
    try:
        with zipfile.ZipFile(path) as z:
            entries=z.infolist()
            if len(entries)!=1 or entries[0].filename!='activations.npy' or entries[0].file_size>100_000_000:
                raise EngineError('ARTIFACT_CORRUPT','Expected one bounded activations.npy member')
            with z.open(entries[0]) as member:
                version=np.lib.format.read_magic(member)
                if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(member,max_header_size=4096)
                elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(member,max_header_size=4096)
                else:raise EngineError('ARTIFACT_CORRUPT','Unsupported NumPy array header version')
                count=math.prod(shape)
                if len(shape)!=3 or any(type(v) is not int or v<=0 for v in shape) or count>25_000_000 or dtype!=np.dtype('float32') or dtype.hasobject:
                    raise EngineError('ARTIFACT_CORRUPT','Invalid activation header dimensions or dtype')
                if member.tell()+count*dtype.itemsize!=entries[0].file_size:
                    raise EngineError('ARTIFACT_CORRUPT','Activation header/payload byte size mismatch')
        with np.load(path,allow_pickle=False,max_header_size=4096) as data:
            array=data['activations']
            if not np.isfinite(array).all():raise EngineError('ARTIFACT_CORRUPT','Nonfinite activation values')
            return array.copy()
    except EngineError:raise
    except (ValueError,KeyError,OSError,EOFError,zipfile.BadZipFile) as error:
        raise EngineError('ARTIFACT_CORRUPT','Unreadable activation payload',{'reason':str(error)}) from error
