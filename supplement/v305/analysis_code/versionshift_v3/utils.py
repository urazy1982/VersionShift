"""VersionShift v3 utility primitives (all experiment runs fail closed)."""
import hashlib, json, os, tempfile
from pathlib import Path

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1 << 20), b''):h.update(chunk)
    return h.hexdigest()

def digest_json(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text())

def write_once(path, data):
    p=Path(path)
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise RuntimeError('refusing to overwrite '+str(p))
    content=json.dumps(data,indent=2,sort_keys=True,default=str)+'\n'
    fd,tmp=tempfile.mkstemp(prefix='.'+p.name+'.',suffix='.tmp',dir=p.parent)
    try:
        with os.fdopen(fd,'w') as f:f.write(content);f.flush();os.fsync(f.fileno())
        os.replace(tmp,p)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    return p

def require(cond,msg):
    if not cond:raise RuntimeError('GATE_FAIL: '+msg)
