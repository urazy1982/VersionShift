"""Independent pre-fit artifact validation, including matched P1/P2 fold identity."""
from pathlib import Path
import numpy as np
from .utils import read_json,require,sha256

def validate_manifest(path, expected_task_id=None):
    path=Path(path);m=read_json(path)
    if expected_task_id is not None:require(m['task_id']==expected_task_id,'manifest task mismatch')
    expected={(f'r{r}_f{f}',p) for r in range(6) for f in range(5) for p in ('P1','P2')}
    keys=[(e['key'],e['preprocessing_id']) for e in m['folds']]
    require(len(keys)==60 and set(keys)==expected,'missing or duplicate fold/preprocessing cells')
    reference={};labels={};tests={};n=None
    for e in m['folds']:
        f=(path.parent/e['file']).resolve()
        require(f.parent==path.parent.resolve(),'matrix path escapes task directory')
        require(f.is_file() and sha256(f)==e['sha256'],'matrix hash mismatch '+str(f))
        with np.load(f,allow_pickle=False) as z:
            tr,te=z['train_indices'],z['test_indices'];a,b=z['X_train'],z['X_test'];y,w=z['y_train'],z['y_test']
            count=len(tr)+len(te)
            if n is None:n=count
            require(count==n and tr.ndim==te.ndim==1,'inconsistent row count')
            require(tr.dtype.kind in 'iu' and te.dtype.kind in 'iu','noninteger indices')
            require(np.array_equal(np.sort(np.concatenate([tr,te])),np.arange(n)),'fold not exact disjoint partition')
            require(a.ndim==b.ndim==2 and a.shape[0]==len(tr)==len(y) and b.shape[0]==len(te)==len(w),'matrix/label row mismatch')
            require(a.shape[1]==b.shape[1]==e['n_features'] and 0<a.shape[1]<=500,'feature cap/shape mismatch')
            require(np.isfinite(a).all() and np.isfinite(b).all(),'nonfinite matrix')
            for indices,ys in ((tr,y),(te,w)):
                for idx,label in zip(indices,ys):
                    idx=int(idx)
                    if idx in labels:require(labels[idx]==label,'inconsistent label by row key')
                    else:labels[idx]=label
            key=e['key']
            if key in reference:
                old=reference[key]
                require(all(np.array_equal(x,v) for x,v in zip(old,(tr,te,y,w))),'P1/P2 fold or label mismatch')
            else:reference[key]=tuple(x.copy() for x in (tr,te,y,w))
            if e['preprocessing_id']=='P1':tests.setdefault(key.split('_')[0],[]).append(te.copy())
    for rep,indices in tests.items():require(np.array_equal(np.sort(np.concatenate(indices)),np.arange(n)),'test rows do not partition repetition '+rep)
    return m
