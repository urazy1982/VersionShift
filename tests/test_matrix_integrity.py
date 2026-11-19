import json
import numpy as np
import pytest
from versionshift_v3.utils import sha256
from versionshift_v3.validate_matrices import validate_manifest

def artifact(tmp_path):
    rows=[]
    for rep in range(6):
        for fold in range(5):
            te=np.array([fold]);tr=np.array([i for i in range(5) if i!=fold])
            for pre in ('P1','P2'):
                name=f'r{rep}_f{fold}_{pre}.npz';f=tmp_path/name
                np.savez(f,X_train=np.ones((4,2)),X_test=np.ones((1,2)),y_train=tr%2,y_test=te%2,train_indices=tr,test_indices=te)
                rows.append(dict(key=f'r{rep}_f{fold}',preprocessing_id=pre,file=name,sha256=sha256(f),n_features=2))
    p=tmp_path/'MATRICES_MANIFEST.json';p.write_text(json.dumps(dict(task_id=7,folds=rows)));return p

def test_valid_and_tampered_artifact(tmp_path):
    p=artifact(tmp_path);validate_manifest(p,7)
    (tmp_path/'r0_f0_P1.npz').write_bytes(b'corrupt')
    with pytest.raises(RuntimeError,match='hash mismatch'):validate_manifest(p,7)

def test_label_drift_even_with_updated_hash(tmp_path):
    p=artifact(tmp_path);f=tmp_path/'r0_f0_P2.npz'
    with np.load(f) as z:data={k:z[k] for k in z.files}
    data['y_test']=np.array([99]);np.savez(f,**data)
    m=json.loads(p.read_text());next(x for x in m['folds'] if x['file']==f.name)['sha256']=sha256(f);p.write_text(json.dumps(m))
    with pytest.raises(RuntimeError,match='inconsistent label'):validate_manifest(p)
