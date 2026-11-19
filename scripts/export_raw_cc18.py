#!/usr/bin/env python3
"""Reconstruct raw source rows in original v2.6 order from OpenML task revisions.
No cross-release model outcomes. Not guaranteed to succeed if OpenML revisions shifted.
"""
import argparse,sys,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from versionshift_v3.utils import read_json,sha256,write_once,require
import hashlib

def seed_h(*parts):return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:4],'big')%(2**31-1)
def reconstruct(task_id,legacy,output):
    import openml
    p=Path(legacy);plan=read_json(p/'PLAN.json')
    d=next((x for x in plan['datasets'] if x['info'].get('task')==task_id),None)
    require(d is not None,'task not in original eligible frame')
    task=openml.tasks.get_task(task_id,download_splits=False)
    ds=task.get_dataset()
    require(int(ds.dataset_id)==int(d['info']['dataset_id']) and int(ds.version)==int(d['info']['version']),
            'OpenML dataset ID or revision mismatches prior data')
    from versionshift_v3.source_integrity import verify_arff
    integrity=verify_arff(ds)
    X,y,categorical,_=ds.get_data(target=ds.default_target_attribute)
    X=pd.DataFrame(X);_,yy=np.unique(np.asarray(y).astype(str),return_inverse=True)
    keep=np.arange(len(yy))
    if len(keep)>d['n']:
        chosen,_=train_test_split(keep,train_size=int(d['n']),stratify=yy,
                                  random_state=seed_h('subsample',d['name']))
        keep=np.sort(chosen)
    require(len(keep)==d['n'],'source rows count mismatch')
    yr=yy[keep].astype(np.int64)
    old=np.load(p/d['dataset_npz'],allow_pickle=False)
    require(sha256(p/d['dataset_npz'])==d['dataset_sha256'],'P0 source hash mismatch')
    require(np.array_equal(old['y'],yr),'target row order does not reproduce old frame')
    x=X.iloc[keep].reset_index(drop=True)
    out=Path(output);require(not out.exists(),'raw export already exists');out.mkdir(parents=True)
    x.to_csv(out/'X.csv',index=False);np.save(out/'labels.npy',yr)
    colspec={'numeric_columns':[str(c) for c,cat in zip(x.columns,categorical) if not cat],
             'categorical_columns':[str(c) for c,cat in zip(x.columns,categorical) if cat]}
    write_once(out/'columns.json',colspec)
    np.save(out/'original_raw_row_positions.npy',keep)
    write_once(out/'RAW_PROVENANCE.json',{'task_id':task_id,'dataset_id':int(ds.dataset_id),
              'dataset_version':int(ds.version),'X_sha256':sha256(out/'X.csv'),
              'labels_sha256':sha256(out/'labels.npy'),
              'indices_sha256':sha256(out/'original_raw_row_positions.npy'),
              'legacy_plan_sha256':sha256(p/'PLAN.json'),
              'source_integrity':integrity,
              'qualification':'Original row order/labels checked, numeric/categorical feature encoding pending P1/P2'})
    print('RAW_TASK_EXPORTED',task_id,'n',len(x),'p',len(x.columns))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--task-id',type=int,required=True)
    a.add_argument('--legacy-plan-dir',required=True);a.add_argument('--out',required=True)
    x=a.parse_args();reconstruct(x.task_id,x.legacy_plan_dir,x.out)
