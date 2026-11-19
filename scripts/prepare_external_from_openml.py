#!/usr/bin/env python3
"""Fetch one prospectively selected external OpenML task and materialize P1/P2.
Creates new 6x5 folds before protocol freeze; no model outcomes inspected.
"""
import argparse,hashlib,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold,train_test_split
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from versionshift_v3.utils import read_json,write_once,sha256,require
from versionshift_v3.prepare import transform_fold

def seed_h(*args):return int.from_bytes(hashlib.sha256(repr(args).encode()).digest()[:4],'big')%(2**31-1)
def go(task_id,project,out):
    import openml
    proj=Path(project);plan=read_json(proj/'runs/plan/PROSPECTIVE_TASK_ALLOCATION.json')
    spec=next((x for x in plan['external_tasks'] if x['task_id']==task_id),None)
    require(spec is not None,'task not prospectively selected as external')
    task=openml.tasks.get_task(task_id,download_splits=False);ds=task.get_dataset()
    require(int(ds.dataset_id)==spec['dataset_id'] and int(ds.version)==spec['dataset_version'],'external source revision mismatch')
    from versionshift_v3.source_integrity import verify_arff
    integrity=verify_arff(ds)
    X,y,cats,_=ds.get_data(target=ds.default_target_attribute)
    X=pd.DataFrame(X);_,yy=np.unique(np.asarray(y).astype(str),return_inverse=True)
    if len(yy)>5000:
        keep,_=train_test_split(np.arange(len(yy)),train_size=5000,stratify=yy,
                                 random_state=seed_h('external_subsample',task_id))
        keep=np.sort(keep);X=X.iloc[keep].reset_index(drop=True);yy=yy[keep]
    require(np.bincount(yy).min()>=10,'insufficient per-class external rows')
    numeric=list(X.columns[np.array(cats)==False]);categorical=list(X.columns[np.array(cats)==True])
    folder=Path(out);require(not folder.exists(),'external output exists');folder.mkdir(parents=True)
    folds=[]
    for rep in range(6):
        splitter=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed_h('external_fold',task_id,rep))
        for fold,(tr,te) in enumerate(splitter.split(X,yy)):
            key='r%d_f%d'%(rep,fold)
            for pre,scale in [('P1',False),('P2',True)]:
                xa,xb,schema=transform_fold(X,tr,te,numeric,categorical,standardize=scale)
                fname=key+'_'+pre+'.npz';path=folder/fname
                np.savez_compressed(path,X_train=xa,X_test=xb,y_train=yy[tr],y_test=yy[te],
                                   train_indices=tr,test_indices=te)
                folds.append({'key':key,'preprocessing_id':pre,'file':fname,'sha256':sha256(path),
                              'n_features':xa.shape[1],
                              'linear_svc_dual_auto_expected':bool(len(tr)<xa.shape[1]),
                              'linear_svc_old_dual_effective':True,
                              'column_schema_sha256':hashlib.sha256(repr(schema).encode()).hexdigest()})
    write_once(folder/'MATRICES_MANIFEST.json',{'task_id':task_id,'external_source':True,
                'dataset_id':int(ds.dataset_id),'dataset_version':int(ds.version),
                'source_integrity':integrity,
                'folds':folds})
    print('EXTERNAL_PREPARED',task_id,len(folds))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task-id',type=int,required=True);p.add_argument('--project',default='.')
    p.add_argument('--out',required=True);a=p.parse_args();go(a.task_id,a.project,a.out)
