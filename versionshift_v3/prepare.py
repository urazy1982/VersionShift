"""Train-only P1/P2 preprocessing and hash-linked frozen fold materialization.
Must run before the prospective freeze; never look at prediction outcomes.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from .utils import sha256,write_once,read_json,require

def transform_fold(X,train_idx,test_idx,numeric_cols,categorical_cols,standardize=False):
    X=pd.DataFrame(X)
    tr=X.iloc[np.asarray(train_idx,dtype=int)].reset_index(drop=True)
    te=X.iloc[np.asarray(test_idx,dtype=int)].reset_index(drop=True)
    require(not set(numeric_cols)&set(categorical_cols),'overlapping categorical and numeric columns')
    require(set(numeric_cols)|set(categorical_cols)==set(X.columns),'unclassified feature column')
    parts_train=[];parts_test=[]; schema=[]
    for col in numeric_cols:
        a=pd.to_numeric(tr[col],errors='coerce').to_numpy(dtype=np.float64)
        b=pd.to_numeric(te[col],errors='coerce').to_numpy(dtype=np.float64)
        median=float(np.nanmedian(a)) if np.isfinite(a).any() else 0.0
        a=np.where(np.isfinite(a),a,median);b=np.where(np.isfinite(b),b,median)
        if standardize:
            mean=float(a.mean());std=float(a.std());std=std if std>0 else 1.0
            a=(a-mean)/std;b=(b-mean)/std
        parts_train.append(a[:,None]);parts_test.append(b[:,None]);schema.append(('numeric',str(col)))
    for col in categorical_cols:
        a=tr[col].astype('object').where(tr[col].notna(),'__MISSING__').map(str).to_numpy()
        b=te[col].astype('object').where(te[col].notna(),'__MISSING__').map(str).to_numpy()
        cats=sorted(set(a))
        for cat in cats:
            parts_train.append((a==cat).astype(np.float64)[:,None]);parts_test.append((b==cat).astype(np.float64)[:,None]);
            schema.append(('categorical',str(col),cat))
    atr=np.concatenate(parts_train,axis=1) if parts_train else np.empty((len(tr),0))
    ate=np.concatenate(parts_test,axis=1) if parts_test else np.empty((len(te),0))
    require(atr.shape[1]==ate.shape[1] and atr.shape[1]<=500,'feature dimension or cap failure')
    require(np.isfinite(atr).all() and np.isfinite(ate).all(),'nonfinite output')
    return atr,ate,schema

def materialize_raw_task(task_id,legacy_plan_dir,raw_frame_csv,raw_labels_npy,numeric_cols,categorical_cols,out):
    """Input raw row ordering must already have been matched to the v2.6 63-task plan."""
    base=Path(legacy_plan_dir);plan=read_json(base/'PLAN.json')
    ds=next((d for d in plan['datasets'] if d['name']=='openml_%d'%task_id),None)
    require(ds is not None,'task not in legacy frozen plan')
    fpath=base/ds['folds_npz']
    require(sha256(fpath)==ds['folds_sha256'],'legacy fold hash mismatch')
    X=pd.read_csv(raw_frame_csv)
    y=np.load(raw_labels_npy,allow_pickle=False)
    require(len(X)==len(y)==ds['n'],'raw row subset and target must match original archive')
    legacy=np.load(base/ds['dataset_npz'],allow_pickle=False)
    require(np.array_equal(legacy['y'],y),'raw labels/order differ from legacy archive')
    out=Path(out);require(not out.exists(),'output exists; frozen preprocessing immutable');out.mkdir(parents=True)
    folds=np.load(fpath,allow_pickle=False)
    summary=[]
    for key in ds['fold_keys']:
        ti=folds[key+'_train'];vi=folds[key+'_test']
        require(not (set(ti)&set(vi)),'train/test overlap')
        require(len(ti)+len(vi)==len(y),'fold does not partition original source rows')
        for mode,scaled in (('P1',False),('P2',True)):
            atr,ate,cols=transform_fold(X,ti,vi,numeric_cols,categorical_cols,scaled)
            path=out/('%s_%s.npz'%(key,mode))
            np.savez_compressed(path,X_train=atr,X_test=ate,y_train=y[ti],y_test=y[vi],
                                train_indices=ti,test_indices=vi)
            summary.append({'key':key,'preprocessing_id':mode,'file':path.name,'sha256':sha256(path),
                            'n_features':atr.shape[1],
                            'linear_svc_dual_auto_expected':bool(len(ti)<atr.shape[1]),
                            'linear_svc_old_dual_effective':True,
                            'column_schema_sha256':hashlib.sha256(repr(cols).encode()).hexdigest()})
    write_once(out/'MATRICES_MANIFEST.json',{'task_id':task_id,'legacy_fold_sha256':sha256(fpath),
                                             'raw_frame_sha256':sha256(raw_frame_csv),
                                             'raw_labels_sha256':sha256(raw_labels_npy),
                                             'folds':summary})
    return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--task-id',type=int,required=True)
    p.add_argument('--legacy-plan-dir',required=True);p.add_argument('--raw-csv',required=True)
    p.add_argument('--labels-npy',required=True);p.add_argument('--columns-json',required=True)
    p.add_argument('--out',required=True);a=p.parse_args()
    cols=read_json(a.columns_json)
    x=materialize_raw_task(a.task_id,a.legacy_plan_dir,a.raw_csv,a.labels_npy,
                           cols['numeric_columns'],cols['categorical_columns'],a.out)
    print('PREPROCESSING_MATERIALIZED',a.task_id,len(x),'matrix files')
if __name__=='__main__':main()
