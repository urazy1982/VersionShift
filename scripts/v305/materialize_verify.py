"""Prepare selected tasks and replay every matrix from raw rows before reuse/index."""
import sys,json,hashlib,fcntl,traceback
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from versionshift_v3.utils import read_json,sha256,write_once,require
from versionshift_v3.prepare import transform_fold
from versionshift_v3.validate_matrices import validate_manifest
from scripts.prepare_external_from_openml import seed_h

def main():
 out=Path(sys.argv[1]);lock=(ROOT/'runs/prepare.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 require(not (ROOT/'runs/plan/FROZEN_PROTOCOL.json').exists(),'freeze already exists')
 plan=read_json(out/'PROSPECTIVE_TASK_ALLOCATION_v305.json');legacy=Path.home()/'versionshift_data/STAGE_B_PLAN_CACHED';lp=read_json(legacy/'PLAN.json');require(sha256(legacy/'PLAN.json')==plan['legacy_plan_sha256'],'legacy plan changed');core={d['info']['task']:d for d in lp['datasets']};selected=plan['pilot_task_ids']+plan['evaluation_task_ids']+plan['external_task_ids'];records=[]
 for tid in selected:
  folder=ROOT/f'runs/data/{tid}';mf=folder/'MATRICES_MANIFEST.json';existed=mf.is_file()
  if tid in core:
   d=core[tid];raw=ROOT/f'runs/raw/{tid}';X=pd.read_csv(raw/'X.csv');y=np.load(raw/'labels.npy',allow_pickle=False);cols=read_json(raw/'columns.json');m=read_json(mf)
   require(sha256(raw/'X.csv')==m['raw_frame_sha256'] and sha256(raw/'labels.npy')==m['raw_labels_sha256'],'core raw hashes changed')
   require(sha256(legacy/d['dataset_npz'])==d['dataset_sha256'] and sha256(legacy/d['folds_npz'])==d['folds_sha256'],'historical archive changed')
   with np.load(legacy/d['dataset_npz'],allow_pickle=False) as z:require(np.array_equal(y,z['y']),'historical row labels differ')
   with np.load(legacy/d['folds_npz'],allow_pickle=False) as z:folds=[(key,z[key+'_train'].copy(),z[key+'_test'].copy()) for key in d['fold_keys']]
   require(m['legacy_fold_sha256']==d['folds_sha256'],'fold manifest provenance differs')
  else:
   raw=out/f'candidates/{tid}';rec=read_json(raw/'POSTCAP.json');require(rec['eligible'] and rec['task_id']==tid,'ineligible/incorrect task')
   spec=next(x for x in plan['external_tasks'] if x['task_id']==tid);require(spec['dataset_id']==rec['dataset_id'] and spec['dataset_version']==rec['dataset_version'],'selected revision differs')
   for name,want in rec['files_sha256'].items():require(sha256(raw/name)==want,'candidate artifact changed '+name)
   X=pd.read_pickle(raw/'X.pkl');y=np.load(raw/'labels.npy',allow_pickle=False);cols=rec['columns'];require(len(X)==len(y)==spec['n'],'row count differs')
   require(np.array_equal(np.bincount(y,minlength=spec['k']),np.array(rec['class_counts_after'])),'postcap classes differ')
   folds=[]
   for rep in range(6):
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed_h('external_fold',tid,rep))
    folds.extend((f'r{rep}_f{f}',tr,te) for f,(tr,te) in enumerate(skf.split(X,y)))
   if existed:
    m=read_json(mf);require(m['dataset_id']==spec['dataset_id'] and m['dataset_version']==spec['dataset_version'],'reused revision differs');require(m['source_integrity']['arff_sha256']==rec['source']['arff_sha256'],'raw ARFF bytes differ')
    prior=ROOT/f'reports/CPU_CONTINUATION_20261117T085621Z/EXTERNAL_RAW_EVIDENCE/{tid}/original_raw_row_positions.npy'
    require(prior.is_file(),'reused external raw row evidence absent');require(np.array_equal(np.load(prior,allow_pickle=False),np.load(raw/'row_positions.npy',allow_pickle=False)),'reused raw positions differ')
  entries=[]
  if not existed:require(not folder.exists(),'preserve incomplete prior output; no overwrite');folder.mkdir(parents=True)
  existing_by={(e['key'],e['preprocessing_id']):e for e in m['folds']} if existed else {}
  for key,tr,te in folds:
   for pre,scale in [('P1',False),('P2',True)]:
    a,b,schema=transform_fold(X,tr,te,cols['numeric_columns'],cols['categorical_columns'],scale);name=f'{key}_{pre}.npz';fp=folder/name;schema_sha=hashlib.sha256(repr(schema).encode()).hexdigest()
    expected={'X_train':a,'X_test':b,'y_train':y[tr],'y_test':y[te],'train_indices':tr,'test_indices':te}
    if not existed:np.savez_compressed(fp,**expected)
    else:
     entry=existing_by[key,pre];require(entry['file']==name and sha256(fp)==entry['sha256'],'existing matrix/hash changed');require(entry['column_schema_sha256']==schema_sha,'preprocessing schema differs')
    with np.load(fp,allow_pickle=False) as z:
     for k,v in expected.items():require(np.array_equal(z[k],v),'exact preprocessing/row/fold mismatch '+str(tid)+' '+key+' '+pre+' '+k)
    entries.append({'key':key,'preprocessing_id':pre,'file':name,'sha256':sha256(fp),'n_features':a.shape[1],'linear_svc_dual_auto_expected':bool(len(tr)<a.shape[1]),'linear_svc_old_dual_effective':True,'column_schema_sha256':schema_sha})
  if not existed:write_once(mf,{'task_id':tid,'external_source':True,'dataset_id':spec['dataset_id'],'dataset_version':spec['dataset_version'],'source_integrity':rec['source'],'raw_frame_sha256':sha256(raw/'X.pkl'),'raw_labels_sha256':sha256(raw/'labels.npy'),'raw_positions_sha256':sha256(raw/'row_positions.npy'),'amendment':'v3.0.5','folds':entries})
  validate_manifest(mf,tid);records.append({'task_id':tid,'reused':existed,'path':str(mf.resolve()),'sha256':sha256(mf),'matrix_files':len(entries),'exact_raw_fold_preprocessing_replay':'PASS'});print(len(records),'/83',tid,'VERIFIED_REUSE' if existed else 'NEW_VERIFIED',flush=True)
  # An append-only ledger preserves successful validation if a later task fails.
  with (out/'MATRIX_VERIFICATION_PROGRESS.jsonl').open('a') as f:f.write(json.dumps(records[-1])+'\n')
 require(len(records)==83 and len(set(x['task_id'] for x in records))==83,'not exactly83 unique tasks')
 write_once(out/'ALL_83_INDEPENDENT_VERIFICATION.json',{'status':'PASS','recorded_utc':datetime.now(timezone.utc).isoformat(),'allocation_sha256':sha256(out/'PROSPECTIVE_TASK_ALLOCATION_v305.json'),'tasks':records,'matrix_files':sum(x['matrix_files'] for x in records)})
 # Preserve the prior proposal byte-for-byte before installing the amended draft.
 prior=ROOT/'runs/plan/PROSPECTIVE_TASK_ALLOCATION_v304_PRESERVED.json';require(not prior.exists(),'prior preservation destination already exists');prior.write_bytes((ROOT/'runs/plan/PROSPECTIVE_TASK_ALLOCATION.json').read_bytes());require(sha256(prior)==sha256(out/'PRIOR_ALLOCATION.json'),'prior allocation changed')
 active=ROOT/'runs/plan/PROSPECTIVE_TASK_ALLOCATION.json';temp=active.with_suffix('.v305.tmp');temp.write_bytes((out/'PROSPECTIVE_TASK_ALLOCATION_v305.json').read_bytes());temp.replace(active)
 write_once(ROOT/'runs/plan/MATRICES_INDEX.json',{'task_manifests':[{k:r[k] for k in ['task_id','path','sha256']} for r in records],'allocation_sha256':sha256(active),'independent_verification_sha256':sha256(out/'ALL_83_INDEPENDENT_VERIFICATION.json'),'status':'VERIFIED_PRE_FREEZE_NOT_AUTHOR_APPROVED'})
 print('83_TASK_INDEX_READY_NOT_FROZEN',flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  out=Path(sys.argv[1]);
  with (out/'FAILURES.log').open('a') as f:f.write(datetime.now(timezone.utc).isoformat()+' | materialization/verification failed\n'+traceback.format_exc()+'\n')
  raise
