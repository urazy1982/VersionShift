"""Outcome-blind deterministic eligibility for the entire prior eligible universe."""
import os,sys,json,csv,hashlib,traceback
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from versionshift_v3.utils import read_json,sha256,require,write_once
from versionshift_v3.plan import external_sample,read_catalog
from scripts.prepare_external_from_openml import seed_h

def assess(item):
 import numpy as np,pandas as pd,openml
 from sklearn.model_selection import train_test_split
 tid,row,out=item;folder=Path(out)/'candidates'/str(tid);folder.mkdir(parents=True,exist_ok=True);rp=folder/'POSTCAP.json'
 if rp.exists():return read_json(rp)
 rec={'task_id':tid,'recorded_utc':datetime.now(timezone.utc).isoformat()}
 try:
  task=openml.tasks.get_task(tid,download_splits=False,download_data=False)
  require(int(task.dataset_id)==int(row['dataset_id']),'task dataset differs')
  ds=openml.datasets.get_dataset(int(row['dataset_id']),download_data=False)
  target=task.target_name
  require(ds.default_target_attribute in (None,target),'dataset and task targets differ')
  require(int(ds.version)==int(row['dataset_version']),'dataset revision differs')
  expected=ds.md5_checksum;cache=ROOT/f'runs/plan/FAST_EXPANSION_SUITE_454/arff/{ds.dataset_id}.arff'
  if cache.is_file():
   require(hashlib.md5(cache.read_bytes()).hexdigest()==expected,'local expanded ARFF MD5 mismatch');proof={'arff_path':str(cache),'published_md5':expected,'arff_sha256':sha256(cache)}
   ds.data_file=str(cache);ds.parquet_file=None;ds.data_pickle_file=None;ds.data_feather_file=None
  else:
   from versionshift_v3.source_integrity import verify_arff
   proof=verify_arff(ds)
  X,y,cats,_=ds.get_data(target=target);X=pd.DataFrame(X);classes,yy=np.unique(np.asarray(y).astype(str),return_inverse=True);before=np.bincount(yy);keep=np.arange(len(yy))
  if len(yy)>5000:
   keep,_=train_test_split(keep,train_size=5000,stratify=yy,random_state=seed_h('external_subsample',tid));keep=np.sort(keep)
  X=X.iloc[keep].reset_index(drop=True);yy=yy[keep];counts=np.bincount(yy,minlength=len(classes))
  X.to_pickle(folder/'X.pkl');np.save(folder/'labels.npy',yy);np.save(folder/'row_positions.npy',keep)
  cols={'numeric_columns':[str(c) for c,cat in zip(X.columns,cats) if not cat],'categorical_columns':[str(c) for c,cat in zip(X.columns,cats) if cat]}
  width=sum(X[c].astype('object').where(X[c].notna(),'__MISSING__').map(str).nunique() if c in cols['categorical_columns'] else 1 for c in X.columns)
  reasons=[]
  if counts.min()<10:reasons.append('POSTCAP_MIN_CLASS_LT_10')
  if width>500:reasons.append('POSTCAP_ENCODED_FEATURES_GT_500')
  rec.update(status='ASSESSED',eligible=not reasons,exclusions=reasons,dataset_id=int(ds.dataset_id),dataset_version=int(ds.version),raw_n=int(sum(before)),n=len(yy),k=len(classes),p_encoded=int(width),class_labels=classes.tolist(),class_counts_before=before.tolist(),class_counts_after=counts.tolist(),min_class_count=int(counts.min()),sampling_seed=seed_h('external_subsample',tid),source=proof,target_name=target,columns=cols,files_sha256={p.name:sha256(p) for p in folder.iterdir() if p.is_file()})
 except Exception as e:rec.update(status='FAILED',eligible=False,exclusions=['INTEGRITY_OR_EXECUTION_FAILURE'],error=repr(e),traceback=traceback.format_exc())
 write_once(rp,rec);return rec

def main():
 out=Path(sys.argv[1]);prior=read_json(out/'PRIOR_ALLOCATION.json');rows=read_catalog(ROOT/'runs/plan/EXTERNAL_CANDIDATES_EXPANDED_v304.csv');by={int(x['task_id']):x for x in rows}
 _,ex,status=external_sample(rows,prior['original_tasks'],prior['seed']);excluded={x['task_id'] for x in ex};universe=sorted(set(by)-excluded);require(len(universe)==52,'universe must be exactly prior 52')
 if not (out/'UNIVERSE_52.json').exists():write_once(out/'UNIVERSE_52.json',{'task_ids':universe,'catalog_sha256':sha256(ROOT/'runs/plan/EXTERNAL_CANDIDATES_EXPANDED_v304.csv'),'prior_exclusions':ex,'seed':prior['seed'],'rule':'same external_subsample hash seed, stratified train_size=5000, sorted original row positions; minimum 10 per class after cap; no outcomes accessed'})
 records=[]
 with ProcessPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(assess,(tid,by[tid],str(out))) for tid in universe]
  for f in as_completed(futures):
   x=f.result();records.append(x);print(len(records),'/52',x['task_id'],x['status'],x.get('eligible'),x.get('min_class_count'),flush=True)
 records.sort(key=lambda x:x['task_id']);write_once(out/'POSTCAP_ALL_52.json',{'records':records,'eligible_count':sum(x['eligible'] for x in records)})
 fields=list(rows[0]);new=[]
 for x in records:
  r=dict(by[x['task_id']]);r['eligible']='yes' if x['eligible'] else 'no'
  if x['status']=='ASSESSED':
   for key in ['n','k','min_class_count','p_encoded']:r[key]=str(x[key])
  new.append(r)
 with (out/'EXTERNAL_CANDIDATES_POSTCAP_v305.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(new)
 require(all(x['status']=='ASSESSED' for x in records),'unresolved candidate execution failure; allocation prohibited')
 selected,ex2,ready=external_sample(new,prior['original_tasks'],prior['seed']);require(len(selected)==20,'fewer than20 eligible')
 plan=dict(prior);plan.update(design_amendment='v3.0.5_PROSPECTIVE_NOT_AUTHOR_APPROVED',catalog_sha256=sha256(out/'EXTERNAL_CANDIDATES_POSTCAP_v305.csv'),external_tasks=selected,external_task_ids=[x['task_id'] for x in selected],external_exclusions=ex2,external_selection_status=ready,previous_allocation_sha256=sha256(out/'PRIOR_ALLOCATION.json'),postcap_evidence_sha256=sha256(out/'POSTCAP_ALL_52.json'))
 write_once(out/'PROSPECTIVE_TASK_ALLOCATION_v305.json',plan);print('NEW_ALLOCATION',plan['external_task_ids'],ready,flush=True)
if __name__=='__main__':main()
