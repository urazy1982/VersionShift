"""Synthetic fold logging evidence, never scientific cross-release outcomes."""
import sys,json,importlib,warnings
from pathlib import Path
import numpy as np,sklearn
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from versionshift_v3.worker import run_job
from versionshift_v3.utils import sha256,write_once
out=Path(sys.argv[1]);out.mkdir(exist_ok=True,parents=True);mats=[];seeds={}
for rep in range(6):
 for fold in range(5):
  key=f'r{rep}_f{fold}';n,p=(20,30) if fold%2 else (60,5);rng=np.random.RandomState(rep*5+fold);X=rng.randn(n+5,p);y=np.arange(n+5)%2;fp=out/(key+'.npz');np.savez_compressed(fp,X_train=X[:n],X_test=X[n:],y_train=y[:n],y_test=y[n:]);mats.append((key,str(fp),sha256(fp)));seeds[key]=42
j={'out':str(out/'unused.json'),'job_id':'SYNTHETIC_LOGGING_ONLY','task_id':-1,'release':sklearn.__version__,'edge_id':'SYNTHETIC','algo':'LinearSVC','policy':'D0','preprocessing_id':'SYNTHETIC','plan_digest':'NOT_SCIENTIFIC','frozen_sha256':'NOT_SCIENTIFIC','matrices':mats,'seeds':seeds}
with warnings.catch_warnings():warnings.simplefilter('ignore');rec,arrays=run_job(j)
assert rec['status']=='SUPPORTED',rec
rows=rec['linear_svc_effective_solver_by_fold'];assert len(rows)==30 and all(x['fit_returned'] for x in rows)
for row in rows:assert row['effective_dual']==(True if sklearn.__version__=='1.4.2' else row['n_train']<row['n_features'])
write_once(out/'OBSERVED_SOLVERS.json',{'status':'PASS','release':sklearn.__version__,'folds':rows,'interpretation':'SYNTHETIC ENGINEERING ONLY; actual backend logging; no scientific performance outcomes retained'})
print(sklearn.__version__,'PASS 30 actual per-fold backend observations')
