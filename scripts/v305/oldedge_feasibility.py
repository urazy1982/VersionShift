"""Original-edge same-stack synthetic feasibility; no historical dataset outcomes."""
import os,sys,json,inspect,hashlib,warnings
from pathlib import Path
import numpy as np,scipy,sklearn
from sklearn.datasets import make_classification
from threadpoolctl import threadpool_info
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from versionshift_v3.worker import cls_map
assert sys.version_info[:3]==(3,7,12)
assert np.__version__=='1.17.2' and scipy.__version__=='1.3.1'
assert sklearn.__version__==sys.argv[1]
assert all(os.environ.get(k)=='1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'])
X,y=make_classification(n_samples=100,n_features=8,n_informative=5,n_redundant=0,random_state=20261116)
rows=[]
for name in ['LR','SVC','kNN','RF','GB','HGB']:
 cls=cls_map()[name];kw={};params=inspect.signature(cls).parameters
 if 'random_state' in params:kw['random_state']=42
 if 'n_jobs' in params:kw['n_jobs']=1
 try:
  with warnings.catch_warnings(record=True) as w:
   warnings.simplefilter('always');m=cls(**kw);m.fit(X[:80],y[:80]);p=m.predict(X[80:])
  assert p.shape==(20,);rows.append({'estimator':name,'status':'PASS','warnings':[str(x.message) for x in w]})
 except Exception as e:rows.append({'estimator':name,'status':'FAIL','exception':repr(e)})
libs=threadpool_info()
for lib in libs:
 p=Path(lib['filepath']);lib['binary_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
r={'status':'PASS' if all(x['status']=='PASS' for x in rows) else 'FAIL','python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'python_binary_sha256':hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),'threadpools':libs,'estimators':rows,'qualification':'SYNTHETIC FEASIBILITY ONLY. No historical performance comparison or fixed-stack attribution executed.'}
Path(sys.argv[2]).write_text(json.dumps(r,indent=2)+'\n');print(sklearn.__version__,r['status']);sys.exit(0 if r['status']=='PASS' else 1)
