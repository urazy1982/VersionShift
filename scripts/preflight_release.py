#!/usr/bin/env python3
"""Synthetic mechanism probe. Never use synthetic probes as research results."""
import argparse,sys,json,os,warnings,inspect,platform,hashlib
from pathlib import Path
import numpy as np,scipy,sklearn
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning
p=argparse.ArgumentParser();p.add_argument('--expected',required=True);p.add_argument('--out',required=True);a=p.parse_args()
versions={'1.4.2','1.5.2','1.6.1'}
checks={}
checks['version_matches']=sklearn.__version__==a.expected
checks['common_stack_candidate']=sys.version_info[:2]==(3,11) and np.__version__=='1.26.4' and scipy.__version__=='1.13.1'
checks['blas_threads_env'] = all(os.getenv(x)=='1' for x in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'))
checks['dual_default']=str(inspect.signature(LinearSVC).parameters['dual'].default)
checks['ada_default']=str(inspect.signature(AdaBoostClassifier).parameters['algorithm'].default)
X,y=make_classification(n_samples=100,n_features=9,n_informative=5,n_redundant=0,random_state=20261116)
for dual in [True,False,'auto']:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        try:
            mdl=LinearSVC(dual=dual,random_state=42,max_iter=500)
            mdl.fit(X,y)
            checks['linear_'+str(dual)]={'status':'OK','accuracy':float((mdl.predict(X)==y).mean()),
                                          'iterations':np.asarray(mdl.n_iter_).tolist(),
                                          'warnings':[str(z.message) for z in w]}
        except Exception as ex:checks['linear_'+str(dual)]={'status':'ERROR','exception':repr(ex)}
for algorithm in ['OMIT','SAMME','SAMME.R']:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        try:
            kw={} if algorithm=='OMIT' else {'algorithm':algorithm}
            mdl=AdaBoostClassifier(n_estimators=8,random_state=42,**kw)
            mdl.fit(X,y)
            checks['ada_'+algorithm]={'status':'OK','accuracy':float((mdl.predict(X)==y).mean()),
                                      'warnings':[str(z.message) for z in w]}
        except Exception as ex:checks['ada_'+algorithm]={'status':'REMOVED_BEHAVIOR' if algorithm=='SAMME.R' and a.expected=='1.6.1' else 'ERROR','exception':repr(ex)}
try:
    from threadpoolctl import threadpool_info
    info=threadpool_info()
except Exception as ex:info={'error':repr(ex)}
requirements={'1.4.2':("warn",'OK'),'1.5.2':("auto",'OK'),'1.6.1':("auto",'REMOVED_BEHAVIOR')}
if a.expected in requirements:
    d,ada=requirements[a.expected]
    checks['linear_dual_default_expected']=checks['dual_default']==d
    checks['ada_samme_r_expected']=checks['ada_SAMME.R']['status']==ada
    if a.expected=='1.6.1':checks['ada_deprecated_signature']=checks['ada_default']=='deprecated'
    if a.expected=='1.4.2':
        # 1.4.2 uses a constructor placeholder 'warn', whose documented effective value is True.
        # Probe on the identical synthetic fixture before freezing.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            old=LinearSVC(random_state=42,max_iter=500).fit(X,y).predict(X)
            explicit=LinearSVC(dual=True,random_state=42,max_iter=500).fit(X,y).predict(X)
        checks['old_warn_effective_dual_true_probe']=bool(np.array_equal(old,explicit))
else:
    checks['linear_dual_default_expected']=False
    checks['ada_samme_r_expected']=False
status='PASS' if (checks['version_matches'] and checks['common_stack_candidate'] and checks['blas_threads_env'] and
                    checks['linear_dual_default_expected'] and checks['ada_samme_r_expected'] and
                    checks.get('ada_deprecated_signature',True) and
                    checks.get('old_warn_effective_dual_true_probe',True)) else 'FAIL'
report={'status':status,'python':sys.version,'python_minor':list(sys.version_info[:2]),'numpy':np.__version__,
        'scipy':scipy.__version__,'sklearn':sklearn.__version__,'threadpool_info':info,'checks':checks,
        'interpretation':'SYNTHETIC_PREFLIGHT_ONLY_NO_EXPERIMENT_OUTCOMES'}
out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
if out.exists():raise SystemExit('STOP preflight file already exists: '+str(out))
out.write_text(json.dumps(report,indent=2,default=str)+'\n')
print(status, a.expected,checks['dual_default'],checks['ada_default'])
if status!='PASS':raise SystemExit(2)
