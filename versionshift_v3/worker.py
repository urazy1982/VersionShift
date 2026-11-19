"""Isolated CPU model fitting for scikit-learn 1.4–1.6. 30 folds/job.
Worker executes one precommitted job; cannot modify the study plan or other results.
"""
import sys,os,hashlib,json,warnings,traceback,inspect,collections
from pathlib import Path
import numpy as np
import sklearn
from sklearn.metrics import balanced_accuracy_score

ALGORITHMS=['LR','SVC','kNN','RF','GB','HGB','LinearSVC','AdaBoost']
STATES=['SUPPORTED','UNSUPPORTED','REMOVED_BEHAVIOR','INEFFECTIVE','API_REMOVED',
        'FIT_FAILED','TIMED_OUT','MISSING_JOB']

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def cls_map():
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC,LinearSVC
    from sklearn.neighbors import KNeighborsClassifier
    try:
        from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401; old versions need this
    except ImportError:pass
    from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,HistGradientBoostingClassifier,AdaBoostClassifier
    return {'LR':LogisticRegression,'SVC':SVC,'kNN':KNeighborsClassifier,
            'RF':RandomForestClassifier,'GB':GradientBoostingClassifier,'HGB':HistGradientBoostingClassifier,
            'LinearSVC':LinearSVC,'AdaBoost':AdaBoostClassifier}

def init_model(name,policy,seed,release,e2,new_settings,old_effective=None):
    cls=cls_map()[name]
    kw={}
    if policy=='D1':
        if name in e2:kw=dict(e2[name])
        elif name=='LinearSVC':kw={'dual':new_settings['LinearSVC']['dual']}
        elif name=='AdaBoost':
            if release=='1.6.1':kw={}
            else:kw={'algorithm':new_settings['AdaBoost']['algorithm_'+release]}
        else:return None,'UNSUPPORTED','D1 map missing algorithm'
    elif policy=='D2':
        if name=='AdaBoost' and release=='1.6.1':
            return None,'REMOVED_BEHAVIOR','AdaBoost SAMME.R removed in scikit-learn 1.6'
        if old_effective is None or name not in old_effective:
            return None,'UNSUPPORTED','pre-frozen D2 map missing estimator'
        kw=dict(old_effective[name])
    elif policy!='D0':return None,'UNSUPPORTED','unknown policy'
    params=inspect.signature(cls).parameters
    for k in kw:
        if k not in params:return None,'API_REMOVED','parameter absent: '+k
    if 'random_state' in params:kw['random_state']=int(seed)
    if 'n_jobs' in params:kw['n_jobs']=1
    try:
        est=cls(**kw)
        if hasattr(est,'_validate_params'):est._validate_params()
    except (TypeError,ValueError) as ex:
        return None,'UNSUPPORTED',repr(ex)
    return est,'SUPPORTED',kw

def fit_with_solver_observation(est, X, y, fold_key, observations):
    """Observe the actual liblinear call; forward all arguments unchanged.

    Workers fit one estimator at a time. The temporary module hook is restored
    even on backend failure. No inference from matrix shape substitutes for it.
    """
    import importlib
    module=importlib.import_module(type(est).__module__)
    backend=module._fit_liblinear
    signature=inspect.signature(backend)
    def observed(*args, **kwargs):
        bound=signature.bind(*args, **kwargs);bound.apply_defaults();params=bound.arguments
        effective=params['dual']
        if not isinstance(effective,(bool,np.bool_)):
            raise RuntimeError('liblinear backend received unresolved dual')
        solver=module._get_liblinear_solver_type(params['multi_class'],params['penalty'],params['loss'],effective)
        rec={'fold_key':fold_key,'constructor_dual':est.dual,
             'effective_dual':bool(effective),'liblinear_solver_type':int(solver),
             'n_train':int(params['X'].shape[0]),'n_features':int(params['X'].shape[1]),
             'loss':params['loss'],'penalty':params['penalty'],'multi_class':params['multi_class'],
             'observation':'actual _fit_liblinear call arguments; forwarded unchanged',
             'backend_module':module.__name__,'fit_returned':False}
        observations.append(rec)
        result=backend(*args,**kwargs);rec['fit_returned']=True
        return result
    module._fit_liblinear=observed
    try:
        result=est.fit(X,y)
        if sum(r['fold_key']==fold_key for r in observations)!=1:
            raise RuntimeError('expected one observed liblinear call per LinearSVC fold')
        return result
    finally:module._fit_liblinear=backend

def spec_digest(j):
    return hashlib.sha256(json.dumps(j,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()

def run_job(j):
    out=Path(j['out']);require=j.get('plan_digest')
    base={'job_id':j['job_id'],'task_id':j['task_id'],'release':j['release'],'edge_id':j['edge_id'],
          'algo':j['algo'],'policy':j['policy'],'preprocessing_id':j['preprocessing_id'],
          'status':'MISSING_JOB','environment':{'python':sys.version,'sklearn':sklearn.__version__,'numpy':np.__version__},
          'worker_sha256':sha(__file__),'environment_lock_sha256':j.get('environment_lock_sha256'),
          'matrix_manifest_digest':hashlib.sha256(repr(j['matrices']).encode()).hexdigest(),
          'plan_digest':require,'frozen_sha256':j['frozen_sha256'],'warnings':[],
          'job_spec_sha256':spec_digest(j),'source_binding':j.get('source_binding'),
          'linear_svc_effective_solver_by_fold':[]}
    if sklearn.__version__!=j['release']:
        base.update(status='UNSUPPORTED',error='wrong sklearn version')
        return base,None
    e2=j.get('legacy_e2',{});new=j.get('new_settings',{})
    labels=[];preds=[];scores=[];iterations=[];warn=[];warn_categories=[];keys=[];test_indices=[];source_ids=[]
    try:
        binding=j['source_binding']
        for name in ('matrix_manifest','labels','row_positions','provenance'):
            if sha(binding[name+'_path'])!=binding[name+'_sha256']:raise ValueError('source binding hash mismatch')
        manifest=json.loads(Path(binding['matrix_manifest_path']).read_text())
        if manifest['task_id']!=j['task_id']:raise ValueError('source task differs')
        source_y=np.load(binding['labels_path'],allow_pickle=False);positions=np.load(binding['row_positions_path'],allow_pickle=False)
        if source_y.ndim!=1 or positions.ndim!=1 or len(source_y)!=len(positions) or len(np.unique(positions))!=len(positions):raise ValueError('source row identity invalid')
        entries={e['key']:e for e in manifest['folds'] if e['preprocessing_id']==j['preprocessing_id']}
        if set(entries)!={f'r{r}_f{f}' for r in range(6) for f in range(5)}:raise ValueError('source fold universe invalid')
    except Exception as ex:return dict(base,status='UNSUPPORTED',error='SOURCE_BINDING: '+repr(ex)),None
    mats=j['matrices'];seeds=j['seeds']
    if len(mats)!=30 or len(seeds)!=30:return dict(base,status='MISSING_JOB',error='not exactly 30 folds'),None
    for key, filename, fhash in mats:
        if sha(filename)!=fhash:return dict(base,status='UNSUPPORTED',error='input matrix hash changed'),None
        entry=entries.get(key)
        if entry is None or entry['sha256']!=fhash or Path(filename).resolve()!=Path(binding['matrix_manifest_path']).resolve().parent/entry['file']:
            return dict(base,status='UNSUPPORTED',error='matrix manifest binding mismatch'),None
        z=np.load(filename,allow_pickle=False)
        try:
            ti=z['test_indices'];tr=z['train_indices']
            if ti.dtype.kind not in 'iu' or tr.dtype.kind not in 'iu' or not np.array_equal(np.sort(np.concatenate([tr,ti])),np.arange(len(source_y))):raise ValueError('row partition differs')
            if not np.array_equal(z['y_test'],source_y[ti]) or not np.array_equal(z['y_train'],source_y[tr]):raise ValueError('source labels differ')
        except Exception as ex:return dict(base,status='UNSUPPORTED',error='SOURCE_ROWS: '+repr(ex)),None
        est,state,details=init_model(j['algo'],j['policy'],seeds[key],j['release'],e2,new,j.get('d2_effective'))
        if state!='SUPPORTED':return dict(base,status=state,error=details),None
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                if j['algo']=='LinearSVC':
                    fit_with_solver_observation(est,z['X_train'],z['y_train'],key,base['linear_svc_effective_solver_by_fold'])
                else:est.fit(z['X_train'],z['y_train'])
                pr=est.predict(z['X_test'])
            pred=np.asarray(pr,dtype=np.int64)
            true=np.asarray(z['y_test'],dtype=np.int64)
            if len(true)!=len(pred):raise RuntimeError('prediction length mismatch')
            scores.append(float(100*balanced_accuracy_score(true,pred)))
            keys.append(key);labels.append(true);preds.append(pred);test_indices.append(ti.copy());source_ids.append(positions[ti].copy())
            warn.extend(str(x.message) for x in w);warn_categories.extend(x.category.__name__ for x in w)
            iterations.append(np.asarray(getattr(est,'n_iter_',[])).tolist())
        except Exception as ex:return dict(base,status='FIT_FAILED',error=repr(ex),traceback=traceback.format_exc()[-3000:]),None
    arrays={}
    for key,y,pr,idx,ids in zip(keys,labels,preds,test_indices,source_ids):
        arrays[key+'_y']=y;arrays[key+'_pred']=pr;arrays[key+'_test_indices']=idx;arrays[key+'_source_row_ids']=ids
    base.update(status='SUPPORTED',balanced_accuracy_pp=scores,fold_keys=keys,
                warning_count=len(warn),warnings=warn[:300],warning_categories=dict(collections.Counter(warn_categories)),
                convergence_warning_count=warn_categories.count('ConvergenceWarning'),n_iter=iterations,
                fitted_params=details,fit_count=len(keys))
    return base,arrays

def main():
    if len(sys.argv)!=2:raise SystemExit('usage: worker.py job.json')
    j=json.loads(Path(sys.argv[1]).read_text());rec,arrays=run_job(j)
    output=Path(j['out']);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise SystemExit('STOP refusing to overwrite existing worker result '+str(output))
    if arrays is not None:
        p=output.with_suffix('.npz');np.savez_compressed(p,**arrays)
        rec['prediction_archive_sha256']=sha(p)
    tmp=output.with_suffix('.tmp');tmp.write_text(json.dumps(rec,indent=2,default=str)+'\n');os.replace(tmp,output)
    print(rec['job_id'],rec['status'])
if __name__=='__main__':main()
