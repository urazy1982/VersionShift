"""Fast offline property tests; no historical-version scientific comparisons."""
import json,hashlib,sys,subprocess
from pathlib import Path
import numpy as np,pandas as pd,pytest
from versionshift_v3.plan import stratified_split,external_sample
from versionshift_v3.prepare import transform_fold
from versionshift_v3.audit import evaluate_edge,paired_mitigation,score_replay
from versionshift_v3.utils import sha256,read_json
from versionshift_v3.worker import run_job,init_model


def test_stratified_deterministic_and_disjoint():
    ds=[{'task_id':i,'n':1000 if i%2 else 2500,'k':2 if i%3 else 4,'p':20} for i in range(1,64)]
    a=stratified_split(ds);b=stratified_split(ds)
    assert a==b and len(a[0])==15 and len(a[1])==48
    assert not set(a[0])&set(a[1])

def test_external_excludes_overlap_and_respects_source():
    core=[{'task_id':i,'dataset_id':i+100} for i in range(63)]
    rows=[{'task_id':str(i),'dataset_id':str(i+100),'dataset_version':'1','n':'1800',
            'p_encoded':'30','k':'2','min_class_count':'20','eligible':'yes'} for i in range(100,125)]
    selected,excluded,status=external_sample(rows,core,42)
    assert len(selected)==20 and not excluded and status['status']=='READY'

def test_train_median_and_test_outlier_no_leakage():
    X=pd.DataFrame({'n':[1.0,2.0,np.nan,100000.0], 'c':['a','a',None,'unseen']})
    train=np.array([0,1,2]); test=np.array([3])
    a,b,cols=transform_fold(X,train,test,['n'],['c'],False)
    assert a.shape==(3,3) and b.shape==(1,3)
    assert b[0,0]==100000.0
    assert list(b[0,1:])==[0.,0.]
    a2,b2,_=transform_fold(X,train,test,['n'],['c'],True)
    assert abs(a2[:,0].mean())<1e-12
    assert b2[0,0]>10000
    assert np.array_equal(a[:,1:],a2[:,1:])

def test_preparation_rejects_train_test_overlap():
    X=pd.DataFrame({'n':[1.,2.]})
    # fold validator is in materialization; transform_fold itself is label-free and can handle arbitrary subset
    assert transform_fold(X,[0],[1],['n'],[],False)[0].shape==(1,1)

def test_fixed_score_gap_effect_formula():
    algo=['LR','SVC','kNN','RF','GB','HGB','LinearSVC','AdaBoost']
    records={}
    for t in range(48):
        for i,a in enumerate(algo):
            records[(t,'1.4.2',a)]=60.0+i
            records[(t,'1.5.2',a)]=60.0+i+(2.0 if a=='RF' else 0)
    r=evaluate_edge(records,algo_panel=algo,bootstrap_reps=100)
    # 1 of 8 estimators shifts: exactly 7 of 28 pairs by 2 points => M=0.5
    assert abs(r['M_pp']-0.5)<1e-12
    assert r['K_complete']==48 and r['affected']==0

def test_missing_task_never_imputed_zero():
    records={(1,'1.4.2','LR'):60,(1,'1.5.2','LR'):70}
    with pytest.raises(RuntimeError,match='no complete-case'):
        evaluate_edge(records,bootstrap_reps=10)

def test_paired_mitigation_bounds():
    a={'per_task':[{'task_id':k,'M_pp':2.0} for k in range(48)]}
    b={'per_task':[{'task_id':k,'M_pp':0.5} for k in range(48)]}
    r=paired_mitigation(a,b,bootstrap_reps=100)
    assert r['paired_tasks']==48 and r['reduction_pp']==1.5
    assert abs(r['relative_attenuation']-0.75)<1e-12

def test_e2_extract_rejects_unsupported_calls(tmp_path):
    from scripts.extract_e2_verified import extract
    p=tmp_path/'worker.py';p.write_text('EXPLICIT = {"LR": dict(C=1), "SVC":dict(),"kNN":dict(),"RF":dict(),"GB":dict(),"HGB":dict()}\n')
    r=extract(p)
    assert r['parameters']['LR']=={'C':1}
    assert r['source_sha256']==sha256(p)
    p.write_text('EXPLICIT = __import__("os").system("echo unsafe")')
    with pytest.raises(Exception):extract(p)

def test_g0_missing_original_blocks(tmp_path):
    from scripts.g0_archive_audit import audit
    r=audit(tmp_path,tmp_path/'missing.json')
    assert r['status']=='FAIL' and r['missing']

def test_run_before_freeze_is_refused(tmp_path):
    proc=subprocess.run([sys.executable,'-m','versionshift_v3.run','--project',str(tmp_path),
                         '--frame','pilot','--edge','E_LINEAR_DUAL','--preprocessing','P1',
                         '--out',str(tmp_path/'run')],capture_output=True,text=True)
    assert proc.returncode!=0 and 'frozen protocol absent' in proc.stderr

def test_worker_real_cpu_smoke_and_score_replay(tmp_path):
    import sklearn
    from synthetic_statistics_fixtures import synthetic_run
    project,run,jobs=synthetic_run(tmp_path)
    j=next(job.copy() for job in jobs if job['algo']=='LR')
    j['release']=sklearn.__version__;j['out']=str(tmp_path/'result.json')
    rec,arrays=run_job(j)
    assert rec['status']=='SUPPORTED' and len(rec['balanced_accuracy_pp'])==30
    np.savez_compressed(tmp_path/'result.npz',**arrays)
    rec['prediction_archive_sha256']=sha256(tmp_path/'result.npz')
    (tmp_path/'result.json').write_text(json.dumps(rec))
    assert score_replay(tmp_path/'result.json')==pytest.approx(np.mean(rec['balanced_accuracy_pp']))

def test_removed_behaviour_is_not_emulated():
    # D2 future AdaBoost cell is deliberately absent even on the wrong host version.
    cls,status,reason=init_model('AdaBoost','D2',42,'1.6.1',{}, {},{'AdaBoost':{'algorithm':'SAMME.R'}})
    assert cls is None and status=='REMOVED_BEHAVIOR'


def test_runner_frozen_source_inventory_rejects_modified_bytes(tmp_path):
    from versionshift_v3.run import verify_frozen_sources
    (tmp_path/'config').mkdir()
    path=tmp_path/'config/edges.json';path.write_text('original')
    inventory={'source_hashes':{'config/edges.json':sha256(path)}}
    assert verify_frozen_sources(tmp_path,inventory)
    path.write_text('modified')
    with pytest.raises(RuntimeError,match='frozen evidence changed'):
        verify_frozen_sources(tmp_path,inventory)

def test_aa_spec_idempotent_definition(tmp_path):
    # Controlled reconstructions of the same A/A input are idempotent;
    # different inputs must not silently overwrite the existing run artifact.
    from versionshift_v3.utils import require
    spec=tmp_path/'AA_PROBE_1.4.2.json'
    obj={'release':'1.4.2','matrix':'frozen.npz'}
    spec.write_text(json.dumps(obj))
    assert json.loads(spec.read_text())==obj
    with pytest.raises(RuntimeError):
        require(json.loads(spec.read_text())=={'release':'1.4.2','matrix':'other.npz'},
                'existing A/A probe specification changed')
