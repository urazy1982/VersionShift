#!/usr/bin/env python3
"""VersionShift G2-E diagnostic pilot; isolated output; eight parallel subprocess workers.
No overwriting, reads original immutable frozen PLAN/selection and data artifacts.
Does NOT claim semantic equivalence from constructor compatibility alone.
"""
import argparse, csv, hashlib, json, os, subprocess, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np

ARMS = (('GB_NEW_PRESORT_AUTO','GB','0.22.2.post1'),('HGB_OLD_MAXBINS255','HGB','0.21.3'))
THREAD_ENV = ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canon(o):return json.dumps(o,sort_keys=True,separators=(',',':'))

def main(a):
    here=Path(__file__).resolve().parent
    worker=here/'worker.py'
    plan_dir=Path(a.plan_dir).resolve()
    plan_file=plan_dir/'PLAN.json'
    sel_file=Path(a.selection).resolve()
    out=Path(a.out).resolve()
    assert out != plan_dir and plan_dir not in out.parents
    assert out != here and here not in out.parents
    assert out != sel_file.parent or out != sel_file
    plan=json.loads(plan_file.read_text());selection=json.loads(sel_file.read_text());plan_sha=sha(plan_file)
    if selection.get('plan_sha256')!=plan_sha: raise SystemExit('FAIL: selection/plan hash mismatch')
    ids=selection.get('stage_b_datasets',[])
    if len(ids)!=16 or len(set(ids))!=16:raise SystemExit('FAIL: original pilot must contain 16 unique datasets')
    dataset_index={d['name']:d for d in plan['datasets']}
    if not set(ids)<=set(dataset_index):raise SystemExit('FAIL: missing frozen dataset IDs')
    versions={'0.21.3':str(Path(a.old_python).resolve()),'0.22.2.post1':str(Path(a.new_python).resolve())}
    for ver,py in versions.items():
        p=subprocess.run([py,'-I','-c','import sklearn;print(sklearn.__version__)'],capture_output=True,text=True,check=True)
        if p.stdout.strip()!=ver:raise SystemExit('FAIL: environment mismatch: '+ver+' '+p.stdout)
    lock=dict(purpose='G2E: two diagnostic interventions, no restored-HGB claim', plan_sha256=plan_sha,
              selection_sha256=sha(sel_file), selection=sorted(ids), source_sha256=sha(worker),
              runner_sha256=sha(__file__), versions=versions, arms=[list(x) for x in ARMS],
              expected_jobs=32, expected_folds_if_all_pass=960, workers=a.workers)
    lockfile=out/'RUN_LOCK.json'
    if lockfile.exists():
        if canon(json.loads(lockfile.read_text()))!=canon(lock):raise SystemExit('FAIL: immutable run lock mismatch')
    else:
        if out.exists() and any(out.iterdir()):raise SystemExit('FAIL: output path already populated')
        (out/'jobs').mkdir(parents=True,exist_ok=True);(out/'results').mkdir(parents=True,exist_ok=True)
        lockfile.write_text(json.dumps(lock,indent=2,sort_keys=True))
    specs=[]
    for ds in sorted(ids):
        d=dataset_index[ds]
        for x,y in [('dataset_npz','dataset_sha256'),('folds_npz','folds_sha256')]:
            if sha(plan_dir/d[x])!=d[y]:raise SystemExit('FAIL: immutable artifact hash mismatch: '+str(d[x]))
        for arm,algo,ver in ARMS:
            jid='G2E__%s__%s__%s'%(arm,ds,algo)
            prefix=out/'results'/jid
            jobpath=out/'jobs'/(jid+'.json')
            job=dict(id=jid, release=ver, dataset=ds, algorithm=algo, policy='G2E',
                     expected_sklearn=ver,dataset_npz=str(plan_dir/d['dataset_npz']),
                     folds_npz=str(plan_dir/d['folds_npz']),fold_keys=d['fold_keys'],seeds=d['seeds'],
                     out_prefix=str(prefix),pinned=None,translations=None,pin_source_sklearn=None)
            job['job_sha256']=hashlib.sha256(canon(job).encode()).hexdigest()
            if jobpath.exists() and json.loads(jobpath.read_text())!=job:raise SystemExit('FAIL: job changed '+jid)
            if not jobpath.exists():jobpath.write_text(json.dumps(job,sort_keys=True))
            specs.append((jid,arm,algo,ver,d,prefix,jobpath,job))
    def one(s):
        jid,arm,algo,ver,d,prefix,jobpath,job=s
        rec=Path(str(prefix)+'.json');npz=Path(str(prefix)+'.npz')
        if not rec.exists():
            if npz.exists():raise RuntimeError('orphan prediction bytes '+jid)
            env=os.environ.copy()
            for key in THREAD_ENV:env[key]='1'
            proc=subprocess.run([versions[ver],'-I','-B',str(worker),str(jobpath)],env=env,
                                capture_output=True,text=True,timeout=a.timeout)
            if proc.returncode:raise RuntimeError('worker crash '+jid+' '+proc.stderr[-500:])
        obj=json.loads(rec.read_text())
        if obj.get('worker_sha256')!=lock['source_sha256'] or obj.get('job_sha256')!=job['job_sha256']:
            raise RuntimeError('provenance mismatch '+jid)
        status=obj.get('status')
        if status!='ok':
            if status not in ('fit_failed','not_portable'):raise RuntimeError('unexpected status '+jid+' '+str(status))
            if npz.exists():raise RuntimeError('failed job has prediction bytes '+jid)
            return jid,status,[],obj.get('error') or obj.get('detail') or 'unknown'
        if not npz.exists() or sha(npz)!=obj.get('npz_sha256'):raise RuntimeError('prediction hash mismatch '+jid)
        if obj.get('fold_keys')!=d['fold_keys'] or len(obj.get('ba',[]))!=30:raise RuntimeError('fold coverage mismatch '+jid)
        with np.load(plan_dir/d['dataset_npz'],allow_pickle=False) as data:y=data['y']
        with np.load(plan_dir/d['folds_npz'],allow_pickle=False) as folds, np.load(npz,allow_pickle=False) as predictions:
            k=int(y.max())+1;rows=[]
            for foldkey,ba in zip(d['fold_keys'],obj['ba']):
                te=folds[foldkey+'_test'];pred=predictions[foldkey+'_pred']
                if len(pred)!=d['test_sizes'][foldkey]:raise RuntimeError('prediction length '+jid)
                score=float(np.mean([np.mean(pred[y[te]==c]==c) for c in range(k) if np.any(y[te]==c)]))
                if abs(score-ba)>1e-12:raise RuntimeError('score replay mismatch '+jid)
                rep,fold=foldkey[1:].split('_f')
                rows.append((arm,ver,d['name'],algo,int(rep),int(fold),ba,hashlib.sha256(pred.tobytes()).hexdigest()[:16]))
        return jid,status,rows,''
    results={};errors=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futs={pool.submit(one,s):s[0] for s in specs}
        for fut in as_completed(futs):
            jid=futs[fut]
            try:
                results[jid]=fut.result()
                print('%d/32 %s %s'%(len(results),jid,results[jid][1]),flush=True)
            except Exception as exc:
                errors.append(jid+': '+str(exc));print('ERROR '+errors[-1],file=sys.stderr,flush=True)
    if errors:raise SystemExit('FAIL: worker validation errors: '+'; '.join(errors[:8]))
    rows=[];diagnostics=[]
    for s in specs:
        jid=s[0];_,status,r,reason=results[jid]
        rows.extend(r);diagnostics.append(dict(id=jid,arm=s[1],dataset=s[4]['name'],status=status,reason=reason))
    with (out/'G2E_FOLD_SCORES.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['arm','release','dataset','algo','rep','fold','ba','pred_sha']);w.writerows(rows)
    n_success=sum(d['status']=='ok' for d in diagnostics)
    report=dict(status='PASS' if n_success==32 else 'PARTIAL_DIAGNOSTIC', jobs_planned=32,jobs_ok=n_success,
                jobs_unavailable=32-n_success,folds=len(rows),datasets=16,worker_sha256=lock['source_sha256'],
                plan_sha256=plan_sha,warning='Experimental diagnostic; do not interpret nonzero errors as failed original baselines',
                jobs=diagnostics)
    (out/'G2E_SUMMARY.json').write_text(json.dumps(report,indent=2))
    print('G2E status=%s, successful jobs=%d/32, folds=%d'%(report['status'],n_success,len(rows)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan-dir',required=True);p.add_argument('--selection',required=True)
    p.add_argument('--out',required=True);p.add_argument('--old-python',required=True);p.add_argument('--new-python',required=True)
    p.add_argument('--workers',type=int,default=8,choices=range(1,17));p.add_argument('--timeout',type=int,default=7200)
    main(p.parse_args())
