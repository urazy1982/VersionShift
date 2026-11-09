#!/usr/bin/env python3
"""Frozen-input G2-D holdout47 runner. Writes only to a new --out directory."""
import argparse, hashlib, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np

ALGOS=('LR','SVC','RF','kNN')
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canon(x):
    return json.dumps(x,sort_keys=True,separators=(',',':'))
def run(a):
    here=Path(__file__).resolve().parent
    worker=here/'worker.py'
    plan_dir=Path(a.plan_dir).resolve()
    selection_path=Path(a.selection).resolve()
    out=Path(a.out).resolve()
    assert out != plan_dir and plan_dir not in out.parents, 'output cannot be within frozen plan'
    assert out != here and here not in out.parents, 'output cannot be within source'
    plan_path=plan_dir/'PLAN.json'
    plan=json.loads(plan_path.read_text())
    selection=json.loads(selection_path.read_text())
    plan_sha=sha(plan_path)
    if selection['plan_sha256']!=plan_sha: raise SystemExit('FAIL: pilot selection plan SHA mismatch')
    selected=selection['stage_b_datasets']
    if len(selected)!=16 or len(set(selected))!=16: raise SystemExit('FAIL: original pilot selection must contain 16 unique tasks')
    all_ds={d['name']:d for d in plan['datasets']}
    if len(all_ds)!=63 or len(plan['datasets'])!=63: raise SystemExit('FAIL: original eligible frame must contain exactly 63 unique datasets')
    if not set(selected) <= set(all_ds): raise SystemExit('FAIL: pilot IDs missing from original plan')
    chosen={name:d for name,d in all_ds.items() if name not in set(selected)}
    if len(chosen)!=47: raise SystemExit('FAIL: expected 47 untouched datasets')
    r=subprocess.run([a.python,'-I','-c','import sklearn;print(sklearn.__version__)'],capture_output=True,text=True,check=True)
    if r.stdout.strip()!='0.22.2.post1': raise SystemExit('FAIL: expected scikit-learn 0.22.2.post1, got '+r.stdout)
    lock=dict(purpose='VersionShift v2.2 G2-D holdout47 semantic restoration',plan_sha256=plan_sha,
              pilot_selection_sha256=sha(selection_path),excluded_pilot_datasets=sorted(selected),selected_datasets=sorted(chosen),
              worker_sha256=sha(worker),runner_sha256=sha(__file__),python=a.python,
              sklearn='0.22.2.post1',algorithms=list(ALGOS),policy='R',expected_jobs=188,
              expected_folds=5640, workers=a.workers)
    lock_path=out/'RUN_LOCK.json'
    if lock_path.exists():
        if canon(json.loads(lock_path.read_text()))!=canon(lock):raise SystemExit('FAIL: run lock changed')
    else:
        if out.exists() and any(out.iterdir()):raise SystemExit('FAIL: output directory not empty')
        (out/'jobs').mkdir(parents=True,exist_ok=True)
        (out/'results').mkdir(exist_ok=True)
        lock_path.write_text(json.dumps(lock,indent=2,sort_keys=True))
    # All planned jobs have unique output prefixes, so they can run in parallel.
    # Only this coordinator writes the final CSV and summary after all jobs pass.
    jobs=[]
    for dname in sorted(chosen):
        d=chosen[dname]
        for x,y in [('dataset_npz','dataset_sha256'),('folds_npz','folds_sha256')]:
            if sha(plan_dir/d[x])!=d[y]:raise SystemExit('FAIL: frozen artifact mismatch '+d[x])
        for algo in ALGOS:
            jid='G2RESTORE__%s__%s__R'%(dname,algo)
            op=out/'results'/jid
            jp=out/'jobs'/(jid+'.json')
            j=dict(id=jid,release='G2RESTORE',dataset=dname,algorithm=algo,policy='R',
                   expected_sklearn='0.22.2.post1',dataset_npz=str(plan_dir/d['dataset_npz']),
                   folds_npz=str(plan_dir/d['folds_npz']),fold_keys=d['fold_keys'],seeds=d['seeds'],
                   out_prefix=str(op),pinned=None,translations=None,pin_source_sklearn=None)
            j['job_sha256']=hashlib.sha256(canon(j).encode()).hexdigest()
            if jp.exists() and json.loads(jp.read_text())!=j:raise SystemExit('FAIL: job file changed '+jid)
            if not jp.exists():jp.write_text(json.dumps(j,sort_keys=True))
            jobs.append((dname,algo,jid,op,jp,j,d))

    def run_one(spec):
        dname,algo,jid,op,jp,j,d=spec
        rec=Path(str(op)+'.json'); pred=Path(str(op)+'.npz')
        if rec.exists():
            obj=json.loads(rec.read_text())
            if obj.get('status')!='ok' or not pred.exists() or sha(pred)!=obj.get('npz_sha256'):
                raise RuntimeError('FAIL: existing result invalid; stop, do not overwrite '+jid)
        else:
            if pred.exists():raise RuntimeError('FAIL: orphan prediction file '+str(pred))
            p=subprocess.run([a.python,'-I','-B',str(worker),str(jp)],capture_output=True,text=True,timeout=a.timeout)
            if p.returncode:raise RuntimeError('FAIL: worker process '+jid+' '+p.stderr[-800:])
            obj=json.loads(rec.read_text())
            if obj.get('status')!='ok':raise RuntimeError('FAIL: worker '+jid+' '+str(obj.get('error') or obj.get('detail')))
        if obj.get('worker_sha256')!=lock['worker_sha256'] or obj.get('job_sha256')!=j['job_sha256']:
            raise RuntimeError('FAIL: record provenance mismatch '+jid)
        if len(obj.get('ba',[]))!=30 or obj.get('fold_keys')!=d['fold_keys']:
            raise RuntimeError('FAIL: fold coverage '+jid)
        if sha(pred)!=obj.get('npz_sha256'):
            raise RuntimeError('FAIL: prediction SHA256 mismatch '+jid)
        z=np.load(pred,allow_pickle=False)
        y=np.load(plan_dir/d['dataset_npz'],allow_pickle=False)['y']
        folds=np.load(plan_dir/d['folds_npz'],allow_pickle=False)
        k=int(y.max())+1
        rows=[]
        for key,score in zip(d['fold_keys'],obj['ba']):
            te=folds[key+'_test']; predy=z[key+'_pred']
            if len(predy)!=d['test_sizes'][key]:raise RuntimeError('FAIL: prediction length '+jid+' '+key)
            again=float(np.mean([np.mean(predy[y[te]==c]==c) for c in range(k) if np.any(y[te]==c)]))
            if abs(again-score)>1e-12:raise RuntimeError('FAIL: score replay '+jid+' '+key)
            rno,fno=key[1:].split('_f')
            rows.append((dname,algo,int(rno),int(fno),score,hashlib.sha256(predy.tobytes()).hexdigest()[:16]))
        return jid,rows

    completed={}
    errors=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(run_one,j):j[2] for j in jobs}
        for future in as_completed(futures):
            try:
                jid,rows=future.result()
                completed[jid]=rows
                print('%d/188 %s PASS'%(len(completed),jid),flush=True)
            except Exception as exc:
                errors.append('%s: %s'%(futures[future],exc))
                print('FAIL: '+errors[-1],file=sys.stderr,flush=True)
    if errors:
        raise SystemExit('FAIL: %d jobs; results preserved for inspection/resume.\n%s'%(len(errors),'\n'.join(errors[:15])))
    results=[]
    for dname,algo,jid,op,jp,j,d in jobs:
        results.extend(completed[jid])
    if len(results)!=5640:
        raise SystemExit('FAIL: unexpected fold total '+str(len(results)))
    import csv
    with (out/'G2_FOLD_SCORES.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['dataset','algo','rep','fold','ba','pred_sha']);w.writerows(results)
    report=dict(status='PASS',jobs=len(results)//30,folds=len(results),datasets=len(chosen),
                source_sha256=lock['worker_sha256'],plan_sha256=plan_sha,
                note='G2-D holdout47 restoration fits only; comparisons require separate analysis')
    (out/'G2_SUMMARY.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan-dir',required=True);p.add_argument('--selection',required=True)
    p.add_argument('--out',required=True);p.add_argument('--python',required=True);p.add_argument('--timeout',type=int,default=7200)
    p.add_argument('--workers',type=int,choices=range(1,17),default=8)
    run(p.parse_args())
