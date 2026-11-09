#!/usr/bin/env python3
"""Read-only frozen-input G2 pilot runner. Writes only beneath a new --out directory."""
import argparse, hashlib, json, os, subprocess, sys, time
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
    if len(selected)!=16 or len(set(selected))!=16: raise SystemExit('FAIL: pilot selection must contain 16 unique tasks')
    chosen={d['name']:d for d in plan['datasets'] if d['name'] in selected}
    if len(chosen)!=16: raise SystemExit('FAIL: 16 selected tasks not found in plan')
    r=subprocess.run([a.python,'-I','-c','import sklearn;print(sklearn.__version__)'],capture_output=True,text=True,check=True)
    if r.stdout.strip()!='0.22.2.post1': raise SystemExit('FAIL: expected scikit-learn 0.22.2.post1, got '+r.stdout)
    lock=dict(purpose='VersionShift v2.2 post-review G2 semantic restoration pilot',plan_sha256=plan_sha,
              selection_sha256=sha(selection_path),selected_datasets=sorted(selected),
              worker_sha256=sha(worker),runner_sha256=sha(__file__),python=a.python,
              sklearn='0.22.2.post1',algorithms=list(ALGOS),policy='R',expected_jobs=64,
              expected_folds=1920)
    lock_path=out/'RUN_LOCK.json'
    if lock_path.exists():
        if canon(json.loads(lock_path.read_text()))!=canon(lock):raise SystemExit('FAIL: run lock changed')
    else:
        if out.exists() and any(out.iterdir()):raise SystemExit('FAIL: output directory not empty')
        (out/'jobs').mkdir(parents=True,exist_ok=True)
        (out/'results').mkdir(exist_ok=True)
        lock_path.write_text(json.dumps(lock,indent=2,sort_keys=True))
    results=[]
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
            rec=Path(str(op)+'.json'); pred=Path(str(op)+'.npz')
            if rec.exists():
                obj=json.loads(rec.read_text())
                if obj.get('status')!='ok' or not pred.exists() or sha(pred)!=obj.get('npz_sha256'):
                    raise SystemExit('FAIL: existing result invalid; stop, do not overwrite '+jid)
            else:
                if pred.exists():raise SystemExit('FAIL: orphan prediction file '+str(pred))
                p=subprocess.run([a.python,'-I','-B',str(worker),str(jp)],capture_output=True,text=True,timeout=a.timeout)
                if p.returncode:raise SystemExit('FAIL: worker process '+jid+' '+p.stderr[-800:])
                obj=json.loads(rec.read_text())
                if obj.get('status')!='ok':raise SystemExit('FAIL: worker '+jid+' '+str(obj.get('error') or obj.get('detail')))
            if obj.get('worker_sha256')!=lock['worker_sha256'] or obj.get('job_sha256')!=j['job_sha256']:
                raise SystemExit('FAIL: record provenance mismatch '+jid)
            if len(obj.get('ba',[]))!=30 or obj.get('fold_keys')!=d['fold_keys']:
                raise SystemExit('FAIL: fold coverage '+jid)
            z=np.load(pred,allow_pickle=False)
            y=np.load(plan_dir/d['dataset_npz'],allow_pickle=False)['y']
            folds=np.load(plan_dir/d['folds_npz'],allow_pickle=False)
            k=int(y.max())+1
            for key,score in zip(d['fold_keys'],obj['ba']):
                te=folds[key+'_test']; predy=z[key+'_pred']
                if len(predy)!=d['test_sizes'][key]:raise SystemExit('FAIL: prediction length '+jid+' '+key)
                again=float(np.mean([np.mean(predy[y[te]==c]==c) for c in range(k) if np.any(y[te]==c)]))
                if abs(again-score)>1e-12:raise SystemExit('FAIL: score replay '+jid+' '+key)
                rno,fno=key[1:].split('_f')
                results.append((dname,algo,int(rno),int(fno),score,hashlib.sha256(predy.tobytes()).hexdigest()[:16]))
            print('%d/64 %s PASS'%(len(results)//30,jid),flush=True)
    import csv
    with (out/'G2_FOLD_SCORES.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['dataset','algo','rep','fold','ba','pred_sha']);w.writerows(results)
    report=dict(status='PASS',jobs=len(results)//30,folds=len(results),datasets=len(chosen),
                source_sha256=lock['worker_sha256'],plan_sha256=plan_sha,
                note='Within-new-runtime semantic restoration fits only; no baseline outcome comparison performed')
    (out/'G2_SUMMARY.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan-dir',required=True);p.add_argument('--selection',required=True)
    p.add_argument('--out',required=True);p.add_argument('--python',required=True);p.add_argument('--timeout',type=int,default=7200)
    run(p.parse_args())
