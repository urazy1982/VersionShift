#!/usr/bin/env python3
"""G3-E: fill only missing HGB explicit-map cells; isolated, 8-way resumable execution.
Historical plan and original v2.1 results are read-only. Python 3.8+ controller.
"""
import argparse,csv,hashlib,json,os,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(o):return json.dumps(o,sort_keys=True,separators=(',',':'))
THREAD_KEYS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')
VERS=('0.21.3','0.22.2.post1')

def main(a):
    root=Path(__file__).resolve().parent;worker=root/'worker.py';plan_dir=Path(a.plan_dir).resolve();out=Path(a.out).resolve()
    if not worker.is_file():raise SystemExit('Missing worker.py')
    if out==plan_dir or plan_dir in out.parents or out==root or root in out.parents:raise SystemExit('Unsafe output directory')
    plan_path=plan_dir/'PLAN.json';plan=json.loads(plan_path.read_text());planhash=sha(plan_path)
    selection=json.loads(Path(a.selection).read_text())
    if selection.get('plan_sha256')!=planhash:raise SystemExit('Frozen plan-selection mismatch')
    pilot=set(selection.get('stage_b_datasets',[]))
    if len(pilot)!=16:raise SystemExit('Pilot selection not 16 tasks')
    all_ds={d['name']:d for d in plan['datasets']}
    if len(all_ds)!=63 or not pilot<=set(all_ds):raise SystemExit('Frozen 63-task frame mismatch')
    chosen=sorted(pilot if a.scope=='pilot' else (set(all_ds)-pilot if a.scope=='holdout' else set(all_ds)))
    versions={'0.21.3':str(Path(a.old_python).absolute()),'0.22.2.post1':str(Path(a.new_python).absolute())}
    for ver,py in versions.items():
        probe=subprocess.run([py,'-I','-c','import sklearn;print(sklearn.__version__)'],capture_output=True,text=True)
        if probe.returncode or probe.stdout.strip()!=ver:raise SystemExit('Environment failed: %s %s'%(ver,probe.stderr[-300:]))
    lock=dict(purpose='G3-E post-review six-estimator E2 comparison: only HGB missing cells',
              plan_sha256=planhash,selection_sha256=sha(a.selection),worker_sha256=sha(worker),
              runner_sha256=sha(__file__),scope=a.scope,datasets=chosen,versions=versions,workers=a.workers,
              job_count=len(chosen)*2,configuration='HGB: max_iter=200, learning_rate=0.1, n_iter_no_change=None, max_bins=255')
    lockfile=out/'RUN_LOCK.json'
    if lockfile.exists():
        if canonical(json.loads(lockfile.read_text()))!=canonical(lock):raise SystemExit('Run lock mismatch; do not overwrite')
    else:
        if out.exists() and any(out.iterdir()):raise SystemExit('Output directory not empty; choose a new --out')
        (out/'jobs').mkdir(parents=True,exist_ok=True);(out/'results').mkdir(parents=True,exist_ok=True)
        lockfile.write_text(json.dumps(lock,indent=2,sort_keys=True))
    specs=[]
    for ds in chosen:
        d=all_ds[ds]
        for name,hashname in (('dataset_npz','dataset_sha256'),('folds_npz','folds_sha256')):
            if sha(plan_dir/d[name])!=d[hashname]:raise SystemExit('Tampered frozen artifact '+d[name])
        for ver in VERS:
            jid='G3E2__%s__%s__HGB'%(ver.replace('.','_'),ds)
            pre=out/'results'/jid;jobfile=out/'jobs'/(jid+'.json')
            job=dict(id=jid,release=ver,dataset=ds,algorithm='HGB',policy='E',expected_sklearn=ver,
                dataset_npz=str(plan_dir/d['dataset_npz']),folds_npz=str(plan_dir/d['folds_npz']),
                fold_keys=d['fold_keys'],seeds=d['seeds'],out_prefix=str(pre),pinned=None,
                translations=None,pin_source_sklearn=None)
            job['job_sha256']=hashlib.sha256(canonical(job).encode()).hexdigest()
            if jobfile.exists():
                if json.loads(jobfile.read_text())!=job:raise SystemExit('Job spec changed '+jid)
            else:jobfile.write_text(json.dumps(job,sort_keys=True))
            specs.append((jid,ver,d,pre,jobfile,job))
    def execute(spec):
        jid,ver,d,pre,jobfile,job=spec;rec=Path(str(pre)+'.json');npz=Path(str(pre)+'.npz')
        if not rec.exists():
            if npz.exists():raise RuntimeError('Orphan prediction file '+jid)
            env=os.environ.copy()
            for key in THREAD_KEYS:env[key]='1'
            run=subprocess.run([versions[ver],'-I','-B',str(worker),str(jobfile)],capture_output=True,text=True,env=env,timeout=a.timeout)
            if run.returncode:raise RuntimeError('Worker crashed '+jid+' '+run.stderr[-800:])
        obj=json.loads(rec.read_text())
        if obj.get('status')!='ok':raise RuntimeError('Model job not successful '+jid+' '+str(obj.get('error') or obj.get('detail')))
        if obj.get('worker_sha256')!=lock['worker_sha256'] or obj.get('job_sha256')!=job['job_sha256']:
            raise RuntimeError('Source/job mismatch '+jid)
        if not npz.is_file() or sha(npz)!=obj.get('npz_sha256'):raise RuntimeError('Prediction artifact changed '+jid)
        if obj.get('fold_keys')!=d['fold_keys'] or len(obj.get('ba',[]))!=30:raise RuntimeError('Fold coverage '+jid)
        params=obj.get('constructor') or {}
        if params.get('n_iter_no_change','__missing__') is not None or params.get('max_bins')!=255 or params.get('max_iter')!=200:
            raise RuntimeError('E2 HGB map not applied '+jid+' '+repr(params))
        rows=[]
        with np.load(plan_dir/d['dataset_npz'],allow_pickle=False) as data, np.load(plan_dir/d['folds_npz'],allow_pickle=False) as foldz, np.load(npz,allow_pickle=False) as predz:
            y=data['y']; k=int(y.max())+1
            for key,score in zip(d['fold_keys'],obj['ba']):
                te=foldz[key+'_test']; pred=predz[key+'_pred']
                if len(pred)!=d['test_sizes'][key]:raise RuntimeError('Prediction length mismatch '+jid)
                again=float(np.mean([np.mean(pred[y[te]==c]==c) for c in range(k) if np.any(y[te]==c)]))
                if abs(again-score)>1e-12:raise RuntimeError('Score replay mismatch '+jid)
                r,f=key[1:].split('_f')
                rows.append((ver,d['name'],'HGB','E2',int(r),int(f),score,hashlib.sha256(pred.tobytes()).hexdigest()[:16]))
        return rows
    found={};errors=[]
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futures={ex.submit(execute,s):s[0] for s in specs}
        for future in as_completed(futures):
            jid=futures[future]
            try:
                found[jid]=future.result();print('%d/%d %s ok'%(len(found),len(specs),jid),flush=True)
            except Exception as exc:
                errors.append('%s: %s'%(jid,exc));print(errors[-1],file=sys.stderr,flush=True)
    if errors:raise SystemExit('FAILED: '+ '; '.join(errors[:6]))
    rows=[row for s in specs for row in found[s[0]]]
    with (out/'G3E_HGB_FOLDS.csv').open('w',newline='') as fp:
        w=csv.writer(fp);w.writerow(('release','dataset','algo','policy','rep','fold','ba','pred_sha'));w.writerows(rows)
    report=dict(status='PASS',scope=a.scope,jobs=len(specs),folds=len(rows),datasets=len(chosen),
                worker_sha256=sha(worker),plan_sha256=planhash)
    (out/'G3E_SUMMARY.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--plan-dir',required=True);p.add_argument('--selection',required=True);p.add_argument('--out',required=True)
    p.add_argument('--old-python',required=True);p.add_argument('--new-python',required=True)
    p.add_argument('--workers',type=int,default=8,choices=range(1,17));p.add_argument('--timeout',type=int,default=7200)
    p.add_argument('--scope',choices=('pilot','holdout','all'),default='pilot')
    main(p.parse_args())
