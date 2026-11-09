#!/usr/bin/env python3
"""Independent G3-G HGB E2 job provenance and fold replay (read-only).

Run with --plan-dir and --pilot / --holdout G3E output folders.
Requires numpy and the frozen data/NPZ files; does not fit models or write into inputs.
"""
import argparse, hashlib, json
from pathlib import Path
from collections import Counter
import numpy as np

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()

def canonical(x): return json.dumps(x,sort_keys=True,separators=(',',':'))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--plan-dir',required=True)
    p.add_argument('--pilot',required=True)
    p.add_argument('--holdout',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    planroot=Path(a.plan_dir)
    plan=json.loads((planroot/'PLAN.json').read_text())
    planhash=sha(planroot/'PLAN.json')
    dataset_map={d['name']:d for d in plan['datasets']}
    assert len(dataset_map)==63
    problems=[];checked=0;folds=0;status=Counter();seen=set();params_seen={};srcs=set()
    for root,scope,number in [(Path(a.pilot),'pilot',16),(Path(a.holdout),'holdout',47)]:
        lock=json.loads((root/'RUN_LOCK.json').read_text())
        if lock.get('scope')!=scope or lock.get('plan_sha256')!=planhash:problems.append(f'{scope}: lock mismatch')
        dslist=lock.get('datasets',[])
        if len(dslist)!=number or len(set(dslist))!=number:problems.append(f'{scope}: dataset selection invalid')
        srcs.add(lock.get('worker_sha256'))
        expect=int(number*2)
        if lock.get('job_count')!=expect:problems.append(f'{scope}: job count in lock incorrect')
        for name in dslist:
            if name in seen:problems.append(f'dataset overlap: {name}')
            seen.add(name)
            if name not in dataset_map:
                problems.append(f'unknown dataset {name}');continue
            d=dataset_map[name]
            for key,expected in [('dataset_npz','dataset_sha256'),('folds_npz','folds_sha256')]:
                if sha(planroot/d[key])!=d[expected]:problems.append(f'{name}: corrupted {key}')
            with np.load(planroot/d['dataset_npz'],allow_pickle=False) as arr:
                y=arr['y'];nclass=int(y.max())+1
            with np.load(planroot/d['folds_npz'],allow_pickle=False) as x:
                fold_ix={key:x[key+'_test'].copy() for key in d['fold_keys']}
            for ver in ['0.21.3','0.22.2.post1']:
                jid='G3E2__%s__%s__HGB'%(ver.replace('.','_'),name)
                jf=root/'jobs'/(jid+'.json')
                rf=root/'results'/(jid+'.json')
                pf=root/'results'/(jid+'.npz')
                if not (jf.exists() and rf.exists() and pf.exists()):
                    problems.append(f'{jid}: missing job / record / prediction');continue
                job=json.loads(jf.read_text());rec=json.loads(rf.read_text());checked+=1
                status[rec.get('status')]+=1
                if job.get('id')!=jid or job.get('release')!=ver or job.get('algorithm')!='HGB' or job.get('policy')!='E':
                    problems.append(f'{jid}: job spec mismatch')
                if job.get('job_sha256')!=hashlib.sha256(canonical({k:v for k,v in job.items() if k!='job_sha256'}).encode()).hexdigest():
                    problems.append(f'{jid}: job spec SHA mismatch')
                if rec.get('job_sha256')!=job.get('job_sha256') or rec.get('worker_sha256')!=lock.get('worker_sha256'):
                    problems.append(f'{jid}: provenance hash mismatch')
                if rec.get('status')!='ok' or rec.get('npz_sha256')!=sha(pf):
                    problems.append(f'{jid}: run status / NPZ SHA mismatch')
                prm=rec.get('constructor',{});params_seen[jid]=prm
                wanted={'n_iter_no_change':None,'max_bins':255,'max_iter':200,'learning_rate':0.1}
                if any(prm.get(k,'UNSET')!=v for k,v in wanted.items()):problems.append(f'{jid}: E2 constructor differs')
                if rec.get('fold_keys')!=d['fold_keys'] or len(rec.get('ba',[]))!=30:
                    problems.append(f'{jid}: fold plan mismatch');continue
                with np.load(pf,allow_pickle=False) as pred:
                    for key,score in zip(d['fold_keys'],rec['ba']):
                        te=fold_ix[key];yp=pred[key+'_pred']
                        if len(yp)!=len(te):
                            problems.append(f'{jid}: prediction length {key}');continue
                        ba=float(np.mean([np.mean(yp[y[te]==c]==c) for c in range(nclass) if np.any(y[te]==c)]))
                        folds+=1
                        if abs(ba-score)>1e-12:problems.append(f'{jid}: fold replay mismatch {key}')
    if len(seen)!=63:problems.append(f'only {len(seen)} datasets covered')
    if checked!=126:problems.append(f'only {checked} jobs checked')
    if folds!=3780:problems.append(f'only {folds} folds checked')
    if len(srcs)!=1:problems.append('pilot and holdout source hashes differ')
    report={'status':'PASS' if not problems else 'FAIL','jobs_verified':checked,'folds_replayed':folds,
            'dataset_count':len(seen),'worker_sha256_values':sorted(str(x) for x in srcs),
            'statuses':dict(status),'errors':problems[:100],
            'note':'Validates archived job specifications, claimed model parameters, stored prediction arrays and numeric score replay. It does not refit models.'}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(report,indent=2,sort_keys=True))
    print(json.dumps(report,indent=2))
    if problems:raise SystemExit(1)

if __name__=='__main__':main()
