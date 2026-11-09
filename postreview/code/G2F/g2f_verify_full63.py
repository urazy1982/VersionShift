#!/usr/bin/env python3
"""Independent G2E/G2F full-frame diagnostic audit. Reads immutable baseline ZIP.
No model execution; GB and HGB interventions are DIFFERENT within-runtime controls.
"""
import argparse, csv, gzip, hashlib, io, json, math, statistics, zipfile
from collections import defaultdict
from pathlib import Path

ARMS={
 'GB_NEW_PRESORT_AUTO':('GB','0.22.2.post1'),
 'HGB_OLD_MAXBINS255':('HGB','0.21.3')
}
BASE_MEMBER='VersionShift/results/STAGE_C_CPU_COMPLETION/merged/POS.csv.gz'
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def key(r):return (r['dataset'],r['algo'],int(float(r['rep'])),int(float(r['fold'])))
def load_intervention(path):
    with open(path,newline='') as f: rows=list(csv.DictReader(f))
    need={'arm','release','dataset','algo','rep','fold','ba','pred_sha'}
    if not rows or not need.issubset(rows[0]):raise ValueError('Invalid intervention CSV '+str(path))
    out={}
    for r in rows:
        arm=r['arm']; k=(arm,)+key(r)
        if arm not in ARMS:raise ValueError('Unknown arm '+arm)
        if (r['algo'],r['release'])!=ARMS[arm]:raise ValueError('Wrong arm release '+str(k))
        if k in out:raise ValueError('Duplicate intervention fold '+str(k))
        ba=float(r['ba'])
        if not math.isfinite(ba) or not 0<=ba<=1 or len(r['pred_sha'])!=16:raise ValueError('Invalid score/hash '+str(k))
        out[k]=r
    return out

def main(a):
    z=Path(a.zip); pilot=Path(a.pilot); holdout=Path(a.holdout); out=Path(a.out)
    if not z.exists():raise SystemExit('Missing baseline ZIP')
    p=load_intervention(pilot/'G2E_FOLD_SCORES.csv')
    h=load_intervention(holdout/'G2F_FOLD_SCORES.csv')
    ps=json.loads((pilot/'G2E_SUMMARY.json').read_text())
    hs=json.loads((holdout/'G2F_SUMMARY.json').read_text())
    if ps['status']!='PASS' or hs['status']!='PASS':raise ValueError('Pilot/holdout run not PASS')
    if ps['worker_sha256']!=hs['worker_sha256']:raise ValueError('Different worker source')
    if ps['plan_sha256']!=hs['plan_sha256']:raise ValueError('Different frozen plan')
    if len(p)!=960 or len(h)!=2820:raise ValueError('Wrong intervention fold counts: %s %s'%(len(p),len(h)))
    if set(p)&set(h):raise ValueError('Pilot and holdout overlap')
    combined={**p,**h}
    if len(combined)!=3780:raise ValueError('Expected 3780 fold rows')
    all_ds=set(r[1] for r in combined)
    if len(all_ds)!=63:raise ValueError('Expected 63 datasets, got '+str(len(all_ds)))
    counts=defaultdict(set)
    for (arm,ds,algo,rep,fold) in combined:
        counts[(arm,ds)].add((rep,fold))
    if len(counts)!=126 or any(len(x)!=30 or x!={(r,f) for r in range(6) for f in range(5)} for x in counts.values()):
        raise ValueError('Fold coverage invalid')
    baselines={}
    with zipfile.ZipFile(z) as archive:
        with gzip.GzipFile(fileobj=io.BytesIO(archive.read(BASE_MEMBER))) as stream:
            for row in csv.DictReader(io.TextIOWrapper(stream)):
                if row['policy']!='D' or row['status']!='ok' or row['algo'] not in ('GB','HGB') or row['dataset'] not in all_ds:continue
                k=(row['release'],)+key(row)
                if k in baselines:raise ValueError('Duplicate baseline '+str(k))
                baselines[k]=row
    fold_changes=defaultdict(list); dataset_out=[]; arm_agg={}; missing=[]
    for (arm,ds,algo,rep,fold),r in sorted(combined.items()):
        release=ARMS[arm][1]
        b=baselines.get((release,ds,algo,rep,fold))
        if b is None:
            missing.append((arm,ds,rep,fold));continue
        diff=100*(float(r['ba'])-float(b['ba']))
        fold_changes[(arm,ds)].append((diff,r['pred_sha']!=b['pred_sha']))
    if missing:raise ValueError('Missing %d baseline fold rows'%len(missing))
    for arm in ARMS:
        per=[]; all_fold=[]; all_abs=[]; hash_n=score_n=0
        for ds in sorted(all_ds):
            ar=fold_changes[(arm,ds)]
            if len(ar)!=30:raise ValueError('Not 30 paired folds '+arm+ds)
            ds_mean=statistics.mean(x for x,_ in ar)
            ds_abs=statistics.mean(abs(x) for x,_ in ar)
            n_hash=sum(bool(h) for _,h in ar)
            n_score=sum(abs(x)>1e-10 for x,_ in ar)
            dataset_out.append([arm,ds,ds_mean,ds_abs,n_score,n_hash])
            per.append(ds_mean)
            all_fold.extend(x for x,_ in ar)
            all_abs.extend(abs(x) for x,_ in ar)
            hash_n+=n_hash;score_n+=n_score
        arm_agg[arm]={
          'datasets':63,'paired_folds':1890,
          'prediction_hash_mismatch_folds':hash_n,
          'score_mismatch_folds':score_n,
          'mean_signed_change_pp':statistics.mean(all_fold),
          'mean_absolute_change_pp':statistics.mean(all_abs),
          'mean_absolute_dataset_mean_change_pp':statistics.mean(abs(v) for v in per),
          'max_absolute_dataset_mean_change_pp':max(abs(v) for v in per),
          'datasets_with_nonzero_mean_change':sum(abs(v)>1e-10 for v in per),
          'interpretation': 'within-runtime configuration intervention, not direct cross-release causal attribution'
        }
    out.mkdir(parents=True,exist_ok=True)
    with (out/'G2F_full63_dataset_changes.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['arm','dataset','mean_signed_change_pp','mean_absolute_fold_change_pp','score_mismatch_folds','hash_mismatch_folds']);w.writerows(dataset_out)
    report={'status':'PASS','scope':'63 datasets x 30 folds, two distinct within-runtime boosting interventions',
            'pilot_folds':len(p),'holdout_folds':len(h),'paired_folds_total':len(combined),
            'pilot_worker_sha256':ps['worker_sha256'],'holdout_worker_sha256':hs['worker_sha256'],
            'plan_sha256':ps['plan_sha256'],'baseline_archive_sha256':sha(z),'arms':arm_agg,
            'limitations':['GB compares new-release presort auto against new shipped defaults',
                           'HGB compares max_bins 255 versus 256 WITHIN old release',
                           'Cannot infer six-estimator default-vs-runtime decomposition from these interventions alone']}
    (out/'G2F_full63_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--zip',required=True)
    p.add_argument('--pilot',required=True)
    p.add_argument('--holdout',required=True)
    p.add_argument('--out',required=True)
    main(p.parse_args())
