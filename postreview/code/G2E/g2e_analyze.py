#!/usr/bin/env python3
"""Compare G2E diagnostic folds with original archived D-policy folds; no model execution."""
import argparse,csv,gzip,io,json,zipfile
from collections import defaultdict
from pathlib import Path

def key(r):return (r['dataset'],r['algo'],int(float(r['rep'])),int(float(r['fold'])))
def main(a):
    result_dir=Path(a.results); report=json.loads((result_dir/'G2E_SUMMARY.json').read_text())
    with (result_dir/'G2E_FOLD_SCORES.csv').open(newline='') as f:interventions=list(csv.DictReader(f))
    targets={r['dataset'] for r in interventions}
    if not targets:
        print('No successful interventions; inspect G2E_SUMMARY.json');return
    baseline={}
    member='VersionShift/results/STAGE_C_CPU_COMPLETION/merged/POS.csv.gz'
    with zipfile.ZipFile(a.zip) as z:
        with gzip.GzipFile(fileobj=io.BytesIO(z.read(member))) as gz:
            for r in csv.DictReader(io.TextIOWrapper(gz)):
                if r['status']!='ok' or r['policy']!='D' or r['dataset'] not in targets or r['algo'] not in ('GB','HGB'):continue
                k=(r['release'],)+key(r)
                if k in baseline:raise ValueError('Duplicate baseline '+str(k))
                baseline[k]=r
    breakdown={}
    for arm,refver,algo in [('GB_NEW_PRESORT_AUTO','0.22.2.post1','GB'),('HGB_OLD_MAXBINS255','0.21.3','HGB')]:
        rows=[r for r in interventions if r['arm']==arm]
        mismatch=0;score_mismatch=0;changes=[];delta_by_dataset=defaultdict(list)
        for r in rows:
            k=(refver,)+key(r)
            if k not in baseline:raise ValueError('Missing baseline '+str(k))
            b=baseline[k];d=100*(float(r['ba'])-float(b['ba']))
            delta_by_dataset[r['dataset']].append(d)
            if abs(d)>1e-10:score_mismatch+=1
            if r['pred_sha']!=b['pred_sha']:mismatch+=1
            if abs(d)>1e-10 or r['pred_sha']!=b['pred_sha']:
                changes.append(dict(dataset=r['dataset'],rep=r['rep'],fold=r['fold'],delta_pp=d,
                                    baseline_sha=b['pred_sha'],intervention_sha=r['pred_sha']))
        completed_ds=sorted(delta_by_dataset)
        breakdown[arm]=dict(status='NOT_RUN' if not rows else 'ANALYZED',n_folds=len(rows),n_datasets=len(completed_ds),
            score_mismatch_folds=score_mismatch,hash_mismatch_folds=mismatch,
            mean_signed_score_change_pp=(sum(sum(v) for v in delta_by_dataset.values())/len(rows) if rows else None),
            per_dataset_mean_changes_pp={k:sum(v)/len(v) for k,v in sorted(delta_by_dataset.items())},
            mismatch_examples=changes[:15],compare_against=refver+' shipped defaults')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    (out/'G2E_ANALYSIS.json').write_text(json.dumps({'execution_status':report['status'],'arms':breakdown},indent=2))
    with (out/'G2E_dataset_changes.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['arm','dataset','mean_signed_delta_pp'])
        for arm,d in breakdown.items():
            for ds,v in d['per_dataset_mean_changes_pp'].items():w.writerow([arm,ds,v])
    for arm,d in breakdown.items():
        print(arm,'folds',d['n_folds'],'score mismatches',d['score_mismatch_folds'],
              'prediction mismatches',d['hash_mismatch_folds'],'mean change pp',d['mean_signed_score_change_pp'])
    print('Saved',out/'G2E_ANALYSIS.json')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--zip',required=True);p.add_argument('--results',required=True);p.add_argument('--out',required=True)
    main(p.parse_args())
